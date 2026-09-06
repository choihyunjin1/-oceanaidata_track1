"""Frozen source-only 11-feature covariance/CDF residual correction."""
import core as base
import numpy as np
from scipy.special import ndtr, ndtri
from sklearn.covariance import LedoitWolf


def actual_profile(frame):
    """Actual-depth piecewise public temperature profile; duplicate depths averaged."""
    nominal = frame.target_depth.to_numpy(float)
    actual = frame.target_actual_depth.to_numpy(float)
    target = np.where(np.isfinite(actual) & (actual > 0), actual, nominal)
    temperatures = frame[[f"temp_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    actuals = frame[[f"depth_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    nominals = frame[[f"nominal_{layer}" for layer in base.PUBLIC_LAYERS]].to_numpy(float)
    depths = np.where(np.isfinite(actuals) & (actuals > 0), actuals, nominals)
    interpolated, gradient = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for index in range(len(frame)):
        good = np.isfinite(temperatures[index]) & np.isfinite(depths[index])
        z, inv = np.unique(depths[index, good], return_inverse=True)
        if not len(z):
            continue
        temp = np.bincount(inv, weights=temperatures[index, good]) / np.bincount(inv)
        interpolated[index] = np.interp(target[index], z, temp)
        gradient[index] = 0.0
        if len(z) >= 2 and z[0] <= target[index] < z[-1]:
            left = int(np.searchsorted(z, target[index], side="right") - 1)
            gradient[index] = (temp[left + 1] - temp[left]) / (z[left + 1] - z[left])
    return target, interpolated, gradient


def physical_features(frame, c):
    if len(frame) != len(c) or not np.isfinite(c).all():
        raise ValueError("C3 requires aligned finite predictions")
    target, interp, gradient = actual_profile(frame)
    nominal = frame.target_depth.to_numpy(float)
    columns = [c, target, nominal, interp - frame.baseline.to_numpy(float), gradient]
    for variable, left, right in (("temp", 1, 5), ("temp", 1, 6), ("temp", 5, 6), ("psal", 1, 5), ("psal", 5, 6)):
        columns.append(frame[f"{variable}_{left}"].to_numpy(float) - frame[f"{variable}_{right}"].to_numpy(float))
    columns.append(target - nominal)
    result = np.column_stack(columns)
    if np.isinf(result).any():
        raise ValueError("infinite physical input")
    return result


def covariance_checks(model, x, residual):
    """Independent analytic Ledoit-Wolf/CDF verification; no estimator.fit call."""
    checks = {}
    zcols = []
    for index, col in enumerate([*x.T, residual]):
        marginal = np.sort(col[np.isfinite(col)])
        saved = model[f"marginal_{index}"] if index < 11 else model["response"]
        checks[f"cdf_{index}"] = bool(np.array_equal(marginal, saved))
        latent = np.zeros(len(col))
        available = np.isfinite(col)
        if len(marginal):
            probability = (np.searchsorted(marginal, col[available], "left")
                           + np.searchsorted(marginal, col[available], "right")) / (2 * len(marginal))
            latent[available] = ndtri(np.clip(probability, .5 / len(marginal), 1 - .5 / len(marginal)))
        zcols.append(latent)
    z = np.column_stack(zcols)
    centered = z - z.mean(axis=0)
    n, p = centered.shape
    gram = centered.T @ centered
    empirical = gram / n
    trace = np.sum(centered ** 2, axis=0) / n
    mu = float(trace.sum() / p)
    delta_raw = float(np.sum(gram ** 2) / n ** 2)
    squared = centered ** 2
    beta = float((np.sum(squared.T @ squared) / n - delta_raw) / (p * n))
    delta = float((delta_raw - 2 * mu * trace.sum() + p * mu ** 2) / p)
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    covariance = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(p)
    checks["location"] = bool(np.allclose(model["location"], z.mean(axis=0), rtol=0, atol=1e-12))
    checks["shrinkage"] = bool(np.isclose(model["shrinkage"], shrinkage, rtol=1e-10, atol=1e-12))
    checks["covariance"] = bool(np.allclose(model["covariance"], covariance, rtol=1e-10, atol=1e-12))
    assert all(checks.values()), "copula CDF/covariance provenance mismatch"
    return checks


def latent(values, sorted_values):
    result = np.zeros(len(values))
    present = np.isfinite(values)
    if not len(sorted_values):
        return result
    left = np.searchsorted(sorted_values, values[present], side="left")
    right = np.searchsorted(sorted_values, values[present], side="right")
    p = (left + right) / (2 * len(sorted_values))
    p = np.clip(p, 0.5 / len(sorted_values), 1 - 0.5 / len(sorted_values))
    result[present] = ndtri(p)
    return result


def fit_copula(x, residual):
    if len(x) != len(residual) or not np.isfinite(residual).all():
        raise ValueError("finite aligned training residual required")
    marginals = [np.sort(col[np.isfinite(col)]) for col in x.T]
    response = np.sort(residual)
    z = np.column_stack([latent(x[:, i], marginal) for i, marginal in enumerate(marginals)] + [latent(residual, response)])
    fitted = LedoitWolf().fit(z)
    return {"location": fitted.location_, "covariance": fitted.covariance_, "shrinkage": np.array(fitted.shrinkage_), "response": response, **{f"marginal_{i}": v for i, v in enumerate(marginals)}}


def predict_copula(model, x):
    d = x.shape[1]
    present = np.isfinite(x)
    for i in range(d):
        present[:, i] &= len(model[f"marginal_{i}"]) > 0
    z = np.column_stack([latent(x[:, i], model[f"marginal_{i}"]) for i in range(d)])
    locations, cov = model["location"], model["covariance"]
    nodes, weights = np.polynomial.hermite.hermgauss(15)
    support = (np.arange(len(model["response"])) + 0.5) / len(model["response"])
    result = np.empty(len(x))
    for pattern in np.unique(present, axis=0):
        rows = np.all(present == pattern, axis=1)
        observed = np.flatnonzero(pattern)
        if len(observed):
            beta = np.linalg.pinv(cov[np.ix_(observed, observed)], hermitian=True) @ cov[observed, d]
            mean = locations[d] + (z[rows][:, observed] - locations[observed]) @ beta
            variance = max(0.0, float(cov[d, d] - cov[d, observed] @ beta))
        else:
            mean, variance = np.full(int(rows.sum()), locations[d]), max(0.0, float(cov[d, d]))
        probability = ndtr(mean[:, None] + np.sqrt(2 * variance) * nodes[None, :])
        transformed = np.interp(probability, support, model["response"])
        result[rows] = transformed @ (weights / np.sqrt(np.pi))
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite correction")
    return result
