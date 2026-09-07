import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bracketb_test", ROOT / "scripts/p1_champion_bracketb_20260907_v1.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_original_plus_bracket_and_order():
    r = m.runtime()
    n = 120
    frame = r.pd.DataFrame(
        {
            "station": ["S-ORS"] * n,
            "year": [2025] * n,
            "layer": [1] * n,
            "time": r.pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul"),
            "temp": 15 + r.np.sin(r.np.arange(n) / 10),
            "psal": [32.0] * n,
            "depth": [5.0] * n,
        }
    )
    cfg = m.read(ROOT / "configs/experiments/p1_champion_bracketb_20260907_v1.json")
    base, new = m.bundle(r, frame, False, cfg), m.bundle(r, frame, True, cfg)
    assert base.frame.equals(new.frame[list(base.feature_columns)])
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    reverse = m.bundle(r, shuffled, True, cfg)
    cols = [c for c in new.frame if c.startswith("bracket_")]
    assert r.np.allclose(
        reverse.frame[cols].to_numpy()[::-1], new.frame[cols].to_numpy(), equal_nan=True
    )
    model = r.lgb.LGBMClassifier(
        n_estimators=2, num_leaves=3, min_child_samples=2, n_jobs=4, verbosity=-1
    )
    encoder = r.TabularEncoder().fit(new, r.np.arange(n))
    x = encoder.transform(new)
    model.fit(x, r.np.arange(n) % 2)
    assert r.np.isfinite(model.predict_proba(x)).all()


def test_config_fixed_original_decoder():
    cfg = m.read(ROOT / "configs/experiments/p1_champion_bracketb_20260907_v1.json")
    assert cfg["fits"] == 17
    assert cfg["seeds"] == [20260813, 20260829, 20260847]
    assert cfg["postprocess"]["high_threshold"] == 0.2
    assert cfg["hard_runtime_cap"] is None


def test_original_gi_anomaly_type_parity():
    r = m.runtime()
    cfg = m.read(ROOT / "configs/experiments/p1_champion_bracketb_20260907_v1.json")
    n = 80
    frame = r.pd.DataFrame(
        {
            "station": ["G-ORS"] * n,
            "layer": [1] * n,
            "time": r.pd.date_range("2025-01-01", periods=n, freq="10min", tz="Asia/Seoul"),
            "temp": r.np.sin(r.np.arange(n) / 20),
            "psal": [32.0] * n,
            "depth": [5.0] * n,
        }
    )
    frame.loc[30, "temp"] = 99
    frame.loc[40:50, "temp"] = 5
    rng = r.np.random.default_rng(6)
    op, bp = rng.random(n), rng.random(n)
    ms = rng.integers(0, 2, size=n)
    plateau = r.detect_plateaus(frame).to_numpy()
    spike = r.detect_singleton_spikes(frame).to_numpy()
    o = r.apply_postprocess(frame, op, plateau, spike, cfg["postprocess"])
    b = r.apply_postprocess(frame, bp, plateau, spike, cfg["postprocess"])
    o_types = r.np.full(n, "", dtype=object)
    o_types[(o == 1) & plateau] = "flatline"
    confirmed = (o == 1) & spike & (op >= 0.2)
    o_types[confirmed & ~plateau] = "spike"
    router, gi = r.composition.compose_tree(
        frame.station,
        frame.layer,
        o,
        b,
        add_cells=r.composition.ADD_CELLS,
        remove_cells=r.composition.REMOVE_CELLS,
    )
    types = r.np.full(n, "", dtype=object)
    types[plateau & b.astype(bool)] = "flatline"
    types[spike & b.astype(bool)] = "spike"
    added = (gi == 1) & (b == 0)
    types[added] = o_types[added]
    expected = (
        router.astype(bool) | ms.astype(bool) | (gi.astype(bool) & (types == "spike"))
    ).astype(int)
    assert r.np.array_equal(m.compose(r, frame, op, bp, ms, cfg), expected)


def test_portable_adapter_changes_B_only_and_no_timer():
    spec = importlib.util.spec_from_file_location(
        "bracketb_pack_test", ROOT / "scripts/p1_champion_bracketb_20260907_pack.py"
    )
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    original = (ROOT / "final_packages/P1/02_code/run.py").read_text(encoding="utf-8")
    result = pack.adapt_runner(original)
    before, after = [ast.parse(text) for text in (original, result)]
    fns_before = {n.name: ast.dump(n) for n in before.body if isinstance(n, ast.FunctionDef)}
    fns_after = {n.name: ast.dump(n) for n in after.body if isinstance(n, ast.FunctionDef)}
    for name in ("infer_ms", "combine", "tree_qa", "verify_manifest"):
        assert fns_before[name] == fns_after[name]
    assert result.index("original = train_full_model") < result.index(
        "bundle = extend(bundle, frame)"
    )
    calls = [
        n
        for n in ast.walk(after)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
    ]
    assert not any(kw.arg == "timeout" for n in calls for kw in n.keywords)
