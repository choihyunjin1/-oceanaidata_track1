"""Command-line interface for the complete P1 workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from p1_qc.audit import audit_train_test
from p1_qc.config import P1QCConfig, load_config
from p1_qc.data import load_dataset
from p1_qc.experiment import RunRecorder, sha256_file
from p1_qc.pipeline import (
    load_model,
    load_or_build_features,
    predict_submission,
    reproduce_submission,
    run_cross_validation,
    save_model,
    train_full_model,
)
from p1_qc.submission import validate_submission, write_submission

PROJECT_ROOT = Path(__file__).resolve().parents[2]
P1_REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "baseline_rule.csv",
    "README.md",
)


def resolve_data_dir(config: P1QCConfig, override: str | Path | None = None) -> Path:
    """Resolve the immutable P1 input directory without a machine-specific path.

    An explicit CLI override or ``P1_DATA_DIR`` (already materialised in
    ``config.paths.data_dir`` by :func:`load_config`) is authoritative.  The
    repository fallback accepts exactly one directory containing the complete
    official file set; zero or multiple matches are errors.
    """

    if override is not None:
        candidate = Path(override)
    elif config.paths.data_dir is not None:
        candidate = config.paths.data_dir
    else:
        candidates = {
            path.parent.resolve()
            for path in PROJECT_ROOT.rglob("train.csv")
            if all((path.parent / name).is_file() for name in P1_REQUIRED_FILES)
        }
        if len(candidates) != 1:
            raise FileNotFoundError(
                "set P1_DATA_DIR or --data-dir; repository fallback requires "
                f"exactly one complete P1 file set, found {len(candidates)}"
            )
        candidate = next(iter(candidates))

    candidate = candidate.expanduser().resolve()
    missing = [name for name in P1_REQUIRED_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P1 data directory is missing {missing}: {candidate}")
    return candidate


def _config(args: argparse.Namespace) -> P1QCConfig:
    config = load_config(args.config)
    if getattr(args, "mode", None):
        config = replace(
            config,
            mode=args.mode,
            features=replace(config.features, mode=args.mode),
        )
    return config


def _recorder(command: str, config: P1QCConfig, args: argparse.Namespace) -> RunRecorder:
    root = config.paths.artifacts_dir
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    recorder = RunRecorder(command, config.to_dict(), root=root, seed=config.seed)
    recorder.copy_config(args.config)
    return recorder


def _load_train_test(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_dataset(data_dir / "train.csv", kind="train", audit=True)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    return train, test


def command_audit(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    recorder = _recorder("audit", config, args)
    train, test = _load_train_test(data_dir)
    recorder.add_inputs(train=data_dir / "train.csv", test=data_dir / "test.csv")
    audit = audit_train_test(train, test, cadence_minutes=config.data.cadence_minutes)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    baseline = pd.read_csv(data_dir / "baseline_rule.csv")
    keys = ["station", "year", "layer", "time"]
    report = audit.to_dict()
    report["sample_key_order_match"] = sample[keys].equals(test[keys])
    report["baseline_key_order_match"] = baseline[keys].equals(test[keys])
    recorder.record_json("audit.json", report)
    audit.raise_for_errors()
    recorder.finish(status="complete")
    print(json.dumps({"run_id": recorder.run_id, **report}, ensure_ascii=False, indent=2))
    return 0


def command_cv(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    recorder = _recorder("cv", config, args)
    train, test = _load_train_test(data_dir)
    recorder.add_inputs(train=data_dir / "train.csv", test=data_dir / "test.csv")
    bundle = load_or_build_features(train, config, kind="train", use_cache=not args.no_cache)
    oof, metrics, selection = run_cross_validation(
        train,
        test,
        bundle,
        config,
        backend=args.backend,
        bootstrap_replicates=args.bootstrap_replicates,
        augmentation=args.augment,
    )
    oof_path = recorder.path / "oof.parquet"
    oof.to_parquet(oof_path, index=False, compression="zstd")
    recorder.record_file(oof_path)
    recorder.record_json("metrics.json", metrics)
    recorder.record_json("selection.json", selection)
    recorder.finish(
        status="complete",
        weighted_f1=metrics["aggregate"]["weighted"]["f1"],
        micro_f1=metrics["aggregate"]["micro"]["f1"],
    )
    print(
        json.dumps(
            {
                "run_id": recorder.run_id,
                "run_path": str(recorder.path.resolve()),
                "micro_f1": metrics["aggregate"]["micro"]["f1"],
                "weighted_f1": metrics["aggregate"]["weighted"]["f1"],
                "selection": selection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    recorder = _recorder("train", config, args)
    train = load_dataset(data_dir / "train.csv", kind="train", audit=True)
    recorder.add_inputs(train=data_dir / "train.csv", selection=args.selection)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if selection["feature_mode"] != config.features.mode:
        raise ValueError("selection and config feature modes differ")
    bundle = load_or_build_features(train, config, kind="train", use_cache=not args.no_cache)
    model = train_full_model(train, bundle, config, selection)
    model_path = Path(args.output) if args.output else recorder.path / "model.joblib"
    save_model(model, model_path)
    metadata = {
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "backend": model.backend,
        "feature_mode": model.feature_mode,
        "iteration_count": model.iteration_count,
        "postprocess": model.postprocess,
    }
    recorder.record_file(model_path)
    recorder.record_json("model_metadata.json", metadata)
    recorder.finish(status="complete")
    print(json.dumps({"run_id": recorder.run_id, **metadata}, ensure_ascii=False, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    recorder = _recorder("predict", config, args)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    recorder.add_inputs(test=data_dir / "test.csv", model=args.model)
    model = load_model(args.model)
    if model.feature_mode != config.features.mode:
        config = replace(
            config,
            mode=model.feature_mode,
            features=replace(config.features, mode=model.feature_mode),
        )
    bundle = load_or_build_features(test, config, kind="test", use_cache=not args.no_cache)
    submission, probability = predict_submission(model, test, bundle)
    output = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "submissions" / f"{recorder.run_id}.csv"
    )
    write_submission(submission, output)
    report = validate_submission(output, test)
    probability_path = recorder.path / "test_probabilities.parquet"
    test.loc[:, ["station", "year", "layer", "time"]].assign(
        probability=probability.astype("float32"), label=submission["label"].to_numpy()
    ).to_parquet(probability_path, index=False, compression="zstd")
    recorder.record_file(probability_path)
    recorder.record_file(output, "submission.csv")
    recorder.record_json("submission_validation.json", report)
    recorder.finish(status="complete", positive_rate=report["positive_rate"])
    print(json.dumps({"run_id": recorder.run_id, **report}, ensure_ascii=False, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    report = validate_submission(args.submission, data_dir / "test.csv")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    config = _config(args)
    data_dir = resolve_data_dir(config, args.data_dir)
    recorder = _recorder("reproduce", config, args)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    model = load_model(args.model)
    if model.feature_mode != config.features.mode:
        config = replace(
            config,
            mode=model.feature_mode,
            features=replace(config.features, mode=model.feature_mode),
        )
    bundle = load_or_build_features(test, config, kind="test", use_cache=not args.no_cache)
    report = reproduce_submission(
        args.model,
        test,
        bundle,
        args.output,
        expected_path=args.expected,
    )
    recorder.add_inputs(test=data_dir / "test.csv", model=args.model)
    recorder.record_file(args.output, "reproduced_submission.csv")
    recorder.record_json("reproduction.json", report)
    recorder.finish(status="complete")
    print(json.dumps({"run_id": recorder.run_id, **report}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m p1_qc")
    _add_common_arguments(parser, suppress_defaults=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    _add_common_arguments(audit, suppress_defaults=True)
    audit.set_defaults(handler=command_audit)

    cv = subparsers.add_parser("cv")
    _add_common_arguments(cv, suppress_defaults=True)
    cv.add_argument("--backend", choices=("lightgbm", "xgboost", "catboost"), default="lightgbm")
    cv.add_argument("--bootstrap-replicates", type=int, default=2000)
    cv.add_argument("--augment", action="store_true", help="inject fold-local synthetic anomalies")
    cv.add_argument("--no-cache", action="store_true")
    cv.set_defaults(handler=command_cv)

    train = subparsers.add_parser("train")
    _add_common_arguments(train, suppress_defaults=True)
    train.add_argument("--selection", type=Path, required=True)
    train.add_argument("--output", type=Path)
    train.add_argument("--no-cache", action="store_true")
    train.set_defaults(handler=command_train)

    predict = subparsers.add_parser("predict")
    _add_common_arguments(predict, suppress_defaults=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--no-cache", action="store_true")
    predict.set_defaults(handler=command_predict)

    validate = subparsers.add_parser("validate")
    _add_common_arguments(validate, suppress_defaults=True)
    validate.add_argument("submission", type=Path)
    validate.set_defaults(handler=command_validate)

    reproduce = subparsers.add_parser("reproduce")
    _add_common_arguments(reproduce, suppress_defaults=True)
    reproduce.add_argument("--model", type=Path, required=True)
    reproduce.add_argument("--output", type=Path, required=True)
    reproduce.add_argument("--expected", type=Path)
    reproduce.add_argument("--no-cache", action="store_true")
    reproduce.set_defaults(handler=command_reproduce)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser, *, suppress_defaults: bool) -> None:
    """Add common options while preserving values parsed before a subcommand."""

    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--config",
        type=Path,
        default=(argparse.SUPPRESS if suppress_defaults else PROJECT_ROOT / "configs" / "p1.toml"),
    )
    parser.add_argument("--data-dir", type=Path, default=default)
    parser.add_argument("--mode", choices=("offline", "causal"), default=default)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
