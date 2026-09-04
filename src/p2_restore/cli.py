"""Command line interface for P2 audit, validation, training, and prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from p2_restore.data import audit_p2_data, load_p2_data
from p2_restore.features import build_test_features, build_training_features
from p2_restore.model import blocked_validation, fit_model
from p2_restore.submission import build_submission, validate_submission


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m p2_restore")
    parser.add_argument("--data-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("cv")
    train = sub.add_parser("train")
    train.add_argument("--output", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("submission")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = load_p2_data(args.data_dir)
    if args.command == "audit":
        print(json.dumps(audit_p2_data(data), ensure_ascii=False, indent=2, default=str))
    elif args.command == "cv":
        result = blocked_validation(build_training_features(data.observations))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "train":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model = fit_model(build_training_features(data.observations))
        joblib.dump(model, output)
        print(json.dumps({"model": str(output.resolve()), "sha256": _sha256(output)}, indent=2))
    elif args.command == "predict":
        model = joblib.load(args.model)
        table = build_test_features(data)
        submission = build_submission(data.test_index, model.predict(table))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
        result = validate_submission(output, data.test_index)
        result.update({"submission": str(output.resolve()), "sha256": _sha256(output)})
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = validate_submission(args.submission, data.test_index)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
