"""Read-only independent QA of a completed standalone P2 package."""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = "46d194a1ef40a1deaebd084916644d9359433d2e6ce7d5c0b53d9f515bbec071"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    package = args.package.resolve()

    def load(relative):
        return json.loads((package / relative).read_text(encoding="utf-8"))

    train, infer, replay, build, cfg = [
        load(p)
        for p in (
            "04_logs/training-result.json",
            "04_logs/inference-result.json",
            "04_logs/replay-result.json",
            "06_docs/BUILD_MANIFEST.json",
            "config.json",
        )
    ]
    checks = []

    def check(name, value):
        checks.append({"check": name, "pass": bool(value)})

    check("three_fresh_processes", len({train["pid"], infer["pid"], replay["pid"]}) == 3)
    check(
        "scratch_three_fits",
        train["empty_model_directory_before_training"]
        and train["new_full_fits"] == len(train["fits"]) == 3,
    )
    check(
        "same_fixed_seeds_epochs",
        [r["seed"] for r in train["fits"]] == cfg["seeds"]
        and all(r["epochs"] == 60 for r in train["fits"]),
    )
    check(
        "training_population_mass",
        train["training"]["original_rows"] == 166268
        and train["training"]["augmented_rows"] == 51354
        and np.isclose(train["training"]["training_weight_sum"], 166268),
    )
    check(
        "source_unchanged",
        sha(args.data / "observations.csv") == train["source_sha256"] == cfg["source_sha256"],
    )
    check(
        "no_prior_models_answers",
        build["copied_prior_models"]
        == build["copied_prior_answers"]
        == build["packaged_source_data_files"]
        == 0,
    )
    check(
        "training_no_official",
        train["official_access_rows"] == train["csv_written"] == train["upload"] == 0,
    )
    check(
        "sample_value_access_zero",
        all(r["access"]["sample_value_rows"] == 0 for r in (train, infer, replay)),
    )
    check(
        "official_key_scope",
        all(
            r["access"]["official_index_key_rows"]
            == r["access"]["official_sample_key_rows"]
            == 26061
            for r in (infer, replay)
        ),
    )
    check(
        "outside_original_repository",
        not package.is_relative_to(Path(__file__).resolve().parents[3]),
    )
    check(
        "local_module_origins",
        train["module_origins"] == {"core": "02_code/core.py", "entry": "02_code/run.py"},
    )
    for relative, checksum in build["files"].items():
        check(f"build/{relative}/unchanged", sha(package / relative) == checksum)
    for stage in (train, infer, replay):
        for relative, checksum in stage["code_hashes"].items():
            check(f"stage{stage['pid']}/{relative}/unchanged", sha(package / relative) == checksum)
    for row in train["fits"]:
        check(f"model/{row['seed']}/sha", sha(package / "03_model" / row["file"]) == row["sha256"])
    names = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "os",
        "sys",
        "time",
        "pathlib",
        "types",
        "typing",
        "numpy",
        "pandas",
        "torch",
        "threadpoolctl",
        "core",
    }
    for path in (package / "02_code").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                check(
                    f"import/{path.name}/{node.lineno}",
                    all(n.name.split(".")[0] in names for n in node.names),
                )
            elif isinstance(node, ast.ImportFrom):
                check(f"import/{path.name}/{node.lineno}", node.module.split(".")[0] in names)
    key_columns = ["station", "layer", "time"]
    sample = pd.read_csv(args.data / "sample_submission.csv", usecols=key_columns)
    index = pd.read_csv(args.data / "test_index.csv", usecols=key_columns)

    def keys(frame):
        return list(
            zip(
                frame.station.astype(str),
                frame.layer.astype(int),
                pd.to_datetime(frame.time, utc=True).astype(str),
                strict=True,
            )
        )

    answers = []
    for receipt in (infer, replay):
        path = package / receipt["answer"]
        output = pd.read_csv(path)
        answers.append(output.temp.to_numpy(float))
        actual_keys = keys(output)
        check(f"answer/{path.name}/sha", sha(path) == receipt["answer_sha256"])
        check(f"answer/{path.name}/schema", list(output.columns) == [*key_columns, "temp"])
        check(f"answer/{path.name}/rows_unique", len(output) == len(set(actual_keys)) == 26061)
        check(
            f"answer/{path.name}/order_index",
            actual_keys == keys(sample) and set(actual_keys) == set(keys(index)),
        )
        check(f"answer/{path.name}/finite", np.isfinite(answers[-1]).all())
    check("full_replay_values_exact", np.array_equal(*answers))
    check(
        "full_replay_sha_exact",
        infer["answer_sha256"] == replay["answer_sha256"]
        and replay["exact_first_inference_sha_match"],
    )
    check(
        "under_six_hours_current_machine",
        train["runtime_seconds"] + infer["runtime_seconds"] + replay["runtime_seconds"] < 21600,
    )
    check("source_still_unchanged", sha(args.data / "observations.csv") == cfg["source_sha256"])
    success = all(row["pass"] for row in checks)
    result = {
        "status": "PASS" if success else "FAIL",
        "checks_passed": sum(c["pass"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "answer_sha256": infer["answer_sha256"],
        "historical_expected_sha256": EXPECTED,
        "historical_sha_exact": infer["answer_sha256"] == EXPECTED,
        "training_seconds": train["runtime_seconds"],
        "inference_seconds": infer["runtime_seconds"],
        "replay_seconds": replay["runtime_seconds"],
        "new_full_fits": 3,
        "new_qa_fits": 0,
        "sample_values_read": 0,
        "qa_official_key_rows": 52122,
        "old_answer_values_read": 0,
        "fresh_venv_verified": False,
        "original_repo_outside_execution_verified": True,
        "physical_network_disconnection_verified": False,
        "network_control": "Python socket audit hook denies connect; not OS network isolation",
        "organizer_hardware_verified": False,
        "upload": 0,
        "qa_runtime_seconds": time.monotonic() - started,
        "qa_runner_sha256": sha(Path(__file__)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "checks_passed",
                    "checks_total",
                    "answer_sha256",
                    "historical_sha_exact",
                    "training_seconds",
                    "inference_seconds",
                    "replay_seconds",
                )
            }
        )
    )
    if not success:
        raise AssertionError([r for r in checks if not r["pass"]])


if __name__ == "__main__":
    main()
