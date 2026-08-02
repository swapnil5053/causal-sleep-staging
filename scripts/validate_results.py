"""Validate archived cross-validation result files without modifying them.

Examples:
    python scripts/validate_results.py
    python scripts/validate_results.py results/sleep78_causal
    python scripts/validate_results.py --expected-folds 5 --json

The command exits with status 1 when it finds an integrity error and status 0 when all
discovered summaries pass. Missing optional reports are warnings rather than errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


SUMMARY_FILENAMES = ("test_metrics_summary.csv", "summary.csv")
REQUIRED_COLUMNS = ("fold", "accuracy", "kappa", "macro_f1")
OPTIONAL_BOUNDED_COLUMNS = (
    "f1_W",
    "f1_N1",
    "f1_N2",
    "f1_N3",
    "f1_REM",
    "acc_30s",
    "kappa_30s",
    "macro_f1_30s",
)


def find_summaries(path: Path) -> list[Path]:
    """Return supported result summaries below *path* in deterministic order."""
    if path.is_file():
        return [path]
    summaries = []
    for filename in SUMMARY_FILENAMES:
        summaries.extend(path.rglob(filename))
    return sorted(set(summaries))


def parse_number(value: str, column: str, row_number: int, errors: list[str]) -> float | None:
    """Parse a finite numeric CSV value, recording a readable error on failure."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: {column!r} is not numeric: {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"row {row_number}: {column!r} is not finite: {value!r}")
        return None
    return number


def validate_summary(summary_path: Path, expected_folds: int) -> dict:
    """Validate one fold summary and return a serializable report."""
    errors: list[str] = []
    warnings: list[str] = []
    parsed_rows: list[dict[str, float]] = []
    folds: list[int] = []

    try:
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            missing = [column for column in REQUIRED_COLUMNS if column not in columns]
            if missing:
                errors.append(f"missing required columns: {', '.join(missing)}")

            metric_columns = [
                column
                for column in REQUIRED_COLUMNS[1:] + OPTIONAL_BOUNDED_COLUMNS
                if column in columns
            ]
            for row_number, row in enumerate(reader, start=2):
                fold_value = parse_number(row.get("fold", ""), "fold", row_number, errors)
                if fold_value is not None:
                    if not fold_value.is_integer():
                        errors.append(f"row {row_number}: fold must be an integer, got {fold_value}")
                    else:
                        folds.append(int(fold_value))

                parsed: dict[str, float] = {}
                for column in metric_columns:
                    raw_value = row.get(column, "")
                    if raw_value == "" and column in OPTIONAL_BOUNDED_COLUMNS:
                        continue
                    value = parse_number(raw_value, column, row_number, errors)
                    if value is None:
                        continue
                    lower_bound = -1.0 if "kappa" in column else 0.0
                    if not lower_bound <= value <= 1.0:
                        errors.append(
                            f"row {row_number}: {column}={value} is outside "
                            f"[{lower_bound}, 1.0]"
                        )
                    parsed[column] = value
                parsed_rows.append(parsed)
    except OSError as exc:
        errors.append(f"could not read file: {exc}")

    duplicates = sorted(fold for fold, count in Counter(folds).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate fold rows: {duplicates}")

    expected = set(range(expected_folds))
    observed = set(folds)
    if observed != expected:
        missing_folds = sorted(expected - observed)
        unexpected_folds = sorted(observed - expected)
        if missing_folds:
            errors.append(f"missing folds: {missing_folds}")
        if unexpected_folds:
            errors.append(f"unexpected folds: {unexpected_folds}")

    result_dir = summary_path.parent
    report_folds = {
        int(path.stem.split("_")[1])
        for path in result_dir.glob("fold_*_test_report.txt")
        if len(path.stem.split("_")) >= 2 and path.stem.split("_")[1].isdigit()
    }
    missing_reports = sorted(observed - report_folds)
    if missing_reports:
        warnings.append(f"no text test report for folds: {missing_reports}")

    means = {}
    for column in REQUIRED_COLUMNS[1:] + OPTIONAL_BOUNDED_COLUMNS:
        values = [row[column] for row in parsed_rows if column in row]
        if values:
            means[column] = sum(values) / len(values)

    return {
        "summary": str(summary_path),
        "folds": sorted(folds),
        "means": means,
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }


def print_human_report(reports: list[dict]) -> None:
    for report in reports:
        print(f"[{report['status']}] {report['summary']}")
        if report["folds"]:
            print(f"  folds: {report['folds']}")
        if report["means"]:
            headline = ("accuracy", "kappa", "macro_f1")
            values = [
                f"{name}={report['means'][name]:.4f}"
                for name in headline
                if name in report["means"]
            ]
            print(f"  calculated mean: {', '.join(values)}")
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")
        for error in report["errors"]:
            print(f"  ERROR: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("results"),
        help="Summary CSV or directory to scan (default: results)",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=5,
        help="Expected number of cross-validation folds (default: 5)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    if args.expected_folds < 1:
        parser.error("--expected-folds must be at least 1")

    summaries = find_summaries(args.path)
    if not summaries:
        print(f"No supported summary CSV found under {args.path}", file=sys.stderr)
        return 1

    reports = [validate_summary(path, args.expected_folds) for path in summaries]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print_human_report(reports)

    failures = sum(report["status"] == "FAIL" for report in reports)
    print(f"Validated {len(reports)} summaries; {failures} failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
