#!/usr/bin/env python3
"""
Run auto-evaluation (Perspective API + BERTScore, λ preference) on each
multi-generator output CSV and aggregate into a single comparison table.

Expects CSVs produced by run_multi_generator.py (human_eval_ready_{name}_30.csv)
with columns: sample_id, original_text, toxicity_strength, harmful_span_texts,
output_unguided, output_guided.

Outputs:
  - experiments/auto_eval_results_multi_{name}.json (per-model)
  - experiments/multi_model_auto_eval_comparison.json
  - experiments/multi_model_auto_eval_comparison.csv (table for paper)

Use --n_samples 300 for 300-sample runs (human_eval_ready_*_300.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_SLUR_SCRIPT = ROOT / "experiments" / "auto_evaluate_gold_slur.py"
HUMAN_EVAL_DIR = ROOT / "human_eval"
EXPERIMENTS_DIR = ROOT / "experiments"


def run_auto_eval_one(
    input_csv: Path,
    model_name: str,
    out_dir: Path,
    lambda_tox: float = 0.6,
    dry_run: bool = False,
    n_samples: int = 30,
) -> bool:
    """Run auto_evaluate_gold_slur.py on one CSV. Returns True on success."""
    suffix = f"_{n_samples}" if n_samples != 30 else ""
    out_json = out_dir / f"auto_eval_results_multi_{model_name}{suffix}.json"
    out_csv = out_dir / f"auto_eval_results_multi_{model_name}{suffix}.csv"
    cmd = [
        sys.executable,
        str(GOLD_SLUR_SCRIPT),
        "--input_csv", str(input_csv),
        "--output_json", str(out_json),
        "--output_csv", str(out_csv),
        "--lambda_tox", str(lambda_tox),
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0 and out_json.exists()


def aggregate_summaries(out_dir: Path, model_names: list[str], n_samples: int = 30) -> list[dict]:
    """Load per-model JSON summaries and return a list of summary rows for comparison."""
    suffix = f"_{n_samples}" if n_samples != 30 else ""
    rows = []
    for name in model_names:
        p = out_dir / f"auto_eval_results_multi_{name}{suffix}.json"
        if not p.exists():
            rows.append({
                "model": name,
                "error": "missing result file",
                "guided_ratio": None,
                "guided_bertscore_mean": None,
                "unguided_bertscore_mean": None,
                "guided_reduction_mean": None,
                "unguided_reduction_mean": None,
            })
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        s = data.get("summary", {})
        mp = s.get("meaning_preservation", {})
        tr = s.get("toxicity_reduction", {})
        rows.append({
            "model": name,
            "error": None,
            "guided_ratio": s.get("overall_guided_ratio"),
            "guided_preferred": s.get("guided_preferred"),
            "unguided_preferred": s.get("unguided_preferred"),
            "ties": s.get("ties"),
            "guided_bertscore_mean": mp.get("guided", {}).get("mean"),
            "unguided_bertscore_mean": mp.get("unguided", {}).get("mean"),
            "guided_reduction_mean": tr.get("guided", {}).get("mean"),
            "unguided_reduction_mean": tr.get("unguided", {}).get("mean"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-eval all multi-model output CSVs and aggregate")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(HUMAN_EVAL_DIR),
        help="Directory containing human_eval_ready_{name}_30.csv files",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (e.g. gold_qwen25_7b,gold_llama31_8b). If not set, discover from input_dir.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        choices=[30, 300],
        help="Sample size to match input CSVs: human_eval_ready_*_{n}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(EXPERIMENTS_DIR),
        help="Directory for per-model results and comparison table",
    )
    parser.add_argument("--lambda_tox", type=float, default=0.6)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    if args.models:
        model_names = [s.strip() for s in args.models.split(",")]
    else:
        # Discover: human_eval_ready_{name}_{n}.csv
        model_names = []
        for f in input_dir.glob(f"human_eval_ready_*_{n}.csv"):
            stem = f.stem  # e.g. human_eval_ready_gold_qwen25_7b_300
            if stem.startswith("human_eval_ready_") and stem.endswith(f"_{n}"):
                name = stem.replace("human_eval_ready_", "").replace(f"_{n}", "")
                model_names.append(name)
        model_names.sort()
        if not model_names:
            print(f"No human_eval_ready_*_{n}.csv found in", input_dir)
            return

    print(f"Models to evaluate: {model_names}")

    for name in model_names:
        csv_path = input_dir / f"human_eval_ready_{name}_{n}.csv"
        if not csv_path.exists():
            print(f"Skip {name}: {csv_path} not found")
            continue
        ok = run_auto_eval_one(csv_path, name, out_dir, args.lambda_tox, args.dry_run, args.n_samples)
        if not ok and not args.dry_run:
            print(f"Warning: auto-eval failed for {name}")

    if args.dry_run:
        print("Dry run: skipping aggregate")
        return

    rows = aggregate_summaries(out_dir, model_names, n)
    suffix = f"_{n}" if n != 30 else ""
    comparison_json = out_dir / f"multi_model_auto_eval_comparison{suffix}.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump({"models": rows, "n_samples": n}, f, ensure_ascii=False, indent=2)
    print(f"Comparison JSON: {comparison_json}")

    comparison_csv = out_dir / f"multi_model_auto_eval_comparison{suffix}.csv"
    fieldnames = [
        "model", "guided_ratio", "guided_preferred", "unguided_preferred", "ties",
        "guided_bertscore_mean", "unguided_bertscore_mean",
        "guided_reduction_mean", "unguided_reduction_mean", "error",
    ]
    with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Comparison CSV: {comparison_csv}")


if __name__ == "__main__":
    main()
