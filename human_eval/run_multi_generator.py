#!/usr/bin/env python3
"""
Run span-guided detoxification generation with multiple generators (open + commercial)
using the same input CSV, for parallel comparison.

Same experiment: one input CSV → N generators → N output CSVs
(human_eval_ready_{name}_30.csv). Each output CSV has the same columns expected by
auto_evaluate_gold_slur.py / auto_evaluate_from_csv.py (output_guided, output_unguided, etc.).

Usage:
  python run_multi_generator.py --input_csv ... --out_dir ...
  (Optional) Set OPENAI_API_KEY, ANTHROPIC_API_KEY for commercial models.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Default generator list: Qwen2.5 (2024) + Mistral 7B + Llama 3.1 8B + Gemma 2 9B + Qwen3 8B (2025) (+ 2 commercial)
DEFAULT_OPEN_MODELS = [
    {"name": "qwen25_7b", "type": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct"},
    {"name": "qwen3_8b", "type": "huggingface", "model": "Qwen/Qwen3-8B"},
    {"name": "mistral7b", "type": "huggingface", "model": "mistralai/Mistral-7B-Instruct-v0.2"},
    {"name": "llama31_8b", "type": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"name": "gemma2_9b", "type": "huggingface", "model": "google/gemma-2-9b-it"},
]
DEFAULT_COMMERCIAL_MODELS = [
    {"name": "gpt4o", "type": "openai", "model": "gpt-4o"},
    {"name": "claude", "type": "anthropic", "model": "claude-3-5-sonnet-20241022"},
]


def run_generation(
    script_path: Path,
    input_csv: str,
    out_dir: Path,
    gen: dict,
    hf_cache_dir: str,
    hf_token: str | None,
    api_key: str | None,
    dry_run: bool,
    output_suffix: str = "",
    local_files_only: bool = False,
    n_samples: int = 30,
) -> bool:
    """Run generate_human_eval_from_csv.py for one generator. Returns True on success."""
    name = gen["name"]
    stem = f"human_eval_ready_{output_suffix}_{name}_{n_samples}" if output_suffix else f"human_eval_ready_{name}_{n_samples}"
    out_csv = out_dir / f"{stem}.csv"
    cmd = [
        sys.executable,
        str(script_path),
        "--input_csv", input_csv,
        "--output_csv", str(out_csv),
        "--hf_cache_dir", hf_cache_dir,
    ]
    if gen["type"] == "huggingface":
        cmd.extend(["--generator_model_name", gen["model"]])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
        if local_files_only:
            cmd.append("--local_files_only")
    else:
        provider = "openai" if gen["type"] == "openai" else "anthropic"
        cmd.extend(["--api_provider", provider, "--api_model", gen["model"]])
        if api_key:
            cmd.extend(["--api_key", api_key])

    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        return True
    print(f"\n>>> Running generator: {name} ({gen['type']} / {gen['model']})")
    result = subprocess.run(cmd, cwd=script_path.parent)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation with multiple generators (same input CSV)")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Input CSV (sample_id, original_text, toxicity_strength, harmful_span_texts). If not set, uses gold span or detector default per --use_gold.",
    )
    parser.add_argument(
        "--use_gold",
        action="store_true",
        help="Use gold-span input (HateXplain rationales). Sets default input per --n_samples.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        choices=[30, 300],
        help="Sample size: 30 or 300. Input: human_eval_gold_spans_{n}.csv, output: human_eval_ready_gold_{name}_{n}.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/multilingual-hate-detection-4/human_eval",
        help="Directory for output CSVs (human_eval_ready_{name}_30.csv)",
    )
    parser.add_argument(
        "--generators",
        type=str,
        default="all",
        choices=["all", "open", "commercial", "qwen_only"],
        help="Which generators to run: all (open+commercial), open only, commercial only, or qwen_only",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model by name (e.g. llama31_8b, gemma2_9b, qwen25_7b). Overrides --generators.",
    )
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load HF models from local cache only (no auth needed when cached)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    human_eval_dir = Path(__file__).resolve().parent
    if args.input_csv is None:
        if args.use_gold:
            args.input_csv = str(human_eval_dir / f"human_eval_gold_spans_{args.n_samples}.csv")
        else:
            args.input_csv = str(human_eval_dir / "human_eval_inputs_from_json_30_balanced.csv")
    output_suffix = "gold" if args.use_gold else ""

    all_models = {g["name"]: g for g in (DEFAULT_OPEN_MODELS + DEFAULT_COMMERCIAL_MODELS)}
    if args.model:
        if args.model not in all_models:
            raise SystemExit(f"Unknown model: {args.model}. Available: {list(all_models.keys())}")
        gens = [all_models[args.model]]
    elif args.generators == "all":
        gens = DEFAULT_OPEN_MODELS + DEFAULT_COMMERCIAL_MODELS
    elif args.generators == "open":
        gens = DEFAULT_OPEN_MODELS
    elif args.generators == "commercial":
        gens = DEFAULT_COMMERCIAL_MODELS
    else:
        gens = [DEFAULT_OPEN_MODELS[0]]  # qwen25_7b only

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "generate_human_eval_from_csv.py"
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # Use default HF cache when local_files_only (model is typically in ~/.cache/huggingface/hub)
    hf_cache_dir = args.hf_cache_dir
    if args.local_files_only:
        hf_cache_dir = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")

    ok = 0
    for gen in gens:
        if run_generation(
            script_path,
            args.input_csv,
            out_dir,
            gen,
            hf_cache_dir,
            hf_token,
            api_key,
            args.dry_run,
            output_suffix=output_suffix,
            local_files_only=args.local_files_only,
            n_samples=args.n_samples,
        ):
            ok += 1

    print(f"\nDone: {ok}/{len(gens)} generators succeeded.")


if __name__ == "__main__":
    main()
