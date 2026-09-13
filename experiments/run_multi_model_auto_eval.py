#!/usr/bin/env python3
"""
Multi-model auto-evaluation: from one detector run (detailed_results with
guided/unguided reduction and BERTScore), recompute preference for multiple
λ values (overall = λ * toxicity_reduction + (1-λ) * bertscore).

Produces one result per "model" (λ=0.5, 0.6, 0.7) so we can later:
  A. Compare guided vote share across models
  B. Check Δtoxicity vs ΔBERTScore correlation (same underlying data)
  C. Check threshold effect (strong vs mild) consistency across models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def compute_preference_and_summary(
    detailed_results: List[Dict[str, Any]],
    lambda_tox: float,
) -> Dict[str, Any]:
    """Recompute preference per sample and summary for given λ."""
    assert 0 <= lambda_tox <= 1
    lambda_bert = 1.0 - lambda_tox

    results = []
    guided_preferred = 0
    unguided_preferred = 0
    ties = 0
    counts_per_strength: Dict[str, Dict[str, int]] = {
        "strong": {"guided": 0, "unguided": 0, "tie": 0},
        "mild": {"guided": 0, "unguided": 0, "tie": 0},
    }

    for r in detailed_results:
        g_overall = lambda_tox * r["guided_reduction"] + lambda_bert * r["guided_bertscore"]
        u_overall = lambda_tox * r["unguided_reduction"] + lambda_bert * r["unguided_bertscore"]

        if g_overall > u_overall:
            preference = "guided"
            guided_preferred += 1
        elif g_overall < u_overall:
            preference = "unguided"
            unguided_preferred += 1
        else:
            preference = "tie"
            ties += 1

        strength = (r.get("strength") or "").lower()
        if strength not in counts_per_strength:
            counts_per_strength[strength] = {"guided": 0, "unguided": 0, "tie": 0}
        counts_per_strength[strength][preference] += 1

        results.append({
            **{k: v for k, v in r.items() if k != "preference" and k != "guided_overall" and k != "unguided_overall"},
            "guided_overall": g_overall,
            "unguided_overall": u_overall,
            "preference": preference,
        })

    total = len(detailed_results)
    total_evaluated = total - ties
    overall_ratio_guided = guided_preferred / total_evaluated if total_evaluated else 0.0

    guided_scores = [x["guided_bertscore"] for x in results]
    unguided_scores = [x["unguided_bertscore"] for x in results]
    guided_reductions = [x["guided_reduction"] for x in results]
    unguided_reductions = [x["unguided_reduction"] for x in results]

    summary = {
        "total_samples": total,
        "guided_preferred": guided_preferred,
        "unguided_preferred": unguided_preferred,
        "ties": ties,
        "lambda_toxicity": lambda_tox,
        "lambda_bertscore": lambda_bert,
        "overall_guided_ratio": float(overall_ratio_guided),
        "counts_per_strength": counts_per_strength,
        "meaning_preservation": {
            "guided": {"mean": sum(guided_scores) / len(guided_scores), "std": _std(guided_scores)},
            "unguided": {"mean": sum(unguided_scores) / len(unguided_scores), "std": _std(unguided_scores)},
        },
        "toxicity_reduction": {
            "guided": {"mean": sum(guided_reductions) / len(guided_reductions), "std": _std(guided_reductions)},
            "unguided": {"mean": sum(unguided_reductions) / len(unguided_reductions), "std": _std(unguided_reductions)},
        },
    }
    return {"summary": summary, "detailed_results": results}


def _std(xs: List[float]) -> float:
    import math
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model auto-eval from one detector run")
    parser.add_argument(
        "--input_json",
        type=str,
        default="/root/multilingual-hate-detection-4/experiments/auto_eval_results_detector.json",
        help="Path to detector-based auto eval JSON (must contain detailed_results)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/multilingual-hate-detection-4/experiments/multi_model_auto_eval_results.json",
        help="Output path for multi-model results",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.5,0.6,0.7",
        help="Comma-separated λ (weight for toxicity reduction); 1-λ = weight for BERTScore",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    detailed = data["detailed_results"]
    if not detailed:
        raise SystemExit("No detailed_results in input JSON.")

    lambdas = [float(x.strip()) for x in args.lambdas.split(",")]
    out: Dict[str, Dict[str, Any]] = {}
    for lam in lambdas:
        key = f"model_lambda_{lam:.2f}".replace(".", "_")
        out[key] = compute_preference_and_summary(detailed, lam)
        print(f"λ={lam}: guided_ratio={out[key]['summary']['overall_guided_ratio']:.3f} "
              f"(guided={out[key]['summary']['guided_preferred']}, unguided={out[key]['summary']['unguided_preferred']}, ties={out[key]['summary']['ties']})")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved multi-model results to {out_path}")


if __name__ == "__main__":
    main()
