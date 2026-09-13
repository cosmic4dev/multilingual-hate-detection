#!/usr/bin/env python3
"""
Lambda sensitivity analysis for gold-span auto-evaluation.

Reads existing auto_eval_results_gold_slur.json (detailed_results with
per-sample toxicity reduction and BERTScore), recomputes preference for
each λ in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], and outputs:
  - experiments/lambda_sensitivity_results.json
  - experiments/lambda_sensitivity_table.tex (for paper)
  - experiments/figures/lambda_sensitivity.png (optional bar chart)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_LAMBDAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def run_sensitivity(
    detailed_results: List[Dict],
    lambdas: List[float],
) -> List[Dict]:
    out = []
    for lam in lambdas:
        guided = unguided = tie = 0
        by_strength: Dict[str, Dict[str, int]] = {
            "strong": {"guided": 0, "unguided": 0, "tie": 0},
            "mild": {"guided": 0, "unguided": 0, "tie": 0},
        }
        for r in detailed_results:
            g_overall = lam * r["guided_reduction"] + (1 - lam) * r["guided_bertscore"]
            u_overall = lam * r["unguided_reduction"] + (1 - lam) * r["unguided_bertscore"]
            if g_overall > u_overall:
                pref = "guided"
                guided += 1
            elif u_overall > g_overall:
                pref = "unguided"
                unguided += 1
            else:
                pref = "tie"
                tie += 1
            stre = r.get("strength", "").lower()
            if stre in by_strength:
                by_strength[stre][pref] += 1
        total = len(detailed_results)
        decided = total - tie
        out.append({
            "lambda_tox": lam,
            "lambda_bertscore": 1 - lam,
            "guided_preferred": guided,
            "unguided_preferred": unguided,
            "ties": tie,
            "total": total,
            "guided_ratio": guided / decided if decided else 0.0,
            "counts_per_strength": by_strength,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="/root/multilingual-hate-detection-4/experiments/auto_eval_results_gold_slur.json",
    )
    parser.add_argument(
        "--output_json",
        default="/root/multilingual-hate-detection-4/experiments/lambda_sensitivity_results.json",
    )
    parser.add_argument(
        "--output_tex",
        default="/root/multilingual-hate-detection-4/experiments/lambda_sensitivity_table.tex",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.2,0.4,0.5,0.6,0.8,1.0",
        help="Comma-separated λ values (toxicity weight)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate lambda_sensitivity.png bar chart",
    )
    args = parser.parse_args()

    lambdas = [float(x.strip()) for x in args.lambdas.split(",")]

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    detailed = data["detailed_results"]

    results = run_sensitivity(detailed, lambdas)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"lambdas": lambdas, "results": results}, f, indent=2)

    # LaTeX table
    out_tex = Path(args.output_tex)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\lambda$ (tox weight) & Guided & Unguided & Guided \% \\",
        r"\midrule",
    ]
    for r in results:
        lam = r["lambda_tox"]
        g = r["guided_preferred"]
        u = r["unguided_preferred"]
        pct = 100.0 * r["guided_ratio"] if r["guided_preferred"] + r["unguided_preferred"] else 0
        lines.append(f"{lam:.1f} & {g} & {u} & {pct:.1f}\\% \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            x = np.array(lambdas)
            guided_pct = np.array([100.0 * r["guided_ratio"] for r in results])
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(x - 0.04, guided_pct, width=0.08, label="Guided %", color="steelblue")
            ax.set_xlabel(r"$\lambda$ (weight on toxicity reduction)")
            ax.set_ylabel("Guided preference (%)")
            ax.set_xticks(lambdas)
            ax.legend()
            ax.set_ylim(0, 100)
            fig.tight_layout()
            plot_path = Path(args.output_json).parent / "figures" / "lambda_sensitivity.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Plot → {plot_path}")
        except Exception as e:
            print(f"  Plot skip: {e}")

    print("Lambda sensitivity summary:")
    for r in results:
        print(f"  λ={r['lambda_tox']:.1f} → Guided {r['guided_preferred']}/{r['total']} ({100*r['guided_ratio']:.1f}%)")
    print(f"\n✅ JSON → {out_json}")
    print(f"✅ TEX  → {out_tex}")


if __name__ == "__main__":
    main()
