#!/usr/bin/env python3
"""
Auto-evaluation for gold-span guided vs unguided detoxification outputs.

Input CSV columns expected:
  sample_id, original_text, toxicity_strength,
  harmful_span_texts, output_unguided, output_guided

Metrics:
  - Toxicity score via Perspective API (TOXICITY attribute)
  - Meaning preservation via BERTScore F1 (output vs original)
  - Combined preference score: λ * tox_reduction + (1-λ) * bertscore
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from bert_score import score as bert_score_fn
from googleapiclient import discovery

PERSPECTIVE_API_KEY_DEFAULT = None
PERSPECTIVE_QPS_DELAY = 1.1  # Perspective API: ~1 QPS for free tier


def build_perspective_client(api_key: str):
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl=(
            "https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
        ),
        static_discovery=False,
    )


def perspective_toxicity(text: str, client, retries: int = 3) -> float:
    for attempt in range(retries):
        try:
            req = {
                "comment": {"text": text},
                "requestedAttributes": {"TOXICITY": {}},
            }
            resp = client.comments().analyze(body=req).execute()
            return float(resp["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\n  [Perspective API error] '{text[:60]}...' → {e}")
                return 0.5  # neutral fallback
    return 0.5


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _std(xs: List[float]) -> float:
    return float(np.std(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="/root/multilingual-hate-detection-4/human_eval/human_eval_ready_gold_slur_30.csv",
    )
    parser.add_argument(
        "--output_json",
        default="/root/multilingual-hate-detection-4/experiments/auto_eval_results_gold_slur.json",
    )
    parser.add_argument(
        "--output_csv",
        default="/root/multilingual-hate-detection-4/experiments/auto_eval_results_gold_slur.csv",
    )
    parser.add_argument(
        "--perspective_api_key",
        default=None,
        help="Override Perspective API key (or set PERSPECTIVE_API_KEY env var)",
    )
    parser.add_argument(
        "--lambda_tox",
        type=float,
        default=0.6,
        help="Weight for toxicity reduction in combined score (0-1). BERTScore weight = 1-λ",
    )
    args = parser.parse_args()

    api_key = (
        args.perspective_api_key
        or os.environ.get("PERSPECTIVE_API_KEY")
        or PERSPECTIVE_API_KEY_DEFAULT
    )
    if not api_key:
        raise SystemExit(
            "Error: Perspective API key required. Use --perspective_api_key "
            "or set the PERSPECTIVE_API_KEY environment variable."
        )

    print(f"Loading samples from {args.input_csv} ...")
    rows = load_csv(args.input_csv)
    print(f"  → {len(rows)} samples loaded")

    originals = [r["original_text"] for r in rows]
    guideds = [r["output_guided"] for r in rows]
    unguideds = [r["output_unguided"] for r in rows]
    strengths = [r.get("toxicity_strength", "").lower() for r in rows]

    # ── BERTScore ───────────────────────────────────────────────────────────
    print("\nComputing BERTScore (guided vs original) ...")
    _, _, f1_guided = bert_score_fn(
        guideds, originals, lang="en", rescale_with_baseline=False, verbose=True
    )
    print("Computing BERTScore (unguided vs original) ...")
    _, _, f1_unguided = bert_score_fn(
        unguideds, originals, lang="en", rescale_with_baseline=False, verbose=True
    )
    bs_guided = [float(v) for v in f1_guided.tolist()]
    bs_unguided = [float(v) for v in f1_unguided.tolist()]

    # ── Perspective API toxicity ─────────────────────────────────────────────
    print("\nInitializing Perspective API ...")
    client = build_perspective_client(api_key)

    # Deduplicate texts to minimise API calls
    unique_texts: List[str] = list(
        {t for t in (originals + guideds + unguideds) if t.strip()}
    )
    tox_cache: Dict[str, float] = {}

    print(f"Scoring {len(unique_texts)} unique texts via Perspective API ...")
    for i, text in enumerate(unique_texts, 1):
        print(f"  [{i:3d}/{len(unique_texts)}] {text[:60]!r}", end="\r")
        tox_cache[text] = perspective_toxicity(text, client)
        time.sleep(PERSPECTIVE_QPS_DELAY)
    print(f"\n  → Toxicity scoring complete")

    # ── Per-sample results ───────────────────────────────────────────────────
    lam = args.lambda_tox
    results = []
    guided_preferred = unguided_preferred = ties = 0
    counts: Dict[str, Dict[str, int]] = {
        "strong": {"guided": 0, "unguided": 0, "tie": 0},
        "mild": {"guided": 0, "unguided": 0, "tie": 0},
    }

    for idx, row in enumerate(rows):
        orig_tox = tox_cache.get(originals[idx], 0.5)
        g_tox = tox_cache.get(guideds[idx], 0.5)
        u_tox = tox_cache.get(unguideds[idx], 0.5)

        g_red = max(0.0, orig_tox - g_tox)
        u_red = max(0.0, orig_tox - u_tox)

        g_overall = lam * g_red + (1 - lam) * bs_guided[idx]
        u_overall = lam * u_red + (1 - lam) * bs_unguided[idx]

        if g_overall > u_overall:
            pref = "guided"
            guided_preferred += 1
        elif u_overall > g_overall:
            pref = "unguided"
            unguided_preferred += 1
        else:
            pref = "tie"
            ties += 1

        stre = strengths[idx]
        if stre not in counts:
            counts[stre] = {"guided": 0, "unguided": 0, "tie": 0}
        counts[stre][pref] += 1

        results.append(
            {
                "sample_id": row.get("sample_id", ""),
                "strength": stre,
                "original_toxicity": orig_tox,
                "guided_toxicity": g_tox,
                "unguided_toxicity": u_tox,
                "guided_reduction": g_red,
                "unguided_reduction": u_red,
                "guided_bertscore": bs_guided[idx],
                "unguided_bertscore": bs_unguided[idx],
                "guided_overall": g_overall,
                "unguided_overall": u_overall,
                "preference": pref,
            }
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    total = len(results)
    decided = total - ties
    g_scores = [r["guided_bertscore"] for r in results]
    u_scores = [r["unguided_bertscore"] for r in results]
    g_reds = [r["guided_reduction"] for r in results]
    u_reds = [r["unguided_reduction"] for r in results]

    summary = {
        "total_samples": total,
        "guided_preferred": guided_preferred,
        "unguided_preferred": unguided_preferred,
        "ties": ties,
        "overall_guided_ratio": guided_preferred / decided if decided else 0.0,
        "lambda_toxicity": lam,
        "lambda_bertscore": 1 - lam,
        "counts_per_strength": counts,
        "meaning_preservation": {
            "guided": {"mean": float(np.mean(g_scores)), "std": _std(g_scores)},
            "unguided": {"mean": float(np.mean(u_scores)), "std": _std(u_scores)},
        },
        "toxicity_reduction": {
            "guided": {"mean": float(np.mean(g_reds)), "std": _std(g_reds)},
            "unguided": {"mean": float(np.mean(u_reds)), "std": _std(u_reds)},
        },
        "toxicity_absolute": {
            "guided": {
                "mean": float(np.mean([r["guided_toxicity"] for r in results])),
                "std": _std([r["guided_toxicity"] for r in results]),
            },
            "unguided": {
                "mean": float(np.mean([r["unguided_toxicity"] for r in results])),
                "std": _std([r["unguided_toxicity"] for r in results]),
            },
            "original": {
                "mean": float(np.mean([r["original_toxicity"] for r in results])),
                "std": _std([r["original_toxicity"] for r in results]),
            },
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": results}, f, ensure_ascii=False, indent=2)

    out_csv = Path(args.output_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "strength", "preference",
            "original_toxicity", "guided_toxicity", "unguided_toxicity",
            "guided_reduction", "unguided_reduction",
            "guided_bertscore", "unguided_bertscore",
            "guided_overall", "unguided_overall",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 60)
    print("AUTO-EVAL SUMMARY  (gold-span guided vs unguided)")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\n✅  JSON → {out_json}")
    print(f"✅  CSV  → {out_csv}")


if __name__ == "__main__":
    main()
