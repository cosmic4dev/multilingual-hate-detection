#!/usr/bin/env python3
"""Compute human-eval preference/meaning stats and render a bar chart."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))

import matplotlib.pyplot as plt
import numpy as np
from bert_score import score as bert_score_fn

from backend.datasets.english_detector_generator_unified import EnglishDetectorPipeline

SAMPLES_CSV = Path("human_eval/human_eval_ready_qwen25_30_spans.csv")
SUMMARY_JSON = Path("experiments/human_eval_summary.json")
FIGURE_PATH = Path("experiments/figures/preference_by_strength.png")


def load_samples(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def toxicity_score(result: Dict) -> float:
    probs = result["detection_results"]["intensity_probabilities"]
    return probs["offensive"] + probs["hate"]


def main() -> None:
    samples = load_samples(SAMPLES_CSV)
    if not samples:
        raise SystemExit("No samples found in the CSV file.")

    detector = EnglishDetectorPipeline(
        model_path="/root/multilingual-hate-detection/backend/model/retrained_english_xlmr_model.pt",
        device="cpu",
    )

    originals = [sample["original_text"] for sample in samples]
    unguided_texts = [sample["output_unguided"] for sample in samples]
    guided_texts = [sample["output_span_guided"] for sample in samples]

    print("Calculating BERTScores ...")
    _, _, f1_unguided = bert_score_fn(
        unguided_texts,
        originals,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )
    _, _, f1_guided = bert_score_fn(
        guided_texts,
        originals,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )

    f1_unguided_vals = [float(score) for score in f1_unguided.tolist()]
    f1_guided_vals = [float(score) for score in f1_guided.tolist()]

    unique_texts = set(originals) | set(unguided_texts) | set(guided_texts)
    print("Running detector on", len(unique_texts), "unique strings ...")
    detection_cache: Dict[str, Dict] = {}
    for text in unique_texts:
        if not text:
            continue
        detection_cache[text] = detector.predict(text)

    data = []
    counts_per_strength = {
        "strong": {"guided": 0, "unguided": 0, "tie": 0},
        "mild": {"guided": 0, "unguided": 0, "tie": 0},
    }
    guided_preferred = 0
    unguided_preferred = 0
    ties = 0

    for idx, (sample, unguided_score, guided_score) in enumerate(
        zip(samples, f1_unguided_vals, f1_guided_vals)
    ):
        orig_text = sample["original_text"]
        orig_toxicity = toxicity_score(detection_cache[orig_text])
        unguided_toxicity = toxicity_score(detection_cache[unguided_texts[idx]])
        guided_toxicity = toxicity_score(detection_cache[guided_texts[idx]])

        red_unguided = max(0.0, orig_toxicity - unguided_toxicity)
        red_guided = max(0.0, orig_toxicity - guided_toxicity)

        overall_unguided = 0.6 * red_unguided + 0.4 * unguided_score
        overall_guided = 0.6 * red_guided + 0.4 * guided_score

        if overall_guided > overall_unguided:
            preference = "guided"
            guided_preferred += 1
        elif overall_guided < overall_unguided:
            preference = "unguided"
            unguided_preferred += 1
        else:
            preference = "tie"
            ties += 1

        strength = sample["toxicity_strength"].lower()
        if strength not in counts_per_strength:
            counts_per_strength[strength] = {"guided": 0, "unguided": 0, "tie": 0}
        counts_per_strength[strength][preference] += 1

        data.append(
            {
                "sample_id": sample["sample_id"],
                "strength": strength,
                "bertscore_guided": guided_score,
                "bertscore_unguided": unguided_score,
                "toxicity_reduction_guided": red_guided,
                "toxicity_reduction_unguided": red_unguided,
                "preference": preference,
            }
        )

    total_evaluated = len(samples) - ties
    overall_ratio_guided = guided_preferred / total_evaluated if total_evaluated else 0.0

    guided_scores = [row["bertscore_guided"] for row in data]
    unguided_scores = [row["bertscore_unguided"] for row in data]

    summary = {
        "total_samples": len(samples),
        "guided_preferred": guided_preferred,
        "unguided_preferred": unguided_preferred,
        "ties": ties,
        "overall_guided_ratio": overall_ratio_guided,
        "counts_per_strength": counts_per_strength,
        "meaning_preservation": {
            "guided": {
                "mean": float(np.mean(guided_scores)),
                "variance": float(np.var(guided_scores)),
            },
            "unguided": {
                "mean": float(np.mean(unguided_scores)),
                "variance": float(np.var(unguided_scores)),
            },
        },
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("Summary:")
    print(json.dumps(summary, indent=2))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    strengths = ["strong", "mild"]
    guided_counts = [counts_per_strength[s]["guided"] for s in strengths]
    unguided_counts = [counts_per_strength[s]["unguided"] for s in strengths]
    x = np.arange(len(strengths))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, guided_counts, width, label="Guided")
    plt.bar(x + width / 2, unguided_counts, width, label="Unguided")
    plt.xticks(x, [s.capitalize() for s in strengths])
    plt.ylabel("Preferred samples")
    plt.title("Preference by toxicity strength (score = 0.6*toxicity reduction + 0.4*BERTScore)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
