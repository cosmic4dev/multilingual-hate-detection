#!/usr/bin/env python3
from __future__ import annotations

"""
Select human evaluation samples from HateXplain for localized/mild/opinion spans.
Outputs JSON/CSV with category metadata.
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.datasets.english_dataset import EnglishHateXplainDataset


@dataclass
class Sample:
    sample_id: str
    text: str
    intensity: str
    target_labels: List[str]
    bio_labels: List[str]
    spans: List[Dict[str, int]]
    category: str
    reason: str


def extract_spans(bio_labels: List[str]) -> Tuple[List[Dict[str, int]], int]:
    spans: List[Dict[str, int]] = []
    current: Dict[str, int] | None = None
    max_span_len = 0
    for idx, label in enumerate(bio_labels):
        if label == "O":
            if current:
                spans.append(current)
                span_len = current["end"] - current["start"] + 1
                max_span_len = max(max_span_len, span_len)
                current = None
            continue

        tag_type = label.split("-")[1]
        if label.startswith("B-"):
            if current:
                spans.append(current)
                span_len = current["end"] - current["start"] + 1
                max_span_len = max(max_span_len, span_len)
            current = {"type": tag_type, "start": idx, "end": idx}
        elif label.startswith("I-") and current and current["type"] == tag_type:
            current["end"] = idx
        else:
            if current:
                spans.append(current)
                span_len = current["end"] - current["start"] + 1
                max_span_len = max(max_span_len, span_len)
            current = None

    if current:
        spans.append(current)
        span_len = current["end"] - current["start"] + 1
        max_span_len = max(max_span_len, span_len)

    return spans, max_span_len


def matches_category(
    category: str,
    text: str,
    intensity: str,
    spans: List[Dict[str, int]],
    max_span_len: int,
    non_o_count: int,
    targets: List[str],
    relaxed: bool,
) -> Tuple[bool, str]:
    tokens = len(text.split())
    lower = text.lower()

    if category == "localized_toxicity":
        limit = 6 if relaxed else 4
        token_req = 10 if relaxed else 12
        span_len_req = 4 if relaxed else 3
        if non_o_count <= limit and tokens >= token_req and max_span_len <= span_len_req and intensity in {
            "normal",
            "offensive",
        }:
            return True, "Short harmful span within mostly neutral sentence"
        return False, ""

    if category == "mild_medium_toxicity":
        limit = 10 if relaxed else 8
        count_cond = non_o_count <= limit
        intensity_cond = intensity in {"offensive", "normal"} if relaxed else intensity == "offensive"
        if intensity_cond and 3 <= non_o_count <= limit:
            return True, "Borderline offensive tone without heavy slurs"
        return False, ""

    if category == "opinionated_target":
        pronoun_cond = "you" in lower or "they" in lower
        target_cond = any(t != "non-hate" for t in targets)
        if tokens >= 10 and (pronoun_cond or target_cond or relaxed):
            return True, "Opinionated statement with a defined target"
        return False, ""

    if category == "indirect_toxicity":
        max_span = 5 if relaxed else 4
        if max_span_len <= max_span and 1 <= non_o_count <= 6:
            return True, "Implicit or less direct toxicity"
        return False, ""

    if category == "longer_mixed":
        token_req = 20 if relaxed else 25
        non_o_req = 4 if relaxed else 5
        if tokens >= token_req and non_o_count >= non_o_req:
            return True, "Long sentence mixing neutral context with harmful clause"
        return False, ""

    return False, ""


def format_category_name(category: str) -> str:
    mapping = {
        "localized_toxicity": "Localized toxicity",
        "mild_medium_toxicity": "Mild–medium toxicity",
        "opinionated_target": "Opinionated but targeted",
        "indirect_toxicity": "Indirect or implicit toxicity",
        "longer_mixed": "Longer sentences with mixed content",
    }
    return mapping.get(category, category)


def main() -> None:
    dataset = EnglishHateXplainDataset("test", tokenizer_name="xlm-roberta-base")
    categories_target = {
        "localized_toxicity": 8,
        "mild_medium_toxicity": 8,
        "opinionated_target": 7,
        "indirect_toxicity": 4,
        "longer_mixed": 3,
    }
    selected: Dict[str, List[Sample]] = {cat: [] for cat in categories_target}
    seen_texts: set[str] = set()

    for relaxed in [False, True]:
        for item in dataset.data:
            text = item["text"]
            if text in seen_texts:
                continue
            bio_labels = item["bio_labels"]
            intensity = item["intensity_label"]
            targets = item["target_labels"]
            spans, max_span_len = extract_spans(bio_labels)
            non_o_count = sum(1 for lbl in bio_labels if lbl != "O")

            for category in categories_target:
                if len(selected[category]) >= categories_target[category]:
                    continue

                match, reason = matches_category(
                    category, text, intensity, spans, max_span_len, non_o_count, targets, relaxed
                )
                if not match:
                    continue

                sample = Sample(
                    sample_id=f"heval_{category}_{len(selected[category]) + 1:02d}",
                    text=text,
                    intensity=intensity,
                    target_labels=targets,
                    bio_labels=bio_labels,
                    spans=spans,
                    category=format_category_name(category),
                    reason=f"{reason} (non-O tokens: {non_o_count}, spans: {len(spans)})",
                )
                selected[category].append(sample)
                seen_texts.add(text)
                break

        if all(len(selected[cat]) >= categories_target[cat] for cat in categories_target):
            break

    output_dir = os.path.join("human_eval", "hatexplain_eval_30")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "samples.json")
    csv_path = os.path.join(output_dir, "samples.csv")

    all_samples: List[Sample] = []
    for cat in categories_target:
        all_samples.extend(selected[cat])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "sample_id": s.sample_id,
                    "text": s.text,
                    "intensity": s.intensity,
                    "targets": s.target_labels,
                    "category": s.category,
                    "reason": s.reason,
                    "spans": s.spans,
                }
                for s in all_samples
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text", "intensity", "category", "reason", "spans"])
        for s in all_samples:
            writer.writerow(
                [
                    s.sample_id,
                    s.text,
                    s.intensity,
                    s.category,
                    s.reason,
                    "; ".join(f"{span['type']}[{span['start']}:{span['end']}]" for span in s.spans),
                ]
            )

    print(f"✅ Wrote {len(all_samples)} samples to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()

