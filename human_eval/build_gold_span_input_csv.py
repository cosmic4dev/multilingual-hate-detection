#!/usr/bin/env python3
"""
Build human-eval input CSV with HateXplain GOLD span text (not detector output).

We avoid HuggingFace `datasets` and its deprecated script loading by reading the
official HateXplain JSONs directly from GitHub:
- dataset.json: full post dictionary (id -> {post_tokens, annotators, rationales, ...})
- post_id_divisions.json: train/valid/test split IDs

Then:
1. Restrict to the test split.
2. For each sample, compute majority intensity from annotators:
   - labels are strings: "hatespeech", "normal", "offensive"
   - we map to intensity_label in {"normal", "offensive", "severe"}.
3. Convert rationales (list of 0/1 lists) into token-level majority mask,
   then into span texts (consecutive 1-runs over post_tokens).
4. Map intensity_label to toxicity_strength (offensive -> mild, others -> strong).
5. [Optional] If --slur_list is provided: any sample whose original_text or
   harmful_span_texts contains a token from the list (case-insensitive) is
   (re)assigned toxicity_strength = "strong". This refines mild/strong using
   a lexical severity cue (slur presence).
6. Randomly sample N examples with non-empty gold spans, roughly balanced
   between mild/strong, and write:
   sample_id, original_text, toxicity_strength, harmful_span_texts

The output CSV is directly usable as --input_csv for
generate_human_eval_from_csv.py (gold-guided vs unguided Qwen).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
SPLIT_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
DEFAULT_SLUR_LIST = "human_eval/slur_list_en.txt"


def _load_slur_set(path: str) -> set[str]:
    """Load one word per line (lowercased), skip empty and comments."""
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                out.add(w)
    return out


def _text_contains_slur(text: str, slur_set: set[str]) -> bool:
    """True if any token (split by whitespace) in text, lowercased, is in slur_set."""
    if not slur_set or not text:
        return False
    tokens = text.replace("|", " ").split()
    for t in tokens:
        if t.lower().strip() in slur_set:
            return True
    return False


def _load_json_from_url(url: str) -> Any:
    with urlopen(url) as resp:
        data_bytes = resp.read()
    return json.loads(data_bytes.decode("utf-8"))


def _get_intensity_from_annotators(annotators: List[Dict[str, Any]]) -> str:
    """
    HateXplain annotators label per sample with strings:
    - "hatespeech"
    - "normal"
    - "offensive"

    We map majority label to a simplified intensity space:
    - "normal"   -> "normal"
    - "offensive" -> "offensive"
    - "hatespeech" -> "severe"
    """
    if not annotators:
        return "normal"

    labels: List[str] = []
    for ann in annotators:
        lab = ann.get("label")
        if isinstance(lab, str):
            labels.append(lab)
    if not labels:
        return "normal"

    majority = max(set(labels), key=labels.count)
    if majority == "normal":
        return "normal"
    if majority == "offensive":
        return "offensive"
    return "severe"


def _majority_rationale_mask(rationales: List[List[int]], num_tokens: int) -> List[int]:
    """
    From list of annotator rationales (each len ~ num_tokens, entries 0/1),
    compute per-token majority mask (1 = harmful token, 0 = non-harmful).
    """
    if not rationales or num_tokens <= 0:
        return [0] * num_tokens

    counts = [0] * num_tokens
    for r in rationales:
        for i, v in enumerate(r):
            if i >= num_tokens:
                break
            if v:
                counts[i] += 1

    # Any positive vote counts as harmful (we could require > len(rationales)/2, but
    # using >=1 keeps more spans when annotators disagree).
    return [1 if c > 0 else 0 for c in counts]


def _span_texts_from_mask(tokens: List[str], mask: List[int]) -> List[str]:
    """
    Given tokens and a 0/1 mask over them, return span texts for 1-runs.
    """
    if not tokens:
        return []
    n = min(len(tokens), len(mask))
    tokens = tokens[:n]
    mask = mask[:n]

    spans: List[str] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        span_tokens = tokens[start:i]
        span_text = " ".join(span_tokens).strip()
        if span_text:
            spans.append(span_text)
    return spans


def _process_item(item: Dict[str, Any]) -> Tuple[str, str, str, str] | None:
    """
    Convert one HateXplain raw item into:
    (post_id, original_text, toxicity_strength, harmful_span_texts)
    or return None if non-toxic / no spans.
    """
    post_id = str(item.get("post_id") or item.get("id") or "")
    post_tokens = item.get("post_tokens") or []
    annotators = item.get("annotators") or []
    rationales = item.get("rationales") or []

    if not post_tokens:
        return None

    text = " ".join(post_tokens)
    intensity_label = _get_intensity_from_annotators(annotators)
    if intensity_label == "normal":
        return None

    # Majority rationale mask -> span texts
    mask = _majority_rationale_mask(rationales, len(post_tokens))
    span_texts = _span_texts_from_mask(post_tokens, mask)
    if not span_texts:
        return None

    toxicity_strength = "mild" if intensity_label == "offensive" else "strong"
    harmful_span_texts = " | ".join(span_texts)
    return (post_id, text, toxicity_strength, harmful_span_texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold-span input CSV from HateXplain (GitHub JSON)")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (default: human_eval/human_eval_gold_spans_30.csv)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of samples to select (default 30)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Balance mild/strong (default True)",
    )
    parser.add_argument(
        "--slur_list",
        type=str,
        default=None,
        help="Path to slur list (one word per line, lowercased). If provided, any sample with a slur token is set to strong. Default: human_eval/slur_list_en.txt if it exists.",
    )
    parser.add_argument(
        "--no_slur_override",
        action="store_true",
        help="Disable slur override; use annotator labels only (offensive→mild, hatespeech→strong). Enables larger balanced pools.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "valid", "val", "test"],
        help="HateXplain splits to scan (default: test only, matching auto300).",
    )
    parser.add_argument(
        "--exclude_csv",
        type=str,
        default=None,
        help="CSV with sample_id column; exclude these IDs (e.g. auto300 pool for disjoint auto600).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling (default 13).",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    slur_set: set[str] = set()
    if args.no_slur_override:
        print("Slur override disabled (annotator-only severity)")
    else:
        slur_path = args.slur_list
        if slur_path is None:
            slur_path = str(ROOT / DEFAULT_SLUR_LIST)
        if Path(slur_path).is_file():
            slur_set = _load_slur_set(slur_path)
            print(f"Loaded {len(slur_set)} slur tokens from {slur_path}")
        else:
            if args.slur_list is not None:
                print(f"Warning: slur_list file not found: {slur_path} (skipping slur override)")

    out_path = args.output_csv
    if not out_path:
        n = args.n_samples
        out_path = str(ROOT / "human_eval" / f"human_eval_gold_spans_{n}.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print("Downloading HateXplain dataset.json from GitHub...")
    dataset: Dict[str, Any] = _load_json_from_url(DATASET_URL)
    print(f"Loaded {len(dataset)} posts from dataset.json")

    print("Downloading post_id_divisions.json (train/valid/test splits)...")
    splits: Dict[str, List[str]] = _load_json_from_url(SPLIT_URL)

    exclude_ids: set[str] = set()
    if args.exclude_csv:
        with open(args.exclude_csv, newline="", encoding="utf-8") as f_ex:
            for r in csv.DictReader(f_ex):
                sid = (r.get("sample_id") or r.get("item_id") or "").strip()
                if sid:
                    exclude_ids.add(sid)
        print(f"Excluding {len(exclude_ids)} IDs from {args.exclude_csv}")

    pool_ids: List[str] = []
    for sp in args.splits:
        key = "val" if sp == "valid" else sp
        ids = splits.get(key) or []
        print(f"Split {sp} ({key}): {len(ids)} IDs")
        pool_ids.extend(ids)

    rows: List[Tuple[str, str, str, str]] = []
    slur_overrides = 0
    skipped_exclude = 0
    for pid in pool_ids:
        if pid in exclude_ids:
            skipped_exclude += 1
            continue
        item = dataset.get(pid)
        if not item:
            continue
        row = _process_item(item)
        if row is not None:
            # Slur override: if currently mild and text or harmful_span_texts contains a slur -> strong
            if slur_set and row[2] == "mild":
                if _text_contains_slur(row[1], slur_set) or _text_contains_slur(row[3], slur_set):
                    row = (row[0], row[1], "strong", row[3])
                    slur_overrides += 1
            rows.append(row)
    if slur_set and slur_overrides > 0:
        print(f"Slur override: {slur_overrides} samples reclassified mild -> strong")

    if skipped_exclude:
        print(f"Skipped {skipped_exclude} IDs due to --exclude_csv")

    if not rows:
        print(f"No rows with non-empty gold spans found in splits {args.splits}.")
        return

    print(f"Found {len(rows)} candidate samples with non-empty gold spans")

    # Shuffle for randomness before balancing
    random.shuffle(rows)

    # Balance: aim for half mild, half strong (unique by original_text)
    n = args.n_samples
    strong = [r for r in rows if r[2] == "strong"]
    mild = [r for r in rows if r[2] == "mild"]

    seen_texts: set[str] = set()
    strong_sel: List[Tuple[str, str, str, str]] = []
    for r in strong:
        if r[1] in seen_texts:
            continue
        seen_texts.add(r[1])
        strong_sel.append(r)
        if len(strong_sel) >= (n // 2):
            break

    mild_sel: List[Tuple[str, str, str, str]] = []
    for r in mild:
        if r[1] in seen_texts:
            continue
        seen_texts.add(r[1])
        mild_sel.append(r)
        if len(mild_sel) >= n - len(strong_sel):
            break

    selected = strong_sel + mild_sel
    if len(selected) < n:
        for r in rows:
            if len(selected) >= n:
                break
            if r[1] in seen_texts:
                continue
            seen_texts.add(r[1])
            selected.append(r)
    selected = selected[:n]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "original_text", "toxicity_strength", "harmful_span_texts"])
        for r in selected:
            writer.writerow(list(r))

    print(f"Wrote {len(selected)} rows to {out_path}")
    print(f"  strong: {sum(1 for r in selected if r[2] == 'strong')}, mild: {sum(1 for r in selected if r[2] == 'mild')}")


if __name__ == "__main__":
    main()

