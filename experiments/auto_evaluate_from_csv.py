#!/usr/bin/env python3
"""
Automated evaluation of detoxification outputs from CSV file.
Computes:
- Toxicity scores (using detector)
- Semantic similarity (BERTScore)
- Toxicity reduction
- Overall preference score
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from bert_score import score as bert_score_fn
from googleapiclient import discovery
import os

import sys
ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))

PERSPECTIVE_API_KEY_DEFAULT = None


def get_perspective_api_client(api_key: str):
    """Initialize Perspective API client."""
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )


def toxicity_score_perspective(text: str, api_client) -> float:
    """Get toxicity score from Perspective API."""
    try:
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = api_client.comments().analyze(body=analyze_request).execute()
        toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        return float(toxicity_score)
    except Exception as e:
        print(f"Warning: Perspective API error for text '{text[:50]}...': {e}")
        return 0.5  # Default neutral score on error


def load_samples(csv_path: str) -> List[Dict[str, str]]:
    """Load samples from CSV file."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up the data
            cleaned = {}
            for k, v in row.items():
                if v:
                    # Remove extra quotes if present
                    v = v.strip().strip('"').strip("'")
                cleaned[k] = v or ""
            rows.append(cleaned)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/root/multilingual-hate-detection/human_eval/human_eval_form_ready_randomized_ko_questions.csv',
    )
    parser.add_argument(
        '--perspective_api_key',
        type=str,
        default=None,
        help='Perspective API key (or set PERSPECTIVE_API_KEY env var)',
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='/root/multilingual-hate-detection/experiments/auto_eval_results.json',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='/root/multilingual-hate-detection/experiments/auto_eval_results.csv',
    )
    args = parser.parse_args()

    print(f"Loading samples from {args.input_csv}...")
    samples = load_samples(args.input_csv)
    if not samples:
        raise SystemExit("No samples found in the CSV file.")
    
    print(f"Loaded {len(samples)} samples")

    # Initialize Perspective API
    api_key = (
        args.perspective_api_key 
        or os.environ.get('PERSPECTIVE_API_KEY')
        or PERSPECTIVE_API_KEY_DEFAULT
    )
    if not api_key:
        raise SystemExit(
            "Error: Perspective API key required. Use --perspective_api_key "
            "or set the PERSPECTIVE_API_KEY environment variable."
        )
    
    print("Initializing Perspective API client...")
    perspective_client = get_perspective_api_client(api_key)
    print("✅ Perspective API client initialized")

    # Extract texts
    originals = []
    output_a_texts = []
    output_b_texts = []
    a_is_guided = []
    b_is_guided = []
    
    for sample in samples:
        originals.append(sample.get('original_text', ''))
        output_a_texts.append(sample.get('output_A', ''))
        output_b_texts.append(sample.get('output_B', ''))
        a_is_guided.append(sample.get('A_is_span_guided', '').lower() == 'true')
        b_is_guided.append(sample.get('B_is_span_guided', '').lower() == 'true')

    # Calculate BERTScore for semantic similarity
    print("Calculating BERTScore for Output A...")
    _, _, f1_a = bert_score_fn(
        output_a_texts,
        originals,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )
    
    print("Calculating BERTScore for Output B...")
    _, _, f1_b = bert_score_fn(
        output_b_texts,
        originals,
        lang="en",
        rescale_with_baseline=False,
        verbose=False,
    )

    f1_a_vals = [float(score) for score in f1_a.tolist()]
    f1_b_vals = [float(score) for score in f1_b.tolist()]

    # Get unique texts for toxicity scoring
    unique_texts = set(originals) | set(output_a_texts) | set(output_b_texts)
    unique_texts = {t for t in unique_texts if t.strip()}
    
    print(f"Getting toxicity scores from Perspective API for {len(unique_texts)} unique strings...")
    toxicity_cache: Dict[str, float] = {}
    for i, text in enumerate(unique_texts, 1):
        if not text:
            continue
        print(f"  [{i}/{len(unique_texts)}] Processing...", end='\r')
        toxicity_cache[text] = toxicity_score_perspective(text, perspective_client)
    print(f"\n✅ Completed toxicity scoring")

    # Calculate metrics for each sample
    results = []
    guided_preferred = 0
    unguided_preferred = 0
    ties = 0
    
    counts_per_strength = {
        "strong": {"guided": 0, "unguided": 0, "tie": 0},
        "mild": {"guided": 0, "unguided": 0, "tie": 0},
    }

    for idx, sample in enumerate(samples):
        orig_text = originals[idx]
        output_a = output_a_texts[idx]
        output_b = output_b_texts[idx]
        
        # Get toxicity scores from Perspective API
        orig_toxicity = toxicity_cache.get(orig_text, 0.5)
        a_toxicity = toxicity_cache.get(output_a, 0.5)
        b_toxicity = toxicity_cache.get(output_b, 0.5)
        
        # Calculate toxicity reduction
        a_reduction = max(0.0, orig_toxicity - a_toxicity)
        b_reduction = max(0.0, orig_toxicity - b_toxicity)
        
        # Get BERTScore
        a_bertscore = f1_a_vals[idx]
        b_bertscore = f1_b_vals[idx]
        
        # Overall score: 0.6 * toxicity_reduction + 0.4 * bertscore
        a_overall = 0.6 * a_reduction + 0.4 * a_bertscore
        b_overall = 0.6 * b_reduction + 0.4 * b_bertscore
        
        # Determine which is guided/unguided
        if a_is_guided[idx]:
            guided_score = a_overall
            unguided_score = b_overall
            guided_bertscore = a_bertscore
            unguided_bertscore = b_bertscore
            guided_reduction = a_reduction
            unguided_reduction = b_reduction
            guided_toxicity = a_toxicity
            unguided_toxicity = b_toxicity
        else:
            guided_score = b_overall
            unguided_score = a_overall
            guided_bertscore = b_bertscore
            unguided_bertscore = a_bertscore
            guided_reduction = b_reduction
            unguided_reduction = a_reduction
            guided_toxicity = b_toxicity
            unguided_toxicity = a_toxicity
        
        # Determine preference
        if guided_score > unguided_score:
            preference = "guided"
            guided_preferred += 1
        elif guided_score < unguided_score:
            preference = "unguided"
            unguided_preferred += 1
        else:
            preference = "tie"
            ties += 1
        
        strength = sample.get('toxicity_strength', '').lower()
        if strength not in counts_per_strength:
            counts_per_strength[strength] = {"guided": 0, "unguided": 0, "tie": 0}
        counts_per_strength[strength][preference] += 1
        
        result = {
            "item_id": sample.get('item_id', ''),
            "strength": strength,
            "original_toxicity": float(orig_toxicity),
            "guided_toxicity": float(guided_toxicity),
            "unguided_toxicity": float(unguided_toxicity),
            "guided_reduction": float(guided_reduction),
            "unguided_reduction": float(unguided_reduction),
            "guided_bertscore": float(guided_bertscore),
            "unguided_bertscore": float(unguided_bertscore),
            "guided_overall": float(guided_score),
            "unguided_overall": float(unguided_score),
            "preference": preference,
        }
        results.append(result)

    # Calculate summary statistics
    total_evaluated = len(samples) - ties
    overall_ratio_guided = guided_preferred / total_evaluated if total_evaluated else 0.0
    
    guided_scores = [r["guided_bertscore"] for r in results]
    unguided_scores = [r["unguided_bertscore"] for r in results]
    guided_reductions = [r["guided_reduction"] for r in results]
    unguided_reductions = [r["unguided_reduction"] for r in results]

    summary = {
        "total_samples": len(samples),
        "guided_preferred": guided_preferred,
        "unguided_preferred": unguided_preferred,
        "ties": ties,
        "overall_guided_ratio": float(overall_ratio_guided),
        "counts_per_strength": counts_per_strength,
        "meaning_preservation": {
            "guided": {
                "mean": float(np.mean(guided_scores)),
                "std": float(np.std(guided_scores)),
            },
            "unguided": {
                "mean": float(np.mean(unguided_scores)),
                "std": float(np.std(unguided_scores)),
            },
        },
        "toxicity_reduction": {
            "guided": {
                "mean": float(np.mean(guided_reductions)),
                "std": float(np.std(guided_reductions)),
            },
            "unguided": {
                "mean": float(np.mean(unguided_reductions)),
                "std": float(np.std(unguided_reductions)),
            },
        },
    }

    # Save JSON results
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open('w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "detailed_results": results,
        }, f, ensure_ascii=False, indent=2)

    # Save CSV results
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = [
            "item_id", "strength", "preference",
            "original_toxicity", "guided_toxicity", "unguided_toxicity",
            "guided_reduction", "unguided_reduction",
            "guided_bertscore", "unguided_bertscore",
            "guided_overall", "unguided_overall",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "="*60)
    print("AUTOMATED EVALUATION SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print(f"\n✅ Saved detailed results to {args.output_json}")
    print(f"✅ Saved CSV results to {args.output_csv}")


if __name__ == "__main__":
    main()
