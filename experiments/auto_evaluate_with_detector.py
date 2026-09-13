#!/usr/bin/env python3
"""
Automated evaluation of detoxification outputs using detector-based toxicity scoring.
Computes:
- Toxicity scores (using XLM-R detector)
- Semantic similarity (BERTScore)
- Toxicity reduction
- Overall preference score
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from bert_score import score as bert_score_fn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Add parent directory to path
ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH))


class XLMRobertaDetector(nn.Module):
    """XLM-RoBERTa based hate span detector."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_intensity_labels: int = 3,
        num_bio_labels: int = 5,
        num_target_labels: int = 6,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.intensity_head = nn.Linear(self.config.hidden_size, num_intensity_labels)
        self.bio_head = nn.Linear(self.config.hidden_size, num_bio_labels)
        self.target_head = nn.Linear(self.config.hidden_size, num_target_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output.mean(dim=1)
        return {
            "intensity_logits": self.intensity_head(pooled_output),
            "bio_logits": self.bio_head(sequence_output),
            "target_logits": self.target_head(pooled_output),
        }


class DetectorToxicityScorer:
    """Toxicity scorer using XLM-R detector."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        
        # Load detector
        ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt.get("model_config", {})
        self.detector = XLMRobertaDetector(
            model_name="xlm-roberta-base",
            num_intensity_labels=cfg.get("intensity_labels", 3),
            num_bio_labels=cfg.get("bio_labels", 5),
            num_target_labels=cfg.get("target_labels", 6),
        )
        self.detector.load_state_dict(ckpt["model_state_dict"])
        self.detector.to(self.device)
        self.detector.eval()
        
        # Intensity labels: ["normal", "offensive", "hate"]
        # We'll use the probability of "hate" + "offensive" as toxicity score
        self.intensity_labels = ["normal", "offensive", "hate"]
    
    def get_toxicity_score(self, text: str) -> float:
        """Get toxicity score (0-1) for a text using detector."""
        if not text or not text.strip():
            return 0.5  # Neutral for empty text
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.detector(**inputs)
            intensity_probs = torch.softmax(outputs["intensity_logits"], dim=-1)
            
            # Get probabilities for "offensive" and "hate"
            # Index 1 = offensive, Index 2 = hate
            offensive_prob = intensity_probs[0][1].item()
            hate_prob = intensity_probs[0][2].item()
            
            # Toxicity score = probability of being offensive or hateful
            # We can use a weighted combination or just sum
            toxicity_score = offensive_prob + hate_prob
            
            # Normalize to 0-1 range (already in range, but ensure)
            return min(1.0, max(0.0, toxicity_score))


def load_samples(csv_path: str) -> List[Dict[str, str]]:
    """Load samples from CSV file."""
    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if v:
                    v = v.strip().strip('"').strip("'")
                cleaned[k] = v or ""
            rows.append(cleaned)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_30_for_eval.csv',
    )
    parser.add_argument(
        '--detector_model',
        type=str,
        default='/root/multilingual-hate-detection/backend/model/english_xlmr_model.pt',
        help='Path to detector model checkpoint',
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default='/root/multilingual-hate-detection/experiments/auto_eval_results_detector.json',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='/root/multilingual-hate-detection/experiments/auto_eval_results_detector.csv',
    )
    args = parser.parse_args()

    print(f"Loading samples from {args.input_csv}...")
    samples = load_samples(args.input_csv)
    if not samples:
        raise SystemExit("No samples found in the CSV file.")
    
    print(f"Loaded {len(samples)} samples")

    # Initialize detector
    print(f"Loading detector from {args.detector_model}...")
    scorer = DetectorToxicityScorer(args.detector_model)
    print("✅ Detector loaded")

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

    # Get toxicity scores using detector
    print(f"Getting toxicity scores from detector for {len(originals) + len(output_a_texts) + len(output_b_texts)} texts...")
    toxicity_cache: Dict[str, float] = {}
    all_texts = set(originals) | set(output_a_texts) | set(output_b_texts)
    all_texts = {t for t in all_texts if t.strip()}
    
    for i, text in enumerate(all_texts, 1):
        if not text:
            continue
        print(f"  [{i}/{len(all_texts)}] Processing...", end='\r')
        toxicity_cache[text] = scorer.get_toxicity_score(text)
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
        
        # Get toxicity scores from detector
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
    print("AUTOMATED EVALUATION SUMMARY (Detector-based)")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print(f"\n✅ Saved detailed results to {args.output_json}")
    print(f"✅ Saved CSV results to {args.output_csv}")


if __name__ == "__main__":
    main()


