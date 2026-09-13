#!/usr/bin/env python3
"""
Randomize output_guided and output_unguided into output_a and output_b
to avoid order bias in human evaluation.
"""

import argparse
import csv
import random
from typing import Dict, List

def randomize_outputs(input_csv: str, output_csv: str, seed: int = 42) -> None:
    """Randomize guided/unguided outputs into output_a/output_b."""
    random.seed(seed)
    
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    results = []
    for row in rows:
        output_unguided = row.get('output_unguided', '')
        output_guided = row.get('output_guided', '')
        
        # Randomly assign to output_a and output_b
        if random.random() < 0.5:
            output_a = output_unguided
            output_b = output_guided
            which_is_guided = 'a'
        else:
            output_a = output_guided
            output_b = output_unguided
            which_is_guided = 'b'
        
        result = {
            'sample_id': row.get('sample_id', ''),
            'original_text': row.get('original_text', ''),
            'toxicity_strength': row.get('toxicity_strength', ''),
            'harmful_span_texts': row.get('harmful_span_texts', ''),
            'output_a': output_a,
            'output_b': output_b,
            'which_is_guided': which_is_guided,
        }
        results.append(result)
    
    # Write output
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'sample_id',
            'original_text',
            'toxicity_strength',
            'harmful_span_texts',
            'output_a',
            'output_b',
            'which_is_guided',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    guided_in_a = sum(1 for r in results if r['which_is_guided'] == 'a')
    guided_in_b = sum(1 for r in results if r['which_is_guided'] == 'b')
    print(f"✅ Randomized {len(results)} samples")
    print(f"   Guided in output_a: {guided_in_a}")
    print(f"   Guided in output_b: {guided_in_b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/root/multilingual-hate-detection/human_eval/human_eval_ready_from_csv_30.csv',
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='/root/multilingual-hate-detection/human_eval/human_eval_ready_from_csv_30_randomized.csv',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    randomize_outputs(args.input_csv, args.output_csv, args.seed)


if __name__ == '__main__':
    main()


