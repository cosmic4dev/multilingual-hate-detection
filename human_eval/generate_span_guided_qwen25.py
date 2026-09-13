#!/usr/bin/env python3
"""
This experiment is NOT to compare model performance.
The goal is to generate paired outputs (unguided vs span-guided)
to analyze when span-guided detoxification helps or hurts,
under a fixed detector (XLM-R) and fixed generator (Qwen2.5-8B-Instruct).

Generate span-guided outputs for 30 samples using:
- Detector: XLM-R (fixed checkpoint)
- Generator: Qwen2.5-8B-Instruct (fixed parameters)
- Output: human_eval_ready_qwen25_30.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


class XLMRobertaDetector(nn.Module):
    """XLM-RoBERTa based hate span detector (matches our saved checkpoint heads)."""

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


@dataclass
class Models:
    device: str
    detector: XLMRobertaDetector
    detector_tokenizer: Any
    generator: Any
    generator_tokenizer: Any


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def load_models(
    detector_ckpt_path: str,
    generator_model_name: str,
    hf_cache_dir: str,
    hf_token: Optional[str],
) -> Models:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detector
    tok_kwargs = {"cache_dir": hf_cache_dir}
    if hf_token:
        tok_kwargs["token"] = hf_token

    detector_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", **tok_kwargs)
    ckpt = torch.load(detector_ckpt_path, map_location=device)
    cfg = ckpt.get("model_config", {})
    detector = XLMRobertaDetector(
        model_name="xlm-roberta-base",
        num_intensity_labels=cfg.get("intensity_labels", 3),
        num_bio_labels=cfg.get("bio_labels", 5),
        num_target_labels=cfg.get("target_labels", 6),
    )
    detector.load_state_dict(ckpt["model_state_dict"])
    detector.to(device)
    detector.eval()

    # Generator
    print("Loading generator tokenizer...")
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_name, **tok_kwargs)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    print(f"Loading generator model: {generator_model_name} (this may take a while for large models)...")
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "dtype": pick_dtype(),
        "cache_dir": hf_cache_dir,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    gen = AutoModelForCausalLM.from_pretrained(
        generator_model_name, 
        **model_kwargs,
        resume_download=True,  # Resume interrupted downloads
    )
    print("✅ Generator model loaded")

    return Models(
        device=device,
        detector=detector,
        detector_tokenizer=detector_tokenizer,
        generator=gen,
        generator_tokenizer=gen_tokenizer,
    )


def detect_spans(models: Models, text: str) -> Dict[str, Any]:
    """Detect hate spans using XLM-R detector and return structured information."""
    inputs = models.detector_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(models.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = models.detector(**inputs)
        intensity_probs = torch.softmax(out["intensity_logits"], dim=-1)
        bio_probs = torch.softmax(out["bio_logits"], dim=-1)
        intensity_pred = torch.argmax(intensity_probs, dim=-1).item()
        bio_pred = torch.argmax(bio_probs, dim=-1)[0]  # [seq_len]

    # BIO spans (token-level indices) - following existing code pattern
    bio_labels = ["O", "B-SOFT", "I-SOFT", "B-HARD", "I-HARD"]
    tokens = models.detector_tokenizer.tokenize(text)
    spans = []
    current = None
    
    bio_pred_list = bio_pred.tolist()
    # Align tokens with predictions (skip special tokens if needed)
    min_len = min(len(tokens), len(bio_pred_list))
    
    for i in range(min_len):
        tok = tokens[i]
        lab_idx = bio_pred_list[i]
        lab = bio_labels[lab_idx]
        
        if lab.startswith("B-"):
            if current:
                spans.append(current)
            current = {"type": lab.split("-")[1], "start": i, "end": i, "text": tok}
        elif lab.startswith("I-") and current and lab.split("-")[1] == current["type"]:
            current["end"] = i
            current["text"] += tok.replace("▁", " ")
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)

    # Extract only SOFT/HARD spans
    hate_spans = [s for s in spans if s.get("type") in ["SOFT", "HARD"]]

    # Get offset mapping for character positions
    encoded = models.detector_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoded["offset_mapping"]

    detected_spans_text = []
    detected_spans_offsets = []
    detected_spans_intensity = []

    for span in hate_spans:
        # Get character offsets from offset_mapping
        start_idx = span["start"]
        end_idx = span["end"]
        
        # Ensure indices are within bounds
        if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx][1]
            span_text = text[start_char:end_char].strip()

            if span_text:
                detected_spans_text.append(span_text)
                detected_spans_offsets.append((start_char, end_char))
                # Map SOFT->mild, HARD->strong
                intensity = "mild" if span["type"] == "SOFT" else "strong"
                detected_spans_intensity.append(intensity)

    span_empty = len(detected_spans_text) == 0

    return {
        "detected_spans_text": " | ".join(detected_spans_text) if detected_spans_text else "",
        "detected_spans_offsets": detected_spans_offsets,
        "detected_spans_intensity": " | ".join(detected_spans_intensity) if detected_spans_intensity else "",
        "span_empty": span_empty,
        "span_list": detected_spans_text,  # For prompt construction
        "span_list_with_intensity": [
            f"{text} ({intensity})"
            for text, intensity in zip(detected_spans_text, detected_spans_intensity)
        ] if detected_spans_text else [],
    }


def build_span_guided_prompt(original_text: str, span_info: Dict[str, Any]) -> str:
    """Build span-guided prompt according to the specified template."""
    span_list_with_intensity = span_info["span_list_with_intensity"]

    if span_info["span_empty"]:
        # Case (B): No spans detected
        prompt = f'''You are given a sentence.

Original sentence:
"{original_text}"

No harmful spans were detected.

Instruction:
- Do NOT rewrite the sentence unless it is absolutely necessary.
- If rewriting is unnecessary, return the original sentence unchanged.
- Output ONE sentence only.'''
    else:
        # Case (A): Spans exist
        span_list_str = "\n".join([f"- {s}" for s in span_list_with_intensity])
        prompt = f'''You are given a sentence that contains harmful expressions.

Original sentence:
"{original_text}"

Detected harmful spans:
{span_list_str}

Instruction:
- Rewrite ONLY the harmful spans to make them non-toxic.
- Do NOT rewrite or paraphrase the entire sentence.
- Preserve the original meaning, tone, and structure as much as possible.
- If the span is mild, apply minimal softening.
- If the span is strong, replace it with a neutral but semantically appropriate expression.
- Output ONE sentence only.'''

    return prompt


def generate_span_guided(
    models: Models, original_text: str, span_info: Dict[str, Any], max_retries: int = 3
) -> str:
    """Generate span-guided output using Qwen2.5-8B-Instruct with fixed parameters."""
    prompt = build_span_guided_prompt(original_text, span_info)

    messages = [{"role": "user", "content": prompt}]

    tok = models.generator_tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_str = prompt

    for attempt in range(max_retries):
        inputs = tok(prompt_str, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(models.generator.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = models.generator.generate(
                **inputs,
                max_new_tokens=80,  # Fixed
                temperature=0.7,  # Fixed
                top_p=0.9,  # Fixed
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # Decode ONLY newly generated tokens
        generated_ids = outputs[0][prompt_len:]
        out = tok.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean-up: remove common prefixes
        out = re.sub(
            r'^(Here is the rewritten text:|Rewritten text:|Answer:|Result:)\s*',
            "",
            out,
            flags=re.IGNORECASE,
        ).strip()

        # Extract only first sentence (up to first period, exclamation, or question mark)
        sentence_end = re.search(r'[.!?]\s', out)
        if sentence_end:
            out = out[: sentence_end.end() - 1].strip()
        else:
            # If no sentence ending found, take first line
            out = out.split("\n")[0].strip()

        # Quality checks
        if not quality_check(out, span_info):
            if attempt < max_retries - 1:
                continue  # Retry
            else:
                # Last attempt failed quality check
                if span_info["span_empty"]:
                    # Return original if no spans and quality check failed
                    return original_text

        return out

    # Fallback
    return original_text if span_info["span_empty"] else out


def quality_check(output: str, span_info: Dict[str, Any]) -> bool:
    """Check output quality according to specified criteria."""
    # Check 1: If span_empty=True, output should not be drastically different
    # (This is logged but not enforced as rejection)

    # Check 2: Output should be one sentence
    sentence_count = len(re.split(r'[.!?]\s+', output))
    if sentence_count > 1:
        return False

    # Check 3: No forbidden patterns
    forbidden_patterns = ["Okay", "Let's", "I think"]
    for pattern in forbidden_patterns:
        if pattern in output:
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate span-guided outputs for human evaluation using Qwen2.5-8B-Instruct"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/root/multilingual-hate-detection/human_eval/samples_for_form_30.csv",
        help="Input CSV with original_text and output_unguided",
    )
    parser.add_argument(
        "--detector_model_path",
        type=str,
        default="/root/multilingual-hate-detection/backend/model/retrained_english_xlmr_model.pt",
        help="Path to detector checkpoint (.pt)",
    )
    parser.add_argument(
        "--generator_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace repo id for Qwen2.5-7B-Instruct (Qwen2.5-8B-Instruct does not exist)",
    )
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/multilingual-hate-detection/human_eval/human_eval_ready_qwen25_30.csv",
    )
    args = parser.parse_args()

    # Prefer explicit arg, else env, else rely on `huggingface-cli login` cached token.
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    print("Loading models...")
    models = load_models(
        detector_ckpt_path=args.detector_model_path,
        generator_model_name=args.generator_model_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=hf_token,
    )
    print("✅ Models loaded")

    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Processing {len(rows)} samples...")

    results = []
    for i, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        original_text = row["original_text"]
        toxicity_strength = row["toxicity_strength"]
        output_unguided = row.get("output_unguided", "")

        print(f"[{i}/{len(rows)}] Processing {sample_id}...")

        # Detect spans
        span_info = detect_spans(models, original_text)

        # Quality check logging
        if span_info["span_empty"]:
            print(f"  ⚠️  No spans detected (span_empty=True)")

        # Generate span-guided output
        output_span_guided = generate_span_guided(models, original_text, span_info)

        # Post-generation quality check
        if span_info["span_empty"] and output_span_guided != original_text:
            # Check if output is significantly different
            if len(output_span_guided) > len(original_text) * 1.5 or len(output_span_guided) < len(original_text) * 0.5:
                print(f"  ⚠️  WARNING: span_empty=True but output changed significantly")
                print(f"     Original: {original_text[:50]}...")
                print(f"     Output: {output_span_guided[:50]}...")

        results.append(
            {
                "sample_id": sample_id,
                "language": "EN",
                "toxicity_strength": toxicity_strength,
                "original_text": original_text,
                "detected_spans_text": span_info["detected_spans_text"],
                "detected_spans_intensity": span_info["detected_spans_intensity"],
                "span_empty": span_info["span_empty"],
                "output_unguided": output_unguided,
                "output_span_guided": output_span_guided,
            }
        )

    # Write output CSV
    print(f"\nWriting output CSV: {args.output_csv}")
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "language",
            "toxicity_strength",
            "original_text",
            "detected_spans_text",
            "detected_spans_intensity",
            "span_empty",
            "output_unguided",
            "output_span_guided",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Successfully wrote {len(results)} samples to {args.output_csv}")

    # Summary statistics
    span_empty_count = sum(1 for r in results if r["span_empty"])
    print(f"\nSummary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Samples with spans: {len(results) - span_empty_count}")
    print(f"  Samples without spans (span_empty=True): {span_empty_count}")


if __name__ == "__main__":
    main()
