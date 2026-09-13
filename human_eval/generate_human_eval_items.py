#!/usr/bin/env python3
"""
Generate human-evaluation items in the same format as `human_eval/samples_raw.json`
and `human_eval/samples_for_form.csv`.

Creates 30 items:
- Strong toxicity: 15
- Mild toxicity: 15

Then generates two mitigations per item:
- output_unguided
- output_span_guided (guided by detector analysis)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


class XLMRobertaDetector(nn.Module):
    """XLM-RoBERTa based hate/toxicity detector (matches our saved checkpoint heads)."""

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
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_name, **tok_kwargs)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "dtype": pick_dtype(),
        "cache_dir": hf_cache_dir,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    gen = AutoModelForCausalLM.from_pretrained(generator_model_name, **model_kwargs)

    return Models(
        device=device,
        detector=detector,
        detector_tokenizer=detector_tokenizer,
        generator=gen,
        generator_tokenizer=gen_tokenizer,
    )


def detect(models: Models, text: str) -> Dict[str, Any]:
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
        target_probs = torch.softmax(out["target_logits"], dim=-1)
        intensity_pred = torch.argmax(intensity_probs, dim=-1).item()
        bio_pred = torch.argmax(bio_probs, dim=-1)[0]  # [seq_len]
        target_pred = torch.argmax(target_probs, dim=-1).item()

    # BIO spans (token-level indices)
    bio_labels = ["O", "B-SOFT", "I-SOFT", "B-HARD", "I-HARD"]
    tokens = models.detector_tokenizer.tokenize(text)
    spans = []
    current = None
    for i, (tok, lab_idx) in enumerate(zip(tokens, bio_pred.tolist())):
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

    return {
        "intensity": intensity_pred,
        "intensity_prob": intensity_probs[0][intensity_pred].item(),
        "intensity_probs": intensity_probs[0].tolist(),
        "bio_spans": spans,
        "target": target_pred,
        "target_prob": target_probs[0][target_pred].item(),
        "target_probs": target_probs[0].tolist(),
    }


def build_prompt(text: str, det: Dict[str, Any], mode: str) -> str:
    # Prepare a simple span list
    hate_spans = [s for s in det.get("bio_spans", []) if s.get("type") in ["SOFT", "HARD"]]
    span_list = []
    if hate_spans:
        span_list.append("Harmful spans: " + ", ".join([s.get("text", "") for s in hate_spans]))

    if mode == "unguided":
        system_msg = "You rewrite text to be respectful and appropriate while preserving meaning and intent."
        user_msg = f"""Rewrite the following text.

Text: "{text}"

Output ONLY the rewritten text, with no additional commentary."""
    else:
        system_msg = "You are an expert in content moderation and hate speech mitigation."
        user_msg = f"""Rewrite the following text to make it more respectful and appropriate while preserving the original meaning and intent.

Original Text: "{text}"

Detection Analysis:
- Intensity Level: {det['intensity']}/2 (0=NORMAL, 1=OFFENSIVE, 2=HATE)
- Confidence: {det['intensity_prob']:.2f}
- Detected Elements: {'; '.join(span_list) if span_list else 'None detected'}

Guidelines:
1. Maintain the original meaning and intent
2. Use respectful and appropriate language
3. Avoid counter-speech or aggressive responses
4. Focus on constructive communication
5. Preserve the core message while removing harmful elements

Output ONLY the rewritten text, with no additional commentary."""

    return system_msg, user_msg


def generate(models: Models, text: str, det: Dict[str, Any], mode: str, max_new_tokens: int) -> str:
    system_msg, user_msg = build_prompt(text, det, mode)
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    tok = models.generator_tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system_msg + "\n\n" + user_msg

    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(models.generator.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = models.generator.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    # Decode ONLY newly generated tokens (robust even with chat templates)
    generated_ids = outputs[0][prompt_len:]
    out = tok.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean-up
    out = re.sub(r'^(Here is the rewritten text:|Rewritten text:)\s*', "", out, flags=re.IGNORECASE).strip()
    # If the model added multiple paragraphs, keep the first non-empty line
    if "\n" in out:
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        out = lines[0] if lines else out
    out = out.strip().strip('"').strip("'").strip()
    return out


def build_items() -> List[Dict[str, str]]:
    # Note: We intentionally avoid slurs or protected-group targeting in these prompts.
    strong = [
        "You're a disgusting piece of trash and nobody can stand you.",
        "Shut up. You're pathetic and completely useless.",
        "You're a horrible person and you ruin everything you touch.",
        "You're so stupid it hurts. Stop embarrassing yourself.",
        "You're a liar and a fraud. Everyone knows you're fake.",
        "You're an absolute failure and a complete joke.",
        "You're unbearable—no one wants you around.",
        "You're selfish, cruel, and you don't deserve any respect.",
        "You're incompetent and you should be ashamed of yourself.",
        "You're trash. That's all you are.",
        "You're disgusting and I can't stand looking at you.",
        "You're the worst. You make everything worse for everyone.",
        "You're so obnoxious and annoying—just stop.",
        "You're a coward and you never take responsibility.",
        "You're worthless and you contribute nothing.",
    ]

    mild = [
        "You're being really rude right now.",
        "That was a dumb thing to say.",
        "You're annoying—please stop.",
        "I'm tired of your nonsense.",
        "You're acting like a jerk.",
        "Can you not be so rude?",
        "That's a pretty ignorant comment.",
        "You're being disrespectful.",
        "Please stop talking like that.",
        "You're being petty and unfair.",
        "That comment was unnecessary and mean.",
        "You're overreacting and it's getting old.",
        "You're not making any sense.",
        "You're being childish.",
        "This is ridiculous—just calm down.",
    ]

    items: List[Dict[str, str]] = []
    for i, t in enumerate(strong, start=1):
        items.append({"sample_id": f"heval_strong_{i:02d}", "original_text": t, "toxicity_strength": "strong"})
    for i, t in enumerate(mild, start=1):
        items.append({"sample_id": f"heval_mild_{i:02d}", "original_text": t, "toxicity_strength": "mild"})
    return items


def write_outputs(raw_items: List[Dict[str, Any]], out_json: str, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "original_text", "toxicity_strength", "output_unguided", "output_span_guided"])
        for it in raw_items:
            writer.writerow(
                [
                    it["sample_id"],
                    it["original_text"],
                    it["toxicity_strength"],
                    it["output_unguided"],
                    it["output_span_guided"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector_model_path",
        type=str,
        default="/root/multilingual-hate-detection/backend/model/english_xlmr_model.pt",
        help="Path to detector checkpoint (.pt)",
    )
    parser.add_argument(
        "--generator_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace repo id or local path",
    )
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token (prefer `huggingface-cli login`)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/multilingual-hate-detection/human_eval/samples_raw_30.json",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/multilingual-hate-detection/human_eval/samples_for_form_30.csv",
    )
    args = parser.parse_args()

    # Prefer explicit arg, else env, else rely on `huggingface-cli login` cached token.
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    models = load_models(
        detector_ckpt_path=args.detector_model_path,
        generator_model_name=args.generator_model_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=hf_token,
    )

    items = build_items()
    out: List[Dict[str, Any]] = []

    for it in items:
        text = it["original_text"]
        det = detect(models, text)
        unguided = generate(models, text, det, mode="unguided", max_new_tokens=args.max_new_tokens)
        guided = generate(models, text, det, mode="guided", max_new_tokens=args.max_new_tokens)

        out.append(
            {
                "sample_id": it["sample_id"],
                "original_text": text,
                "toxicity_strength": it["toxicity_strength"],
                "output_unguided": unguided,
                "output_span_guided": guided,
                "_detected_spans": det.get("bio_spans", []),
            }
        )

    write_outputs(out, args.output_json, args.output_csv)
    print(f"✅ Wrote: {args.output_json}")
    print(f"✅ Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()


