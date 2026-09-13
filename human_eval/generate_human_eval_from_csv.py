#!/usr/bin/env python3
"""
Generate human evaluation samples from CSV input.
- Guided: uses toxicity_strength and harmful span information
- Unguided: uses only original_text
- Generator: HuggingFace (e.g. Qwen2.5-7B-Instruct) or API (OpenAI, Anthropic)
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
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Models:
    device: str
    generator: Any
    generator_tokenizer: Any


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def load_generator(
    generator_model_name: str,
    hf_cache_dir: str,
    hf_token: Optional[str],
    local_files_only: bool = False,
) -> Models:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_kwargs = {"cache_dir": hf_cache_dir, "local_files_only": local_files_only}
    if hf_token:
        tok_kwargs["token"] = hf_token

    print("Loading generator tokenizer...")
    gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_name, **tok_kwargs)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    print(f"Loading generator model: {generator_model_name}...")
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "dtype": pick_dtype(),
        "cache_dir": hf_cache_dir,
        "local_files_only": local_files_only,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    gen = AutoModelForCausalLM.from_pretrained(
        generator_model_name,
        **model_kwargs,
    )
    print("✅ Generator model loaded")

    return Models(
        device=device,
        generator=gen,
        generator_tokenizer=gen_tokenizer,
    )


def parse_harmful_spans(harmful_spans_str: str) -> List[Dict[str, Any]]:
    """Parse harmful_spans JSON string."""
    try:
        spans = json.loads(harmful_spans_str)
        return spans if isinstance(spans, list) else []
    except:
        return []


def build_guided_prompt(original_text: str, toxicity_strength: str, harmful_span_texts: str) -> str:
    """Build guided prompt with toxicity_strength and harmful spans."""
    span_list = [s.strip() for s in harmful_span_texts.split("|") if s.strip()]
    span_list_str = "\n".join([f"- {s}" for s in span_list]) if span_list else "None detected"
    
    prompt = f'''You are given a sentence that contains harmful expressions.

Original sentence:
"{original_text}"

Toxicity strength: {toxicity_strength}
Detected harmful spans:
{span_list_str}

Instruction:
- Rewrite ONLY the harmful spans to make them non-toxic.
- Do NOT rewrite or paraphrase the entire sentence.
- Preserve the original meaning, tone, and structure as much as possible.
- If the toxicity is {toxicity_strength}, apply appropriate mitigation.
- Output ONE sentence only.'''

    return prompt


def build_unguided_prompt(original_text: str) -> str:
    """Build unguided prompt."""
    prompt = f'''You are given a sentence.

Original sentence:
"{original_text}"

Instruction:
- Rewrite the sentence to make it respectful and appropriate.
- Preserve the original meaning and intent as much as possible.
- Output ONE sentence only.'''

    return prompt


def clean_generation_output(out: str) -> str:
    """Normalize model output to a single sentence (shared by HF and API)."""
    # Qwen3 thinking mode: use content after </think> if present
    if "</think>" in out:
        out = out.split("</think>", 1)[-1].strip()
    out = re.sub(
        r'^(Here is the rewritten text:|Rewritten text:|Answer:|Result:)\s*',
        "",
        out,
        flags=re.IGNORECASE,
    ).strip()
    sentence_end = re.search(r'[.!?]\s', out)
    if sentence_end:
        out = out[: sentence_end.end() - 1].strip()
    else:
        out = out.split("\n")[0].strip()
    return out


def generate(
    models: Models, prompt: str, max_retries: int = 3
) -> str:
    """Generate output using Qwen2.5-7B-Instruct."""
    messages = [{"role": "user", "content": prompt}]

    tok = models.generator_tokenizer
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        # Qwen3 uses enable_thinking; disable for direct detox output
        try:
            prompt_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
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
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        generated_ids = outputs[0][prompt_len:]
        out = tok.decode(generated_ids, skip_special_tokens=True).strip()
        out = clean_generation_output(out)
        if quality_check(out):
            return out

    return out


def quality_check(output: str) -> bool:
    """Check output quality."""
    sentence_count = len(re.split(r'[.!?]\s+', output))
    if sentence_count > 1:
        return False

    forbidden_patterns = ["Okay", "Let's", "I think"]
    for pattern in forbidden_patterns:
        if pattern in output:
            return False

    return True


def generate_with_openai(prompt: str, model: str, api_key: Optional[str], max_retries: int = 3) -> str:
    """Generate one completion via OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = clean_generation_output(text)
            if quality_check(text):
                return text
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"OpenAI API error: {e}")
            return ""
    return ""


def generate_with_anthropic(prompt: str, model: str, api_key: Optional[str], max_retries: int = 3) -> str:
    """Generate one completion via Anthropic API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("Install anthropic: pip install anthropic")
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.content[0].text if resp.content else "").strip()
            text = clean_generation_output(text)
            if quality_check(text):
                return text
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"Anthropic API error: {e}")
            return ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/root/multilingual-hate-detection-4/human_eval/human_eval_inputs_from_json_30_balanced.csv",
    )
    parser.add_argument(
        "--generator_model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id (used when --api_provider is not set)",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default=None,
        choices=["openai", "anthropic"],
        help="Use API instead of HuggingFace: openai or anthropic",
    )
    parser.add_argument(
        "--api_model",
        type=str,
        default=None,
        help="API model id (e.g. gpt-4o, claude-3-5-sonnet-20241022). Required if --api_provider is set.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI or Anthropic (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env).",
    )
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_cache")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model from local cache only (no HF auth needed when model is cached)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/multilingual-hate-detection-4/human_eval/human_eval_ready_from_csv_30.csv",
    )
    args = parser.parse_args()

    use_api = args.api_provider is not None
    if use_api and not args.api_model:
        raise SystemExit("--api_model is required when --api_provider is set (e.g. gpt-4o, claude-3-5-sonnet-20241022)")

    models: Optional[Models] = None
    if not use_api:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        print("Loading generator model...")
        models = load_generator(
            generator_model_name=args.generator_model_name,
            hf_cache_dir=args.hf_cache_dir,
            hf_token=hf_token,
            local_files_only=args.local_files_only,
        )
        print("✅ Models loaded")
    else:
        print(f"Using API: {args.api_provider} / {args.api_model}")

    print(f"Reading input CSV: {args.input_csv}")
    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Processing {len(rows)} samples...")

    api_key = args.api_key or (os.environ.get("OPENAI_API_KEY") if args.api_provider == "openai" else os.environ.get("ANTHROPIC_API_KEY"))

    def do_generate_unguided(original_text: str) -> str:
        prompt = build_unguided_prompt(original_text)
        if use_api:
            if args.api_provider == "openai":
                return generate_with_openai(prompt, args.api_model, api_key)
            return generate_with_anthropic(prompt, args.api_model, api_key)
        assert models is not None
        return generate(models, prompt)

    def do_generate_guided(original_text: str, toxicity_strength: str, harmful_span_texts: str) -> str:
        prompt = build_guided_prompt(original_text, toxicity_strength, harmful_span_texts)
        if use_api:
            if args.api_provider == "openai":
                return generate_with_openai(prompt, args.api_model, api_key)
            return generate_with_anthropic(prompt, args.api_model, api_key)
        assert models is not None
        return generate(models, prompt)

    results = []
    for i, row in enumerate(rows, 1):
        sample_id = row.get("sample_id", f"sample_{i}")
        original_text = row["original_text"]
        toxicity_strength = row.get("toxicity_strength", "unknown")
        harmful_span_texts = row.get("harmful_span_texts", "")

        print(f"[{i}/{len(rows)}] Processing {sample_id}...")

        output_unguided = do_generate_unguided(original_text)
        output_guided = do_generate_guided(original_text, toxicity_strength, harmful_span_texts)

        results.append(
            {
                "sample_id": sample_id,
                "original_text": original_text,
                "toxicity_strength": toxicity_strength,
                "harmful_span_texts": harmful_span_texts,
                "output_unguided": output_unguided,
                "output_guided": output_guided,
            }
        )

    print(f"\nWriting output CSV: {args.output_csv}")
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "original_text",
            "toxicity_strength",
            "harmful_span_texts",
            "output_unguided",
            "output_guided",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Successfully wrote {len(results)} samples to {args.output_csv}")


if __name__ == "__main__":
    main()


