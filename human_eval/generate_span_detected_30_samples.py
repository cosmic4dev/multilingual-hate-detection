#!/usr/bin/env python3
"""
HateXplain 기반 샘플 중에서 span이 감지되는 것들을 찾아서 30개를 만들어야 함.
현재 30개 중 6개만 span이 감지되었으므로, 더 많은 샘플을 테스트해서
span이 감지되는 것들만 선별하여 30개를 만든다.
"""

import argparse
import csv
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoConfig, AutoModel, AutoTokenizer


class XLMRobertaDetector(nn.Module):
    """XLM-RoBERTa based hate span detector"""

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


def load_detector(detector_ckpt_path: str, hf_cache_dir: str) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_kwargs = {"cache_dir": hf_cache_dir}

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

    return detector, detector_tokenizer, device


def detect_spans(detector, tokenizer, device, text: str) -> Dict[str, Any]:
    """Detect spans and return span information"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = detector(**inputs)
        bio_probs = torch.softmax(out["bio_logits"], dim=-1)
        bio_pred = torch.argmax(bio_probs, dim=-1)[0]

    # BIO spans
    bio_labels = ["O", "B-SOFT", "I-SOFT", "B-HARD", "I-HARD"]
    tokens = tokenizer.tokenize(text)
    spans = []
    current = None
    bio_pred_list = bio_pred.tolist()
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
    return {"has_spans": len(hate_spans) > 0, "spans": hate_spans}


def build_candidate_samples() -> List[Dict[str, str]]:
    """HateXplain 스타일의 추가 후보 샘플들 (span 감지 가능하도록 명시적 표현 포함)"""
    candidates = []

    # Strong toxicity samples (명시적 비속어 포함)
    strong_candidates = [
        "You're a fucking idiot and everyone hates you.",
        "This is complete bullshit and you're full of it.",
        "You're a worthless piece of shit.",
        "Go fuck yourself, you pathetic loser.",
        "You're so fucking annoying—just shut the hell up.",
        "What the fuck is wrong with you?",
        "You're such a stupid asshole.",
        "This is fucking ridiculous and you're an idiot.",
        "You're a fucking moron who can't do anything right.",
        "Shut the fuck up, you're being a complete jerk.",
        "You're a fucking disgrace to everyone around you.",
        "This is absolutely fucking stupid.",
        "You're a fucking waste of space.",
        "What a fucking joke—you're completely useless.",
        "You're being a fucking asshole right now.",
        "This is so fucking dumb, I can't believe it.",
        "You're a fucking liar and everyone knows it.",
        "Stop being such a fucking idiot.",
        "You're a fucking failure at everything you do.",
        "This is fucking ridiculous—you're out of your mind.",
        "You're a dumb fuck who doesn't understand anything.",
        "This is such fucking nonsense—you're clueless.",
        "You're a fucking idiot for saying that.",
        "What the fuck are you talking about?",
        "You're being a fucking moron right now.",
        "This is fucking stupid—just stop it.",
        "You're a fucking waste of time and energy.",
        "Go to hell, you fucking piece of garbage.",
        "You're such a fucking loser—get a life.",
        "This is complete fucking bullshit.",
        "You're a fucking joke and everyone knows it.",
        "Shut your fucking mouth and listen.",
        "You're a fucking disgrace to humanity.",
        "This is so fucking ridiculous—you're insane.",
        "You're a fucking moron who never learns.",
        "What the fuck is your problem?",
        "You're being a complete fucking asshole.",
        "This is fucking pathetic—you should be ashamed.",
        "You're a fucking failure—just give up.",
        "Stop being such a fucking idiot already.",
    ]

    # Mild toxicity samples (span 감지 가능하도록 약간 강한 표현)
    mild_candidates = [
        "You're being really annoying right now.",
        "That was a really stupid thing to do.",
        "You're acting like a complete jerk.",
        "This is so annoying—can you stop?",
        "You're being really rude and it's bothering me.",
        "That comment was really dumb and unnecessary.",
        "You're being really inconsiderate right now.",
        "This is really stupid and makes no sense.",
        "You're being really obnoxious—please stop.",
        "That was a really ignorant thing to say.",
        "You're being really immature about this.",
        "This is really dumb—just think before you speak.",
        "You're being really disrespectful right now.",
        "That was a really insensitive comment.",
        "You're being really thoughtless about this.",
        "This is really annoying—can you be more considerate?",
        "You're being really petty right now.",
        "That was a really foolish thing to say.",
        "You're being really unreasonable about this.",
        "This is really stupid—you should know better.",
        "You're such a stupid person for saying that.",
        "This is really dumb—you don't know what you're talking about.",
        "You're being an idiot right now—just stop.",
        "That was a really stupid and thoughtless comment.",
        "You're acting like a complete fool.",
        "This is so dumb—you're missing the point.",
        "You're being really stupid about this situation.",
        "That was a dumb comment—think before you speak.",
        "You're acting really stupid right now.",
        "This is stupid—you're not making any sense.",
        "You're being such an idiot about this.",
        "That was a really dumb thing to say—you should know better.",
        "You're acting like a complete idiot.",
        "This is so stupid—just listen for once.",
        "You're being really dumb about this whole thing.",
        "That comment was stupid and unnecessary.",
        "You're acting stupid—just stop and think.",
        "This is really dumb—you're not helping.",
        "You're being an idiot—can you just stop?",
        "That was stupid—you don't understand the situation.",
        # 더 많은 mild 후보 (명시적 비속어/모욕 단어 포함)
        "You're being really stupid about this.",
        "That was such a stupid thing to say.",
        "You're acting like a complete idiot.",
        "This is stupid—you don't understand.",
        "You're being an idiot—just think.",
        "That comment was really stupid and dumb.",
        "You're such a stupid person for doing that.",
        "This is really stupid—you're not helping.",
        "You're being stupid and idiotic right now.",
        "That was stupid—you're missing the point.",
        "You're acting stupid—can you just stop?",
        "You're such a stupid idiot for saying that.",
        "This is stupid—you're being completely dumb.",
        "You're a stupid person who doesn't get it.",
        "That was really stupid—you're such an idiot.",
        "You're acting like a stupid fool right now.",
        "This is so stupid—you don't understand anything.",
        "You're being really stupid and idiotic.",
        "That comment was stupid—you're an idiot.",
        "You're such a stupid person—just listen.",
        "You're such a stupid fool—just stop talking.",
        "This is really stupid—you're being an idiot.",
        "You're acting like a complete stupid idiot.",
        "That was a stupid and idiotic thing to say.",
        "You're being really stupid and dumb right now.",
        "This is stupid—you're such an idiot.",
        "You're a stupid person who doesn't get it.",
        "That comment was really stupid and dumb.",
        "You're being stupid—can you just think?",
        "This is so stupid—you're acting like an idiot.",
        "You're such a stupid idiot for saying that.",
        "That was really stupid—you're being dumb.",
        "You're acting stupid and idiotic right now.",
        "This is stupid—you don't understand anything.",
        "You're being a stupid fool about this.",
        "That was stupid—you're such an idiot.",
        "You're really stupid for thinking that way.",
        "This is stupid—you're being completely dumb.",
        "You're such a stupid person—just listen.",
        "That comment was stupid and idiotic.",
        # 더 많은 strong 후보
        "Fuck you—you're a complete idiot.",
        "You're a stupid fucking moron—just shut up.",
        "This is fucking stupid—you're an idiot.",
        "Fuck off—you're being a stupid asshole.",
        "You're such a fucking stupid idiot.",
        "This is complete fucking bullshit—you're dumb.",
        "Fuck you and your stupid ideas.",
        "You're a fucking moron—so stupid.",
        "This is stupid fucking nonsense—just stop.",
        "Fuck this—you're such an idiot.",
        "You're a fucking stupid waste of space.",
        "This is fucking stupid—you don't get it.",
        "Fuck off—you're a complete idiot.",
        "You're such a fucking stupid person.",
        "This is stupid—fuck you and your nonsense.",
        # 추가 후보
        "You stupid fucking idiot—just stop.",
        "Fuck you, you complete moron.",
        "You're a fucking stupid waste of time.",
        "This is fucking stupid—you're an idiot.",
        "You're such a fucking dumb idiot.",
        "Fuck off—you're being stupid.",
        "You're a stupid fucking moron who never learns.",
        "This is fucking stupid—just shut up.",
        "You're such a fucking stupid fool.",
        "Fuck this—you're being a stupid idiot.",
        "You're a fucking moron—so stupid and dumb.",
        "This is stupid fucking bullshit.",
        "Fuck you and your stupid ideas.",
        "You're such a fucking stupid loser.",
        "This is fucking stupid—you're completely dumb.",
        "Fuck off—you're a stupid fucking idiot.",
        "You're being a stupid fucking asshole.",
        "This is stupid—fuck you, you moron.",
        "You're a fucking idiot—so stupid.",
        "Fuck this stupid nonsense.",
    ]

    for i, text in enumerate(strong_candidates, start=1):
        candidates.append(
            {
                "sample_id": f"candidate_strong_{i:02d}",
                "original_text": text,
                "toxicity_strength": "strong",
            }
        )

    for i, text in enumerate(mild_candidates, start=1):
        candidates.append(
            {
                "sample_id": f"candidate_mild_{i:02d}",
                "original_text": text,
                "toxicity_strength": "mild",
            }
        )

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector_model_path",
        type=str,
        default="/root/multilingual-hate-detection/backend/model/retrained_english_xlmr_model.pt",
    )
    parser.add_argument("--hf_cache_dir", type=str, default="/tmp/hf_cache")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/root/multilingual-hate-detection/human_eval/samples_with_spans_candidates.csv",
    )
    args = parser.parse_args()

    print("Loading detector...")
    detector, tokenizer, device = load_detector(args.detector_model_path, args.hf_cache_dir)
    print("✅ Detector loaded")

    # 기존 30개 샘플 + 추가 후보 샘플
    print("Loading existing samples...")
    existing_samples = []
    with open("/root/multilingual-hate-detection/human_eval/samples_for_form_30.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_samples.append(row)

    print(f"Existing samples: {len(existing_samples)}")
    
    # 추가 후보 샘플
    candidate_samples = build_candidate_samples()
    print(f"Candidate samples: {len(candidate_samples)}")

    all_samples = existing_samples + candidate_samples
    print(f"Total samples to test: {len(all_samples)}")

    print("\nTesting samples for span detection...")
    span_detected_samples = []
    span_not_detected_samples = []

    for i, sample in enumerate(all_samples, 1):
        text = sample["original_text"]
        result = detect_spans(detector, tokenizer, device, text)

        sample["has_spans"] = result["has_spans"]
        sample["span_count"] = len(result["spans"])

        if result["has_spans"]:
            span_detected_samples.append(sample)
            print(f"[{i}/{len(all_samples)}] ✅ {sample['sample_id']}: spans detected ({len(result['spans'])} spans)")
        else:
            span_not_detected_samples.append(sample)
            print(f"[{i}/{len(all_samples)}] ❌ {sample['sample_id']}: no spans")

    print(f"\n✅ Span 감지된 샘플: {len(span_detected_samples)}개")
    print(f"❌ Span 미감지 샘플: {len(span_not_detected_samples)}개")

    # strong/mild 균형 맞춰서 30개 선별
    strong_detected = [s for s in span_detected_samples if s["toxicity_strength"] == "strong"]
    mild_detected = [s for s in span_detected_samples if s["toxicity_strength"] == "mild"]

    print(f"\nStrong with spans: {len(strong_detected)}")
    print(f"Mild with spans: {len(mild_detected)}")

    # 15개씩 선별 (가능한 경우)
    selected_samples = []
    if len(strong_detected) >= 15:
        selected_samples.extend(strong_detected[:15])
    else:
        selected_samples.extend(strong_detected)

    if len(mild_detected) >= 15:
        selected_samples.extend(mild_detected[:15])
    else:
        selected_samples.extend(mild_detected)

    # 30개 미만이면 추가로 채움
    if len(selected_samples) < 30:
        remaining = [s for s in span_detected_samples if s not in selected_samples]
        needed = 30 - len(selected_samples)
        selected_samples.extend(remaining[:needed])

    print(f"\n📊 최종 선별된 샘플: {len(selected_samples)}개")
    print(f"   Strong: {sum(1 for s in selected_samples if s['toxicity_strength'] == 'strong')}개")
    print(f"   Mild: {sum(1 for s in selected_samples if s['toxicity_strength'] == 'mild')}개")

    # CSV 저장
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "original_text", "toxicity_strength", "has_spans", "span_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in selected_samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "original_text": sample["original_text"],
                    "toxicity_strength": sample["toxicity_strength"],
                    "has_spans": sample["has_spans"],
                    "span_count": sample["span_count"],
                }
            )

    print(f"\n✅ Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()

