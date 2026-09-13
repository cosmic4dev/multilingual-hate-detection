#!/usr/bin/env python3
"""
다양한 유형의 샘플을 생성하여 span이 감지되는 30개를 선별
- 문장 유형 다양화 (명령문, 질문, 평서문, 감탄문)
- 표현 방식 다양화 (직접적, 간접적, 비유적)
- 길이 다양화 (짧은 문장, 긴 문장)
- 주제 다양화
"""

import argparse
import csv
import os
import torch
import torch.nn as nn
from typing import Any, Dict, List

from transformers import AutoConfig, AutoModel, AutoTokenizer


class XLMRobertaDetector(nn.Module):
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
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = detector(**inputs)
        bio_probs = torch.softmax(out["bio_logits"], dim=-1)
        bio_pred = torch.argmax(bio_probs, dim=-1)[0]

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

    hate_spans = [s for s in spans if s.get("type") in ["SOFT", "HARD"]]
    return {"has_spans": len(hate_spans) > 0, "spans": hate_spans, "span_count": len(hate_spans)}


def build_diverse_candidates() -> List[Dict[str, str]]:
    """다양한 유형의 후보 샘플 생성"""
    candidates = []

    # 1. 명령문 유형 (강한) - span 감지 가능하도록
    candidates.extend([
        {"text": "Get the hell out of here, you worthless piece of trash.", "strength": "strong", "type": "명령문"},
        {"text": "Stop being such an idiot and think before you speak.", "strength": "strong", "type": "명령문"},
        {"text": "Shut your stupid mouth and listen for once.", "strength": "strong", "type": "명령문"},
        {"text": "Go away, you're being really annoying right now.", "strength": "mild", "type": "명령문"},
        {"text": "Please stop saying such dumb things—it's embarrassing.", "strength": "mild", "type": "명령문"},
        {"text": "Shut up, you stupid idiot.", "strength": "strong", "type": "명령문"},
        {"text": "Stop being so dumb and think clearly.", "strength": "mild", "type": "명령문"},
        {"text": "Get out, you're being an idiot.", "strength": "strong", "type": "명령문"},
        {"text": "Please stop acting like a fool.", "strength": "mild", "type": "명령문"},
        {"text": "Just stop being stupid, okay?", "strength": "mild", "type": "명령문"},
    ])

    # 2. 질문형 유형 - span 감지 가능하도록
    candidates.extend([
        {"text": "What the hell is wrong with you?", "strength": "strong", "type": "질문형"},
        {"text": "Are you really that stupid?", "strength": "strong", "type": "질문형"},
        {"text": "Why are you being such a jerk?", "strength": "mild", "type": "질문형"},
        {"text": "Can you stop being so annoying?", "strength": "mild", "type": "질문형"},
        {"text": "What makes you think that's acceptable?", "strength": "mild", "type": "질문형"},
        {"text": "How stupid can you be?", "strength": "strong", "type": "질문형"},
        {"text": "Are you really that dumb?", "strength": "mild", "type": "질문형"},
        {"text": "What's wrong with you, you idiot?", "strength": "strong", "type": "질문형"},
        {"text": "Why are you acting like such a fool?", "strength": "mild", "type": "질문형"},
        {"text": "Can't you see how stupid this is?", "strength": "mild", "type": "질문형"},
    ])

    # 3. 간접적/비유적 표현 - span 감지 가능하도록
    candidates.extend([
        {"text": "You're acting like a complete fool right now.", "strength": "strong", "type": "비유적"},
        {"text": "This behavior is absolutely ridiculous and unacceptable.", "strength": "strong", "type": "간접적"},
        {"text": "You're making yourself look really stupid here.", "strength": "mild", "type": "간접적"},
        {"text": "That was a foolish thing to do, don't you think?", "strength": "mild", "type": "간접적"},
        {"text": "You're behaving like an idiot right now.", "strength": "strong", "type": "비유적"},
        {"text": "You're acting stupid and it's not helping.", "strength": "mild", "type": "간접적"},
        {"text": "You look like a complete moron doing that.", "strength": "strong", "type": "비유적"},
        {"text": "You're being really dumb about this situation.", "strength": "mild", "type": "간접적"},
    ])

    # 4. 짧은 문장 (강도 높음)
    candidates.extend([
        {"text": "You're an idiot.", "strength": "strong", "type": "짧은문장"},
        {"text": "This is stupid.", "strength": "mild", "type": "짧은문장"},
        {"text": "You're pathetic.", "strength": "strong", "type": "짧은문장"},
        {"text": "That's ridiculous.", "strength": "mild", "type": "짧은문장"},
    ])

    # 5. 긴 문장 (복합 표현) - span 감지 가능하도록
    candidates.extend([
        {"text": "I can't believe how stupid and thoughtless you're being right now, and it's really disappointing to see someone act this way.", "strength": "strong", "type": "긴문장"},
        {"text": "Your behavior is not only inappropriate but also shows a complete lack of consideration for others, which is really frustrating.", "strength": "mild", "type": "긴문장"},
        {"text": "You're being such an idiot about this whole situation, and frankly, it's embarrassing to watch you make these mistakes over and over again.", "strength": "strong", "type": "긴문장"},
        {"text": "You're acting like such a stupid fool right now, and I can't believe you would do something so dumb.", "strength": "strong", "type": "긴문장"},
        {"text": "That was really stupid of you, and you should have known better than to act like an idiot.", "strength": "mild", "type": "긴문장"},
    ])

    # 6. 감탄문
    candidates.extend([
        {"text": "How stupid can you possibly be?", "strength": "strong", "type": "감탄문"},
        {"text": "What a complete idiot!", "strength": "strong", "type": "감탄문"},
        {"text": "How ridiculous is this?", "strength": "mild", "type": "감탄문"},
    ])

    # 7. 조건문/가정
    candidates.extend([
        {"text": "If you keep acting like an idiot, people will lose respect for you.", "strength": "strong", "type": "조건문"},
        {"text": "You would be less annoying if you just thought before speaking.", "strength": "mild", "type": "조건문"},
    ])

    # 8. 부정문
    candidates.extend([
        {"text": "I can't stand how stupid you're being right now.", "strength": "strong", "type": "부정문"},
        {"text": "There's nothing intelligent about what you just said.", "strength": "mild", "type": "부정문"},
    ])

    # 9. 복수 명령/요청
    candidates.extend([
        {"text": "Stop being an idiot and start using your brain for once.", "strength": "strong", "type": "복수명령"},
        {"text": "Please stop saying dumb things and think before you speak.", "strength": "mild", "type": "복수명령"},
    ])

    # 10. 일반적인 모욕 (다양한 단어) - span 감지 가능하도록 명시적 표현
    candidates.extend([
        {"text": "You're being a complete moron about this situation.", "strength": "strong", "type": "모욕"},
        {"text": "That was a really foolish decision on your part.", "strength": "mild", "type": "모욕"},
        {"text": "You're acting like a total fool and it's not helping anyone.", "strength": "strong", "type": "모욕"},
        {"text": "Your comment was really thoughtless and inappropriate.", "strength": "mild", "type": "모욕"},
        {"text": "You're such a clueless person sometimes.", "strength": "mild", "type": "모욕"},
        {"text": "You're such a stupid moron—just stop it.", "strength": "strong", "type": "모욕"},
        {"text": "That was a really stupid thing to do.", "strength": "mild", "type": "모욕"},
        {"text": "You're being an absolute idiot right now.", "strength": "strong", "type": "모욕"},
        {"text": "This is such a stupid idea—just think for once.", "strength": "mild", "type": "모욕"},
        {"text": "You're such a dumb fool—can't you see the problem?", "strength": "strong", "type": "모욕"},
        {"text": "That was really dumb—you should have known better.", "strength": "mild", "type": "모욕"},
        {"text": "You're acting like a complete idiot—it's embarrassing.", "strength": "strong", "type": "모욕"},
    ])

    # 11. 비교 표현
    candidates.extend([
        {"text": "You're being even more stupid than usual today.", "strength": "strong", "type": "비교"},
        {"text": "That was dumber than your usual comments.", "strength": "mild", "type": "비교"},
    ])

    # 12. 시간/과거 표현
    candidates.extend([
        {"text": "You've always been such an idiot about these things.", "strength": "strong", "type": "시간"},
        {"text": "You were really being stupid back there.", "strength": "mild", "type": "시간"},
    ])

    # 13. 추가 다양한 유형 (기존 span 감지 패턴 활용)
    candidates.extend([
        # 직접 비속어 포함 (span 감지 높음)
        {"text": "You're a fucking idiot—just stop talking.", "strength": "strong", "type": "직접비속어"},
        {"text": "This is fucking stupid and you know it.", "strength": "strong", "type": "직접비속어"},
        {"text": "You're such a stupid asshole—get out of here.", "strength": "strong", "type": "직접비속어"},
        {"text": "Stop being a fucking moron already.", "strength": "strong", "type": "직접비속어"},
        {"text": "You're being really stupid right now—just think.", "strength": "mild", "type": "직접비속어"},
        
        # "You're such a..." 패턴 (span 감지 높음)
        {"text": "You're such a stupid idiot for saying that.", "strength": "strong", "type": "such패턴"},
        {"text": "You're such a dumb fool—can't you see?", "strength": "strong", "type": "such패턴"},
        {"text": "You're such a moron sometimes.", "strength": "mild", "type": "such패턴"},
        {"text": "You're such an idiot—just listen for once.", "strength": "strong", "type": "such패턴"},
        
        # "That was..." 패턴
        {"text": "That was really stupid of you.", "strength": "mild", "type": "that패턴"},
        {"text": "That was such a dumb thing to say.", "strength": "mild", "type": "that패턴"},
        {"text": "That was idiotic—you should have known better.", "strength": "strong", "type": "that패턴"},
        
        # "You're being..." 패턴
        {"text": "You're being really stupid about this.", "strength": "mild", "type": "being패턴"},
        {"text": "You're being such an idiot right now.", "strength": "strong", "type": "being패턴"},
        {"text": "You're being a complete fool—just stop.", "strength": "strong", "type": "being패턴"},
        
        # "acting like" 패턴
        {"text": "You're acting like a stupid idiot.", "strength": "strong", "type": "acting패턴"},
        {"text": "You're acting like a complete moron.", "strength": "strong", "type": "acting패턴"},
        {"text": "You're acting like such a fool right now.", "strength": "mild", "type": "acting패턴"},
        
        # 복합 표현 (명령 + 모욕)
        {"text": "Shut up, you stupid idiot.", "strength": "strong", "type": "복합명령"},
        {"text": "Get out, you're being an idiot.", "strength": "strong", "type": "복합명령"},
        {"text": "Stop it, you're acting stupid.", "strength": "mild", "type": "복합명령"},
        
        # 짧은 강한 표현
        {"text": "You're an idiot.", "strength": "strong", "type": "짧은강한"},
        {"text": "You're stupid.", "strength": "mild", "type": "짧은강한"},
        {"text": "What an idiot!", "strength": "strong", "type": "짧은강한"},
        {"text": "How stupid!", "strength": "mild", "type": "짧은강한"},
        
        # "jerk" 포함 (span 감지 높음)
        {"text": "You're being such a jerk right now.", "strength": "mild", "type": "jerk패턴"},
        {"text": "Stop being a complete jerk.", "strength": "strong", "type": "jerk패턴"},
        {"text": "You're acting like a jerk—just stop.", "strength": "mild", "type": "jerk패턴"},
        
        # 다양한 길이의 문장들
        {"text": "You're stupid and you know it.", "strength": "mild", "type": "중간길이"},
        {"text": "You're such a stupid person who doesn't understand anything.", "strength": "strong", "type": "중간길이"},
        {"text": "That was really stupid, and you should feel bad about it.", "strength": "mild", "type": "중간길이"},
        
        # 14. 기존 span 감지 패턴 기반 다양한 변형
        # "jerk" 패턴 확장
        {"text": "You're acting like a complete jerk right now.", "strength": "mild", "type": "jerk확장"},
        {"text": "Stop being such a jerk about everything.", "strength": "mild", "type": "jerk확장"},
        {"text": "Why are you being such a jerk?", "strength": "mild", "type": "jerk확장"},
        {"text": "You're being a real jerk, you know that?", "strength": "mild", "type": "jerk확장"},
        {"text": "Can you stop acting like a jerk for once?", "strength": "mild", "type": "jerk확장"},
        
        # "stupid" 다양한 위치
        {"text": "That was stupid.", "strength": "mild", "type": "stupid변형"},
        {"text": "You're being really stupid.", "strength": "mild", "type": "stupid변형"},
        {"text": "How stupid can you be?", "strength": "strong", "type": "stupid변형"},
        {"text": "You're so stupid it's unbelievable.", "strength": "strong", "type": "stupid변형"},
        {"text": "This is really stupid—just stop.", "strength": "mild", "type": "stupid변형"},
        {"text": "Stop being stupid, okay?", "strength": "mild", "type": "stupid변형"},
        
        # "dumb" 패턴
        {"text": "That was really dumb of you.", "strength": "mild", "type": "dumb패턴"},
        {"text": "You're being so dumb right now.", "strength": "mild", "type": "dumb패턴"},
        {"text": "How dumb can you possibly be?", "strength": "strong", "type": "dumb패턴"},
        {"text": "You're such a dumb person.", "strength": "mild", "type": "dumb패턴"},
        
        # "idiot" 패턴
        {"text": "You're such an idiot.", "strength": "strong", "type": "idiot패턴"},
        {"text": "Are you really that much of an idiot?", "strength": "strong", "type": "idiot패턴"},
        {"text": "You're being a complete idiot.", "strength": "strong", "type": "idiot패턴"},
        {"text": "Why are you acting like such an idiot?", "strength": "strong", "type": "idiot패턴"},
        
        # "trash" 패턴
        {"text": "You're such trash.", "strength": "strong", "type": "trash패턴"},
        {"text": "You're a piece of trash.", "strength": "strong", "type": "trash패턴"},
        
        # "loser" 패턴
        {"text": "You're such a loser.", "strength": "strong", "type": "loser패턴"},
        {"text": "You're being a complete loser right now.", "strength": "strong", "type": "loser패턴"},
        
        # "fucking" 포함 다양한 구조
        {"text": "You're fucking stupid.", "strength": "strong", "type": "fucking변형"},
        {"text": "This is fucking ridiculous.", "strength": "strong", "type": "fucking변형"},
        {"text": "You're being a fucking idiot.", "strength": "strong", "type": "fucking변형"},
        {"text": "Stop being so fucking stupid.", "strength": "strong", "type": "fucking변형"},
        
        # "pathetic" 패턴
        {"text": "You're being really pathetic right now.", "strength": "strong", "type": "pathetic패턴"},
        {"text": "That was pathetic of you.", "strength": "strong", "type": "pathetic패턴"},
        
        # "asshole" 패턴
        {"text": "You're being such an asshole.", "strength": "strong", "type": "asshole패턴"},
        {"text": "Stop being an asshole for once.", "strength": "strong", "type": "asshole패턴"},
        
        # 다양한 문장 구조
        {"text": "You know what? You're being stupid.", "strength": "mild", "type": "도입구"},
        {"text": "I can't believe how stupid you're being.", "strength": "strong", "type": "도입구"},
        {"text": "Honestly, you're acting like an idiot.", "strength": "strong", "type": "도입구"},
        
        # 복문 구조
        {"text": "You're being stupid, and you know it.", "strength": "mild", "type": "복문"},
        {"text": "You're such an idiot, and it's embarrassing.", "strength": "strong", "type": "복문"},
        {"text": "That was really dumb, so just stop.", "strength": "mild", "type": "복문"},
        
        # 15. 추가 다양한 패턴 (30개 완성을 위해)
        {"text": "You're such a stupid fool.", "strength": "strong", "type": "추가패턴"},
        {"text": "That was really stupid of you to do.", "strength": "mild", "type": "추가패턴"},
        {"text": "You're being an absolute idiot right now.", "strength": "strong", "type": "추가패턴"},
        {"text": "Stop acting like such a stupid person.", "strength": "mild", "type": "추가패턴"},
        {"text": "You're such a dumb idiot—just listen.", "strength": "strong", "type": "추가패턴"},
        {"text": "That was a really stupid thing to say out loud.", "strength": "mild", "type": "추가패턴"},
        {"text": "You're being really dumb about this whole situation.", "strength": "mild", "type": "추가패턴"},
        {"text": "Why would you do something so stupid?", "strength": "strong", "type": "추가패턴"},
        {"text": "You're acting like a complete moron right now.", "strength": "strong", "type": "추가패턴"},
        {"text": "That was such a stupid mistake on your part.", "strength": "mild", "type": "추가패턴"},
    ])

    # sample_id 생성
    result = []
    for i, cand in enumerate(candidates, start=1):
        result.append({
            "sample_id": f"diverse_{cand['type']}_{i:03d}",
            "original_text": cand["text"],
            "toxicity_strength": cand["strength"],
            "type": cand["type"],
        })

    return result


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
        default="/root/multilingual-hate-detection/human_eval/diverse_samples_with_spans.csv",
    )
    args = parser.parse_args()

    print("Loading detector...")
    detector, tokenizer, device = load_detector(args.detector_model_path, args.hf_cache_dir)
    print("✅ Detector loaded")

    print("Generating diverse candidate samples...")
    candidates = build_diverse_candidates()
    print(f"Generated {len(candidates)} diverse candidates")

    print("\nTesting samples for span detection...")
    span_detected = []
    span_not_detected = []

    for i, sample in enumerate(candidates, 1):
        text = sample["original_text"]
        result = detect_spans(detector, tokenizer, device, text)

        sample["has_spans"] = result["has_spans"]
        sample["span_count"] = result["span_count"]

        if result["has_spans"]:
            span_detected.append(sample)
            print(f"[{i}/{len(candidates)}] ✅ {sample['sample_id']}: {result['span_count']} spans ({sample['type']})")
        else:
            span_not_detected.append(sample)
            if i % 10 == 0:
                print(f"[{i}/{len(candidates)}] ❌ {sample['sample_id']}: no spans")

    print(f"\n✅ Span 감지된 샘플: {len(span_detected)}개")
    print(f"❌ Span 미감지 샘플: {len(span_not_detected)}개")

    # 유형별 분류
    by_type = {}
    for sample in span_detected:
        t = sample["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(sample)

    print("\n=== 유형별 span 감지 통계 ===")
    for t, samples in sorted(by_type.items()):
        strong = sum(1 for s in samples if s["toxicity_strength"] == "strong")
        mild = sum(1 for s in samples if s["toxicity_strength"] == "mild")
        print(f"{t}: {len(samples)}개 (strong: {strong}, mild: {mild})")

    # 다양성을 고려하여 30개 선별
    print("\n선별 중: 다양성 고려하여 30개 선택...")
    selected = []
    
    # 각 유형에서 균등하게 선택 (최대 2-3개씩)
    max_per_type = 3
    for t in sorted(by_type.keys()):
        samples = by_type[t]
        # strong/mild 균형
        strong_samples = [s for s in samples if s["toxicity_strength"] == "strong"]
        mild_samples = [s for s in samples if s["toxicity_strength"] == "mild"]
        
        # 각 유형에서 최대 2개씩 먼저 선택 (다양성 확보)
        selected.extend(strong_samples[:1])
        selected.extend(mild_samples[:1])
        
        if len(selected) >= 30:
            break

    # 30개 미만이면 나머지 채우기 (유형 다양성 유지하며)
    if len(selected) < 30:
        # 이미 선택된 유형을 제외하고 나머지에서 추가
        selected_types = set(s["type"] for s in selected)
        remaining_by_type = {t: [s for s in by_type[t] if s not in selected] for t in by_type.keys()}
        
        # 아직 선택되지 않은 유형부터 채우기
        for t in sorted(remaining_by_type.keys()):
            if len(selected) >= 30:
                break
            samples = remaining_by_type[t]
            strong_samples = [s for s in samples if s["toxicity_strength"] == "strong"]
            mild_samples = [s for s in samples if s["toxicity_strength"] == "mild"]
            
            # 필요한 만큼만 추가
            needed = 30 - len(selected)
            if needed > 0:
                selected.extend(strong_samples[:min(1, needed)])
            needed = 30 - len(selected)
            if needed > 0:
                selected.extend(mild_samples[:min(1, needed)])
            
            if len(selected) >= 30:
                break
        
        # 여전히 부족하면 전체에서 strong/mild 균형 유지하며 채우기
        if len(selected) < 30:
            remaining = [s for s in span_detected if s not in selected]
            needed = 30 - len(selected)
            strong_remaining = [s for s in remaining if s["toxicity_strength"] == "strong"]
            mild_remaining = [s for s in remaining if s["toxicity_strength"] == "mild"]
            
            # strong/mild 균형 유지
            strong_count = sum(1 for s in selected if s["toxicity_strength"] == "strong")
            mild_count = sum(1 for s in selected if s["toxicity_strength"] == "mild")
            
            # 목표: strong 15, mild 15
            strong_needed = max(0, min(15 - strong_count, len(strong_remaining), needed))
            mild_needed = needed - strong_needed
            
            if mild_needed > len(mild_remaining):
                mild_needed = len(mild_remaining)
                strong_needed = needed - mild_needed
            
            selected.extend(strong_remaining[:strong_needed])
            selected.extend(mild_remaining[:mild_needed])
    
    # 최종적으로 30개가 아니면 남은 것으로 채우기
    if len(selected) < 30:
        remaining = [s for s in span_detected if s not in selected]
        needed = 30 - len(selected)
        
        # strong/mild 균형 고려
        strong_count = sum(1 for s in selected if s["toxicity_strength"] == "strong")
        mild_count = sum(1 for s in selected if s["toxicity_strength"] == "mild")
        
        # 목표: 각각 15개
        strong_needed = max(0, min(15 - strong_count, needed))
        mild_needed = needed - strong_needed
        
        strong_remaining = [s for s in remaining if s["toxicity_strength"] == "strong"]
        mild_remaining = [s for s in remaining if s["toxicity_strength"] == "mild"]
        
        if strong_needed > len(strong_remaining):
            strong_needed = len(strong_remaining)
            mild_needed = needed - strong_needed
        
        if mild_needed > len(mild_remaining):
            mild_needed = len(mild_remaining)
            strong_needed = needed - mild_needed
        
        selected.extend(strong_remaining[:strong_needed])
        selected.extend(mild_remaining[:mild_needed])

    selected = selected[:30]  # 정확히 30개

    # 최종 확인: 30개 미만이면 강제로 채우기
    strong_count = sum(1 for s in selected if s["toxicity_strength"] == "strong")
    mild_count = sum(1 for s in selected if s["toxicity_strength"] == "mild")
    
    if len(selected) < 30:
        remaining_all = [s for s in span_detected if s not in selected]
        needed = 30 - len(selected)
        
        # strong 15, mild 15 목표
        strong_target = 15
        mild_target = 15
        
        strong_needed = max(0, min(strong_target - strong_count, needed))
        mild_needed = needed - strong_needed
        
        if mild_needed < 0:
            mild_needed = 0
            strong_needed = needed
        
        strong_remaining = [s for s in remaining_all if s["toxicity_strength"] == "strong"]
        mild_remaining = [s for s in remaining_all if s["toxicity_strength"] == "mild"]
        
        # 실제 추가 가능한 만큼만
        strong_to_add = min(strong_needed, len(strong_remaining))
        mild_to_add = min(mild_needed, len(mild_remaining))
        
        selected.extend(strong_remaining[:strong_to_add])
        selected.extend(mild_remaining[:mild_to_add])
        
        # 여전히 부족하면 나머지로 채우기
        if len(selected) < 30:
            remaining = [s for s in remaining_all if s not in selected]
            needed = 30 - len(selected)
            selected.extend(remaining[:needed])
    
    selected = selected[:30]  # 정확히 30개
    
    print(f"\n📊 최종 선별된 샘플: {len(selected)}개")
    strong_count = sum(1 for s in selected if s["toxicity_strength"] == "strong")
    mild_count = sum(1 for s in selected if s["toxicity_strength"] == "mild")
    print(f"   Strong: {strong_count}개")
    print(f"   Mild: {mild_count}개")
    
    print("\n=== 최종 선택된 유형 분포 ===")
    final_by_type = {}
    for sample in selected:
        t = sample["type"]
        final_by_type[t] = final_by_type.get(t, 0) + 1
    for t, count in sorted(final_by_type.items()):
        print(f"{t}: {count}개")

    # CSV 저장
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "original_text", "toxicity_strength", "type", "has_spans", "span_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in selected:
            writer.writerow({
                "sample_id": sample["sample_id"],
                "original_text": sample["original_text"],
                "toxicity_strength": sample["toxicity_strength"],
                "type": sample["type"],
                "has_spans": sample["has_spans"],
                "span_count": sample["span_count"],
            })

    print(f"\n✅ Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()

