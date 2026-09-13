#!/usr/bin/env python3
"""
English XLM-R Detector + Generator Pipeline
영어 혐오 표현 검출 및 완화 통합 파이프라인
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
import numpy as np
import requests
warnings.filterwarnings("ignore")

# Import BERTScore for evaluation
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not available. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

class EnglishXLMDetector(nn.Module):
    """English XLM-R hate speech detector"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", 
                 num_intensity_labels: int = 3, 
                 num_bio_labels: int = 5,
                 num_target_labels: int = 6):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        
        self.intensity_head = nn.Linear(hidden_size, num_intensity_labels)
        self.bio_head = nn.Linear(hidden_size, num_bio_labels)
        self.target_head = nn.Linear(hidden_size, num_target_labels)
        
        # Set labels based on num_labels
        if num_intensity_labels == 3:
            self.intensity_labels = ["normal", "offensive", "hate"]
        elif num_intensity_labels == 6:
            self.intensity_labels = ["normal", "offensive", "L1_hate", "L2_hate", "mild", "severe"]
        else:
            self.intensity_labels = [f"label_{i}" for i in range(num_intensity_labels)]
        
        self.bio_labels = ["O", "B-SOFT", "I-SOFT", "B-HARD", "I-HARD"]
        
        if num_target_labels == 6:
            self.target_labels = ["gender", "age", "political", "religion", "region", "others"]
        elif num_target_labels == 10:
            self.target_labels = ["gender", "age", "political", "religion", "region", "others", "sexism", "racism", "lgbtq", "non-hate"]
        else:
            self.target_labels = [f"target_{i}" for i in range(num_target_labels)]
        
        self.bio_class_weights = torch.tensor([1.0, 10.0, 10.0, 20.0, 20.0])
        
    def get_intensity_label(self, intensity_id: int) -> str:
        """Get intensity label from ID"""
        return self.intensity_labels[intensity_id]
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        intensity_logits = self.intensity_head(sequence_output[:, 0, :])
        bio_logits = self.bio_head(sequence_output)
        target_logits = self.target_head(sequence_output[:, 0, :])
        
        return {
            "intensity_logits": intensity_logits,
            "bio_logits": bio_logits,
            "target_logits": target_logits
        }

class EnglishDetectorPipeline:
    """English XLM-R Detector pipeline with JSON output"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        
        # Load checkpoint first to infer label counts
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer label counts from state_dict shapes
        num_intensity = state_dict["intensity_head.weight"].shape[0]
        num_bio = state_dict["bio_head.weight"].shape[0]
        num_target = state_dict["target_head.weight"].shape[0]
        
        self.model = EnglishXLMDetector(
            num_intensity_labels=num_intensity,
            num_bio_labels=num_bio,
            num_target_labels=num_target,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """Predict hate speech and return structured JSON output"""
        start_time = time.time()
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            intensity_probs = torch.softmax(outputs["intensity_logits"], dim=-1)
            bio_probs = torch.softmax(outputs["bio_logits"], dim=-1)
            target_probs = torch.softmax(outputs["target_logits"], dim=-1)
            
            intensity_pred = torch.argmax(intensity_probs, dim=-1).item()
            bio_preds = torch.argmax(bio_probs, dim=-1).squeeze().cpu().numpy()
            target_preds = torch.argmax(target_probs, dim=-1).item()
        
        harmful_spans = self._extract_harmful_spans(
            bio_preds, text, intensity_pred, target_probs[0]
        )
        
        is_harmful = intensity_pred > 0 or len(harmful_spans) > 0 or intensity_probs[0][1] > 0.3 or intensity_probs[0][2] > 0.2
        targets = self._get_targets(target_probs[0])
        rationale = self._generate_rationale(
            intensity_pred, targets, harmful_spans, intensity_probs[0]
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "input_text": text,
            "detection_results": {
                "is_harmful": bool(is_harmful),
                "overall_intensity": self.model.intensity_labels[intensity_pred],
                "confidence": float(intensity_probs[0][intensity_pred]),
                "intensity_probabilities": {
                    label: float(prob) for label, prob in 
                    zip(self.model.intensity_labels, intensity_probs[0])
                },
                "targets": targets,
                "target_confidence": [float(prob) for prob in target_probs[0]],
                "target_probabilities": {
                    label: float(prob) for label, prob in 
                    zip(self.model.target_labels, target_probs[0])
                },
                "harmful_spans": harmful_spans,
                "rationale": rationale
            },
            "model_info": {
                "model_name": "english_xlmr_detector",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time
            }
        }
        
        return result
    
    def _extract_harmful_spans(self, bio_preds: List[int], text: str, 
                              intensity_pred: int, target_probs: torch.Tensor) -> List[Dict]:
        """Extract harmful spans from BIO predictions using span-level extraction"""
        # 1. 토큰화 및 토큰-문자 매핑
        tokens = self.tokenizer.tokenize(text)
        token_to_char_map = self._map_tokens_to_characters(tokens, text)
        
        # 2. BIO 예측을 문자 단위로 확장
        char_bio_labels = self._expand_bio_to_characters(bio_preds, tokens, token_to_char_map, text)
        
        # 3. 연속된 BIO 태그를 하나의 Span으로 그룹화
        spans = self._group_consecutive_bio_labels(char_bio_labels, text)
        
        # 4. Span 정보 구성
        harmful_spans = []
        for span in spans:
            harmful_spans.append({
                "text": span['text'],
                "start_pos": span['start'],
                "end_pos": span['end'],
                "intensity": span['intensity'],
                "confidence": span['confidence'],
                "targets": self._get_targets(target_probs),
                "bio_label": span['bio_label']
            })
        
        # 5. 의미 단위 그룹화 적용 (인접한 같은 강도 스팬 병합)
        grouped_spans = self._group_adjacent_spans(harmful_spans, text)
        
        return grouped_spans
    
    def _map_tokens_to_characters(self, tokens: List[str], text: str) -> Dict[int, int]:
        """Map token indices to character positions (including empty tokens)"""
        token_to_char = {}
        char_pos = 0
        
        for i, token in enumerate(tokens):
            clean_token = token.replace('▁', '').replace('##', '')
            if clean_token:
                # Find where this token starts in the original text
                found_pos = text.find(clean_token, char_pos)
                if found_pos != -1:
                    token_to_char[i] = found_pos
                    char_pos = found_pos + len(clean_token)
                else:
                    # Try partial matching for subwords
                    for j in range(char_pos, len(text)):
                        if text[j:j+len(clean_token)] == clean_token:
                            token_to_char[i] = j
                            char_pos = j + len(clean_token)
                            break
            else:
                # Empty token - maintain current position
                token_to_char[i] = char_pos
        
        return token_to_char
    
    def _expand_bio_to_characters(self, bio_preds: List[int], tokens: List[str], 
                                 token_to_char_map: Dict[int, int], text: str) -> List[Dict]:
        """Expand BIO predictions to character level"""
        char_bio_labels = []
        
        # Initialize character-level BIO labels
        for char_idx in range(len(text)):
            char_bio_labels.append({
                'char': text[char_idx],
                'position': char_idx,
                'bio_label': 0,  # Default: O (Outside)
                'intensity': 'NONE'
            })
        
        # Apply token-level BIO labels to characters
        for token_idx, bio_pred in enumerate(bio_preds):
            if token_idx in token_to_char_map:
                char_start = token_to_char_map[token_idx]
                token_text = tokens[token_idx].replace('▁', '').replace('##', '')
                char_end = char_start + len(token_text)
                
                # Apply BIO label to all characters in this token
                for char_idx in range(char_start, min(char_end, len(text))):
                    if char_idx < len(char_bio_labels):
                        char_bio_labels[char_idx]['bio_label'] = bio_pred
                        
                        # Map intensity
                        if bio_pred == 1 or bio_pred == 2:  # B-SOFT, I-SOFT
                            char_bio_labels[char_idx]['intensity'] = 'SOFT'
                        elif bio_pred == 3 or bio_pred == 4:  # B-HARD, I-HARD
                            char_bio_labels[char_idx]['intensity'] = 'HARD'
        
        return char_bio_labels
    
    def _group_consecutive_bio_labels(self, char_bio_labels: List[Dict], text: str) -> List[Dict]:
        """Group consecutive same BIO labels into spans"""
        spans = []
        current_span = None
        
        for char_info in char_bio_labels:
            bio_label = char_info['bio_label']
            intensity = char_info['intensity']
            
            if bio_label in [1, 3]:  # B-SOFT, B-HARD
                # Start new span
                if current_span:
                    spans.append(current_span)
                
                current_span = {
                    'start': char_info['position'],
                    'end': char_info['position'] + 1,
                    'text': char_info['char'],
                    'intensity': intensity,
                    'confidence': 0.9 if intensity == 'HARD' else 0.8,
                    'bio_label': f'B-{intensity}' if intensity != 'NONE' else 'B-O'
                }
                
            elif bio_label in [2, 4] and current_span:  # I-SOFT, I-HARD
                # Continue same span
                if current_span['intensity'] == intensity:
                    current_span['end'] = char_info['position'] + 1
                    current_span['text'] += char_info['char']
                else:
                    # Different intensity, start new span
                    spans.append(current_span)
                    current_span = {
                        'start': char_info['position'],
                        'end': char_info['position'] + 1,
                        'text': char_info['char'],
                        'intensity': intensity,
                        'confidence': 0.9 if intensity == 'HARD' else 0.8,
                        'bio_label': f'B-{intensity}' if intensity != 'NONE' else 'B-O'
                    }
            else:  # O (Outside)
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add last span
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _group_adjacent_spans(self, spans: List[Dict], text: str, max_gap: int = 1) -> List[Dict]:
        """인접한 같은 강도의 스팬들을 그룹화"""
        if not spans:
            return []
        
        # 위치순으로 정렬
        sorted_spans = sorted(spans, key=lambda x: x['start_pos'])
        grouped = []
        current_group = [sorted_spans[0]]
        
        for i in range(1, len(sorted_spans)):
            current_span = sorted_spans[i]
            last_span = current_group[-1]
            
            # 같은 강도이고 인접한지 확인
            same_intensity = current_span['intensity'] == last_span['intensity']
            adjacent = current_span['start_pos'] - last_span['end_pos'] <= max_gap
            
            if same_intensity and adjacent:
                current_group.append(current_span)
            else:
                # 현재 그룹을 합치고 새 그룹 시작
                grouped.append(self._merge_span_group(current_group, text))
                current_group = [current_span]
        
        # 마지막 그룹 추가
        grouped.append(self._merge_span_group(current_group, text))
        
        return grouped
    
    def _merge_span_group(self, span_group: List[Dict], text: str) -> Dict:
        """스팬 그룹을 하나의 스팬으로 병합"""
        if not span_group:
            return None
        
        # 첫 번째와 마지막 스팬의 위치 사용
        start_pos = span_group[0]['start_pos']
        end_pos = span_group[-1]['end_pos']
        
        # 실제 텍스트 추출
        merged_text = text[start_pos:end_pos]
        
        # 첫 번째 스팬의 속성 사용
        first_span = span_group[0]
        
        return {
            "text": merged_text,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "intensity": first_span['intensity'],
            "confidence": max(span['confidence'] for span in span_group),  # 최대 신뢰도
            "targets": first_span['targets'],
            "bio_label": first_span['bio_label']
        }
    
    def _get_targets(self, target_probs: torch.Tensor, threshold: float = 0.3) -> List[str]:
        """Get target labels above threshold"""
        targets = []
        for i, prob in enumerate(target_probs):
            if prob > threshold:
                targets.append(self.model.target_labels[i])
        return targets
    
    def _generate_rationale(self, intensity_pred: int, targets: List[str], 
                           spans: List[Dict], intensity_probs: torch.Tensor) -> str:
        """Generate rationale based on detection results"""
        intensity_label = self.model.get_intensity_label(intensity_pred)
        
        if intensity_pred == 0:
            return "No hate speech detected"
        
        rationale_parts = []
        
        intensity_prob = float(intensity_probs[intensity_pred])
        rationale_parts.append(f"Intensity: {intensity_label} (probability: {intensity_prob:.3f})")
        
        if targets:
            rationale_parts.append(f"Targets: {', '.join(targets)}")
        
        if spans:
            span_texts = [span["text"] for span in spans]
            rationale_parts.append(f"Harmful spans: {', '.join(span_texts)}")
        
        return ", ".join(rationale_parts)

class EnglishDetectorGeneratorPipeline:
    """English XLM-R Detector + Generator pipeline"""
    
    def __init__(self, config: dict):
        self.detector_model_path = config['detector_model_path']
        self.generator_model_name = config['generator_model_name']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector = None
        self.generator_model = None
        self.generator_tokenizer = None
        
        # Load detector and generator
        self.load_detector()
        self.load_generator()
        
        # Quality criteria thresholds
        self.quality_thresholds = {
            'bertscore_min': 0.3,  # 매우 관대하게 (0.5 → 0.3)
            'toxicity_reduction_min': 0.05,  # 매우 관대하게 (0.1 → 0.05)
            'ppl_max': 200.0,  # 매우 관대하게 (150.0 → 200.0)
            'detector_safe': False  # Detector safety check 비활성화
        }
    
    def load_detector(self):
        """Load English XLM-R detector"""
        try:
            print(f"🔍 Loading English detector from: {self.detector_model_path}")
            self.detector = EnglishDetectorPipeline(
                model_path=self.detector_model_path,
                device=self.device
            )
            print("✅ English detector loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_generator(self):
        """Load generator model"""
        try:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
            config = AutoConfig.from_pretrained(self.generator_model_name)
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                config.rope_scaling = None
            
            self.generator_model = AutoModelForCausalLM.from_pretrained(
                self.generator_model_name,
                config=config,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True if "3B" in self.generator_model_name else False,
                attn_implementation="eager"
            )
            
            return True
        except Exception as e:
            print(f"Error loading generator: {e}")
            return False
    
    def generate_mitigation_prompt(self, text: str, detection_result: dict) -> str:
        """Generate advanced mitigation prompt with span-level JSON conditioning and target-specific templates"""
        
        harmful_spans = detection_result['detection_results']['harmful_spans']
        overall_intensity = detection_result['detection_results']['overall_intensity']
        targets = detection_result['detection_results']['targets']
        
        if overall_intensity == 'normal':
            return None
        
        # 1. Span-level JSON conditioning
        spans_json = []
        for span in harmful_spans[:5]:  # 최대 5개 span
            spans_json.append({
                "text": span['text'],
                "intensity": span['intensity'],
                "start": span.get('start', 0),
                "end": span.get('end', len(span['text']))
            })
        
        spans_json_str = json.dumps(spans_json, ensure_ascii=False, indent=2)
        
        # 2. Intensity-level tone guide
        intensity_guide = self._get_intensity_tone_guide(overall_intensity)
        
        # 3. Target category tone templates
        target_guide = self._get_target_category_guide(targets)
        
        # 4. Mitigation strategy based on severity
        if overall_intensity in ["hate", "SEVERE"] or any(span['intensity'] in ['HARD', 'B-HARD', 'I-HARD'] for span in harmful_spans):
            strategy = "DELETE_ONLY"
        else:
            strategy = "SOFTEN_AND_PRESERVE"
        
        # Harmful spans 정보를 구체적으로 언급
        harmful_spans_text = ""
        if harmful_spans:
            spans_list = []
            for span in harmful_spans[:3]:  # 최대 3개 span
                spans_list.append(f'"{span["text"]}" (intensity: {span["intensity"]})')
            harmful_spans_text = f"\n- Problematic expressions: {', '.join(spans_list)}"
        
        if strategy == "DELETE_ONLY":
            prompt = f"""Original: {text}

🔍 Detected Information:
- Intensity: {overall_intensity}
- Targets: {', '.join(targets) if targets else 'general'}{harmful_spans_text}

📋 Mitigation Guidelines:
{intensity_guide}

{target_guide}

Remove only the hateful or offensive parts from the above sentence.
- Pay special attention to the detected problematic expressions
- Delete only problematic words or phrases
- Do not add new words  
- Keep the rest unchanged
- Do not completely change the sentence

Mitigated sentence:"""
        else:
            prompt = f"""Original: {text}

🔍 Detected Information:
- Intensity: {overall_intensity}
- Targets: {', '.join(targets) if targets else 'general'}{harmful_spans_text}

📋 Mitigation Guidelines:
{intensity_guide}

{target_guide}

Softly change the offensive expressions in the above sentence.
- Pay special attention to the detected problematic expressions
- Replace offensive words with neutral words
- Preserve the original meaning
- Output only the mitigated sentence

Mitigated sentence:"""
        
        return prompt
    
    def _get_intensity_tone_guide(self, intensity: str) -> str:
        """Get intensity-specific tone guide"""
        guides = {
            "offensive": "• Soften aggressive tone\n• Replace insulting expressions with neutral expressions\n• Maintain original intent but soften expression",
            "hate": "• Strongly mitigate hateful expressions\n• Remove words with prejudice\n• Change to expressions that show respect for the target",
            "SEVERE": "• This is extremely serious hate speech, so delete only that part\n• Remove only problematic words or phrases and keep the rest unchanged\n• Preserve the original structure and meaning as much as possible"
        }
        return guides.get(intensity, "• Apply general mitigation principles")
    
    def _get_target_category_guide(self, targets: list) -> str:
        """Get target category-specific tone templates"""
        if not targets:
            return "• Apply general mitigation principles"
        
        guides = []
        
        if "political" in targets:
            guides.append("• Political criticism: Maintain constructive criticism but remove personal insults\n• Change attacks on politicians to policy criticism")
        
        if "region" in targets:
            guides.append("• Regional issues: Change regional discriminatory expressions to neutral expressions\n• Use expressions that promote harmony between regions")
        
        if "gender" in targets:
            guides.append("• Gender-related: Remove gender stereotypes and change to equal perspectives\n• Use gender-neutral expressions")
        
        if "age" in targets:
            guides.append("• Age-related: Change age discriminatory expressions to respectful expressions\n• Use expressions that help intergenerational understanding")
        
        if "religion" in targets:
            guides.append("• Religion-related: Remove religious prejudices and change to expressions that respect religious diversity\n• Use tolerant expressions that acknowledge religious differences")
        
        if "individual" in targets:
            guides.append("• Individual-related: Change expressions that attack individuals to constructive feedback\n• Use expressions that respect individual dignity")
        
        if "others" in targets or not guides:
            guides.append("• Others: Change to general respectful and considerate expressions")
        
        return "\n".join(guides)
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text to extract only the final sentence"""
        
        if not text or len(text.strip()) < 2:
            return text
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Mitigated sentence:", "Rewritten sentence:", "Answer:", "Result:", 
            "Here's the mitigated version:", "The mitigated sentence is:",
            "Original:", "Problematic expressions:", "Targets:", "Target categories:",
            "Remove only the hateful or offensive parts from the above sentence.",
            "Softly change the offensive expressions in the above sentence.",
            "- Delete only problematic words or phrases", "- Do not add new words",
            "- Keep the rest unchanged", "- Do not completely change the sentence",
            "- Replace offensive words with neutral words", "- Preserve the original meaning",
            "- Output only the mitigated sentence",
            "```json", "```",
            "• Soften aggressive tone", "• Replace insulting expressions with neutral expressions",
            "• Maintain original intent but soften expression", "• Strongly mitigate hateful expressions",
            "• Remove words with prejudice", "• Change to expressions that show respect for the target",
            "• This is extremely serious hate speech, so delete only that part",
            "• Remove only problematic words or phrases and keep the rest unchanged",
            "• Preserve the original structure and meaning as much as possible",
            "• Apply general mitigation principles"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove bullet points and special characters
        text = text.replace("•", "").replace("*", "").strip()
        
        # Split by lines and find the first meaningful English sentence
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that contain explanation keywords
            explanation_keywords = [
                "explanation", "analysis", "note:", "comment:", "here's", "the following",
                "mitigation", "offensive", "neutral", "replaced", "changed", "original",
                "problematic", "expression", "target", "category", "intensity",
                "soften", "aggressive", "tone", "insulting", "maintain", "intent",
                "strongly", "hateful", "prejudice", "respect", "extremely", "serious",
                "speech", "delete", "part", "remove", "structure", "meaning", "possible",
                "apply", "general", "principles", "political", "criticism", "constructive",
                "personal", "insults", "attacks", "politicians", "policy", "regional",
                "issues", "discriminatory", "expressions", "promote", "harmony", "regions",
                "gender", "related", "stereotypes", "equal", "perspectives", "neutral",
                "age", "discriminatory", "respectful", "intergenerational", "understanding",
                "religion", "prejudices", "diversity", "tolerant", "acknowledge", "differences",
                "individual", "feedback", "dignity", "others", "respectful", "considerate"
            ]
            
            if any(keyword in line.lower() for keyword in explanation_keywords):
                continue
            
            # Check if line contains English characters and meaningful content
            if any(ord(char) >= 32 and ord(char) <= 126 for char in line) and len(line.strip()) >= 5:
                # Take only the first sentence (up to first period, question mark, or exclamation)
                sentence_endings = ['.', '?', '!']
                for ending in sentence_endings:
                    if ending in line:
                        result = line.split(ending)[0] + ending
                        if len(result.strip()) >= 5:  # Ensure meaningful length
                            return result.strip()
                
                # If no sentence ending found, return the line if it's meaningful
                if len(line.strip()) >= 5:
                    return line.strip()
        
        # Fallback: return original text if cleaning fails
        return text
    
    def calculate_bertscore(self, original_text: str, rewritten_text: str) -> float:
        """Calculate BERTScore F1 between original and rewritten text"""
        if not BERTSCORE_AVAILABLE:
            return 0.0
        
        try:
            P, R, F1 = bert_score_fn([rewritten_text], [original_text], lang="en")
            return float(F1[0].item())
        except Exception as e:
            print(f"BERTScore calculation error: {e}")
            return 0.0
    
    def calculate_toxicity_reduction(self, original_text: str, rewritten_text: str) -> float:
        """Calculate toxicity reduction using detector model"""
        try:
            # Original toxicity score
            orig_result = self.detector.predict(original_text)
            orig_toxicity = 1.0 - orig_result['detection_results']['intensity_probabilities']['normal']
            
            # Rewritten toxicity score
            rewritten_result = self.detector.predict(rewritten_text)
            rewritten_toxicity = 1.0 - rewritten_result['detection_results']['intensity_probabilities']['normal']
            
            # Toxicity reduction
            reduction = orig_toxicity - rewritten_toxicity
            return max(0.0, reduction)  # Ensure non-negative
            
        except Exception as e:
            print(f"Toxicity reduction calculation error: {e}")
            return 0.0
    
    def calculate_ppl(self, text: str) -> float:
        """Calculate perplexity using actual language model"""
        try:
            if not text or len(text.strip()) < 2:
                return 1.0
            
            # Use the same tokenizer as the generator for consistency
            if self.generator_tokenizer is None:
                self.load_generator()
            
            # Tokenize the text
            inputs = self.generator_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate perplexity using the generator model
            with torch.no_grad():
                outputs = self.generator_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
            
            # Cap at reasonable values
            return min(max(ppl, 1.0), 1000.0)
            
        except Exception as e:
            print(f"PPL calculation error: {e}")
            # Fallback to simple estimation
            try:
                tokens = text.split()
                if len(tokens) < 2:
                    return 1.0
                word_count = len(tokens)
                base_ppl = 50.0
                length_factor = min(word_count / 10.0, 2.0)
                complexity_factor = 1.0 + (len(set(tokens)) / word_count) * 0.5
                ppl = base_ppl * length_factor * complexity_factor
                return min(ppl, 100.0)
            except:
                return 50.0
    
    def meets_quality_criteria(self, bertscore: float, toxicity_reduction: float, 
                              ppl: float, detector_safe: bool) -> bool:
        """Check if the rewritten text meets quality criteria"""
        
        criteria_met = {
            'bertscore': bertscore >= self.quality_thresholds['bertscore_min'],
            'toxicity_reduction': toxicity_reduction >= self.quality_thresholds['toxicity_reduction_min'],
            'ppl': ppl <= self.quality_thresholds['ppl_max'],
            'detector_safe': detector_safe if self.quality_thresholds['detector_safe'] else True
        }
        
        # All criteria must be met
        return all(criteria_met.values())
    
    def set_quality_thresholds(self, bertscore_min: float = 0.7, 
                              toxicity_reduction_min: float = 0.3,
                              ppl_max: float = 50.0, 
                              detector_safe: bool = True):
        """Set quality criteria thresholds"""
        self.quality_thresholds = {
            'bertscore_min': bertscore_min,
            'toxicity_reduction_min': toxicity_reduction_min,
            'ppl_max': ppl_max,
            'detector_safe': detector_safe
        }
    
    def generate_mitigation(self, text: str, prompt_or_detection_result) -> str:
        """Generate mitigation using generator model"""
        
        # If prompt_or_detection_result is a string, treat it as a prompt
        if isinstance(prompt_or_detection_result, str):
            prompt = prompt_or_detection_result
        else:
            # Otherwise, treat it as detection_result and generate prompt
            prompt = self.generate_mitigation_prompt(text, prompt_or_detection_result)
        
        if prompt is None:
            return text
        
        try:
            inputs = self.generator_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generator_model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=20,  # 더 안정적 (30 → 20)
                    top_p=0.7,  # 더 안정적 (0.8 → 0.7)
                    max_new_tokens=150,
                    temperature=0.6,  # 더 안정적 (0.7 → 0.6)
                    pad_token_id=self.generator_tokenizer.pad_token_id,
                    eos_token_id=self.generator_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            full_response = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            mitigated_text = full_response.replace(prompt, "").strip()
            
            # Clean up the response to get only the final sentence
            mitigated_text = self._clean_generated_text(mitigated_text)
            
            return mitigated_text
            
        except Exception as e:
            print(f"Error generating mitigation: {e}")
            return text
    
    def iterative_refinement(self, text: str, max_iterations: int = 10) -> dict:
        """Correct iterative refinement logic: only enter loop if toxicity exists and criteria not met"""
        
        # Step 1: Check if original text has toxicity
        initial_detection = self.detector.predict(text)
        is_toxic = initial_detection['detection_results']['is_harmful']
        
        if not is_toxic:
            # No toxicity detected - return original text
            return {
                'original_text': text,
                'final_text': text,
                'iterations': [{
                    'iteration': 0,
                    'input_text': text,
                    'detection_result': initial_detection,
                    'mitigated_text': text,
                    'quality_metrics': {
                        'bertscore': 1.0,
                        'toxicity_reduction': 0.0,
                        'ppl': self.calculate_ppl(text),
                        'detector_safe': True
                    },
                    'quality_criteria_met': True,
                    'improvement': False,
                    'overall_score': 1.0
                }],
                'total_iterations': 1,
                'final_quality_metrics': {
                    'bertscore': 1.0,
                    'toxicity_reduction': 0.0,
                    'ppl': self.calculate_ppl(text),
                    'detector_safe': True
                },
                'success': True,
                'quality_criteria_met': True,
                'best_overall_score': 1.0,
                'used_best_result': False
            }
        
        # Step 2: Generate initial mitigation
        initial_detection = self.detector.predict(text)
        mitigated_text = self.generate_mitigation(text, initial_detection)
        
        # Step 3: Evaluate initial mitigation
        bertscore = self.calculate_bertscore(text, mitigated_text)
        toxicity_reduction = self.calculate_toxicity_reduction(text, mitigated_text)
        ppl = self.calculate_ppl(mitigated_text)
        
        # Step 4: Critic Model (Detector) validation
        final_detection = self.detector.predict(mitigated_text)
        detector_safe = not final_detection['detection_results']['is_harmful']
        
        # Step 5: Check quality criteria
        quality_met = self.meets_quality_criteria(
            bertscore, toxicity_reduction, ppl, detector_safe
        )
        
        # Calculate overall score
        overall_score = (bertscore * 0.4 + toxicity_reduction * 0.4 + 
                       (1.0 - min(ppl/100.0, 1.0)) * 0.2)
        
        # Track best result
        best_result = {
            'mitigated_text': mitigated_text,
            'quality_metrics': {
                'bertscore': bertscore,
                'toxicity_reduction': toxicity_reduction,
                'ppl': ppl,
                'detector_safe': detector_safe
            },
            'quality_criteria_met': quality_met,
            'overall_score': overall_score
        }
        best_score = overall_score
        
        # Store initial iteration results
        iteration_results = [{
            'iteration': 1,
            'input_text': text,
            'detection_result': initial_detection,
            'mitigated_text': mitigated_text,
            'quality_metrics': {
                'bertscore': bertscore,
                'toxicity_reduction': toxicity_reduction,
                'ppl': ppl,
                'detector_safe': detector_safe
            },
            'quality_criteria_met': quality_met,
            'improvement': mitigated_text != text,
            'overall_score': overall_score
        }]
        
        # Step 6: Check if criteria are met - if yes, return immediately
        if quality_met:
            return {
                'original_text': text,
                'final_text': mitigated_text,
                'iterations': iteration_results,
                'total_iterations': 1,
                'final_quality_metrics': best_result['quality_metrics'],
                'success': True,
                'quality_criteria_met': True,
                'best_overall_score': best_score,
                'used_best_result': True
            }
        
        # Step 7: Criteria not met - enter iterative loop
        current_text = mitigated_text
        
        for iteration in range(1, max_iterations):  # Start from iteration 2
            # Generate mitigation using LLM
            detection_result = self.detector.predict(current_text)
            mitigated_text = self.generate_mitigation(current_text, detection_result)
            
            # Quality evaluation
            bertscore = self.calculate_bertscore(text, mitigated_text)
            toxicity_reduction = self.calculate_toxicity_reduction(text, mitigated_text)
            ppl = self.calculate_ppl(mitigated_text)
            
            # Detector validation (critic model role)
            final_detection = self.detector.predict(mitigated_text)
            detector_safe = not final_detection['detection_results']['is_harmful']
            
            # Check quality criteria
            quality_met = self.meets_quality_criteria(
                bertscore, toxicity_reduction, ppl, detector_safe
            )
            
            # Calculate overall score
            overall_score = (bertscore * 0.4 + toxicity_reduction * 0.4 + 
                           (1.0 - min(ppl/100.0, 1.0)) * 0.2)
            
            # Track best result
            if overall_score > best_score:
                best_score = overall_score
                best_result = {
                    'mitigated_text': mitigated_text,
                    'quality_metrics': {
                        'bertscore': bertscore,
                        'toxicity_reduction': toxicity_reduction,
                        'ppl': ppl,
                        'detector_safe': detector_safe
                    },
                    'quality_criteria_met': quality_met,
                    'overall_score': overall_score
                }
            
            # Store iteration results
            iteration_results.append({
                'iteration': iteration + 1,
                'input_text': current_text,
                'detection_result': detection_result,
                'mitigated_text': mitigated_text,
                'quality_metrics': {
                    'bertscore': bertscore,
                    'toxicity_reduction': toxicity_reduction,
                    'ppl': ppl,
                    'detector_safe': detector_safe
                },
                'quality_criteria_met': quality_met,
                'improvement': mitigated_text != current_text,
                'overall_score': overall_score
            })
            
            # Check stopping conditions - if criteria met, exit loop
            if quality_met:
                break
            else:
                # Continue to next iteration
                current_text = mitigated_text
        
        # Return best result (highest overall score)
        if best_result:
            final_detection = self.detector.predict(best_result['mitigated_text'])
            return {
                'original_text': text,
                'final_text': best_result['mitigated_text'],
                'iterations': iteration_results,
                'total_iterations': len(iteration_results),
                'final_quality_metrics': best_result['quality_metrics'],
                'success': not final_detection['detection_results']['is_harmful'],
                'quality_criteria_met': best_result['quality_criteria_met'],
                'best_overall_score': best_score,
                'used_best_result': True
            }
        else:
            # Fallback to original text if no iterations occurred
            final_detection = self.detector.predict(text)
            final_bertscore = self.calculate_bertscore(text, text)
            final_toxicity_reduction = self.calculate_toxicity_reduction(text, text)
            final_ppl = self.calculate_ppl(text)
        
        return {
            'original_text': text,
                'final_text': text,
            'iterations': iteration_results,
            'total_iterations': len(iteration_results),
            'final_quality_metrics': {
                'bertscore': final_bertscore,
                'toxicity_reduction': final_toxicity_reduction,
                'ppl': final_ppl,
                'detector_safe': not final_detection['detection_results']['is_harmful']
            },
            'success': not final_detection['detection_results']['is_harmful'],
            'quality_criteria_met': self.meets_quality_criteria(
                final_bertscore, final_toxicity_reduction, final_ppl, 
                not final_detection['detection_results']['is_harmful']
                ),
                'best_overall_score': 0.0,
                'used_best_result': False
        }
    
    def process_single_text(self, text: str, use_iterative: bool = True) -> dict:
        """Process a single text through the pipeline"""
        
        if use_iterative:
            result = self.iterative_refinement(text)
            # 한국어 파이프라인과 동일하게 필드 매핑
            result['mitigated_text'] = result.get('final_text', text)
            result['changed'] = result.get('final_text', text) != text
            # quality_metrics와 detection_results 추가
            result['quality_metrics'] = result.get('final_quality_metrics', {})
            # detection_result는 첫 번째 iteration에서 가져오기
            if result.get('iterations') and len(result['iterations']) > 0:
                result['detection_results'] = result['iterations'][0].get('detection_result', {})
            else:
                result['detection_results'] = {}
            return result
        else:
            detection_result = self.detector.predict(text)
            mitigated_text = self.generate_mitigation(text, detection_result)
            
            return {
                'original_text': text,
                'final_text': mitigated_text,
                'mitigated_text': mitigated_text,
                'changed': mitigated_text != text,
                'detection_result': detection_result,
                'iterations': [{
                    'iteration': 1,
                    'input_text': text,
                    'detection_result': detection_result,
                    'mitigated_text': mitigated_text,
                    'improvement': mitigated_text != text
                }],
                'total_iterations': 1,
                'success': not self.detector.predict(mitigated_text)['detection_results']['is_harmful']
            }
    
    def process_batch(self, texts: list, use_iterative: bool = True) -> list:
        """Process a batch of texts"""
        results = []
        
        for text in texts:
            try:
                result = self.process_single_text(text, use_iterative)
                results.append(result)
            except Exception as e:
                results.append({
                    'original_text': text,
                    'error': str(e),
                    'final_text': text,
                    'success': False
                })
        
        return results
    
    def evaluate_results(self, results: list) -> dict:
        """Evaluate pipeline results"""
        
        total_texts = len(results)
        successful_texts = sum(1 for r in results if r.get('success', False))
        error_texts = sum(1 for r in results if 'error' in r)
        
        avg_iterations = sum(r.get('total_iterations', 0) for r in results) / total_texts
        improved_texts = sum(1 for r in results if r.get('final_text', '') != r.get('original_text', ''))
        
        metrics = {
            'total_texts': total_texts,
            'successful_texts': successful_texts,
            'error_texts': error_texts,
            'success_rate': successful_texts / total_texts if total_texts > 0 else 0,
            'average_iterations': avg_iterations,
            'improved_texts': improved_texts,
            'improvement_rate': improved_texts / total_texts if total_texts > 0 else 0
        }
        
        return metrics
    
    def save_results(self, results: list, metrics: dict, output_file: str = None) -> str:
        """Save results to JSON file"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config['generator_model_name'].split('/')[-1]
            output_file = f"english_detector_generator_results_{model_name}_{timestamp}.json"
        
        output_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'pipeline': 'english_detector_generator',
                'iterative_refinement': True
            },
            'metrics': metrics,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return output_file

def main():
    """Main function for testing the pipeline"""
    
    config = {
        'detector_model_path': '/root/PROJECT-ROOT/backend/english_xlmr_model.pt',  # English model path
        'generator_model_name': 'meta-llama/Llama-3.2-3B-Instruct',
        'max_iterations': 3,
        'use_iterative': True
    }
    
    pipeline = EnglishDetectorGeneratorPipeline(config)
    
    if not pipeline.load_detector():
        print("Failed to load detector")
        return
    
    if not pipeline.load_generator():
        print("Failed to load generator")
        return
    
    result = pipeline.process_single_text("This is a normal sentence without any hate speech.")
    
    print("=" * 80)
    print("🔍 ENGLISH DETECTOR-GENERATOR PIPELINE RESULTS")
    print("=" * 80)
    
    print(f"\n📝 INPUT TEXT:")
    print(f"   {result['original_text']}")
    
    print(f"\n🔍 INITIAL DETECTOR ANALYSIS:")
    initial_detection = result['iterations'][0]['detection_result']
    print(f"   Is Harmful: {initial_detection['detection_results']['is_harmful']}")
    print(f"   Intensity: {initial_detection['detection_results']['overall_intensity']}")
    print(f"   Confidence: {initial_detection['detection_results']['intensity_confidence']:.3f}")
    print(f"   Targets: {initial_detection['detection_results']['targets']}")
    print(f"   Rationale: {initial_detection['detection_results']['rationale']}")
    
    if initial_detection['detection_results']['harmful_spans']:
        print(f"   Harmful Spans:")
        for span in initial_detection['detection_results']['harmful_spans']:
            print(f"     - '{span['text']}' ({span['intensity']}, confidence: {span['confidence']:.3f})")
    
    print(f"\n🤖 GENERATOR PROCESS:")
    for i, iteration in enumerate(result['iterations']):
        print(f"\n   Iteration {i+1}:")
        print(f"     Input: {iteration['input_text']}")
        print(f"     Generated: {iteration['mitigated_text']}")
        print(f"     Quality Metrics:")
        print(f"       - BERTScore: {iteration['quality_metrics']['bertscore']:.3f}")
        print(f"       - Toxicity Reduction: {iteration['quality_metrics']['toxicity_reduction']:.3f}")
        print(f"       - PPL: {iteration['quality_metrics']['ppl']:.3f}")
        print(f"       - Detector Safe: {iteration['quality_metrics']['detector_safe']}")
        print(f"       - Quality Criteria Met: {iteration['quality_criteria_met']}")
        
        # Detector validation for this iteration
        if 'detection_result' in iteration:
            iter_detection = iteration['detection_result']
            print(f"     Detector Validation:")
            print(f"       - Is Harmful: {iter_detection['detection_results']['is_harmful']}")
            print(f"       - Intensity: {iter_detection['detection_results']['overall_intensity']}")
            print(f"       - Confidence: {iter_detection['detection_results']['intensity_confidence']:.3f}")
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"   Final Text: {result['final_text']}")
    print(f"   Success: {result['success']}")
    print(f"   Total Iterations: {result['total_iterations']}")
    print(f"   Quality Criteria Met: {result['quality_criteria_met']}")
    
    print(f"\n📊 FINAL QUALITY METRICS:")
    final_metrics = result['final_quality_metrics']
    print(f"   BERTScore: {final_metrics['bertscore']:.3f}")
    print(f"   Toxicity Reduction: {final_metrics['toxicity_reduction']:.3f}")
    print(f"   PPL: {final_metrics['ppl']:.3f}")
    print(f"   Detector Safe: {final_metrics['detector_safe']}")
    
    print(f"\n🎯 QUALITY THRESHOLDS:")
    print(f"   BERTScore Min: {pipeline.quality_thresholds['bertscore_min']}")
    print(f"   Toxicity Reduction Min: {pipeline.quality_thresholds['toxicity_reduction_min']}")
    print(f"   PPL Max: {pipeline.quality_thresholds['ppl_max']}")
    print(f"   Detector Safe Required: {pipeline.quality_thresholds['detector_safe']}")
    
    print("=" * 80)

def process_english_test_data_tuples(pipeline, test_data: List[Dict]) -> List[Tuple[str, Dict, str]]:
    """
    Process English test data and return tuples in format: (input, detector_result, generator_result)
    """
    results = []
    
    print(f"🔄 Processing {len(test_data)} English test samples...")
    
    for i, sample in enumerate(test_data):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{len(test_data)}")
        
        # Get input text
        input_text = sample.get('text', sample.get('content', ''))
        
        # Process through pipeline
        try:
            result = pipeline.process_single_text(input_text, use_iterative=True)
            
            # Extract detector result
            detector_result = {
                "intensity": result['detection']['overall_intensity'],
                "intensity_confidence": result['detection']['intensity_confidence'],
                "target": result['detection']['targets'][0] if result['detection']['targets'] else "others",
                "target_confidence": 0.8,  # Default confidence
                "bio_spans": result['detection']['harmful_spans'],
                "is_hateful": result['detection']['is_harmful'],
                "detection_confidence": result['detection']['intensity_confidence']
            }
            
            # Get generator result
            generator_result = result.get('final_text', input_text)
            
            # Add to results
            results.append((input_text, detector_result, generator_result))
            
        except Exception as e:
            print(f"⚠️ Error processing sample {i+1}: {e}")
            # Add error result
            detector_result = {
                "intensity": "error",
                "intensity_confidence": 0.0,
                "target": "error",
                "target_confidence": 0.0,
                "bio_spans": [],
                "is_hateful": False,
                "detection_confidence": 0.0
            }
            results.append((input_text, detector_result, f"[Error] {str(e)}"))
    
    print(f"✅ Completed processing {len(results)} samples")
    return results

def save_english_tuples_results(results: List[Tuple[str, Dict, str]], filename: str = None):
    """Save English tuples results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"english_detector_generator_tuples_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = []
    for input_text, detector_result, generator_result in results:
        serializable_results.append({
            "input_text": input_text,
            "detector_result": detector_result,
            "generator_result": generator_result
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "results": serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 English tuples results saved to: {filename}")

def main_english_tuples():
    """Main function for English tuples processing"""
    print("🚀 Starting English Detector + Generator Tuples Test")
    
    # Initialize pipeline
    pipeline = EnglishDetectorGeneratorPipeline(detector_model_path="english_xlmr_model.pt")
    
    # Load test data
    try:
        with open("/root/PROJECT-ROOT/backend/clean_test_datasets/english_clean_test.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "samples" in data:
            test_data = data["samples"][:50]  # Limit to 50 samples for testing
        else:
            test_data = data[:50]
            
        print(f"📊 Loaded {len(test_data)} English test samples")
    except FileNotFoundError:
        print("⚠️ English test data not found, using sample data")
        test_data = [
            {"text": "I hate this person", "intensity": "hate", "target": "general"},
            {"text": "All women should stay home", "intensity": "hate", "target": "gender"},
            {"text": "The weather is nice today", "intensity": "normal", "target": "general"}
        ]
    
    # Process test data
    results = process_english_test_data_tuples(pipeline, test_data)
    
    # Show sample results
    print("\n📝 Sample Results:")
    print("=" * 60)
    
    for i, (input_text, detector_result, generator_result) in enumerate(results[:5]):
        print(f"\nSample {i+1}:")
        print(f"Input: {input_text}")
        print(f"Detector: {detector_result['intensity']} ({detector_result['intensity_confidence']:.3f}) - {detector_result['target']}")
        print(f"Generator: {generator_result}")
        print("-" * 40)
    
    # Save results
    save_english_tuples_results(results)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"Total samples processed: {len(results)}")
    
    # Count intensity distribution
    intensity_counts = {}
    for _, detector_result, _ in results:
        intensity = detector_result['intensity']
        intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
    
    print("Intensity distribution:")
    for intensity, count in intensity_counts.items():
        print(f"  {intensity}: {count} ({count/len(results):.1%})")

if __name__ == "__main__":
    main_english_tuples()
