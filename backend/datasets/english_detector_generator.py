#!/usr/bin/env python3
"""
English XLM-R Detector + Generator Pipeline
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
        
        self.intensity_labels = ["normal", "offensive", "hate"]
        self.bio_labels = ["O", "B-SOFT", "I-SOFT", "B-HARD", "I-HARD"]
        self.target_labels = ["gender", "age", "political", "religion", "region", "others"]
        
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
        
        self.model = EnglishXLMDetector()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        """Extract harmful spans from BIO predictions"""
        spans = []
        tokens = self.tokenizer.tokenize(text)
        
        current_span = None
        for i, bio_pred in enumerate(bio_preds):
            if bio_pred == 1:  # B-SOFT
                if current_span:
                    spans.append(current_span)
                current_span = {
                    "text": tokens[i].replace('▁', ''),
                    "start_pos": i,
                    "end_pos": i + 1,
                    "intensity": "SOFT",
                    "confidence": 0.8,
                    "targets": self._get_targets(target_probs),
                    "bio_label": "B-SOFT"
                }
            elif bio_pred == 2:  # I-SOFT
                if current_span and current_span["intensity"] == "SOFT":
                    current_span["text"] += tokens[i].replace('▁', '')
                    current_span["end_pos"] = i + 1
            elif bio_pred == 3:  # B-HARD
                if current_span:
                    spans.append(current_span)
                current_span = {
                    "text": tokens[i].replace('▁', ''),
                    "start_pos": i,
                    "end_pos": i + 1,
                    "intensity": "HARD",
                    "confidence": 0.9,
                    "targets": self._get_targets(target_probs),
                    "bio_label": "B-HARD"
                }
            elif bio_pred == 4:  # I-HARD
                if current_span and current_span["intensity"] == "HARD":
                    current_span["text"] += tokens[i].replace('▁', '')
                    current_span["end_pos"] = i + 1
            else:  # O (Outside)
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        if current_span:
            spans.append(current_span)
        
        return spans
    
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
    
    def __init__(self, detector_model_path: str, generator_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.config = {
            'detector_model_path': detector_model_path,
            'generator_model_name': generator_model_name
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector = None
        self.generator_model = None
        self.generator_tokenizer = None
        
        # Load detector and generator
        self.load_detector()
        self.load_generator()
        
        # Quality criteria thresholds
        self.quality_thresholds = {
            'bertscore_min': 0.3,
            'toxicity_reduction_min': 0.05,
            'ppl_max': 200.0,
            'detector_safe': False
        }
    
    def load_detector(self):
        """Load English XLM-R detector"""
        try:
            self.detector = EnglishDetectorPipeline(
                model_path=self.config['detector_model_path'],
                device=self.device
            )
            return True
        except Exception as e:
            print(f"Error loading detector: {e}")
            return False
    
    def load_generator(self):
        """Load generator model"""
        try:
            self.generator_tokenizer = AutoTokenizer.from_pretrained(self.config['generator_model_name'])
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
            config = AutoConfig.from_pretrained(self.config['generator_model_name'])
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                config.rope_scaling = None
            
            self.generator_model = AutoModelForCausalLM.from_pretrained(
                self.config['generator_model_name'],
                config=config,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True if "3B" in self.config['generator_model_name'] else False,
                attn_implementation="eager"
            )
            
            return True
        except Exception as e:
            print(f"Error loading generator: {e}")
            return False
    
    def generate_mitigation_prompt(self, text: str, detection_result: dict) -> str:
        """Generate mitigation prompt based on detection results"""
        
        harmful_spans = detection_result['detection_results']['harmful_spans']
        overall_intensity = detection_result['detection_results']['overall_intensity']
        targets = detection_result['detection_results']['targets']
        
        if overall_intensity == 'normal':
            return None
        
        if overall_intensity in ['offensive']:
            hate_tokens_str = ""
            if harmful_spans:
                hate_tokens_str = "\nProblematic expressions:\n" + "\n".join([f"• {span['text']} ({span['intensity']})" for span in harmful_spans[:5]])
            
            prompt = f"""You are a professional content moderator. Rewrite the following sentence to remove any offensive or hateful content while maintaining the original meaning.

Original: {text}
{hate_tokens_str}

IMPORTANT: You MUST rewrite this sentence. Do not return the original text unchanged.

Please rewrite the sentence in a more neutral and respectful way:"""
            
        else:  # hate
            hate_tokens_str = ""
            if harmful_spans:
                hate_tokens_str = "\nProblematic expressions:\n" + "\n".join([f"• {span['text']} ({span['intensity']})" for span in harmful_spans[:5]])
            
            prompt = f"""You are a professional content moderator. Rewrite the following sentence to remove any offensive or hateful content while maintaining the original meaning.

Original: {text}
{hate_tokens_str}

IMPORTANT: You MUST rewrite this sentence. Do not return the original text unchanged.

Please rewrite the sentence in a more neutral and respectful way:"""
        
        return prompt
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text to extract only the final sentence"""
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Mitigated sentence:",
            "Rewritten sentence:",
            "Answer:",
            "Result:",
            "Here's the mitigated version:",
            "The mitigated sentence is:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove bullet points and special characters
        text = text.replace("•", "").replace("*", "").strip()
        
        # Remove explanations and metadata
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that contain explanations or metadata
            if (line.startswith('*') or 
                line.startswith('-') or 
                line.startswith('Please') or
                line.startswith('Avoid') or
                line.startswith('Discriminatory') or
                line.startswith('Violent') or
                line.startswith('Threatening') or
                '( HATER )' in line or
                '( SOFT )' in line or
                '( HARD )' in line or
                line == ''):
                continue
            
            # Keep only English sentences
            if any(ord(char) >= 65 and ord(char) <= 122 for char in line):
                clean_lines.append(line)
        
        # Join clean lines and take the first meaningful sentence
        if clean_lines:
            result = ' '.join(clean_lines).strip()
            # Take only the first sentence (up to first period)
            if '.' in result:
                result = result.split('.')[0] + '.'
            return result
        
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
        """Calculate perplexity using Perplexity API"""
        try:
            # Simple PPL estimation based on text characteristics
            tokens = text.split()
            if len(tokens) < 2:
                return 1.0
            
            word_count = len(tokens)
            base_ppl = 50.0
            length_factor = min(word_count / 10.0, 2.0)
            complexity_factor = 1.0 + (len(set(tokens)) / word_count) * 0.5
            ppl = base_ppl * length_factor * complexity_factor
            
            return min(ppl, 100.0)  # Cap at 100
            
        except Exception as e:
            print(f"PPL calculation error: {e}")
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
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.generator_model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=20,
                    top_p=0.7,
                    max_new_tokens=150,
                    temperature=0.6,
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
    
    def iterative_refinement(self, text: str, max_iterations: int = 3) -> dict:
        """Enhanced iterative refinement with quality criteria and detector validation"""
        
        current_text = text
        iteration_results = []
        
        for iteration in range(max_iterations):
            # Step 1: Generate mitigation using LLM
            detection_result = self.detector.predict(current_text)
            mitigated_text = self.generate_mitigation(current_text, detection_result)
            
            # Step 2: Quality evaluation
            bertscore = self.calculate_bertscore(text, mitigated_text)
            toxicity_reduction = self.calculate_toxicity_reduction(text, mitigated_text)
            ppl = self.calculate_ppl(mitigated_text)
            
            # Step 3: Detector validation (critic model role)
            final_detection = self.detector.predict(mitigated_text)
            detector_safe = not final_detection['detection_results']['is_harmful']
            
            # Step 4: Check quality criteria
            quality_met = self.meets_quality_criteria(
                bertscore, toxicity_reduction, ppl, detector_safe
            )
            
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
                'improvement': mitigated_text != current_text
            })
            
            # Step 5: Check stopping conditions
            if quality_met:
                # Quality criteria met - return result
                break
            elif detector_safe and iteration >= 1:
                # Detector says safe and we've tried at least once - acceptable
                break
            else:
                # Continue to next iteration
                current_text = mitigated_text
        
        # Final evaluation
        final_detection = self.detector.predict(current_text)
        final_bertscore = self.calculate_bertscore(text, current_text)
        final_toxicity_reduction = self.calculate_toxicity_reduction(text, current_text)
        final_ppl = self.calculate_ppl(current_text)
        
        return {
            'original_text': text,
            'final_text': current_text,
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
            )
        }
    
    def process_single_text(self, text: str, use_iterative: bool = True) -> dict:
        """Process a single text through the pipeline"""
        
        if use_iterative:
            return self.iterative_refinement(text)
        else:
            detection_result = self.detector.predict(text)
            mitigated_text = self.generate_mitigation(text, detection_result)
            
            return {
                'original_text': text,
                'final_text': mitigated_text,
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
        'detector_model_path': '/root/PROJECT-ROOT/backend/final_english_xlmr_model.pt',  # Final English model path
        'generator_model_name': 'meta-llama/Llama-3.2-3B-Instruct',
        'max_iterations': 3,
        'use_iterative': True
    }
    
    pipeline = EnglishDetectorGeneratorPipeline('final_english_xlmr_model.pt')
    
    result = pipeline.process_single_text("This is a normal sentence without any hate speech.")
    
    print("=" * 80)
    print("ENGLISH DETECTOR-GENERATOR PIPELINE RESULTS")
    print("=" * 80)
    
    print(f"\n INPUT TEXT:")
    print(f"   {result['original_text']}")
    
    print(f"\n INITIAL DETECTOR ANALYSIS:")
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
    
    print(f"\n GENERATOR PROCESS:")
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
    
    print(f"\n FINAL RESULTS:")
    print(f"   Final Text: {result['final_text']}")
    print(f"   Success: {result['success']}")
    print(f"   Total Iterations: {result['total_iterations']}")
    print(f"   Quality Criteria Met: {result['quality_criteria_met']}")
    
    print(f"\n FINAL QUALITY METRICS:")
    final_metrics = result['final_quality_metrics']
    print(f"   BERTScore: {final_metrics['bertscore']:.3f}")
    print(f"   Toxicity Reduction: {final_metrics['toxicity_reduction']:.3f}")
    print(f"   PPL: {final_metrics['ppl']:.3f}")
    print(f"   Detector Safe: {final_metrics['detector_safe']}")
    
    print(f"\n QUALITY THRESHOLDS:")
    print(f"   BERTScore Min: {pipeline.quality_thresholds['bertscore_min']}")
    print(f"   Toxicity Reduction Min: {pipeline.quality_thresholds['toxicity_reduction_min']}")
    print(f"   PPL Max: {pipeline.quality_thresholds['ppl_max']}")
    print(f"   Detector Safe Required: {pipeline.quality_thresholds['detector_safe']}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
