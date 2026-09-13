#!/usr/bin/env python3
"""
Improved English Pipeline with Real Model Inference
Removes hardcoded parts and uses actual model predictions
"""

import torch
import json
import random
import sys
import os
import argparse
import re
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import warnings
warnings.filterwarnings("ignore")

# Import BERTScore for evaluation
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not available. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

# Add project root to path
sys.path.append('/root/PROJECT-ROOT/backend')

# Import detector model
import torch.nn as nn

class XLMRobertaDetector(nn.Module):
    """XLM-RoBERTa based hate speech detector"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", 
                 num_intensity_labels: int = 5, 
                 num_bio_labels: int = 5,
                 num_target_labels: int = 10):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Classification heads
        self.intensity_head = nn.Linear(self.config.hidden_size, num_intensity_labels)
        self.bio_head = nn.Linear(self.config.hidden_size, num_bio_labels)
        self.target_head = nn.Linear(self.config.hidden_size, num_target_labels)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Global average pooling
        pooled_output = sequence_output.mean(dim=1)
        
        # Classification outputs
        intensity_logits = self.intensity_head(pooled_output)
        bio_logits = self.bio_head(sequence_output)
        target_logits = self.target_head(pooled_output)
        
        return {
            'intensity_logits': intensity_logits,
            'bio_logits': bio_logits,
            'target_logits': target_logits
        }

class ImprovedEnglishDetectorGeneratorExperiment:
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector_model = None
        self.detector_tokenizer = None
        self.generator_model = None
        self.generator_tokenizer = None
        
        print(f"Device: {self.device}")
        print(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    def load_detector(self):
        """Load English XLM-R detector with proper error handling"""
        print(f"Loading English XLM-R detector from {self.config['detector_model_path']}...")
        
        try:
            # Load tokenizer
            self.detector_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            
            # Load checkpoint to get model config
            checkpoint = torch.load(self.config['detector_model_path'], map_location=self.device)
            model_config = checkpoint.get('model_config', {})
            num_intensity = model_config.get('intensity_labels', 3)
            num_bio = model_config.get('bio_labels', 5)
            num_target = model_config.get('target_labels', 6)
            
            print(f"Model config: intensity={num_intensity}, bio={num_bio}, target={num_target}")
            
            # Load model with correct configuration from checkpoint
            self.detector_model = XLMRobertaDetector(
                model_name="xlm-roberta-base",
                num_intensity_labels=num_intensity,
                num_bio_labels=num_bio,
                num_target_labels=num_target
            )
            
            self.detector_model.load_state_dict(checkpoint['model_state_dict'])
            self.detector_model.to(self.device)
            self.detector_model.eval()
            
            print("✅ English XLM-R detector loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading detector: {e}")
            print("❌ Cannot proceed without detector model!")
            return False
    
    def load_generator(self):
        """Load Llama3.2 generator model with proper error handling"""
        print(f"Loading generator model: {self.config['generator_model_name']}...")
        
        try:
            hf_token = self.config.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            hf_cache_dir = self.config.get("hf_cache_dir") or os.environ.get("HF_HOME") or "/tmp/hf_cache"

            # Load tokenizer
            tok_kwargs = {"cache_dir": hf_cache_dir}
            if hf_token:
                tok_kwargs["token"] = hf_token
            self.generator_tokenizer = AutoTokenizer.from_pretrained(self.config['generator_model_name'], **tok_kwargs)
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
            # Prefer bf16 when available (often better for 8B); fall back to fp16
            if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            # Load model
            model_kwargs = {
                "dtype": torch_dtype,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "cache_dir": hf_cache_dir,
            }
            if hf_token:
                model_kwargs["token"] = hf_token
            self.generator_model = AutoModelForCausalLM.from_pretrained(self.config['generator_model_name'], **model_kwargs)
            
            print(f"✅ Generator loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading generator model: {e}")
            print("❌ Cannot proceed without generator model!")
            return False
    
    def detect_hate_speech(self, text):
        """Detect hate speech using actual XLM-R model inference"""
        if self.detector_model is None:
            raise RuntimeError("Detector model not loaded!")
        
        inputs = self.detector_tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
            
            # Get predictions
            intensity_logits = outputs['intensity_logits']
            bio_logits = outputs['bio_logits']
            target_logits = outputs['target_logits']
            
            # Convert to probabilities
            intensity_probs = torch.softmax(intensity_logits, dim=-1)
            bio_probs = torch.softmax(bio_logits, dim=-1)
            target_probs = torch.softmax(target_logits, dim=-1)
            
            # Get predicted labels
            intensity_pred = torch.argmax(intensity_probs, dim=-1).item()
            bio_pred = torch.argmax(bio_probs, dim=-1)
            target_pred = torch.argmax(target_probs, dim=-1).item()
            
            # Convert BIO predictions to spans (5 classes: O, B-SOFT, I-SOFT, B-HARD, I-HARD)
            bio_labels = ['O', 'B-SOFT', 'I-SOFT', 'B-HARD', 'I-HARD']
            tokens = self.detector_tokenizer.tokenize(text)
            
            spans = []
            current_span = None
            
            for i, (token, label_idx) in enumerate(zip(tokens, bio_pred[0])):
                label = bio_labels[label_idx.item()]
                
                if label.startswith('B-'):
                    if current_span:
                        spans.append(current_span)
                    current_span = {
                        'type': label.split('-')[1],
                        'start': i,
                        'end': i,
                        'text': token
                    }
                elif label.startswith('I-') and current_span and label.split('-')[1] == current_span['type']:
                    current_span['end'] = i
                    current_span['text'] += token.replace('▁', ' ')
                else:
                    if current_span:
                        spans.append(current_span)
                        current_span = None
            
            if current_span:
                spans.append(current_span)
            
            return {
                'intensity': intensity_pred,
                'bio_spans': spans,
                'target': target_pred,
                'intensity_prob': intensity_probs[0][intensity_pred].item(),
                'target_prob': target_probs[0][target_pred].item(),
                'intensity_probs': intensity_probs[0].tolist(),
                'target_probs': target_probs[0].tolist()
            }
    
    def generate_mitigation(self, text, detection_result, iteration=0):
        """Generate mitigation using actual Llama3.2 model inference"""
        if self.generator_model is None:
            raise RuntimeError("Generator model not loaded!")

        prompt_mode = self.config.get("prompt_mode", "guided")
        
        # Extract span information
        hate_spans = [span for span in detection_result['bio_spans'] if span['type'] in ['SOFT', 'HARD']]
        target_spans = []  # Target is separate classification, not BIO spans
        
        # Create span lists
        span_list = []
        if hate_spans:
            span_list.append(f"Hate spans: {', '.join([span['text'] for span in hate_spans])}")
        if target_spans:
            span_list.append(f"Target spans: {', '.join([span['text'] for span in target_spans])}")
        
        # Create specific instructions based on detection
        specific_instructions = []
        if detection_result['intensity'] >= 1:  # OFFENSIVE or HATE
            specific_instructions.append("Focus on reducing the intensity of harmful language")
        if detection_result['target'] > 0:  # Has target category
            specific_instructions.append("Be especially careful with targeted content")
        if iteration > 0:
            specific_instructions.append("This is a refinement iteration - improve upon the previous attempt")
        
        if prompt_mode == "unguided":
            system_msg = "You rewrite text to be respectful and appropriate while preserving meaning and intent."
            user_msg = f"""Rewrite the following text.

Text: "{text}"

Output ONLY the rewritten text, with no additional commentary."""
        else:
            system_msg = "You are an expert in content moderation and hate speech mitigation."
            user_msg = f"""Rewrite the following text to make it more respectful and appropriate while preserving the original meaning and intent.

Original Text: "{text}"

Detection Analysis:
- Intensity Level: {detection_result['intensity']}/2 (0=NORMAL, 1=OFFENSIVE, 2=HATE)
- Confidence: {detection_result['intensity_prob']:.2f}
- Detected Elements: {'; '.join(span_list) if span_list else 'None detected'}

Specific Instructions:
{chr(10).join([f"- {instruction}" for instruction in specific_instructions]) if specific_instructions else "- General mitigation required"}

Guidelines:
1. Maintain the original meaning and intent
2. Use respectful and appropriate language
3. Avoid counter-speech or aggressive responses
4. Focus on constructive communication
5. Preserve the core message while removing harmful elements

Output ONLY the rewritten text, with no additional commentary."""

        # Prefer chat templates when available (better for Instruct models)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if hasattr(self.generator_tokenizer, "apply_chat_template") and getattr(self.generator_tokenizer, "chat_template", None):
            prompt = self.generator_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = system_msg + "\n\n" + user_msg

        # Generate response
        inputs = self.generator_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.generator_model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            outputs = self.generator_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.generator_tokenizer.eos_token_id,
                eos_token_id=self.generator_tokenizer.eos_token_id
            )
        
        # Decode ONLY newly generated tokens (robust even with chat templates)
        generated_ids = outputs[0][prompt_len:]
        mitigated_text = self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Clean up the text
        mitigated_text = re.sub(r'^["\']|["\']$', '', mitigated_text)
        mitigated_text = mitigated_text.strip()

        # Remove common preambles
        mitigated_text = re.sub(r'^(Here is the rewritten text:|Rewritten text:)\s*', '', mitigated_text, flags=re.IGNORECASE).strip()

        # If the model added extra commentary, keep the first quoted line or first non-empty line
        if "\n" in mitigated_text:
            lines = [ln.strip() for ln in mitigated_text.splitlines() if ln.strip()]
            quoted = [ln for ln in lines if (ln.startswith('"') and ln.endswith('"')) or (ln.startswith("'") and ln.endswith("'"))]
            mitigated_text = (quoted[0] if quoted else lines[0]) if lines else mitigated_text
        mitigated_text = mitigated_text.strip().strip('"').strip("'").strip()
        
        return mitigated_text
    
    def evaluate_mitigation(self, original_text, mitigated_text):
        """Evaluate the quality of mitigation using actual metrics"""
        results = {}
        
        # BERTScore evaluation
        if BERTSCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score_fn([mitigated_text], [original_text], lang="en")
                results['bertscore'] = F1.item()
            except Exception as e:
                print(f"BERTScore evaluation failed: {e}")
                results['bertscore'] = 0.0
        else:
            results['bertscore'] = 0.0
        
            # Toxicity reduction using actual detector model
        try:
            original_detection = self.detect_hate_speech(original_text)
            mitigated_detection = self.detect_hate_speech(mitigated_text)
            
            # Toxicity = probability of OFFENSIVE or HATE (1 - NORMAL probability)
            original_toxicity = 1.0 - original_detection['intensity_probs'][0]  # 1 - NORMAL probability
            mitigated_toxicity = 1.0 - mitigated_detection['intensity_probs'][0]
            
            if original_toxicity > 0:
                results['toxicity_reduction'] = (original_toxicity - mitigated_toxicity) / original_toxicity
            else:
                results['toxicity_reduction'] = 0.0
                
        except Exception as e:
            print(f"Toxicity reduction evaluation failed: {e}")
            results['toxicity_reduction'] = 0.0
        
        # Perplexity using actual language model (simplified)
        try:
            # Use generator model to calculate perplexity
            inputs = self.generator_tokenizer(
                mitigated_text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.generator_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generator_model(**inputs)
                logits = outputs.logits
                
                # Calculate perplexity
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                perplexity = torch.exp(loss).item()
                
                results['ppl'] = perplexity
                
        except Exception as e:
            print(f"Perplexity calculation failed: {e}")
            results['ppl'] = 100.0  # Default fallback
        
        return results
    
    def iterative_refinement(self, text, max_iterations=3):
        """Perform iterative refinement using actual model inference"""
        results = {
            'original_text': text,
            'iterations': [],
            'final_text': text,
            'final_metrics': {},
            'success': False
        }
        
        current_text = text
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Detect hate speech using actual model
            detection_result = self.detect_hate_speech(current_text)
            print(f"Detection: Intensity={detection_result['intensity']}, Confidence={detection_result['intensity_prob']:.2f}")
            
            # Check if mitigation is needed (0 = NORMAL, 1 = OFFENSIVE, 2 = HATE)
            if detection_result['intensity'] == 0:  # NORMAL
                print("✅ Text is already appropriate")
                results['success'] = True
                results['final_text'] = current_text
                # Provide default metrics so summary doesn't crash
                results['final_metrics'] = {
                    'bertscore': 1.0,
                    'toxicity_reduction': 0.0,
                    'ppl': 0.0
                }
                break
            
            # Generate mitigation using actual model
            mitigated_text = self.generate_mitigation(current_text, detection_result, iteration)
            print(f"Mitigated: {mitigated_text}")
            
            # Evaluate mitigation using actual metrics
            metrics = self.evaluate_mitigation(current_text, mitigated_text)
            print(f"Metrics: BERTScore={metrics['bertscore']:.3f}, Toxicity Reduction={metrics['toxicity_reduction']:.3f}, PPL={metrics['ppl']:.1f}")
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'input_text': current_text,
                'detection': detection_result,
                'mitigated_text': mitigated_text,
                'metrics': metrics
            }
            results['iterations'].append(iteration_result)
            
            # Update current text
            current_text = mitigated_text
            
            # Early termination if good enough
            if metrics['toxicity_reduction'] > 0.3 and metrics['bertscore'] > 0.7 and metrics['ppl'] < 100:
                print("✅ Early termination: Good enough results achieved")
                results['success'] = True
                break
        
        # Final evaluation
        if results['iterations']:
            results['final_text'] = current_text
            results['final_metrics'] = results['iterations'][-1]['metrics']
        
        return results
    
    def run_experiment(self, test_data, num_samples=50):
        """Run the complete experiment using actual model inference"""
        print(f"\n🚀 Starting Improved English Detector-Generator Experiment")
        print(f"Model: {self.config['generator_model_name']}")
        print(f"Prompt mode: {self.config.get('prompt_mode', 'guided')}")
        print(f"Test samples: {num_samples}")
        
        # Sample test data
        if len(test_data) > num_samples:
            test_data = random.sample(test_data, num_samples)
        
        results = {
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(test_data),
            'samples': [],
            'summary': {}
        }
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            print(f"\n--- Sample {i+1}/{len(test_data)} ---")
            
            # Get text (assuming sample has 'text' field)
            text = sample.get('text', sample.get('sentence', ''))
            if not text:
                continue
            
            print(f"Original: {text}")
            
            # Run iterative refinement using actual models
            sample_result = self.iterative_refinement(text)
            results['samples'].append(sample_result)
            
            # Print summary
            if sample_result['success']:
                print(f"✅ Success: {sample_result['final_text']}")
            else:
                print(f"❌ Failed: {sample_result['final_text']}")
        
        # Calculate summary statistics
        successful_samples = [s for s in results['samples'] if s['success']]
        if successful_samples:
            bertscores = [s.get('final_metrics', {}).get('bertscore', 0.0) for s in successful_samples]
            toxicity_reductions = [s.get('final_metrics', {}).get('toxicity_reduction', 0.0) for s in successful_samples]
            ppls = [s.get('final_metrics', {}).get('ppl', 0.0) for s in successful_samples]
            
            results['summary'] = {
                'success_rate': len(successful_samples) / len(results['samples']),
                'avg_bertscore': sum(bertscores) / len(bertscores),
                'avg_toxicity_reduction': sum(toxicity_reductions) / len(toxicity_reductions),
                'avg_ppl': sum(ppls) / len(ppls),
                'avg_iterations': sum(len(s['iterations']) for s in successful_samples) / len(successful_samples)
            }
        
        print(f"\n📊 Experiment Summary:")
        print(f"Success Rate: {results['summary'].get('success_rate', 0):.2%}")
        print(f"Avg BERTScore: {results['summary'].get('avg_bertscore', 0):.3f}")
        print(f"Avg Toxicity Reduction: {results['summary'].get('avg_toxicity_reduction', 0):.3f}")
        print(f"Avg PPL: {results['summary'].get('avg_ppl', 0):.1f}")
        print(f"Avg Iterations: {results['summary'].get('avg_iterations', 0):.1f}")
        
        return results

def load_english_test_data():
    """Load English test data from hateXplain test set"""
    print("Loading English test data from hateXplain test set...")
    
    try:
        # Preferred: load pre-split cleaned test file if present in this repo
        candidate_files = [
            "/root/multilingual-hate-detection/backend/clean_test_datasets/english_clean_test.json",
            "/root/PROJECT-ROOT/backend/clean_test_datasets/english_clean_test.json",
        ]

        for test_file in candidate_files:
            if os.path.exists(test_file):
                with open(test_file, 'r', encoding='utf-8') as f:
                    data_info = json.load(f)

                data = []
                for sample in data_info.get('samples', []):
                    data.append({
                        'text': sample.get('text', ''),
                        'intensity': sample.get('intensity', ''),
                        'target_categories': sample.get('target_categories', []),
                        'id': sample.get('id', 0)
                    })

                data = [d for d in data if d.get('text')]
                print(f"✅ Loaded {len(data)} English samples from cleaned test set")
                print(f"Dataset info: {data_info.get('dataset_name', 'unknown')}, Split: {data_info.get('split', 'test')}")
                return data

        # Fallback: load directly from HuggingFace HateXplain test split
        print("Cleaned test file not found; falling back to HuggingFace dataset `hatexplain` (test split).")
        ds = load_dataset('hatexplain', trust_remote_code=True)
        test_ds = ds['test']

        data = []
        for item in test_ds:
            post_tokens = item.get('post_tokens', [])
            text = ' '.join(post_tokens).strip()
            if not text:
                continue
            data.append({
                'text': text,
                'id': item.get('id', 0)
            })

        print(f"✅ Loaded {len(data)} English samples from HateXplain (HF) test split")
        return data
        
    except Exception as e:
        print(f"❌ Failed to load hateXplain test data: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Improved English Detector-Generator Pipeline Experiment')
    parser.add_argument('--detector_model_path', type=str, 
                       default='/root/multilingual-hate-detection/backend/model/english_xlmr_model.pt',
                       help='Path to English XLM-R detector model')
    parser.add_argument('--generator_model_name', type=str, 
                       default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='Generator model name (HuggingFace repo id or local path)')
    parser.add_argument('--prompt_mode', type=str,
                       choices=['guided', 'unguided', 'both'],
                       default='guided',
                       help='Prompting mode for generator')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token (optional; can also set HF_TOKEN env var)')
    parser.add_argument('--hf_cache_dir', type=str, default='/tmp/hf_cache',
                       help='HuggingFace cache dir')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'detector_model_path': args.detector_model_path,
        'generator_model_name': args.generator_model_name,
        'prompt_mode': args.prompt_mode,
        'hf_token': args.hf_token,
        'hf_cache_dir': args.hf_cache_dir,
        'num_samples': args.num_samples
    }
    
    # Initialize experiment
    experiment = ImprovedEnglishDetectorGeneratorExperiment(config)
    
    try:
        # Load models
        detector_loaded = experiment.load_detector()
        generator_loaded = experiment.load_generator()
        
        if not detector_loaded or not generator_loaded:
            print("❌ Cannot proceed without both models loaded!")
            return
        
        # Load test data
        test_data = load_english_test_data()
        if not test_data:
            print("❌ Cannot proceed without test data!")
            return
        
        # Run experiment(s)
        if args.prompt_mode == 'both':
            # Use the same sampled data for fair comparison
            if len(test_data) > args.num_samples:
                sampled_data = random.sample(test_data, args.num_samples)
            else:
                sampled_data = test_data

            all_results = {
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'modes': {}
            }

            for mode in ['guided', 'unguided']:
                experiment.config['prompt_mode'] = mode
                all_results['modes'][mode] = experiment.run_experiment(sampled_data, len(sampled_data))

            results = all_results
        else:
            results = experiment.run_experiment(test_data, args.num_samples)
        
        # Save results
        if args.output_file:
            output_file = args.output_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"improved_english_detector_generator_experiment_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
