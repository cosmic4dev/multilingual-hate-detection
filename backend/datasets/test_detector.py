#!/usr/bin/env python3
"""
Test script to verify detector is working properly
"""

import torch
import sys
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn

# Add project root to path
sys.path.append('/root/multilingual-hate-detection/backend')

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

def load_detector(model_path):
    """Load the detector model"""
    print(f"Loading detector from {model_path}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        
        # Load model with correct label counts from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})
        num_intensity = model_config.get('intensity_labels', 3)
        num_bio = model_config.get('bio_labels', 5)
        num_target = model_config.get('target_labels', 6)
        
        print(f"Model config: intensity={num_intensity}, bio={num_bio}, target={num_target}")
        
        model = XLMRobertaDetector(
            model_name="xlm-roberta-base",
            num_intensity_labels=num_intensity,
            num_bio_labels=num_bio,
            num_target_labels=num_target
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print("✅ Detector loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def detect_hate_speech(model, tokenizer, device, text):
    """Detect hate speech in text"""
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
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
        tokens = tokenizer.tokenize(text)
        
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
        
        # Intensity labels mapping (3 classes)
        intensity_labels = ['NORMAL', 'OFFENSIVE', 'HATE']
        
        return {
            'intensity': intensity_pred,
            'intensity_label': intensity_labels[intensity_pred],
            'bio_spans': spans,
            'target': target_pred,
            'intensity_prob': intensity_probs[0][intensity_pred].item(),
            'target_prob': target_probs[0][target_pred].item(),
            'intensity_probs': intensity_probs[0].tolist(),
            'target_probs': target_probs[0].tolist()
        }

def main():
    # Test cases - more diverse examples
    test_cases = [
        "I love this beautiful day!",
        "You are an idiot and I hate you!",
        "This is a normal conversation about the weather.",
        "I think that group of people should leave our country.",
        "Thank you for your help with the project.",
        "All immigrants are criminals and should be deported.",
        "Women are not capable of leadership roles.",
        "That religion is full of terrorists.",
        "I disagree with your political views.",
        "The weather is nice today, isn't it?",
    ]
    
    # Model path
    model_path = '/root/multilingual-hate-detection/backend/model/retrained_english_xlmr_model.pt'
    
    # Load detector
    model, tokenizer, device = load_detector(model_path)
    
    if model is None:
        print("❌ Failed to load detector. Cannot proceed.")
        return
    
    print("\n" + "="*80)
    print("Testing Detector with Sample Texts")
    print("="*80 + "\n")
    
    # Test each case
    for i, text in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Text: \"{text}\"")
        print("-" * 80)
        
        try:
            result = detect_hate_speech(model, tokenizer, device, text)
            
            print(f"Intensity: {result['intensity']} ({result['intensity_label']})")
            print(f"Confidence: {result['intensity_prob']:.4f}")
            print(f"Intensity Probabilities: {[f'{p:.3f}' for p in result['intensity_probs']]}")
            
            if result['bio_spans']:
                print(f"Detected Spans:")
                for span in result['bio_spans']:
                    print(f"  - {span['type']}: \"{span['text']}\" (tokens {span['start']}-{span['end']})")
            else:
                print("Detected Spans: None")
            
            print(f"Target Probabilities: {[f'{p:.3f}' for p in result['target_probs']]}")
            
            print(f"Target Category: {result['target']}")
            print()
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*80)
    print("✅ Detector test completed!")
    print("="*80)

if __name__ == "__main__":
    main()

