"""
English HateXplain Dataset for Hate Speech Detection
Supports BIO tagging, intensity classification, and target detection
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any
from datasets import load_dataset

class EnglishHateXplainDataset(Dataset):
    """English HateXplain Dataset with unified BIO tagging scheme"""
    
    def __init__(self, split: str, tokenizer_name: str = "microsoft/infoxlm-base", max_length: int = 128):
        self.language = "en"
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Define label mappings (unified with other languages)
        self.bio2id = {"O": 0, "B-SOFT": 1, "I-SOFT": 2, "B-HARD": 3, "I-HARD": 4}
        self.id2bio = {v: k for k, v in self.bio2id.items()}
        
        # Intensity mapping (6-class: normal, offensive, L1_hate, L2_hate, mild, severe)
        self.intensity2id = {"normal": 0, "offensive": 1, "L1_hate": 2, "L2_hate": 3, "mild": 4, "severe": 5}
        self.id2intensity = {v: k for k, v in self.intensity2id.items()}
        
        # Target categories (unified 10 categories)
        self.target_categories = ["gender", "age", "political", "religion", "region", 
                                "others", "sexism", "racism", "lgbtq", "non-hate"]
        self.target2id = {cat: i for i, cat in enumerate(self.target_categories)}
        
        # Load data
        self.data = self._load_hatexplain_data(split)
        
        print(f"Loaded {len(self.data)} English samples for {split} split")
        print(f"Target categories: {self.target_categories}")
    
    def _load_hatexplain_data(self, split: str) -> List[Dict]:
        """Load HateXplain data from HuggingFace datasets"""
        try:
            dataset = load_dataset('hatexplain', trust_remote_code=True)
            
            if split == "train":
                data = dataset['train']
            elif split == "test":
                data = dataset['test']
            else:
                raise ValueError(f"Invalid split: {split}")
            
            processed_data = []
            for item in data:
                processed_item = self._process_hatexplain_item(item)
                if processed_item:
                    processed_data.append(processed_item)
            
            return processed_data
            
        except Exception as e:
            print(f"❌ Error loading HateXplain data: {e}")
            return []
    
    def _process_hatexplain_item(self, item: Dict) -> Dict:
        """Process HateXplain item to unified format"""
        try:
            # HateXplain uses 'post_tokens' instead of 'text'
            text = ' '.join(item['post_tokens'])
            
            # Get intensity label
            post_id = item.get('id', '')
            annotators = item.get('annotators', {})
            rationales = item.get('rationales', [])
            
            # Determine intensity from annotators
            intensity_label = self._get_intensity_from_annotators(annotators)

            # Get target labels
            target_labels = self._get_target_labels_from_annotators(annotators)

            # Get BIO labels from rationales
            bio_labels = self._get_bio_labels_from_rationales(
                rationales,
                len(item['post_tokens']),
                intensity_label
            )
            
            return {
                'text': text,
                'intensity_label': intensity_label,
                'target_labels': target_labels,
                'bio_labels': bio_labels,
                'post_id': post_id
            }
            
        except Exception as e:
            print(f"❌ Error processing item: {e}")
            return None
    
    def _get_intensity_from_annotators(self, annotators: Dict) -> str:
        """Extract intensity from annotators"""
        if not annotators or 'label' not in annotators:
            return "normal"
        
        # Get majority label from annotators
        labels = annotators['label']
        majority_label = max(set(labels), key=labels.count)
        
        # Map HateXplain labels to intensity
        if majority_label == 0:
            return "normal"
        elif majority_label == 1:
            return "offensive"
        else:  # majority_label == 2
            return "severe"
    
    def _get_target_labels_from_annotators(self, annotators: Dict) -> List[str]:
        """Extract target labels from annotators"""
        targets = set()
        
        if 'target' in annotators:
            target_lists = annotators['target']
            for target_list in target_lists:
                if isinstance(target_list, list):
                    targets.update(target_list)
                else:
                    targets.add(target_list)
        
        # Map to unified target categories
        unified_targets = []
        for target in targets:
            if target in self.target2id:
                unified_targets.append(target)
            else:
                # Map unknown targets to 'others'
                unified_targets.append('others')
        
        return unified_targets if unified_targets else ['non-hate']
    
    def _get_bio_labels_from_rationales(self, rationales: List[List[int]], num_tokens: int, intensity_label: str) -> List[str]:
        """Generate BIO labels from HateXplain rationales"""
        if not rationales:
            return ["O"] * num_tokens
        
        # Get majority rationale (most common pattern across annotators)
        rationale_votes = {}
        for rationale in rationales:
            if not rationale:  # Skip empty rationales
                continue
            rationale_str = str(rationale)
            rationale_votes[rationale_str] = rationale_votes.get(rationale_str, 0) + 1
        
        # If no valid rationales, return all O
        if not rationale_votes:
            return ["O"] * num_tokens
        
        # Get the most common rationale
        majority_rationale = max(rationale_votes, key=rationale_votes.get)
        majority_rationale = eval(majority_rationale)  # Convert back to list
        
        # Convert rationale to BIO labels
        label_suffix = self._get_span_type_from_intensity(intensity_label)
        bio_labels = []
        for i in range(num_tokens):
            if i < len(majority_rationale):
                is_hate = majority_rationale[i]
            else:
                is_hate = 0  # Pad with 0 (non-hate)
            
            if is_hate:
                if i == 0 or (i > 0 and i-1 < len(majority_rationale) and majority_rationale[i-1] == 0):
                    bio_labels.append(f"B-{label_suffix}")
                else:
                    bio_labels.append(f"I-{label_suffix}")
            else:
                bio_labels.append("O")
        
        return bio_labels

    def _get_span_type_from_intensity(self, intensity_label: str) -> str:
        """Determine if a span should be tagged as SOFT or HARD"""
        if intensity_label == "offensive":
            return "SOFT"
        return "HARD"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert BIO labels to IDs
        bio_labels = item['bio_labels']
        bio_label_ids = [self.bio2id.get(label, 0) for label in bio_labels]
        
        # Pad or truncate BIO labels
        if len(bio_label_ids) < self.max_length:
            bio_label_ids.extend([0] * (self.max_length - len(bio_label_ids)))
        else:
            bio_label_ids = bio_label_ids[:self.max_length]
        
        # Convert intensity label to ID
        intensity_label_id = self.intensity2id.get(item['intensity_label'], 0)
        
        # Convert target labels to multi-hot vector
        target_labels = [0.0] * len(self.target_categories)
        for target in item['target_labels']:
            if target in self.target2id:
                target_labels[self.target2id[target]] = 1.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'bio_labels': torch.tensor(bio_label_ids, dtype=torch.long),
            'sentence_labels': torch.tensor(intensity_label_id, dtype=torch.long),
            'target_labels': torch.tensor(target_labels, dtype=torch.float),
            'language': 'english'
        }