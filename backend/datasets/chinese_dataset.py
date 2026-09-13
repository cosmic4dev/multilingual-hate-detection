"""
Chinese ToxiCN Dataset for Hate Speech Detection
Supports BIO tagging, intensity classification, and target detection
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple
import pandas as pd
import os


class ChineseToxiCNDataset(Dataset):
    """Chinese ToxiCN Dataset with unified BIO tagging scheme"""
    
    def __init__(self, split: str, tokenizer_name: str = "xlm-roberta-base", max_length: int = 128):
        self.language = "zh"
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_toxicn_data(split)
        
        # Define label mappings (unified with other languages)
        self.bio2id = {"O": 0, "B-MILD": 1, "I-MILD": 2, "B-SEVERE": 3, "I-SEVERE": 4}
        self.id2bio = {v: k for k, v in self.bio2id.items()}
        
        # Intensity mapping (3-class: None/Mild/Severe)
        self.intensity2id = {"NONE": 0, "MILD": 1, "SEVERE": 2}
        self.id2intensity = {v: k for k, v in self.intensity2id.items()}
        
        # Target categories (5 categories from ToxiCN)
        self.target_categories = ["lgbt", "race", "region", "gender", "general"]
        self.target2id = {cat: i for i, cat in enumerate(self.target_categories)}
        
        print(f"Loaded {len(self.data)} Chinese samples for {split} split")
        print(f"Target categories: {self.target_categories}")
    
    def _load_toxicn_data(self, split: str) -> List[Dict]:
        """Load ToxiCN data from JSON files"""
        if split == "train":
            file_path = "/root/STATE-ToxiCN/data/train.json"
        elif split == "test":
            file_path = "/root/STATE-ToxiCN/data/test.json"
        else:
            raise ValueError(f"Invalid split: {split}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def _create_bio_labels(self, text: str, toxic: int, toxic_type: int) -> List[str]:
        """Create BIO labels based on toxicity and type (키워드 기반 근사화)"""
        bio_labels = ["O"] * len(text)
        
        if toxic == 1:  # Toxic content
            # 중국어 혐오표현 키워드 패턴
            severe_keywords = [
                '傻', '蠢', '白痴', '智障', '脑残', '弱智',
                '死', '滚', '去死', '该死', '操', '艹',
                '垃圾', '废物', '人渣', '败类', '杂种',
                '丑', '胖', '矮', '穷', '土', 'low'
            ]
            
            mild_keywords = [
                '烦', '讨厌', '恶心', '无语', '服了',
                '什么', '怎么', '为什么', '真是',
                '呵呵', '哈哈', '笑死', '醉了'
            ]
            
            # 키워드 찾기 및 BIO 태깅
            found_spans = []
            
            # 심각한 혐오표현 키워드 찾기
            for keyword in severe_keywords:
                start = text.find(keyword)
                if start != -1:
                    end = start + len(keyword)
                    found_spans.append((start, end, "SEVERE"))
            
            # 경미한 혐오표현 키워드 찾기
            for keyword in mild_keywords:
                start = text.find(keyword)
                if start != -1:
                    end = start + len(keyword)
                    found_spans.append((start, end, "MILD"))
            
            # 중복 제거 및 정렬
            found_spans = list(set(found_spans))
            found_spans.sort(key=lambda x: x[0])
            
            # BIO 태깅 적용
            for start, end, intensity in found_spans:
                if start < len(bio_labels):
                    bio_labels[start] = f"B-{intensity}"
                    for i in range(start + 1, min(end, len(bio_labels))):
                        bio_labels[i] = f"I-{intensity}"
            
            # 키워드를 찾지 못한 경우 전체 문장을 태깅 (하이브리드 방식)
            if not found_spans:
                if toxic_type == 1:  # Mild toxicity
                    label_prefix = "MILD"
                else:  # toxic_type == 2, Severe toxicity
                    label_prefix = "SEVERE"
                
                if len(text) > 0:
                    bio_labels[0] = f"B-{label_prefix}"
                    for i in range(1, len(text)):
                        bio_labels[i] = f"I-{label_prefix}"
        
        return bio_labels
    
    def _get_intensity_label(self, toxic: int, toxic_type: int) -> int:
        """Convert ToxiCN labels to unified intensity scheme"""
        if toxic == 0:
            return self.intensity2id["NONE"]
        elif toxic_type == 1:
            return self.intensity2id["MILD"]
        else:  # toxic_type == 2
            return self.intensity2id["SEVERE"]
    
    def _get_target_labels(self, topic: str) -> torch.Tensor:
        """Convert topic to target vector"""
        target_vector = torch.zeros(len(self.target_categories), dtype=torch.float)
        
        # Map topic to target category
        topic_mapping = {
            "lgbt": "lgbt",
            "race": "race", 
            "region": "region",
            "gender": "gender",
            "general": "general"  # Default for other topics
        }
        
        target_category = topic_mapping.get(topic, "general")
        if target_category in self.target2id:
            target_vector[self.target2id[target_category]] = 1.0
        
        return target_vector
    
    def _tokenize_and_align_labels(self, text: str, bio_labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize text and align BIO labels with tokens"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        
        # Align BIO labels with tokens
        bio_label_ids = torch.zeros(self.max_length, dtype=torch.long)
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0:  # Special tokens
                bio_label_ids[i] = self.bio2id["O"]
            else:
                # Find corresponding character-level label
                char_start = int(start)
                char_end = int(end)
                
                if char_start < len(bio_labels):
                    # Use the label of the first character in this token
                    char_label = bio_labels[char_start]
                    bio_label_ids[i] = self.bio2id.get(char_label, self.bio2id["O"])
                else:
                    bio_label_ids[i] = self.bio2id["O"]
        
        return input_ids, attention_mask, bio_label_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item['content']
        toxic = int(item['sen_hate'])
        toxic_type = item.get('topic', 'general')
        topic = item['topic']
        
        # Create BIO labels
        bio_labels = self._create_bio_labels(text, toxic, toxic_type)
        
        # Get intensity label
        intensity_label = self._get_intensity_label(toxic, toxic_type)
        
        # Get target labels
        target_labels = self._get_target_labels(topic)
        
        # Tokenize and align
        input_ids, attention_mask, bio_label_ids = self._tokenize_and_align_labels(text, bio_labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bio_labels': bio_label_ids,
            'sentence_labels': torch.tensor(intensity_label, dtype=torch.long),
            'target_labels': target_labels,
            'text': text,
            'original_bio_labels': bio_labels
        }


def test_chinese_dataset():
    """Test the Chinese dataset implementation"""
    print("Testing Chinese ToxiCN Dataset...")
    
    # Test train dataset
    train_dataset = ChineseToxiCNDataset("train")
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test a sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"Text: {sample['text']}")
    print(f"Intensity: {train_dataset.id2intensity[sample['intensity_labels'].item()]}")
    print(f"Target: {[train_dataset.target_categories[i] for i, val in enumerate(sample['target_labels']) if val == 1.0]}")
    print(f"BIO labels: {[train_dataset.id2bio[label.item()] for label in sample['bio_labels'][:20]]}")
    
    # Test test dataset
    test_dataset = ChineseToxiCNDataset("test")
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    # Check label distributions
    train_intensities = [train_dataset[i]['intensity_labels'].item() for i in range(min(100, len(train_dataset)))]
    intensity_counts = {train_dataset.id2intensity[i]: train_intensities.count(i) for i in range(3)}
    print(f"\nIntensity distribution (first 100 samples): {intensity_counts}")
    
    print("Chinese dataset test completed!")


if __name__ == "__main__":
    test_chinese_dataset()
