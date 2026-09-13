import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

class KhatersBioDataset(Dataset):
    """
    K-HATERS 데이터셋을 BIO tagging (O, B-SOFT, I-SOFT, B-HARD, I-HARD)으로 변환
    """
    def __init__(self, split="train", tokenizer_name="beomi/KcELECTRA-base", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label2id = {"O": 0, "B-SOFT": 1, "I-SOFT": 2, "B-HARD": 3, "I-HARD": 4}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 문장 분류 라벨 매핑
        self.sentence_label2id = {
            "normal": 0,
            "offensive": 1,
            "L1_hate": 2,
            "L2_hate": 3
        }
        self.sentence_id2label = {v: k for k, v in self.sentence_label2id.items()}
        
        # Target 라벨 매핑
        self.target2id = {
            "gender": 0,
            "age": 1,
            "political": 2,
            "religion": 3,
            "region": 4,
            "others": 5
        }
        self.id2target = {v: k for k, v in self.target2id.items()}
        
        self.examples = []
        ds = load_dataset("humane-lab/K-HATERS", split=split)
        
        for ex in ds:
            text = ex["text"]
            offensiveness_rationale = ex.get("offensiveness_rationale", [])
            
            # 문장 분류 라벨 결정
            sentence_label = self._get_sentence_label(ex)
            
            # Target 라벨 처리
            target_labels = self._get_target_labels(ex)
            
            # 토크나이저로 토큰화
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            
            # BIO 라벨 시퀀스 초기화
            bio_labels = ["O"] * len(encoding["input_ids"])
            
            # offensiveness_rationale 정보를 BIO로 변환
            if sentence_label > 0:  # normal이 아닌 경우에만 BIO 라벨링
                for start, end in offensiveness_rationale:
                    # 문장 라벨에 따라 BIO 라벨 결정
                    if sentence_label in [1, 2]:  # offensive, L1_hate
                        label_prefix = "SOFT"
                    elif sentence_label == 3:  # L2_hate
                        label_prefix = "HARD"
                    else:
                        continue
                    
                    # offset mapping을 사용하여 토큰과 문자 위치 매칭
                    for i, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
                        if token_start == token_end:  # 특수 토큰 (CLS, SEP, PAD 등)
                            continue
                        
                        # 토큰과 범위가 겹치는지 확인
                        if token_end <= start or token_start >= end:
                            continue
                        
                        # 겹치는 부분이 있으면 라벨 할당
                        if token_start <= start and token_end > start:
                            bio_labels[i] = f"B-{label_prefix}"
                        elif token_start < end and token_end >= end:
                            bio_labels[i] = f"I-{label_prefix}"
                        elif token_start >= start and token_end <= end:
                            bio_labels[i] = f"I-{label_prefix}"
            
            # 라벨을 ID로 변환
            token_labels = [self.label2id[label] for label in bio_labels]
            
            self.examples.append({
                "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
                "bio_labels": torch.tensor(token_labels, dtype=torch.long),
                "sentence_labels": torch.tensor(sentence_label, dtype=torch.long),
                "target_labels": torch.tensor(target_labels, dtype=torch.float)
            })

    def _get_sentence_label(self, ex):
        """문장의 전체 라벨을 결정"""
        # 원본 데이터셋의 label 필드 사용
        label = ex.get("label", "normal")
        
        # 라벨 매핑
        label_mapping = {
            "normal": 0,
            "offensive": 1,
            "L1_hate": 2,
            "L2_hate": 3
        }
        
        return label_mapping.get(label, 0)  # 기본값은 normal
    
    def _get_target_labels(self, ex):
        """Target 라벨을 멀티라벨 형태로 변환"""
        target_labels = [0.0] * len(self.target2id)
        target_list = ex.get("target_label", [])
        
        for target in target_list:
            if target in self.target2id:
                target_labels[self.target2id[target]] = 1.0
        
        return target_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx] 