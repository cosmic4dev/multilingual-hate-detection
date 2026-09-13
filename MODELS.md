# Model Files and Download Instructions

## 📦 Model Files

Due to GitHub's file size limitations (100MB per file), the trained models are stored separately.

### Model Information

| Language | Model File | Size | Non-O F1 | Download |
|----------|------------|------|----------|----------|
| Chinese | `chinese_xlmr_model.pt` | ~1.1GB | 69.89% | [Download](https://huggingface.co/USERNAME/models) |
| English | `english_xlmr_model.pt` | ~1.1GB | 73.65% | [Download](https://huggingface.co/USERNAME/models) |
| Korean | `korean_xlmr_model.pt` | ~1.1GB | 67.56% | [Download](https://huggingface.co/USERNAME/models) |

## 🚀 Quick Setup

### Option 1: Download Models Manually
```bash
# Create models directory
mkdir -p backend/models/

# Download models (replace with actual URLs)
wget https://example.com/chinese_xlmr_model.pt -O backend/models/chinese_xlmr_model.pt
wget https://example.com/english_xlmr_model.pt -O backend/models/english_xlmr_model.pt
wget https://example.com/korean_xlmr_model.pt -O backend/models/korean_xlmr_model.pt
```

### Option 2: Train Models Yourself
```bash
# Train all models (will take several hours)
python backend/training/chinese_xlmr_train.py --epochs 8 --patience 3
python backend/training/english_xlmr_train.py --epochs 8 --patience 3
python backend/training/korean_xlmr_train.py --epochs 8 --patience 3
```

### Option 3: Use Hugging Face Hub (Recommended)
```python
from transformers import AutoModel
import torch

# Load model from Hugging Face Hub
model = AutoModel.from_pretrained("USERNAME/chinese-xlmr-hate-detection")
```

## 🔧 Model Usage

### Loading Trained Models
```python
import torch
from backend.training.chinese_xlmr_train import ChineseXLMDetector

# Load model
model = ChineseXLMDetector(num_targets=6)
model.load_state_dict(torch.load('backend/models/chinese_xlmr_model.pt'))
model.eval()

# Use for inference
# (See individual training scripts for usage examples)
```

### Model Specifications

#### Chinese Model
- **Base**: XLM-RoBERTa-base
- **Targets**: 6 categories (Sexism, Racism, LGBTQ, Region, others, non-hate)
- **BIO Classes**: 5 (O, B-MILD, I-MILD, B-SEVERE, I-SEVERE)
- **Intensity**: 3 levels (NONE, MILD, SEVERE)

#### English Model
- **Base**: XLM-RoBERTa-base
- **Targets**: 10 categories
- **BIO Classes**: 9 (O, B-SOFT, I-SOFT, B-HARD, I-HARD, B-MILD, I-MILD, B-SEVERE, I-SEVERE)
- **Intensity**: 6 levels (normal, offensive, L1_hate, L2_hate, mild, severe)

#### Korean Model
- **Base**: XLM-RoBERTa-base
- **Targets**: Not available in current implementation
- **BIO Classes**: 5 (O, B-MILD, I-MILD, B-SEVERE, I-SEVERE)
- **Intensity**: 4 levels

## 📊 Performance Benchmarks

### Training Configuration
- **Optimizer**: AdamW (lr=2e-5)
- **Batch Size**: 16 (train), 32 (test)
- **Max Epochs**: 8
- **Early Stopping**: Patience 3
- **Class Weights**: Applied for BIO loss

### Results Summary
- **Best Non-O F1**: English (73.65%)
- **Best BIO F1**: Korean (97.27%)
- **Best Intensity F1**: Chinese (78.69%)
- **Most Stable**: Korean (largest dataset)

## 🔄 Model Updates

Models are periodically updated with:
- Better hyperparameters
- Improved preprocessing
- Additional training data
- Architecture optimizations

Check the release notes for update information.

## 📞 Support

For model-related issues:
- Check the training logs in `backend/results/`
- Review the training scripts for configuration
- Open an issue with model performance questions

---

**Note**: Models are trained on specific datasets and may not generalize to all domains. Consider fine-tuning for your specific use case.
