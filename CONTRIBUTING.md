# Contributing to Multilingual Hate Speech Detection

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation
```bash
git clone https://github.com/cosmic4dev/multilingual-hate-detection.git
cd multilingual-hate-detection
cp .env.example .env   # add keys locally; never commit .env
pip install -r requirements.txt
```

## 🧪 Running Experiments

### Training Individual Models
```bash
# Chinese model
python backend/training/chinese_xlmr_train.py --epochs 8 --patience 3

# English model  
python backend/training/english_xlmr_train.py --epochs 8 --patience 3

# Korean model
python backend/training/korean_xlmr_train.py --epochs 8 --patience 3
```

### Evaluation
```python
from backend.training.chinese_xlmr_train import ChineseXLMTrainer

# Load and evaluate model
trainer = ChineseXLMTrainer()
results = trainer.evaluate()
print(f"Non-O F1: {results['non_o_f1']:.4f}")
```

## 🔬 Adding New Languages

1. **Create Dataset Class**:
```python
# backend/datasets/new_language_dataset.py
class NewLanguageDataset(Dataset):
    def __init__(self, split: str, tokenizer_name: str = "xlm-roberta-base"):
        # Implement dataset loading
        pass
```

2. **Create Training Script**:
```python
# backend/training/new_language_xlmr_train.py
class NewLanguageXLMTrainer:
    def __init__(self):
        # Implement training logic
        pass
```

3. **Update Results**: Add results to `experiments/results_summary.md`

## 📊 Benchmarking

### Performance Metrics
- **Non-O F1**: Primary metric for harmful span detection
- **BIO F1**: Overall token-level accuracy
- **Intensity F1**: Severity classification accuracy
- **Target F1**: Target group classification accuracy

### Expected Performance
| Language | Non-O F1 | BIO F1 | Training Time |
|----------|----------|--------|---------------|
| Chinese | 69.89% | 95.42% | ~3 hours |
| English | 73.65% | 94.69% | ~2 hours |
| Korean | 67.56% | 97.27% | ~4 hours |

## 🐛 Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- GPU information (if applicable)
- Complete error traceback
- Steps to reproduce

## 📝 Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Add comments for complex logic

## 🔄 Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit pull request

## 📞 Questions?

- Open an issue for bugs
- Start a discussion for questions
- Contact maintainers for collaboration

---

Thank you for contributing! 🎉
