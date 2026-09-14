# Multilingual Hate Speech Detection & Guided Detoxification

**Awareness-Enhanced Guidance for Iterative Safeguard**

[![arXiv](https://img.shields.io/badge/arXiv-2607.17713-b31b1b)](https://arxiv.org/abs/2607.17713)

AEGIS is an exploratory framework for studying span-guided multilingual text
detoxification across English, Mandarin Chinese, and Korean. It separates a
span-level detector from frozen generator backbones so that the effect of
harmful-span, intensity, and target guidance can be examined without treating
the framework as a state-of-the-art claim.

| Resource | Status |
|---|---|
| Paper | [arXiv:2607.17713](https://arxiv.org/abs/2607.17713) |
| Code | Detector training and guided-generation pipeline available |
| Data | Use the official upstream datasets described in [DATA.md](DATA.md) |
| License | [MIT](LICENSE) for code; upstream terms apply to data and models |

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code for multilingual span-level hate-speech detection and controlled
detoxification across English, Chinese, and Korean. The project connects token
classification, guided generation, and evaluator diagnostics while keeping the
trade-off between toxicity reduction and meaning preservation explicit.

## What this project demonstrates

| Component | Scope |
|---|---|
| Detection | XLM-R token classification with harmful-span BIO labels |
| Generation | Guided and unguided multilingual rewriting pipelines |
| Evaluation | Automatic metrics, sensitivity analyses, and model comparisons |
| Reproducibility | Audited code, configuration guidance, and compact metrics |

The detector is optimized with harmful-token precision, recall, and Non-`O`
F1 rather than relying on majority-dominated token accuracy.

## Recorded detector results

| Language | Dataset | Non-O F1 | BIO F1 | Intensity F1 | Target F1 (macro) |
|---|---|---:|---:|---:|---:|
| Chinese | STATE ToxiCN | 69.89% | 95.42% | 78.69% | 59.75% |
| English | HateXplain | 73.65% | 94.69% | 68.70% | 67.32% |
| Korean | K-HATERS | 67.56% | 97.27% | 66.14% | 68.45% |

These are recorded results from the included experiment configuration, not a
claim that the datasets or languages are directly comparable.

## Setup

Python 3.8 or newer is recommended. CUDA is useful for training and generation.

```bash
git clone https://github.com/cosmic4dev/multilingual-hate-detection.git
cd multilingual-hate-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Provide gated-model and API credentials through environment variables or the
relevant provider CLI. Never place credentials directly in scripts.

## Usage

Train the detector variants:

```bash
python backend/training/chinese_xlmr_train.py --epochs 8 --patience 3
python backend/training/english_xlmr_train.py --epochs 8 --patience 3
python backend/training/korean_xlmr_train.py --epochs 8 --patience 3
```

Generate guided and unguided rewrites from a locally obtained evaluation file:

```bash
python human_eval/run_multi_generator.py \
  --input_csv /path/to/local/evaluation_input.csv \
  --out_dir /path/to/local/output
```

Inspect the available evaluation entry points:

```bash
python experiments/auto_evaluate_from_csv.py --help
python experiments/run_multi_model_auto_eval.py --help
```

## Repository map

```text
backend/
├── datasets/       Dataset loaders and detector integration
├── training/       Language-specific XLM-R training
└── results/        Compact detector metrics
human_eval/         Generation and evaluation utilities
experiments/        Automatic evaluation and sensitivity analyses
```

Raw datasets, row-level human evaluations, generated-text pools, model weights,
and manuscript or submission materials are intentionally excluded. See
[`DATA.md`](DATA.md) and [`PUBLIC.md`](PUBLIC.md) for the release boundary.

## External datasets

Obtain datasets from their official releases and configure local paths in the
loaders:

- [HateXplain](https://github.com/hate-alert/HateXplain)
- [STATE ToxiCN](https://github.com/state-sp/STATE-ToxiCN)
- [K-HATERS](https://github.com/2runo3/K-HATERS)

## Evidence boundary

The included code and metrics support inspection and reproduction of the
documented pipeline. They do not establish that one guidance policy is
universally optimal, that automatic toxicity scores replace human judgment, or
that performance transfers unchanged across languages and domains.

## Collaboration

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Contributions are especially useful
for portable configuration, dataset adapters, multilingual evaluation, and
audits of toxicity–meaning trade-offs.

## Contact

[GitHub Issues](https://github.com/cosmic4dev/multilingual-hate-detection/issues)
