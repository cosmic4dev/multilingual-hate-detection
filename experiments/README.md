# Automatic evaluation experiments

This directory contains reusable evaluation and sensitivity-analysis scripts
for multilingual detoxification outputs.

The public release intentionally omits item-level generations and evaluator
records. Provide locally obtained inputs and write new outputs to a separate or
ignored directory.

Useful entry points include:

- `auto_evaluate_from_csv.py` for one evaluation file;
- `run_multi_model_auto_eval.py` for comparable model runs;
- `run_lambda_sensitivity.py` for guidance-strength analysis;
- `strength_analysis.py` for condition-level summaries.

Treat automatic toxicity and semantic-similarity scores as diagnostic signals,
not as substitutes for audited human evaluation.
