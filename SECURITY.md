# Security

## Reporting vulnerabilities

Please report security issues via [GitHub Security Advisories](https://github.com/cosmic4dev/multilingual-hate-detection/security/advisories) (private report) or open a minimal public issue if the advisory UI is unavailable.

## Secrets

- Never commit `.env`. Use `.env.example` as a template.
- Supported variables include `OPENAI_API_KEY`, `HF_TOKEN`, and
  `HUGGINGFACEHUB_API_TOKEN`; see the scripts in `human_eval/`.
- If a key was ever pasted into a committed file, rotate it immediately and consider rewriting Git history (`git filter-repo`).

## Human evaluation data

Raw evaluator responses (spreadsheets, per-rater CSV) are excluded from this public repository. See `DATA.md`.
