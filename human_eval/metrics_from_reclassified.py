# -*- coding: utf-8 -*-
"""
Build human_eval_*_full_metrics.json structure from docs/human_eval_reclassified.csv
(long-form: one row per evaluator × item).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None

REASON_MAP = {
    "원문의 의미나 의도가 잘 유지되었다": "Meaning preservation",
    "문장 자체가 더 자연스럽고 읽기 쉽다": "Fluency/readability",
    "다른 출력은 원문을 지나치게 많이 바꾼 것 같다": "Over-modification avoidance",
    "두 출력 간에 뚜렷한 차이를 느끼기 어렵다": "No clear difference",
    "안전성과 의미 보존의 균형이 잘 맞는다": "Balance (safety & meaning)",
    "유해하거나 공격적인 표현이 효과적으로 완화되었다": "Effective toxicity reduction",
    "다른 출력은 유해한 표현을 충분히 완화하지 못했다": "Insufficient mitigation",
}


def plurality_label(ng: int, nu: int, nt: int) -> str:
    m = max(ng, nu, nt)
    winners: List[str] = []
    if ng == m:
        winners.append("guided")
    if nu == m:
        winners.append("unguided")
    if nt == m:
        winners.append("tie")
    if len(winners) == 1:
        return winners[0]
    return "ambiguous"


def item_majority_bucket(pl: str) -> str:
    if pl == "guided":
        return "guided"
    if pl == "unguided":
        return "unguided"
    return "tie"


def _bootstrap_delta_p(
    items: List[Dict[str, Any]], n_iter: int = 10_000, seed: int = 42
) -> Tuple[float, float]:
    """Bootstrap CI for p_hat_strong - p_hat_mild among decisive items (no tie-majority)."""
    rng = np.random.default_rng(seed)
    strong = [i for i in items if i["strength"] == "strong" and i["bucket"] in ("guided", "unguided")]
    mild = [i for i in items if i["strength"] == "mild" and i["bucket"] in ("guided", "unguided")]
    if not strong or not mild:
        return float("nan"), float("nan")

    def phat(lst):
        g = sum(1 for x in lst if x["bucket"] == "guided")
        return g / len(lst) if lst else float("nan")

    deltas = []
    for _ in range(n_iter):
        rs = [strong[rng.integers(0, len(strong))] for _ in range(len(strong))]
        rm = [mild[rng.integers(0, len(mild))] for _ in range(len(mild))]
        deltas.append(phat(rs) - phat(rm))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def compute_full_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    n_raters = df.groupby("item_id").size()
    if not (n_raters == n_raters.iloc[0]).all():
        raise ValueError("Inconsistent judgments per item.")
    n_per = int(n_raters.iloc[0])
    n_items = df["item_id"].nunique()
    if n_items != 60:
        raise ValueError(f"Expected 60 items, got {n_items}")

    # --- judgment-level ---
    pref = df["is_guided_preferred"]
    jl = {
        "guided": int(pref.eq(True).sum()),
        "unguided": int(pref.eq(False).sum()),
        "tie": int(pref.isna().sum()),
    }
    total_j = jl["guided"] + jl["unguided"] + jl["tie"]

    jls = {"strong": {"guided": 0, "unguided": 0, "tie": 0}, "mild": {"guided": 0, "unguided": 0, "tie": 0}}
    for strength in ("strong", "mild"):
        sub = df[df["toxicity_strength"].str.lower() == strength]
        p = sub["is_guided_preferred"]
        jls[strength]["guided"] = int(p.eq(True).sum())
        jls[strength]["unguided"] = int(p.eq(False).sum())
        jls[strength]["tie"] = int(p.isna().sum())

    # blocks by question order
    old = df[df["question_no"] <= 30]
    new = df[df["question_no"] > 30]
    jlb = {
        "old30": {
            "guided": int(old["is_guided_preferred"].eq(True).sum()),
            "unguided": int(old["is_guided_preferred"].eq(False).sum()),
            "tie": int(old["is_guided_preferred"].isna().sum()),
        },
        "new30": {
            "guided": int(new["is_guided_preferred"].eq(True).sum()),
            "unguided": int(new["is_guided_preferred"].eq(False).sum()),
            "tie": int(new["is_guided_preferred"].isna().sum()),
        },
    }

    # --- per-item ---
    item_rows: List[Dict[str, Any]] = []
    for item_id, g in df.groupby("item_id", sort=False):
        strength = str(g["toxicity_strength"].iloc[0]).lower()
        qno = int(g["question_no"].iloc[0])
        p = g["is_guided_preferred"]
        ng = int(p.eq(True).sum())
        nu = int(p.eq(False).sum())
        nt = int(p.isna().sum())
        pl = plurality_label(ng, nu, nt)
        bucket = item_majority_bucket(pl)
        item_rows.append(
            {
                "item_id": item_id,
                "question_no": qno,
                "strength": strength,
                "plurality": pl,
                "bucket": bucket,
            }
        )

    im = {"guided": 0, "unguided": 0, "tie": 0}
    for r in item_rows:
        im[r["bucket"]] += 1

    ims = {"strong": {"guided": 0, "unguided": 0, "tie": 0}, "mild": {"guided": 0, "unguided": 0, "tie": 0}}
    for r in item_rows:
        s = r["strength"]
        ims[s][r["bucket"]] += 1

    imb = {
        "old30": {"guided": 0, "unguided": 0, "tie": 0},
        "new30": {"guided": 0, "unguided": 0, "tie": 0},
    }
    for r in item_rows:
        block = "old30" if r["question_no"] <= 30 else "new30"
        imb[block][r["bucket"]] += 1

    # Fisher: 2x2 on decisive items only (guided vs unguided majority)
    strong_d = [r for r in item_rows if r["strength"] == "strong" and r["bucket"] in ("guided", "unguided")]
    mild_d = [r for r in item_rows if r["strength"] == "mild" and r["bucket"] in ("guided", "unguided")]
    sg = sum(1 for r in strong_d if r["bucket"] == "guided")
    su = sum(1 for r in strong_d if r["bucket"] == "unguided")
    mg = sum(1 for r in mild_d if r["bucket"] == "guided")
    mu = sum(1 for r in mild_d if r["bucket"] == "unguided")

    fisher_p = None
    if fisher_exact and sg + su > 0 and mg + mu > 0:
        _, fisher_p = fisher_exact([[sg, su], [mg, mu]])

    lo, hi = _bootstrap_delta_p(item_rows, n_iter=10_000, seed=42)

    item_stats = {
        "strong_guided": sg,
        "strong_unguided": su,
        "mild_guided": mg,
        "mild_unguided": mu,
        "fisher_two_sided_p": fisher_p,
        "delta_p_bootstrap_95": [lo, hi],
    }

    # rationale (only non-tie judgments)
    df_nt = df[df["is_guided_preferred"].notna()].copy()
    df_nt["reason_cat"] = df_nt["reason"].map(REASON_MAP).fillna("Other")
    rationale_distribution: Dict[str, Dict[str, int]] = {"guided": {}, "unguided": {}}
    for col_pref, key in [(True, "guided"), (False, "unguided")]:
        sub = df_nt[df_nt["is_guided_preferred"] == col_pref]
        rationale_distribution[key] = sub["reason_cat"].value_counts().to_dict()

    # meaning scores
    def mean_score(mask):
        s = pd.to_numeric(df.loc[mask, "meaning_preservation_score"], errors="coerce")
        s = s.dropna()
        return float(s.mean()) if len(s) else float("nan")

    m_guided = mean_score(df["is_guided_preferred"] == True)
    m_unguided = mean_score(df["is_guided_preferred"] == False)

    meaning_by_rationale: Dict[str, float] = {}
    for cat in df_nt["reason_cat"].unique():
        if cat == "Other":
            continue
        sub = df_nt[df_nt["reason_cat"] == cat]
        s = pd.to_numeric(sub["meaning_preservation_score"], errors="coerce").dropna()
        if len(s):
            meaning_by_rationale[str(cat)] = float(s.mean())

    out: Dict[str, Any] = {
        "respondents": int(df["evaluator"].nunique()),
        "questions": 60,
        "total_judgments": total_j,
        "judgment_level": jl,
        "judgment_level_strength": jls,
        "judgment_level_blocks": jlb,
        "item_majority": im,
        "item_majority_strength": ims,
        "item_majority_blocks": imb,
        "item_level_stats": item_stats,
        "rationale_distribution": rationale_distribution,
        "meaning_mean_by_pref": {"guided": m_guided, "unguided": m_unguided},
        "meaning_mean_by_rationale": meaning_by_rationale,
    }
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Compute metrics JSON from reclassified CSV")
    ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "human_eval_reclassified.csv")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "human_eval_60_full_metrics.json")
    ap.add_argument("--exclude-evaluator", type=str, default=None, help="Drop rows with this evaluator id (e.g. timestamp)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.exclude_evaluator:
        df = df[df["evaluator"].astype(str) != str(args.exclude_evaluator)]
    metrics = compute_full_metrics(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Wrote", args.out)
    print("respondents", metrics["respondents"], "total_judgments", metrics["total_judgments"])


if __name__ == "__main__":
    main()
