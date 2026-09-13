#!/usr/bin/env python3
"""
Toxicity Strength (Strong vs Mild) 기반 Span-Guided vs Unguided 비교 분석
4개 오픈소스 모델 × 2개 강도 × 4개 지표
"""
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
MODELS = [
    ("qwen25_7b",   "Qwen2.5\n7B",   "2024"),
    ("llama31_8b",  "Llama 3.1\n8B", "2024"),
    ("mistral7b",   "Mistral\n7B",   "2024"),
    ("qwen3_8b",    "Qwen3\n8B",     "2025"),
]
BASE = "/root/multilingual-hate-detection-4/experiments"

data = {}
for key, label, year in MODELS:
    path = f"{BASE}/auto_eval_results_multi_gold_{key}.csv"
    rows = list(csv.DictReader(open(path)))
    data[key] = {
        "label": label,
        "year": year,
        "strong": [r for r in rows if r["strength"] == "strong"],
        "mild":   [r for r in rows if r["strength"] == "mild"],
        "all":    rows,
    }

def agg(rows, field):
    return sum(float(r[field]) for r in rows) / len(rows)

def guided_ratio(rows):
    return sum(1 for r in rows if r["preference"] == "guided") / len(rows)

# ── Figure 1: Guided 선호 비율 (grouped bar) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("Span-Guided vs Unguided: Toxicity Strength Breakdown\n(Gold Spans, 30 samples, λ=0.6)",
             fontsize=13, fontweight='bold', y=1.01)

model_labels = [d["label"] for _, d in data.items() if True]  # ordered
model_labels = [data[k]["label"] for k, _, _ in MODELS]
x = np.arange(len(MODELS))
width = 0.28

colors = {
    "strong_guided":   "#c0392b",
    "strong_unguided": "#e8b4b1",
    "mild_guided":     "#2980b9",
    "mild_unguided":   "#aed6f1",
}

# ── Subplot 1: Guided 비율 by strength ───────────────────────────────────────
ax = axes[0]
g_strong = [guided_ratio(data[k]["strong"]) * 100 for k, _, _ in MODELS]
g_mild   = [guided_ratio(data[k]["mild"])   * 100 for k, _, _ in MODELS]

bars1 = ax.bar(x - width/2, g_strong, width, label="Strong (Guided %)", color=colors["strong_guided"], alpha=0.85)
bars2 = ax.bar(x + width/2, g_mild,   width, label="Mild (Guided %)",   color=colors["mild_guided"],   alpha=0.85)

ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label="50% baseline")
ax.set_ylim(0, 65)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9.5)
ax.set_ylabel("Guided Preferred (%)", fontsize=10)
ax.set_title("① Guided Preference\nby Toxicity Strength", fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, color=colors["strong_guided"])
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, color=colors["mild_guided"])

# ── Subplot 2: Toxicity Reduction by strength ────────────────────────────────
ax = axes[1]
metrics = [
    ("guided_reduction",   "Strong Guided",   colors["strong_guided"],   x - width*1.5),
    ("unguided_reduction", "Strong Unguided", colors["strong_unguided"], x - width*0.5),
    ("guided_reduction",   "Mild Guided",     colors["mild_guided"],     x + width*0.5),
    ("unguided_reduction", "Mild Unguided",   colors["mild_unguided"],   x + width*1.5),
]
strength_keys = ["strong", "strong", "mild", "mild"]

for (field, lbl, color, pos), sk in zip(metrics, strength_keys):
    vals = [agg(data[k][sk], field) for k, _, _ in MODELS]
    ax.bar(pos, vals, width, label=lbl, color=color, alpha=0.85)

ax.set_ylim(0, 0.75)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9.5)
ax.set_ylabel("Mean Toxicity Reduction", fontsize=10)
ax.set_title("② Toxicity Reduction\nby Strength × Guidance", fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7.5, ncol=2)

# ── Subplot 3: BERTScore by strength ─────────────────────────────────────────
ax = axes[2]
metrics2 = [
    ("guided_bertscore",   "Strong Guided",   colors["strong_guided"],   x - width*1.5),
    ("unguided_bertscore", "Strong Unguided", colors["strong_unguided"], x - width*0.5),
    ("guided_bertscore",   "Mild Guided",     colors["mild_guided"],     x + width*0.5),
    ("unguided_bertscore", "Mild Unguided",   colors["mild_unguided"],   x + width*1.5),
]

for (field, lbl, color, pos), sk in zip(metrics2, strength_keys):
    vals = [agg(data[k][sk], field) for k, _, _ in MODELS]
    ax.bar(pos, vals, width, label=lbl, color=color, alpha=0.85)

ax.set_ylim(0.82, 0.99)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9.5)
ax.set_ylabel("Mean BERTScore (F1)", fontsize=10)
ax.set_title("③ Meaning Preservation\nby Strength × Guidance", fontsize=10.5, fontweight='bold')
ax.legend(fontsize=7.5, ncol=2)

plt.tight_layout()
out1 = f"{BASE}/strength_breakdown_3panel.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: 히트맵 — guided_ratio delta (Strong - Mild) ──────────────────
fig, ax = plt.subplots(figsize=(9, 4))

delta = []
row_labels = []
for k, label, year in MODELS:
    gs = guided_ratio(data[k]["strong"])
    gm = guided_ratio(data[k]["mild"])
    delta.append([gs * 100, gm * 100, (gs - gm) * 100])
    row_labels.append(label.replace('\n', ' '))

col_labels = ["Strong\n(Guided %)", "Mild\n(Guided %)", "Δ (Strong−Mild)"]
arr = np.array(delta)

# delta 컬럼 색상 분리: 양수=적색(strong에서 더 유효), 음수=청색(mild에서 더 유효)
cmap_main = plt.cm.Blues
im = ax.imshow(arr[:, :2], cmap=cmap_main, aspect='auto', vmin=0, vmax=60)

# delta 컬럼은 별도 색상
delta_arr = arr[:, 2:3]
vabs = max(abs(delta_arr.max()), abs(delta_arr.min()), 1)
cmap_delta = plt.cm.RdBu_r
im2 = ax.imshow(
    np.hstack([np.full((4,2), np.nan), delta_arr]),
    cmap=cmap_delta, aspect='auto', vmin=-vabs, vmax=vabs, alpha=0.0
)

# 수동으로 delta 셀 색칠
for i, d in enumerate(delta):
    val = d[2]
    color = "#c0392b" if val > 0 else "#2980b9" if val < 0 else "#cccccc"
    alpha = min(abs(val) / 20, 0.85) + 0.15
    ax.add_patch(plt.Rectangle((1.5, i - 0.5), 1, 1, color=color, alpha=alpha))

# 수치 표시
for i in range(4):
    for j in range(3):
        val = arr[i, j]
        txt = f"+{val:.1f}%" if (j == 2 and val > 0) else f"{val:.1f}%"
        weight = 'bold' if j == 2 else 'normal'
        fc = 'white' if (j == 2 and abs(arr[i,2]) > 8) else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=11, fontweight=weight, color=fc)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(col_labels, fontsize=10.5)
ax.set_yticks(range(4))
ax.set_yticklabels([f"{r} ({y})" for (r, (_, _, y)) in zip(row_labels, [(k,l,yr) for k,l,yr in MODELS])], fontsize=10.5)
ax.set_title("Span-Guided Preference Rate by Toxicity Strength\n(Red Δ = guided helps MORE on strong; Blue Δ = guided helps MORE on mild)",
             fontsize=11, fontweight='bold', pad=12)

plt.colorbar(im, ax=ax, shrink=0.6, label="Guided Preferred (%)")
plt.tight_layout()
out2 = f"{BASE}/strength_delta_heatmap.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ── 수치 요약 출력 ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY TABLE: Guided Preference & Metrics by Strength")
print("="*70)
print(f"{'Model':<14} {'Str':>6} {'G%':>6} {'U%':>6}  │  {'G-red':>7} {'U-red':>7}  │  {'G-BS':>7} {'U-BS':>7}")
print("-"*70)
for k, label, year in MODELS:
    for sk in ["strong", "mild"]:
        rows = data[k][sk]
        gr = guided_ratio(rows) * 100
        ur = 100 - gr
        g_red = agg(rows, "guided_reduction")
        u_red = agg(rows, "unguided_reduction")
        g_bs  = agg(rows, "guided_bertscore")
        u_bs  = agg(rows, "unguided_bertscore")
        l = label.replace('\n',' ')
        print(f"{l:<14} {sk:>6} {gr:>5.1f}% {ur:>5.1f}%  │  {g_red:>7.4f} {u_red:>7.4f}  │  {g_bs:>7.4f} {u_bs:>7.4f}")
    # delta
    gs = guided_ratio(data[k]["strong"]) * 100
    gm = guided_ratio(data[k]["mild"]) * 100
    print(f"{'':14} {'Δ(S-M)':>6} {gs-gm:>+5.1f}%")
    print()
