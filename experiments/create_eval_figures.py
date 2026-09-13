#!/usr/bin/env python3
"""
Create figures and tables from automated evaluation results.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_DOCS = Path(__file__).resolve().parents[1] / "docs"
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))

from colm_figure_style import (
    FIG_SINGLE_H,
    FIG_SINGLE_W,
    GUIDED,
    UNGUIDED,
    apply_colm_style,
    style_legend,
)

def load_results(json_path: str):
    """Load evaluation results from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['summary'], data['detailed_results']

def create_preference_figure(summary, output_path: str):
    """Create bar chart for preference ratios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall preference
    total = summary['total_samples']
    guided = summary['guided_preferred']
    unguided = summary['unguided_preferred']
    ties = summary['ties']
    
    ax1.bar(['Guided', 'Unguided', 'Tie'], 
            [guided, unguided, ties],
            color=['#2ecc71', '#e74c3c', '#95a5a6'])
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Overall Preference Distribution')
    ax1.set_ylim(0, total)
    
    # Add percentage labels
    for i, (label, value) in enumerate(zip(['Guided', 'Unguided', 'Tie'], 
                                           [guided, unguided, ties])):
        percentage = (value / total) * 100 if total > 0 else 0
        ax1.text(i, value + 1, f'{value}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    # By toxicity strength
    counts = summary['counts_per_strength']
    strengths = ['Strong', 'Mild']
    guided_counts = [counts['strong']['guided'], counts['mild']['guided']]
    unguided_counts = [counts['strong']['unguided'], counts['mild']['unguided']]
    tie_counts = [counts['strong']['tie'], counts['mild']['tie']]
    
    x = np.arange(len(strengths))
    width = 0.25
    
    ax2.bar(x - width, guided_counts, width, label='Guided', color='#2ecc71')
    ax2.bar(x, unguided_counts, width, label='Unguided', color='#e74c3c')
    ax2.bar(x + width, tie_counts, width, label='Tie', color='#95a5a6')
    
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Preference by Toxicity Strength')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strengths)
    ax2.legend()
    ax2.set_ylim(0, max(max(guided_counts), max(unguided_counts), max(tie_counts)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved preference figure to {output_path}")

def create_meaning_preservation_figure(summary, output_path: str):
    """Create bar chart for meaning preservation metrics."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    meaning = summary['meaning_preservation']
    guided_mean = meaning['guided']['mean']
    guided_std = meaning['guided']['std']
    unguided_mean = meaning['unguided']['mean']
    unguided_std = meaning['unguided']['std']
    
    x = np.arange(2)
    means = [guided_mean, unguided_mean]
    stds = [guided_std, unguided_std]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(['Guided', 'Unguided'], means, yerr=stds, 
                  color=colors, capsize=10, alpha=0.7)
    
    ax.set_ylabel('BERTScore (F1)')
    ax.set_title('Meaning Preservation (BERTScore)')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.02, f'{mean:.3f} ± {std:.3f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved meaning preservation figure to {output_path}")

def create_toxicity_meaning_tradeoff_figure(summary, detailed, output_path: str):
    """Create scatter plot showing toxicity-meaning trade-off."""
    apply_colm_style()
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W + 0.3, FIG_SINGLE_H))

    guided_tox = []
    guided_meaning = []
    unguided_tox = []
    unguided_meaning = []

    for result in detailed:
        guided_tox.append(result["guided_reduction"])
        guided_meaning.append(result["guided_bertscore"])
        unguided_tox.append(result["unguided_reduction"])
        unguided_meaning.append(result["unguided_bertscore"])

    ax.scatter(
        guided_tox,
        guided_meaning,
        alpha=0.72,
        s=52,
        label="Guided",
        color=GUIDED,
        edgecolors="#222222",
        linewidths=0.45,
    )
    ax.scatter(
        unguided_tox,
        unguided_meaning,
        alpha=0.72,
        s=52,
        label="Unguided",
        color=UNGUIDED,
        edgecolors="#222222",
        linewidths=0.45,
    )

    ax.set_xlabel("Toxicity reduction")
    ax.set_ylabel("Meaning preservation (BERTScore)")
    style_legend(ax, loc="lower right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.80, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved toxicity-meaning trade-off figure to {output_path}")

def create_summary_table(summary, output_path: str):
    """Create LaTeX table with summary statistics using booktabs."""
    latex = """\\begin{table}[h]
\\centering
\\caption{Automated Evaluation Results}
\\label{tab:auto_eval}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Metric} & \\textbf{Guided} & \\textbf{Unguided} \\\\
\\midrule
"""
    
    # Preference
    total = summary['total_samples']
    guided_pct = (summary['guided_preferred'] / total) * 100
    unguided_pct = (summary['unguided_preferred'] / total) * 100
    
    latex += f"\\textbf{{Preference}} & {summary['guided_preferred']} ({guided_pct:.1f}\\%) & {summary['unguided_preferred']} ({unguided_pct:.1f}\\%) \\\\\n"
    latex += "\\midrule\n"
    
    # Meaning preservation
    meaning = summary['meaning_preservation']
    guided_mean = meaning['guided']['mean']
    guided_std = meaning['guided']['std']
    unguided_mean = meaning['unguided']['mean']
    unguided_std = meaning['unguided']['std']
    
    latex += f"\\textbf{{Meaning Preservation}} & ${guided_mean:.3f} \\pm {guided_std:.3f}$ & ${unguided_mean:.3f} \\pm {unguided_std:.3f}$ \\\\\n"
    latex += "\\midrule\n"
    
    # Toxicity reduction
    toxicity = summary['toxicity_reduction']
    guided_tox_mean = toxicity['guided']['mean']
    guided_tox_std = toxicity['guided']['std']
    unguided_tox_mean = toxicity['unguided']['mean']
    unguided_tox_std = toxicity['unguided']['std']
    
    latex += f"\\textbf{{Toxicity Reduction}} & ${guided_tox_mean:.3f} \\pm {guided_tox_std:.3f}$ & ${unguided_tox_mean:.3f} \\pm {unguided_tox_std:.3f}$ \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✅ Saved LaTeX table to {output_path}")

def create_detailed_table(summary, output_path: str):
    """Create detailed table with strength breakdown."""
    latex = """\\begin{table}[h]
\\centering
\\caption{Preference by Toxicity Strength}
\\label{tab:preference_by_strength}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Toxicity Strength} & \\textbf{Guided} & \\textbf{Unguided} & \\textbf{Tie} \\\\
\\midrule
"""
    
    counts = summary['counts_per_strength']
    
    for strength in ['strong', 'mild']:
        strength_label = strength.capitalize()
        guided = counts[strength]['guided']
        unguided = counts[strength]['unguided']
        tie = counts[strength]['tie']
        total = guided + unguided + tie
        
        if total > 0:
            guided_pct = (guided / total) * 100
            unguided_pct = (unguided / total) * 100
            tie_pct = (tie / total) * 100
            
            latex += f"{strength_label} & {guided} ({guided_pct:.1f}\\%) & {unguided} ({unguided_pct:.1f}\\%) & {tie} ({tie_pct:.1f}\\%) \\\\\n"
    
    latex += "\\midrule\n"
    
    # Overall
    total = summary['total_samples']
    guided = summary['guided_preferred']
    unguided = summary['unguided_preferred']
    ties = summary['ties']
    guided_pct = (guided / total) * 100
    unguided_pct = (unguided / total) * 100
    tie_pct = (ties / total) * 100
    
    latex += f"\\textbf{{Overall}} & {guided} ({guided_pct:.1f}\\%) & {unguided} ({unguided_pct:.1f}\\%) & {ties} ({tie_pct:.1f}\\%) \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✅ Saved detailed table to {output_path}")

def create_human_eval_summary_table(human_eval_path: str, output_path: str):
    """Create human evaluation summary table."""
    with open(human_eval_path, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    
    latex = """\\begin{table}[h]
\\centering
\\caption{Human Evaluation Summary}
\\label{tab:human_eval_summary}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Toxicity Strength} & \\textbf{Guided} & \\textbf{Unguided} & \\textbf{Tie} \\\\
\\midrule
"""
    
    counts = human_data['counts_per_strength']
    
    for strength in ['strong', 'mild']:
        strength_label = strength.capitalize()
        guided = counts[strength]['guided']
        unguided = counts[strength]['unguided']
        tie = counts[strength]['tie']
        total = guided + unguided + tie
        
        if total > 0:
            guided_pct = (guided / total) * 100
            unguided_pct = (unguided / total) * 100
            tie_pct = (tie / total) * 100
            
            latex += f"{strength_label} & {guided} ({guided_pct:.1f}\\%) & {unguided} ({unguided_pct:.1f}\\%) & {tie} ({tie_pct:.1f}\\%) \\\\\n"
    
    latex += "\\midrule\n"
    
    # Overall
    total = human_data['total_samples']
    guided = human_data['guided_preferred']
    unguided = human_data['unguided_preferred']
    ties = human_data['ties']
    guided_pct = (guided / total) * 100
    unguided_pct = (unguided / total) * 100
    tie_pct = (ties / total) * 100
    
    latex += f"\\textbf{{Overall}} & {guided} ({guided_pct:.1f}\\%) & {unguided} ({unguided_pct:.1f}\\%) & {ties} ({tie_pct:.1f}\\%) \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✅ Saved human evaluation summary table to {output_path}")

def create_auto_vs_human_table(auto_summary, human_data, output_path: str):
    """Create table comparing automatic metrics vs human judgment."""
    latex = """\\begin{table}[h]
\\centering
\\caption{Automatic Metric vs Human Judgment}
\\label{tab:auto_vs_human}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Metric} & \\textbf{Guided} & \\textbf{Unguided} \\\\
\\midrule
"""
    
    # Human preference
    total = human_data['total_samples']
    human_guided = human_data['guided_preferred']
    human_unguided = human_data['unguided_preferred']
    human_guided_pct = (human_guided / total) * 100
    human_unguided_pct = (human_unguided / total) * 100
    
    latex += f"\\textbf{{Human Preference}} & {human_guided} ({human_guided_pct:.1f}\\%) & {human_unguided} ({human_unguided_pct:.1f}\\%) \\\\\n"
    latex += "\\midrule\n"
    
    # Auto preference
    auto_guided = auto_summary['guided_preferred']
    auto_unguided = auto_summary['unguided_preferred']
    auto_total = auto_summary['total_samples']
    auto_guided_pct = (auto_guided / auto_total) * 100
    auto_unguided_pct = (auto_unguided / auto_total) * 100
    
    latex += f"\\textbf{{Auto Preference}} & {auto_guided} ({auto_guided_pct:.1f}\\%) & {auto_unguided} ({auto_unguided_pct:.1f}\\%) \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"✅ Saved auto vs human table to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_json',
        type=str,
        default='/root/multilingual-hate-detection/experiments/auto_eval_results_v2.json',
    )
    parser.add_argument(
        '--human_eval_json',
        type=str,
        default='/root/multilingual-hate-detection/experiments/human_eval_summary.json',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/multilingual-hate-detection/experiments/figures',
    )
    args = parser.parse_args()
    
    # Load results
    summary, detailed = load_results(args.input_json)
    
    # Load human evaluation results
    with open(args.human_eval_json, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figures
    figure_path = str(output_dir / 'toxicity_meaning_tradeoff.png')
    create_toxicity_meaning_tradeoff_figure(summary, detailed, figure_path)
    
    # Also copy to docs/figures if it exists
    docs_figures_path = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'toxicity_meaning_tradeoff.png'
    if docs_figures_path.parent.exists():
        import shutil
        shutil.copy2(figure_path, str(docs_figures_path))
        print(f"✅ Also saved to {docs_figures_path}")
    
    # Create tables
    create_human_eval_summary_table(args.human_eval_json, str(output_dir / 'human_eval_summary.tex'))
    create_auto_vs_human_table(summary, human_data, str(output_dir / 'auto_vs_human.tex'))
    
    print("\n✅ All figures and tables created successfully!")

if __name__ == '__main__':
    main()

