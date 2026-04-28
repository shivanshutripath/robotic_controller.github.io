#!/usr/bin/env python3
"""
plot_metrics.py - Performance metrics visualization for code generation models

Metrics computed:
1. Success Rate (SR): Fraction of attempted iterations that achieve full-suite pass
2. Cumulative Success (CS): Fraction of runs that have succeeded by iteration k

Mathematical definitions:
- success_{r,k} = I{failed_{r,k} = 0}
- τ_r = min{k: success_{r,k} = 1} (stopping time)
- SR = Σ I{failed_{r,k}=0} / |V|  where V is set of valid iterations
- CS(k) = (1/R) Σ I{τ_r ≤ k}
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Raw experiment data extracted from logs
# Format: (run_id, model, iteration, failed_tests)
RAW_DATA = [
    # GPT-4o runs
    ("codegen_4o_1", "gpt-4o", 1, 26),
    ("codegen_4o_1", "gpt-4o", 2, 4),
    ("codegen_4o_1", "gpt-4o", 3, 26),
    ("codegen_4o_1", "gpt-4o", 4, 26),
    ("codegen_4o_1", "gpt-4o", 5, 28),
    ("codegen_4o_1", "gpt-4o", 6, 5),
    ("codegen_4o_1", "gpt-4o", 7, 0),  # SUCCESS
    
    ("codegen_4o_2", "gpt-4o", 1, 3),
    ("codegen_4o_2", "gpt-4o", 2, 4),
    ("codegen_4o_2", "gpt-4o", 3, 4),
    ("codegen_4o_2", "gpt-4o", 4, 0),  # SUCCESS
    
    ("codegen_4o_3", "gpt-4o", 1, 3),
    ("codegen_4o_3", "gpt-4o", 2, 0),  # SUCCESS
    
    ("codegen_4o_4", "gpt-4o", 1, 8),
    ("codegen_4o_4", "gpt-4o", 2, 8),
    ("codegen_4o_4", "gpt-4o", 3, 28),
    ("codegen_4o_4", "gpt-4o", 4, 11),
    ("codegen_4o_4", "gpt-4o", 5, None),  # Generation failed
    ("codegen_4o_4", "gpt-4o", 6, 8),
    ("codegen_4o_4", "gpt-4o", 7, 0),  # SUCCESS
    
    ("codegen_4o_5", "gpt-4o", 1, 4),
    ("codegen_4o_5", "gpt-4o", 2, None),  # Generation failed
    ("codegen_4o_5", "gpt-4o", 3, 0),  # SUCCESS
    
    # GPT-4.1 runs
    ("codegen_4p1_1", "gpt-4.1", 1, 1),
    ("codegen_4p1_1", "gpt-4.1", 2, 0),  # SUCCESS
    
    ("codegen_4p1_2", "gpt-4.1", 1, 0),  # SUCCESS
    
    ("codegen_4p1_3", "gpt-4.1", 1, 1),
    ("codegen_4p1_3", "gpt-4.1", 2, 0),  # SUCCESS
    
    ("codegen_4p1_4", "gpt-4.1", 1, 1),
    ("codegen_4p1_4", "gpt-4.1", 2, 0),  # SUCCESS
    
    ("codegen_4p1_5", "gpt-4.1", 1, 0),  # SUCCESS
    
    # GPT-5.2 runs
    ("codegen_5p2_1", "gpt-5.2", 1, 0),  # SUCCESS
    ("codegen_5p2_2", "gpt-5.2", 1, 0),  # SUCCESS
    ("codegen_5p2_3", "gpt-5.2", 1, 0),  # SUCCESS
    ("codegen_5p2_4", "gpt-5.2", 1, 0),  # SUCCESS
    ("codegen_5p2_5", "gpt-5.2", 1, 0),  # SUCCESS
]

# Model display configuration
MODEL_COLORS = {
    "gpt-4o": "#1f77b4",
    "gpt-4.1": "#ff7f0e", 
    "gpt-5.2": "#2ca02c"
}

MODEL_MARKERS = {
    "gpt-4o": "o",
    "gpt-4.1": "s",
    "gpt-5.2": "^"
}

MODEL_ORDER = ["gpt-4o", "gpt-4.1", "gpt-5.2"]


def compute_metrics(data):
    """
    Compute Success Rate (SR) and Cumulative Success (CS) for each model.
    
    Returns:
        sr_by_model: dict mapping model -> success rate
        cs_by_model: dict mapping model -> list of CS values for k=1..K
        stopping_times: dict mapping model -> list of τ_r values
    """
    # Organize data by model and run
    runs_by_model = defaultdict(lambda: defaultdict(list))
    
    for run_id, model, iteration, failed in data:
        runs_by_model[model][run_id].append((iteration, failed))
    
    sr_by_model = {}
    cs_by_model = {}
    stopping_times_by_model = {}
    stats_by_model = {}
    
    for model in MODEL_ORDER:
        if model not in runs_by_model:
            continue
            
        runs = runs_by_model[model]
        R = len(runs)  # Number of runs
        
        # Compute stopping times τ_r for each run
        stopping_times = []
        valid_iterations = []  # (run_id, k, failed) for valid iterations
        
        for run_id, iterations in runs.items():
            iterations.sort(key=lambda x: x[0])
            tau_r = float('inf')
            
            for k, failed in iterations:
                if failed is not None:  # Valid iteration
                    valid_iterations.append((run_id, k, failed))
                    if failed == 0 and tau_r == float('inf'):
                        tau_r = k
            
            stopping_times.append(tau_r)
        
        # Success Rate: fraction of valid iterations with 0 failures
        num_successes = sum(1 for _, _, failed in valid_iterations if failed == 0)
        sr = num_successes / len(valid_iterations) if valid_iterations else 0
        sr_by_model[model] = sr
        
        # Cumulative Success: CS(k) = fraction of runs with τ_r ≤ k
        K_max = max(k for _, k, _ in valid_iterations) if valid_iterations else 1
        cs_values = []
        for k in range(1, K_max + 1):
            n_succ_k = sum(1 for tau in stopping_times if tau <= k)
            cs_k = n_succ_k / R
            cs_values.append(cs_k)
        
        cs_by_model[model] = cs_values
        stopping_times_by_model[model] = stopping_times
        
        # Additional stats
        finite_taus = [t for t in stopping_times if t != float('inf')]
        stats_by_model[model] = {
            'num_runs': R,
            'num_valid_iters': len(valid_iterations),
            'num_successes': num_successes,
            'mean_tau': np.mean(finite_taus) if finite_taus else float('inf'),
            'median_tau': np.median(finite_taus) if finite_taus else float('inf'),
            'completion_rate': len(finite_taus) / R
        }
    
    return sr_by_model, cs_by_model, stopping_times_by_model, stats_by_model


def plot_success_rate(sr_by_model, ax=None):
    """Plot Success Rate bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    models = [m for m in MODEL_ORDER if m in sr_by_model]
    rates = [sr_by_model[m] * 100 for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    
    bars = ax.bar(models, rates, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Success Rate (SR)\nFraction of iterations achieving full-suite pass', fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    return ax


def plot_cumulative_success(cs_by_model, ax=None):
    """Plot Cumulative Success curves."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    max_k = max(len(cs) for cs in cs_by_model.values())
    
    for model in MODEL_ORDER:
        if model not in cs_by_model:
            continue
        cs_values = cs_by_model[model]
        k_values = list(range(1, len(cs_values) + 1))
        
        # Extend to max_k if needed (values stay at final CS)
        if len(cs_values) < max_k:
            cs_values = cs_values + [cs_values[-1]] * (max_k - len(cs_values))
            k_values = list(range(1, max_k + 1))
        
        ax.plot(k_values, [v * 100 for v in cs_values], 
                marker=MODEL_MARKERS[model], 
                color=MODEL_COLORS[model],
                label=model, 
                linewidth=2.5, 
                markersize=8)
    
    ax.set_xlabel('Iteration (k)', fontsize=12)
    ax.set_ylabel('Cumulative Success (%)', fontsize=12)
    ax.set_title('Cumulative Success CS(k)\nFraction of runs succeeded by iteration k', fontsize=13)
    ax.set_xlim(0.5, max_k + 0.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(1, max_k + 1))
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_stopping_time_distribution(stopping_times_by_model, ax=None):
    """Plot distribution of stopping times (box plot)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    data_to_plot = []
    labels = []
    colors = []
    
    for model in MODEL_ORDER:
        if model not in stopping_times_by_model:
            continue
        taus = [t for t in stopping_times_by_model[model] if t != float('inf')]
        if taus:
            data_to_plot.append(taus)
            labels.append(model)
            colors.append(MODEL_COLORS[model])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points
    for i, (taus, model) in enumerate(zip(data_to_plot, labels)):
        x = np.random.normal(i + 1, 0.04, size=len(taus))
        ax.scatter(x, taus, alpha=0.6, color=MODEL_COLORS[model], 
                   edgecolor='black', s=50, zorder=3)
    
    ax.set_ylabel('Stopping Time (τ)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Distribution of Stopping Times\nIterations needed to achieve success', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    
    return ax


def plot_iterations_summary(stats_by_model, ax=None):
    """Plot summary statistics table."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.axis('off')
    
    # Prepare table data
    columns = ['Model', 'Runs (R)', 'Valid Iters', 'SR (%)', 'Mean τ', 'Median τ', 'Completion']
    rows = []
    
    for model in MODEL_ORDER:
        if model not in stats_by_model:
            continue
        s = stats_by_model[model]
        sr = s['num_successes'] / s['num_valid_iters'] * 100 if s['num_valid_iters'] > 0 else 0
        rows.append([
            model,
            s['num_runs'],
            s['num_valid_iters'],
            f"{sr:.1f}",
            f"{s['mean_tau']:.2f}" if s['mean_tau'] != float('inf') else "∞",
            f"{s['median_tau']:.1f}" if s['median_tau'] != float('inf') else "∞",
            f"{s['completion_rate']*100:.0f}%"
        ])
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color model column
    for i, model in enumerate([r[0] for r in rows]):
        table[(i+1, 0)].set_facecolor(MODEL_COLORS.get(model, 'white'))
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    return ax


def create_full_report(data):
    """Generate complete metrics report with all plots."""
    
    # Compute all metrics
    sr_by_model, cs_by_model, stopping_times, stats = compute_metrics(data)
    
    # Print summary to console
    print("=" * 60)
    print("MODEL COMPARISON METRICS REPORT")
    print("=" * 60)
    
    for model in MODEL_ORDER:
        if model not in stats:
            continue
        s = stats[model]
        sr = sr_by_model[model]
        print(f"\n{model}:")
        print(f"  Runs: {s['num_runs']}")
        print(f"  Valid iterations: {s['num_valid_iters']}")
        print(f"  Success Rate: {sr*100:.1f}%")
        print(f"  Mean stopping time: {s['mean_tau']:.2f}")
        print(f"  Completion rate: {s['completion_rate']*100:.0f}%")
    
    print("\n" + "=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    plot_success_rate(sr_by_model, ax1)
    plot_cumulative_success(cs_by_model, ax2)
    plot_stopping_time_distribution(stopping_times, ax3)
    plot_iterations_summary(stats, ax4)
    
    plt.suptitle('Code Generation Model Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('metrics_report.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('metrics_report.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("\nPlots saved to: metrics_report.png, metrics_report.pdf")
    
    return fig


if __name__ == "__main__":
    fig = create_full_report(RAW_DATA)
    plt.show()