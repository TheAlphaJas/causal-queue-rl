"""
Analyze and compare regret and sample complexity across algorithms
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_metrics_json(log_dir):
    """Load metrics.json if it exists"""
    metrics_file = log_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def load_tensorboard_regret(log_dir):
    """Load regret data from TensorBoard logs"""
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None, None
    
    event_file = event_files[0]
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    data = {}
    
    # Load regret metrics
    for tag in ['regret/cumulative', 'regret/average', 'regret/instantaneous']:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (steps, values)
    
    return data


def find_algorithm_dirs(experiment_dir):
    """Find all algorithm directories in experiment folder"""
    exp_path = Path(experiment_dir)
    
    algorithm_map = {
        "PPO": ["PPO"],
        "REINFORCE": ["reinforce_*"],
        "Causal-REINFORCE": ["causal_*"],
        "Causal-PPO": ["CausalPPO"]
    }
    
    found = {}
    for algo_name, patterns in algorithm_map.items():
        for pattern in patterns:
            matches = list(exp_path.glob(pattern))
            if matches:
                found[algo_name] = matches[0]
                break
    
    return found


def plot_regret_comparison(experiment_dir):
    """Generate comprehensive regret and sample complexity plots"""
    algo_dirs = find_algorithm_dirs(experiment_dir)
    
    if not algo_dirs:
        print("No algorithm directories found!")
        return
    
    print(f"\nFound {len(algo_dirs)} algorithms:")
    for name, path in algo_dirs.items():
        print(f"  - {name}: {path.name}")
    
    # Load all data
    algo_data = {}
    for algo_name, log_dir in algo_dirs.items():
        metrics_json = load_metrics_json(log_dir)
        regret_data = load_tensorboard_regret(log_dir)
        
        if metrics_json or regret_data:
            algo_data[algo_name] = {
                'metrics': metrics_json,
                'regret': regret_data
            }
    
    if not algo_data:
        print("No data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Regret and Sample Complexity Analysis', fontsize=16, fontweight='bold')
    
    # Color scheme
    colors = {
        'PPO': '#1f77b4',
        'REINFORCE': '#ff7f0e',
        'Causal-REINFORCE': '#2ca02c',
        'Causal-PPO': '#d62728'
    }
    
    # Plot 1: Cumulative Regret over Time
    ax1 = axes[0, 0]
    for algo_name, data in algo_data.items():
        if data['regret'] and 'regret/cumulative' in data['regret']:
            steps, values = data['regret']['regret/cumulative']
            ax1.plot(steps, values, label=algo_name, color=colors.get(algo_name, 'gray'), linewidth=2)
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('Cumulative Regret (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Regret over Time
    ax2 = axes[0, 1]
    for algo_name, data in algo_data.items():
        if data['regret'] and 'regret/average' in data['regret']:
            steps, values = data['regret']['regret/average']
            ax2.plot(steps, values, label=algo_name, color=colors.get(algo_name, 'gray'), linewidth=2)
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Average Regret per Episode', fontsize=12)
    ax2.set_title('Average Regret (Lower is Better)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample Complexity (Steps to Milestones)
    ax3 = axes[1, 0]
    milestone_data = {}
    for algo_name, data in algo_data.items():
        if data['metrics'] and 'milestones' in data['metrics']:
            milestone_data[algo_name] = data['metrics']['milestones']
    
    if milestone_data:
        # Get all thresholds
        all_thresholds = set()
        for milestones in milestone_data.values():
            all_thresholds.update(int(k) for k in milestones.keys())
        
        thresholds = sorted(all_thresholds)
        x_pos = np.arange(len(thresholds))
        width = 0.2
        
        for i, (algo_name, milestones) in enumerate(milestone_data.items()):
            steps = [milestones.get(str(t), None) for t in thresholds]
            # Replace None with a high value for visualization
            steps = [s if s is not None else 0 for s in steps]
            
            offset = width * (i - len(milestone_data) / 2)
            bars = ax3.bar(x_pos + offset, steps, width, 
                          label=algo_name, color=colors.get(algo_name, 'gray'))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Performance Threshold', fontsize=12)
        ax3.set_ylabel('Steps to Reach', fontsize=12)
        ax3.set_title('Sample Complexity: Steps to Reach Thresholds (Lower is Better)', 
                     fontsize=13, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([str(t) for t in thresholds])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'No milestone data available', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Final Metrics Summary (Bar Chart)
    ax4 = axes[1, 1]
    final_metrics = {}
    for algo_name, data in algo_data.items():
        if data['metrics']:
            final_metrics[algo_name] = {
                'avg_regret': data['metrics'].get('average_regret', 0),
                'cum_regret': data['metrics'].get('cumulative_regret', 0)
            }
    
    if final_metrics:
        algos = list(final_metrics.keys())
        avg_regrets = [final_metrics[a]['avg_regret'] for a in algos]
        
        bars = ax4.barh(algos, avg_regrets, color=[colors.get(a, 'gray') for a in algos])
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, avg_regrets)):
            ax4.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val:.2f}',
                    va='center', fontsize=11, fontweight='bold')
        
        ax4.set_xlabel('Average Regret per Episode', fontsize=12)
        ax4.set_title('Final Average Regret (Lower is Better)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis so best (lowest regret) is on top
        ax4.invert_yaxis()
    else:
        ax4.text(0.5, 0.5, 'No final metrics available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(experiment_dir) / "regret_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Regret analysis plot saved to: {output_path}")
    
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED REGRET AND SAMPLE COMPLEXITY ANALYSIS")
    print("="*80)
    
    for algo_name, data in sorted(algo_data.items()):
        print(f"\n{algo_name}:")
        print("-" * 60)
        
        if data['metrics']:
            m = data['metrics']
            print(f"  Total Episodes:        {m.get('total_episodes', 'N/A')}")
            print(f"  Total Steps:           {m.get('total_steps', 'N/A')}")
            print(f"  Cumulative Regret:     {m.get('cumulative_regret', 0):.2f}")
            print(f"  Average Regret/Ep:     {m.get('average_regret', 0):.2f}")
            
            if 'milestones' in m and m['milestones']:
                print(f"\n  Milestones Reached:")
                for threshold, steps in sorted(m['milestones'].items(), 
                                              key=lambda x: float(x[0]), reverse=True):
                    print(f"    Avg Return >= {threshold}: {steps} steps")
            else:
                print(f"\n  Milestones: None reached")
    
    print("\n" + "="*80)
    print("RANKING (Lower regret = Better)")
    print("="*80)
    
    # Rank by average regret
    rankings = []
    for algo_name, data in algo_data.items():
        if data['metrics']:
            avg_regret = data['metrics'].get('average_regret', float('inf'))
            rankings.append((algo_name, avg_regret))
    
    rankings.sort(key=lambda x: x[1])
    
    medals = ['🥇', '🥈', '🥉', '  ']
    for i, (algo_name, avg_regret) in enumerate(rankings):
        medal = medals[i] if i < len(medals) else '  '
        print(f"  {medal} {i+1}. {algo_name:20s} Avg Regret: {avg_regret:.2f}")
    
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_regret.py <experiment_directory>")
        print("\nExample:")
        print("  python analyze_regret.py experiments_20251124-141252")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Directory '{experiment_dir}' not found!")
        sys.exit(1)
    
    print(f"\nAnalyzing regret and sample complexity for: {experiment_dir}")
    plot_regret_comparison(experiment_dir)


if __name__ == "__main__":
    main()

