"""
Quick comparison script for analyzing experiment results
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_all_runs(log_dir, tag="charts/episode_return"):
    """Load all runs from a log directory"""
    runs_data = {}
    
    # Tags to try (in order of preference)
    if isinstance(tag, str):
        tags_to_try = [tag, "charts/episode_return", "rollout/ep_rew_mean", "eval/mean_reward"]
    else:
        tags_to_try = tag
    
    # Find all subdirectories with tensorboard event files
    for run_dir in Path(log_dir).iterdir():
        if not run_dir.is_dir():
            continue
        
        event_files = list(run_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            continue
        
        # Get algorithm name from directory
        # Handle both "algo_timestamp" and exact "algo" formats
        dir_name = run_dir.name
        if '_' in dir_name:
            algo_name = dir_name.split('_')[0]
        else:
            algo_name = dir_name  # Exact match (e.g., "PPO")
        
        # Load the event file
        event_file = str(sorted(event_files)[-1])
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            available_tags = ea.Tags().get("scalars", [])
            
            # Try each tag
            loaded = False
            for try_tag in tags_to_try:
                if try_tag in available_tags:
                    events = ea.Scalars(try_tag)
                    if len(events) > 0:
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        
                        if algo_name not in runs_data:
                            runs_data[algo_name] = []
                        runs_data[algo_name].append((steps, values, run_dir.name))
                        loaded = True
                        break
            
            if not loaded:
                print(f"Warning: No suitable tag found in {run_dir.name}")
                print(f"  Available tags: {available_tags[:5]}")
                
        except Exception as e:
            print(f"Warning: Could not load {event_file}: {e}")
            continue
    
    return runs_data


def print_statistics(runs_data):
    """Print statistics for all runs"""
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS")
    print("="*80)
    
    for algo_name, runs in sorted(runs_data.items()):
        print(f"\n{algo_name.upper()}:")
        print("-" * 60)
        
        all_final_returns = []
        all_max_returns = []
        all_mean_returns = []
        
        for i, (steps, values, run_name) in enumerate(runs):
            if len(values) < 10:
                continue
            
            final_10 = values[-10:]
            final_avg = np.mean(final_10)
            max_return = max(values)
            mean_return = np.mean(values)
            
            all_final_returns.append(final_avg)
            all_max_returns.append(max_return)
            all_mean_returns.append(mean_return)
            
            print(f"  Run {i+1} ({run_name}):")
            print(f"    Final Avg (last 10):  {final_avg:8.2f}")
            print(f"    Max Return:           {max_return:8.2f}")
            print(f"    Overall Mean:         {mean_return:8.2f}")
            print(f"    Total Episodes:       {len(values):8d}")
        
        if all_final_returns:
            print(f"\n  Summary across {len(all_final_returns)} runs:")
            print(f"    Final Avg:  {np.mean(all_final_returns):8.2f} ± {np.std(all_final_returns):6.2f}")
            print(f"    Max Return: {np.mean(all_max_returns):8.2f} ± {np.std(all_max_returns):6.2f}")
            print(f"    Mean:       {np.mean(all_mean_returns):8.2f} ± {np.std(all_mean_returns):6.2f}")


def plot_comparison(runs_data, output_file="comparison.png", smooth_window=10):
    """Generate comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All runs
    ax1 = axes[0]
    ax1.set_title("Learning Curves - All Runs", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Return")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed average
    ax2 = axes[1]
    ax2.set_title(f"Learning Curves - Smoothed (window={smooth_window})", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episode Return")
    ax2.grid(True, alpha=0.3)
    
    colors = {'ppo': '#1f77b4', 'reinforce': '#ff7f0e', 'causal': '#2ca02c'}
    
    for algo_name, runs in sorted(runs_data.items()):
        color = colors.get(algo_name.lower(), None)
        
        for i, (steps, values, run_name) in enumerate(runs):
            # Plot raw data with transparency
            ax1.plot(steps, values, alpha=0.3, color=color, linewidth=1)
            
            # Plot smoothed data
            if len(values) >= smooth_window:
                smoothed = np.convolve(values, np.ones(smooth_window)/smooth_window, mode='valid')
                smoothed_steps = steps[:len(smoothed)]
                ax2.plot(smoothed_steps, smoothed, label=f"{algo_name}", 
                        color=color, linewidth=2, alpha=0.8)
    
    # Add legend (deduplicate)
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), loc='best')
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Quick comparison of experiment results")
    parser.add_argument("log_dir", type=str, 
                       help="Path to log directory (e.g., experiments_20231124-120000)")
    parser.add_argument("--tag", type=str, default="charts/episode_return",
                       help="TensorBoard tag to plot (default: charts/episode_return)")
    parser.add_argument("--smooth-window", type=int, default=10,
                       help="Smoothing window size (default: 10)")
    parser.add_argument("--output", type=str, default="comparison.png",
                       help="Output filename for plot (default: comparison.png)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Don't generate plots, just print statistics")
    args = parser.parse_args()
    
    print(f"\nLoading data from: {args.log_dir}")
    runs_data = load_all_runs(args.log_dir, tag=args.tag)
    
    if not runs_data:
        print(f"No data found in {args.log_dir}")
        return
    
    print(f"Found {sum(len(runs) for runs in runs_data.values())} runs across {len(runs_data)} algorithms")
    
    # Print statistics
    print_statistics(runs_data)
    
    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_comparison(runs_data, output_file=args.output, smooth_window=args.smooth_window)
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

