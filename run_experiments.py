"""
Parallel Experiment Runner for RL Algorithms
Trains PPO, REINFORCE, and Causal-REINFORCE in parallel and generates comparison plots.
"""

import os
import time
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def run_ppo(total_steps, seed, log_dir, device):
    """Run PPO training using stable-baselines3"""
    print(f"[PPO] Starting training with {total_steps} steps...")
    cmd = [
        "python", "train_sb3.py",
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--log-dir", str(log_dir),
        "--device", device
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[PPO] Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[PPO] Training failed with error:\n{e.stderr}")
        return False


def run_reinforce(total_steps, seed, log_dir, device):
    """Run vanilla REINFORCE training"""
    print(f"[REINFORCE] Starting training with {total_steps} steps...")
    cmd = [
        "python", "train_custom.py",
        "--algo", "reinforce",
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--log-dir", str(log_dir),
        "--device", device
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[REINFORCE] Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[REINFORCE] Training failed with error:\n{e.stderr}")
        return False


def run_causal_reinforce(total_steps, seed, log_dir, device):
    """Run Causal-REINFORCE training"""
    print(f"[Causal-REINFORCE] Starting training with {total_steps} steps...")
    cmd = [
        "python", "train_custom.py",
        "--algo", "causal",
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--log-dir", str(log_dir),
        "--device", device
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Causal-REINFORCE] Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Causal-REINFORCE] Training failed with error:\n{e.stderr}")
        return False


def run_causal_ppo(total_steps, seed, log_dir, device, cf_weight=0.5):
    """Run Causal-PPO training"""
    print(f"[Causal-PPO] Starting training with {total_steps} steps (cf_weight={cf_weight})...")
    cmd = [
        "python", "train_causal_ppo.py",
        "--total-steps", str(total_steps),
        "--seed", str(seed),
        "--log-dir", str(log_dir),
        "--cf-weight", str(cf_weight),
        "--device", device
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[Causal-PPO] Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Causal-PPO] Training failed with error:\n{e.stderr}")
        return False


def run_algorithm(algo_name, total_steps, seed, log_dir, device, cf_weight=0.5):
    """Wrapper function to run an algorithm"""
    if algo_name == "ppo":
        return run_ppo(total_steps, seed, log_dir, device)
    elif algo_name == "reinforce":
        return run_reinforce(total_steps, seed, log_dir, device)
    elif algo_name == "causal":
        return run_causal_reinforce(total_steps, seed, log_dir, device)
    elif algo_name == "causal-ppo":
        return run_causal_ppo(total_steps, seed, log_dir, device, cf_weight)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def load_tensorboard_data(log_dir, tags=None):
    """Load scalar data from TensorBoard event files
    
    Args:
        log_dir: Directory containing event files
        tags: List of tags to try (in order of preference), or single tag string
    
    Returns:
        (steps, values) tuple or (None, None) if loading fails
    """
    # Default tags to try
    if tags is None:
        tags = ["charts/episode_return", "rollout/ep_rew_mean", "eval/mean_reward"]
    elif isinstance(tags, str):
        tags = [tags]
    
    event_files = list(Path(log_dir).rglob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return None, None
    
    # Use the most recent event file
    event_file = str(sorted(event_files)[-1])
    
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        available_tags = ea.Tags().get("scalars", [])
        
        # Try each tag in order
        for tag in tags:
            if tag in available_tags:
                events = ea.Scalars(tag)
                if len(events) > 0:
                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    print(f"[OK] Loaded {len(events)} data points from '{tag}' in {log_dir.name if hasattr(log_dir, 'name') else log_dir}")
                    return steps, values
        
        print(f"Warning: None of the tags {tags} found in {event_file}")
        print(f"Available tags: {available_tags[:10]}")  # Show first 10 tags
        return None, None
        
    except Exception as e:
        print(f"Error loading {event_file}: {e}")
        return None, None


def smooth_curve(values, window=10):
    """Apply moving average smoothing to a curve"""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    return smoothed


def plot_comparison(base_log_dir, output_file="experiment_comparison.png"):
    """Generate comparison plots for all algorithms"""
    print("\nGenerating comparison plots...")
    
    algorithms = {
        "PPO": "PPO",
        "REINFORCE": "reinforce",
        "Causal-REINFORCE": "causal",
        "Causal-PPO": "CausalPPO"
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RL Algorithms Comparison on Tandem Queue Environment", fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Returns (Raw)
    ax1 = axes[0, 0]
    ax1.set_title("Episode Returns (Raw)")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Return")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Returns (Smoothed)
    ax2 = axes[0, 1]
    ax2.set_title("Episode Returns (Smoothed, window=10)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episode Return")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Progress (normalized)
    ax3 = axes[1, 0]
    ax3.set_title("Normalized Learning Progress")
    ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Normalized Return")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title("Performance Statistics", fontweight='bold', pad=20)
    
    stats_data = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    all_data = {}
    for i, (display_name, algo_key) in enumerate(algorithms.items()):
        # Find the most recent run directory for this algorithm
        # Try exact match first (for PPO), then pattern match
        exact_dir = Path(base_log_dir) / algo_key
        if exact_dir.exists() and exact_dir.is_dir():
            runs = [exact_dir]
        else:
            pattern = f"{algo_key}_*"
            runs = sorted(Path(base_log_dir).glob(pattern))
        
        if not runs:
            print(f"Warning: No runs found for {display_name}")
            continue
        
        log_dir = runs[-1]  # Use most recent run
        steps, values = load_tensorboard_data(log_dir)
        
        if steps is None or values is None:
            print(f"Warning: Could not load data for {display_name}")
            continue
        
        all_data[display_name] = (steps, values)
        
        # Plot raw data
        ax1.plot(steps, values, label=display_name, alpha=0.4, color=colors[i])
        
        # Plot smoothed data
        if len(values) >= 10:
            smoothed_vals = smooth_curve(values, window=10)
            smoothed_steps = steps[:len(smoothed_vals)]
            ax2.plot(smoothed_steps, smoothed_vals, label=display_name, 
                    linewidth=2, color=colors[i])
        
        # Plot normalized data
        if len(values) > 0:
            min_val, max_val = min(values), max(values)
            if max_val - min_val > 0:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
                ax3.plot(steps, normalized, label=display_name, 
                        linewidth=2, color=colors[i])
        
        # Calculate statistics
        final_returns = values[-10:] if len(values) >= 10 else values
        stats_data.append([
            display_name,
            f"{np.mean(values):.2f}",
            f"{np.mean(final_returns):.2f}",
            f"{max(values):.2f}",
            f"{min(values):.2f}",
            f"{np.std(values):.2f}"
        ])
    
    # Add legends
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    
    # Create statistics table
    if stats_data:
        table_headers = ['Algorithm', 'Mean Return', 'Final Avg', 'Max', 'Min', 'Std Dev']
        table = ax4.table(
            cellText=stats_data,
            colLabels=table_headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the header
        for i in range(len(table_headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(stats_data) + 1):
            for j in range(len(table_headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    output_path = Path(base_log_dir) / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to: {output_path}")
    
    # Also save individual algorithm plots
    for display_name, (steps, values) in all_data.items():
        fig_individual = plt.figure(figsize=(10, 6))
        plt.plot(steps, values, alpha=0.3, label='Raw')
        if len(values) >= 10:
            smoothed_vals = smooth_curve(values, window=10)
            smoothed_steps = steps[:len(smoothed_vals)]
            plt.plot(smoothed_steps, smoothed_vals, linewidth=2, label='Smoothed')
        plt.xlabel('Environment Steps')
        plt.ylabel('Episode Return')
        plt.title(f'{display_name} - Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        individual_path = Path(base_log_dir) / f"{display_name.lower().replace('-', '_')}_plot.png"
        plt.savefig(individual_path, dpi=200, bbox_inches='tight')
        plt.close(fig_individual)
        print(f"[OK] Individual plot saved: {individual_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run parallel RL experiments")
    parser.add_argument("--total-steps", type=int, default=15000,
                      help="Total training steps per algorithm (default: 15000)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use: cpu or cuda (default: cpu)")
    parser.add_argument("--sequential", action="store_true",
                      help="Run algorithms sequentially instead of in parallel")
    parser.add_argument("--algorithms", nargs='+', 
                      choices=['ppo', 'reinforce', 'causal', 'causal-ppo', 'all'],
                      default=['all'],
                      help="Which algorithms to run (default: all)")
    parser.add_argument("--cf-weight", type=float, default=0.5,
                      help="Counterfactual weight for Causal-PPO (0=pure PPO, 1=pure CF, default: 0.5)")
    args = parser.parse_args()
    
    # Determine which algorithms to run
    if 'all' in args.algorithms:
        algorithms_to_run = ['ppo', 'reinforce', 'causal', 'causal-ppo']
    else:
        algorithms_to_run = args.algorithms
    
    # Create timestamped experiment directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_log_dir = f"experiments_{timestamp}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    print("="*80)
    print(f"Starting Parallel RL Experiments")
    print("="*80)
    print(f"Algorithms: {', '.join(algorithms_to_run)}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Log directory: {base_log_dir}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print("="*80)
    print()
    
    start_time = time.time()
    
    if args.sequential:
        # Run sequentially
        results = {}
        for algo in algorithms_to_run:
            print(f"\n{'='*60}")
            print(f"Training {algo.upper()}")
            print(f"{'='*60}")
            results[algo] = run_algorithm(algo, args.total_steps, args.seed, 
                                         base_log_dir, args.device, args.cf_weight)
    else:
        # Run in parallel using multiprocessing
        print("Starting parallel training processes...\n")
        
        with mp.Pool(processes=len(algorithms_to_run)) as pool:
            # Create async tasks
            tasks = []
            for algo in algorithms_to_run:
                task = pool.apply_async(
                    run_algorithm,
                    args=(algo, args.total_steps, args.seed, base_log_dir, args.device, args.cf_weight)
                )
                tasks.append((algo, task))
            
            # Wait for all tasks to complete
            results = {}
            for algo, task in tasks:
                try:
                    results[algo] = task.get(timeout=3600)  # 1 hour timeout
                except Exception as e:
                    print(f"[{algo.upper()}] Failed with exception: {e}")
                    results[algo] = False
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print("\nResults:")
    for algo, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {algo.upper():20s} {status}")
    print("="*80)
    
    # Generate comparison plots
    if any(results.values()):
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        try:
            plot_path = plot_comparison(base_log_dir)
            print(f"\n[OK] All plots generated successfully!")
            print(f"[OK] Results saved in: {base_log_dir}/")
        except Exception as e:
            print(f"[ERROR] Error generating plots: {e}")
        
        # Run regret analysis
        print("\n" + "="*80)
        print("ANALYZING REGRET & SAMPLE COMPLEXITY")
        print("="*80)
        
        try:
            import subprocess
            result = subprocess.run(
                ["python", "analyze_regret.py", str(base_log_dir)],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                # Print only the summary part
                output_lines = result.stdout.split('\n')
                print_section = False
                for line in output_lines:
                    if "DETAILED REGRET" in line or print_section:
                        print_section = True
                        print(line)
            else:
                print("[WARNING] Regret analysis failed")
        except Exception as e:
            print(f"[WARNING] Could not run regret analysis: {e}")
    else:
        print("\n[ERROR] No successful runs to plot")
    
    print("\n" + "="*80)
    print(f"[OK] Experiment completed! Check {base_log_dir}/ for results")
    print(f"[OK] View live training with: tensorboard --logdir {base_log_dir}")
    print(f"[OK] Regret analysis: {base_log_dir}/regret_analysis.png")
    print("="*80)


if __name__ == "__main__":
    main()

