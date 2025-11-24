# Experiment Runner Guide

This guide explains how to use the parallel experiment runner to compare RL algorithms.

## Quick Start

### 1. Run All Experiments (Parallel)

Run all three algorithms (PPO, REINFORCE, Causal-REINFORCE) in parallel:

```bash
python run_experiments.py --total-steps 15000
```

This will:
- Train all 3 algorithms simultaneously
- Save logs to a timestamped directory (`experiments_YYYYMMDD-HHMMSS/`)
- Generate comparison plots automatically
- Print performance statistics

### 2. Customize Your Experiment

```bash
python run_experiments.py --total-steps 10000 --seed 42 --device cuda
```

**Available options:**
- `--total-steps`: Number of training steps per algorithm (default: 15000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device to use - `cpu` or `cuda` (default: cpu)
- `--sequential`: Run algorithms one after another instead of parallel
- `--algorithms`: Choose specific algorithms to run (e.g., `--algorithms ppo causal`)

### 3. Run Specific Algorithms

```bash
# Only train PPO and Causal-REINFORCE
python run_experiments.py --algorithms ppo causal --total-steps 20000

# Only train REINFORCE
python run_experiments.py --algorithms reinforce --total-steps 10000
```

### 4. Sequential Training (for debugging)

```bash
python run_experiments.py --sequential --total-steps 5000
```

## Analyzing Results

### View Live Training

While experiments are running, monitor progress with TensorBoard:

```bash
tensorboard --logdir experiments_YYYYMMDD-HHMMSS
```

Then open http://localhost:6006 in your browser.

### Quick Analysis of Existing Results

Use the quick comparison script to analyze any experiment directory:

```bash
python quick_compare.py experiments_20231124-120000
```

This will:
- Print detailed statistics for all runs
- Generate comparison plots
- Show performance metrics

**Options:**
```bash
python quick_compare.py experiments_20231124-120000 \
    --smooth-window 20 \
    --output my_comparison.png \
    --tag "charts/episode_return"
```

### Original Analyze Script

You can also use the original analyze.py:

```bash
python analyze.py --log-dir experiments_20231124-120000
```

## Output Files

After running experiments, you'll find:

```
experiments_YYYYMMDD-HHMMSS/
├── PPO_YYYYMMDD-HHMMSS/           # PPO logs
│   └── events.out.tfevents.*
├── reinforce_YYYYMMDD-HHMMSS/     # REINFORCE logs
│   └── events.out.tfevents.*
├── causal_YYYYMMDD-HHMMSS/        # Causal-REINFORCE logs
│   └── events.out.tfevents.*
├── experiment_comparison.png       # Main comparison plot
├── ppo_plot.png                   # Individual PPO plot
├── reinforce_plot.png             # Individual REINFORCE plot
└── causal_reinforce_plot.png      # Individual Causal-REINFORCE plot
```

## Example Workflows

### Quick Test (5k steps)
```bash
python run_experiments.py --total-steps 5000 --seed 0
```

### Standard Comparison (15k steps)
```bash
python run_experiments.py --total-steps 15000 --seed 42
```

### Extended Training (50k steps)
```bash
python run_experiments.py --total-steps 50000 --seed 123 --device cuda
```

### Multiple Seeds Comparison
```bash
# Run with different seeds
for seed in 0 1 2 3 4; do
    python run_experiments.py --total-steps 10000 --seed $seed
done

# Then compare all results
python quick_compare.py experiments_*/
```

## Understanding the Plots

The main comparison plot includes:

1. **Episode Returns (Raw)**: Shows actual returns with high variance
2. **Episode Returns (Smoothed)**: Moving average for clearer trends
3. **Normalized Learning Progress**: All algorithms on same scale (0-1)
4. **Performance Statistics Table**: Key metrics for each algorithm

### Key Metrics

- **Mean Return**: Average across all episodes
- **Final Avg**: Average of last 10 episodes (convergence performance)
- **Max**: Best episode return achieved
- **Min**: Worst episode return
- **Std Dev**: Standard deviation (variance indicator)

## Tips

1. **Parallel vs Sequential**: Use parallel (default) for speed, sequential for debugging
2. **Memory**: Running all algorithms in parallel uses ~3x memory
3. **Time**: With 15k steps, expect 10-30 minutes total (parallel)
4. **Reproducibility**: Use the same seed for fair comparison
5. **GPU**: Use `--device cuda` if available for faster training

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Out of Memory
Run sequentially or reduce `--update-steps` in train_custom.py:
```bash
python run_experiments.py --sequential --total-steps 10000
```

### No Plots Generated
Check that matplotlib and tensorboard are installed:
```bash
pip install matplotlib tensorboard
```

## Advanced Usage

### Custom Analysis

You can also write custom analysis scripts using the utilities:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_data(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars("charts/episode_return")
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# Your custom analysis here...
```

### Comparing Specific Metrics

```bash
# Compare training losses
python quick_compare.py experiments_20231124-120000 --tag "losses/loss"

# Compare episode lengths
python quick_compare.py experiments_20231124-120000 --tag "charts/episode_length"
```

## Questions?

- Check the tensorboard logs: `tensorboard --logdir experiments_*/`
- View the source code: `run_experiments.py` and `quick_compare.py`
- Read the algorithm implementations: `algos/` directory

