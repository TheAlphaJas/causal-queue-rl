import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalar(path, tag="charts/episode_return"):
    ea = EventAccumulator(path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

    runs = glob.glob(os.path.join(args.log_dir, "*"))
    plt.figure()
    for run in runs:
        algo = os.path.basename(run).split("_")[0]
        # Find event files
        event_files = glob.glob(os.path.join(run, "events.*"))
        if not event_files:
            continue
        steps, values = load_scalar(event_files[0])
        if steps is None:
            continue
        plt.plot(steps, values, label=algo + " (" + os.path.basename(run) + ")")

    plt.xlabel("Environment steps")
    plt.ylabel("Episode return")
    plt.legend()
    plt.grid(True)
    plt.title("RL algorithms comparison on TandemQueueEnv")
    plt.show()

if __name__ == "__main__":
    main()
