import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from envs.tandem_queue_env import TandemQueueEnv
from utils.seeding import set_seed

def make_env(seed: int = 0):
    def _init():
        env = TandemQueueEnv(seed=seed)
        env = Monitor(env)  # episode stats in info, SB3-friendly
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs_sb3")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # VecEnv with a single env
    env = DummyVecEnv([make_env(args.seed)])
    eval_env = DummyVecEnv([make_env(args.seed + 1)])

    # Create log dir + SB3 logger with TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "PPO")
    new_logger = configure(log_path, ["tensorboard", "stdout"])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
    )
    model.set_logger(new_logger)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # SB3's learn will handle rollout collection, training, and TensorBoard logging
    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name="PPO_TandemQueue",
        callback=eval_callback,
    )

    model.save(os.path.join(args.log_dir, "ppo_tandemqueue_final"))

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
