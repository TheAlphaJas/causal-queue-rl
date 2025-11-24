import argparse
import os
import numpy as np

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from algos.causal_ppo import CausalPPO
from envs.tandem_queue_env import TandemQueueEnv
from utils.seeding import set_seed


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging episode returns to TensorBoard
    in the same format as the custom algorithms, plus regret tracking
    """
    def __init__(self, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.current_lengths = []
        
        # Regret tracking
        self.optimal_episode_reward = -200.0
        self.cumulative_regret = 0.0
        self.episode_count = 0
        
        # Sample complexity tracking
        self.milestones = {-600: None, -500: None, -400: None, -350: None, -300: None, -250: None}
        self.recent_returns = []
        self.log_dir = log_dir
        
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                # Accumulate rewards
                if len(self.current_rewards) <= i:
                    self.current_rewards.append(0)
                    self.current_lengths.append(0)
                    
                self.current_rewards[i] += self.locals["rewards"][i]
                self.current_lengths[i] += 1
                
                if done:
                    ep_return = self.current_rewards[i]
                    
                    # Update regret
                    instantaneous_regret = self.optimal_episode_reward - ep_return
                    self.cumulative_regret += instantaneous_regret
                    self.episode_count += 1
                    
                    # Update rolling window for milestones
                    self.recent_returns.append(ep_return)
                    if len(self.recent_returns) > 10:
                        self.recent_returns.pop(0)
                    
                    # Check milestones
                    if len(self.recent_returns) >= 10:
                        avg_recent = sum(self.recent_returns) / len(self.recent_returns)
                        for threshold in sorted(self.milestones.keys(), reverse=True):
                            if self.milestones[threshold] is None and avg_recent >= threshold:
                                self.milestones[threshold] = self.num_timesteps
                                print(f"[Causal-PPO] MILESTONE: Avg return >= {threshold} at step {self.num_timesteps}")
                                self.logger.record("milestones/steps_to_threshold", self.num_timesteps)
                    
                    # Log episode return
                    self.logger.record("charts/episode_return", ep_return)
                    self.logger.record("charts/episode_length", self.current_lengths[i])
                    
                    # Log regret
                    self.logger.record("regret/instantaneous", instantaneous_regret)
                    self.logger.record("regret/cumulative", self.cumulative_regret)
                    self.logger.record("regret/average", self.cumulative_regret / self.episode_count)
                    
                    # Reset for next episode
                    self.current_rewards[i] = 0
                    self.current_lengths[i] = 0
        
        return True
    
    def _on_training_end(self) -> None:
        """Save final metrics"""
        if self.log_dir:
            import json
            metrics = {
                "algorithm": "Causal-PPO",
                "total_steps": self.num_timesteps,
                "total_episodes": self.episode_count,
                "cumulative_regret": float(self.cumulative_regret),
                "average_regret": float(self.cumulative_regret / max(self.episode_count, 1)),
                "milestones": {k: v for k, v in self.milestones.items() if v is not None}
            }
            
            with open(f"{self.log_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\n[Causal-PPO] Final Metrics:")
            print(f"  Cumulative Regret: {self.cumulative_regret:.2f}")
            print(f"  Average Regret: {self.cumulative_regret / max(self.episode_count, 1):.2f}")
            print(f"  Milestones reached: {sum(1 for v in self.milestones.values() if v is not None)}/{len(self.milestones)}")


def make_env(seed: int = 0):
    def _init():
        env = TandemQueueEnv(seed=seed)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs_causal_ppo")
    parser.add_argument("--cf-weight", type=float, default=0.5,
                       help="Weight for counterfactual baseline (0=pure PPO, 1=pure CF)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # Create environment
    env = DummyVecEnv([make_env(args.seed)])
    
    # Get environment parameters
    base_env = env.envs[0].env
    w0 = base_env.w0
    w1 = base_env.w1
    max_queue = base_env.max_queue

    # Create log dir + logger with TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "CausalPPO")
    new_logger = configure(log_path, ["tensorboard", "stdout"])

    # Create Causal-PPO model
    model = CausalPPO(
        "MlpPolicy",
        env,
        cf_weight=args.cf_weight,
        w0=w0,
        w1=w1,
        max_queue=max_queue,
        verbose=1,
        tensorboard_log=args.log_dir,
        device=args.device,
        n_steps=200,
        batch_size=50,  # Divisor of 200 to avoid warnings
    )
    model.set_logger(new_logger)

    # Custom callback for consistent logging
    tensorboard_callback = TensorboardCallback(log_dir=log_path)

    # Train the model
    model.learn(
        total_timesteps=args.total_steps,
        tb_log_name="CausalPPO_TandemQueue",
        callback=tensorboard_callback,
    )

    model.save(os.path.join(args.log_dir, "causal_ppo_tandemqueue_final"))

    env.close()

if __name__ == "__main__":
    main()

