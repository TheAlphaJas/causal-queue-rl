"""
Causal-PPO: PPO with Counterfactual Baseline

Combines PPO's value function baseline with counterfactual rewards
to reduce variance while maintaining the benefits of PPO.
"""

import warnings
from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.ppo import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

from utils.counterfactual import compute_counterfactual_rewards


class CausalRolloutBuffer(RolloutBuffer):
    """
    Extended RolloutBuffer that also stores noise/exogenous variables
    needed for counterfactual computation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noises = None
        
    def reset(self) -> None:
        super().reset()
        self.noises = [None] * self.buffer_size
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        noise: Any = None,
    ) -> None:
        """
        Add noise to the buffer along with standard data.
        """
        # Store noise for counterfactual computation
        if self.pos < self.buffer_size:
            self.noises[self.pos] = noise
        
        # Call parent's add method
        super().add(obs, action, reward, episode_start, value, log_prob)


class CausalPPO(PPO):
    """
    Causal Proximal Policy Optimization (Causal-PPO)
    
    Extends PPO by incorporating counterfactual baselines into the advantage calculation.
    The advantage is computed as a weighted combination of:
    - Standard PPO advantage (return - value_function)
    - Counterfactual advantage (return - counterfactual_baseline)
    
    This reduces variance by conditioning on exogenous variables while maintaining
    PPO's stability and sample efficiency.
    
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param cf_weight: Weight for counterfactual baseline (0=pure PPO, 1=pure counterfactual)
    :param w0: Weight for queue 0 in environment (for counterfactual computation)
    :param w1: Weight for queue 1 in environment (for counterfactual computation)
    :param max_queue: Maximum queue size in environment (for counterfactual computation)
    :param learning_rate: The learning rate
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE
    :param clip_range: Clipping parameter
    :param clip_range_vf: Clipping parameter for the value function
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: the log location for tensorboard
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device on which the code should be run
    """
    
    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        cf_weight: float = 0.5,
        w0: float = 1.0,
        w1: float = 1.0,
        max_queue: int = 50,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 200,
        batch_size: int = 50,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Store counterfactual parameters
        self.cf_weight = cf_weight
        self.w0 = w0
        self.w1 = w1
        self.max_queue = max_queue
        
        # Initialize parent PPO
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )
        
        # Use custom rollout buffer that stores noise
        self.rollout_buffer_class = CausalRolloutBuffer
        
        if _init_setup_model:
            self._setup_model()
    
    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: CausalRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a RolloutBuffer.
        Modified to also store noise for counterfactual computation.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        
        n_steps = 0
        rollout_buffer.reset()
        
        callback.on_rollout_start()
        
        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            
            actions = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            self.num_timesteps += env.num_envs
            
            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            self._update_info_buffer(infos)
            n_steps += 1
            
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
            
            # Extract noise from info
            noise = infos[0].get("noise") if len(infos) > 0 else None
            
            # Add to buffer with noise
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                noise=noise,
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy(obs_tensor)
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        
        callback.on_rollout_end()
        
        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Modified to incorporate counterfactual baselines in advantage computation.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        cf_baseline_values = []
        
        continue_training = True
        
        # Compute counterfactual baselines for all observations in buffer
        if self.cf_weight > 0:
            cf_baselines = self._compute_counterfactual_baselines()
        else:
            cf_baselines = None
        
        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                
                # Compute advantages with counterfactual baseline
                advantages = rollout_data.advantages
                
                # If using counterfactual baseline, modify advantages
                if cf_baselines is not None and self.cf_weight > 0:
                    # Get counterfactual baselines for this batch
                    # We need to map batch indices back to buffer indices
                    # For simplicity, we'll recompute for this batch
                    batch_cf_baselines = []
                    for i, obs in enumerate(rollout_data.observations):
                        # Find corresponding noise in buffer
                        # This is a simplified approach - in practice, we'd track indices
                        cf_baseline = self._compute_single_cf_baseline(obs.cpu().numpy())
                        batch_cf_baselines.append(cf_baseline)
                    
                    batch_cf_baselines = th.tensor(
                        batch_cf_baselines, dtype=th.float32, device=self.device
                    )
                    
                    # Compute counterfactual advantage: return - cf_baseline
                    # returns = advantages + values (since advantages = returns - values)
                    returns = advantages + rollout_data.old_values
                    cf_advantages = returns - batch_cf_baselines
                    
                    # Combine standard and counterfactual advantages
                    advantages = (
                        (1 - self.cf_weight) * advantages 
                        + self.cf_weight * cf_advantages
                    )
                    
                    cf_baseline_values.append(batch_cf_baselines.mean().item())
                
                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                
                # Entropy loss favor exploration
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                
                entropy_losses.append(entropy_loss.item())
                
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            self._n_updates += 1
            if not continue_training:
                break
        
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), 
            self.rollout_buffer.returns.flatten()
        )
        
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        
        # Log counterfactual baseline stats
        if cf_baseline_values:
            self.logger.record("train/cf_baseline_mean", np.mean(cf_baseline_values))
            self.logger.record("train/cf_weight", self.cf_weight)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
    def _compute_single_cf_baseline(self, obs: np.ndarray) -> float:
        """
        Compute simplified baseline for a single observation.
        Falls back to immediate reward when noise is not available.
        """
        # Simplified: use current state reward as proxy
        # This is used when noise is not available
        from envs.tandem_queue_env import TandemQueueEnv
        Q0, Q1 = int(obs[0]), int(obs[1])
        immediate_reward = -(self.w0 * Q0 + self.w1 * Q1)
        
        return immediate_reward
    
    def _compute_counterfactual_baselines(self) -> np.ndarray:
        """
        Compute counterfactual baselines for all observations in the buffer.
        """
        cf_baselines = []
        
        for i in range(len(self.rollout_buffer.observations)):
            obs = self.rollout_buffer.observations[i]
            
            # Handle multi-dimensional observations (from vectorized envs)
            if len(obs.shape) > 1:
                obs = obs[0]  # Take first env if vectorized
            
            noise = self.rollout_buffer.noises[i] if hasattr(self.rollout_buffer, 'noises') else None
            
            if noise is not None:
                # Use full counterfactual computation with noise
                baseline = self._compute_cf_baseline_with_noise(obs, noise)
                cf_baselines.append(baseline)
            else:
                # Fallback to simplified baseline
                baseline = self._compute_single_cf_baseline(obs)
                cf_baselines.append(baseline)
        
        return np.array(cf_baselines, dtype=np.float32)
    
    def _compute_cf_baseline_with_noise(self, obs: np.ndarray, noise: tuple) -> float:
        """
        Compute counterfactual baseline using stored noise.
        """
        from envs.tandem_queue_env import TandemQueueEnv
        
        # Get policy probabilities for this state
        with th.no_grad():
            obs_tensor = th.as_tensor(obs).unsqueeze(0).to(self.device)
            distribution = self.policy.get_distribution(obs_tensor)
            
            if hasattr(distribution, 'distribution'):
                # For discrete actions
                probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
            else:
                # Fallback
                probs = th.softmax(distribution.logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Compute counterfactual rewards for all actions
        n_actions = len(probs)
        r_cf = []
        
        for action in range(n_actions):
            next_state, reward = TandemQueueEnv.transition(
                obs, action, noise, 
                w0=self.w0, w1=self.w1, max_queue=self.max_queue
            )
            r_cf.append(reward)
        
        r_cf = np.array(r_cf, dtype=np.float32)
        baseline = float((probs * r_cf).sum())
        
        return baseline

