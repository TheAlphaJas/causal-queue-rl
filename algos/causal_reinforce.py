import torch
from torch.distributions import Categorical
from agents.networks import MLPPolicy
from utils.counterfactual import compute_counterfactual_rewards

class CausalREINFORCEAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99, w0=1.0, w1=1.0, max_queue=50, device="cpu"):
        self.device = device
        self.policy = MLPPolicy(obs_dim, n_actions).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.w0 = w0
        self.w1 = w1
        self.max_queue = max_queue
        self.n_actions = n_actions

    def get_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def update(self, buffer):
        n = len(buffer)
        obs_np = buffer.obs[:n]
        actions = torch.tensor(buffer.actions[:n], dtype=torch.int64, device=self.device)
        rewards = buffer.rewards[:n]
        noises = buffer.noises[:n]

        # Compute discounted returns (for reference, but core is per-step baseline)
        returns = []
        G = 0.0
        for r, done in zip(reversed(rewards), reversed(buffer.dones[:n])):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Compute causal baselines per step
        baselines = []
        for s, noise in zip(obs_np, noises):
            _, _, b = compute_counterfactual_rewards(
                s, noise, self.policy,
                w0=self.w0, w1=self.w1, max_queue=self.max_queue,
                device=self.device
            )
            baselines.append(b)
        baselines_t = torch.tensor(baselines, dtype=torch.float32, device=self.device)

        # One-step causal advantage: r_t - b_cf_t  (you may also use returns - baseline)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        advantages = rewards_t - baselines_t

        loss = -(log_probs * advantages).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": float(loss.item()),
            "avg_return": float(returns_t.mean().item()),
            "avg_baseline": float(baselines_t.mean().item())
        }
