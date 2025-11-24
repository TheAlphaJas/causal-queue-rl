import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from algos.networks import MLPPolicy, ValueNet

class REINFORCEAgent:
    def __init__(self, obs_dim, n_actions, lr=3e-4, gamma=0.99, device="cpu"):
        self.device = device
        self.policy = MLPPolicy(obs_dim, n_actions).to(device)
        self.value = ValueNet(obs_dim).to(device)
        self.optim = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        self.gamma = gamma

    def get_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def update(self, buffer):
        # Compute returns and advantages
        n = len(buffer)
        rewards = buffer.rewards[:n]
        obs = torch.tensor(buffer.obs[:n], dtype=torch.float32, device=self.device)
        actions = torch.tensor(buffer.actions[:n], dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs[:n], dtype=torch.float32, device=self.device)

        returns = []
        G = 0.0
        for r, done in zip(reversed(rewards), reversed(buffer.dones[:n])):
            if done:
                G = 0.0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        values = self.value(obs)
        advantages = returns_t - values.detach()

        # Policy loss (with learned baseline)
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)

        loss = policy_loss + 0.5 * value_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "avg_return": float(returns_t.mean().item())
        }
