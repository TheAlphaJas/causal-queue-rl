import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, obs_dim)
        return self.net(x)  # logits

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
