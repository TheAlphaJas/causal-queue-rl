import numpy as np

class RolloutBuffer:
    def __init__(self, max_steps, obs_dim):
        self.obs = np.zeros((max_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps,), dtype=np.int64)
        self.rewards = np.zeros((max_steps,), dtype=np.float32)
        self.dones = np.zeros((max_steps,), dtype=np.bool_)
        self.log_probs = np.zeros((max_steps,), dtype=np.float32)
        self.noises = [None] * max_steps
        self.ptr = 0
        self.max_steps = max_steps

    def add(self, obs, action, reward, done, log_prob, noise):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.noises[self.ptr] = noise
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr
