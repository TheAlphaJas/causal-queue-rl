import numpy as np
from envs.tandem_queue_env import TandemQueueEnv

def all_actions(num_actions=8):
    return list(range(num_actions))

def compute_counterfactual_rewards(state, noise, policy, w0=1.0, w1=1.0, max_queue=50, device="cpu"):
    """
    state: np.array, single state
    noise: tuple (A_A, A_B, F_0, F_1) from env.info["noise"]
    policy: PyTorch policy network; returns logits over actions.
    Returns:
      r_cf: np.array shape (num_actions,)
      pi: np.array shape (num_actions,) of policy probs at this state
      baseline: scalar sum_a pi(a|s) * r_cf(a)
    """
    import torch

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = policy(state_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    actions = all_actions(len(probs))
    r_cf = []
    for a in actions:
        next_s, r = TandemQueueEnv.transition(
            state, a, noise, w0=w0, w1=w1, max_queue=max_queue
        )
        r_cf.append(r)
    r_cf = np.array(r_cf, dtype=np.float32)
    baseline = float((probs * r_cf).sum())
    return r_cf, probs, baseline
