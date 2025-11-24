import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TandemQueueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 lambda_A=0.3,
                 lambda_B=0.3,
                 mu_0=0.9,
                 mu_1=0.9,
                 w0=1.0,
                 w1=1.0,
                 max_queue=50,
                 max_episode_steps=200,
                 seed=None):
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.mu_0 = mu_0
        self.mu_1 = mu_1
        self.w0 = w0
        self.w1 = w1
        self.max_queue = max_queue
        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=0,
            high=self.max_queue,
            shape=(2,),
            dtype=np.int32,
        )
        # 8 actions: routeA (0/1) x routeB (0/1) x serve_node (0/1)
        self.action_space = spaces.Discrete(8)

        self.np_random = np.random.RandomState(seed)
        self.state = None
        self.time = 0

    @staticmethod
    def decode_action(a: int):
        route_A = (a // 4) % 2
        route_B = (a // 2) % 2
        serve_node = a % 2
        return route_A, route_B, serve_node

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random.seed(seed)
        self.state = np.array([0, 0], dtype=np.int32)
        self.time = 0
        info = {}
        return self.state.copy(), info

    def _sample_noise(self):
        A_A = int(self.np_random.rand() < self.lambda_A)
        A_B = int(self.np_random.rand() < self.lambda_B)
        F_0 = int(self.np_random.rand() < self.mu_0)
        F_1 = int(self.np_random.rand() < self.mu_1)
        return A_A, A_B, F_0, F_1

    @staticmethod
    def transition(state, action, noise, w0=1.0, w1=1.0, max_queue=50):
        """
        Pure SCM transition used for both env.step and counterfactuals.
        state: np.array([Q0, Q1])
        action: int in [0,7]
        noise: tuple (A_A, A_B, F_0, F_1)
        """
        Q0, Q1 = int(state[0]), int(state[1])
        A_A, A_B, F_0, F_1 = noise
        route_A, route_B, serve_node = TandemQueueEnv.decode_action(action)

        # Service first
        if serve_node == 0 and Q0 > 0 and F_0 == 1:
            Q0 -= 1
        elif serve_node == 1 and Q1 > 0 and F_1 == 1:
            Q1 -= 1

        # Arrivals and routing
        if A_A == 1:
            if route_A == 0:
                Q0 += 1
            else:
                Q1 += 1
        if A_B == 1:
            if route_B == 0:
                Q0 += 1
            else:
                Q1 += 1

        # Optional clipping for safety
        Q0 = min(Q0, max_queue)
        Q1 = min(Q1, max_queue)
        next_state = np.array([Q0, Q1], dtype=np.int32)

        # Reward: negative weighted total queue length
        reward = -(w0 * Q0 + w1 * Q1)
        return next_state, reward

    def step(self, action):
        assert self.action_space.contains(action)
        noise = self._sample_noise()
        next_state, reward = self.transition(
            self.state, action, noise,
            w0=self.w0, w1=self.w1, max_queue=self.max_queue
        )
        self.state = next_state
        self.time += 1
        terminated = False  # continuing task
        truncated = self.time >= self.max_episode_steps  # truncate after max steps
        info = {"noise": noise}
        return next_state.copy(), float(reward), terminated, truncated, info
