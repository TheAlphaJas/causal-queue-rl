# Causal Reinforcement Learning in Tandem Queue Systems
## Technical Documentation for Presentation

---

## 1. The Environment: Tandem Queue System

### Overview
The tandem queue environment simulates a two-node queueing network where arrivals must be routed and servers must be scheduled. This is a **continuing task** with a structural causal model (SCM) formulation.

### State Space
**State**: \( s_t = (Q_0, Q_1) \in \mathbb{Z}_+^2 \)
- \( Q_0 \): Queue length at node 0
- \( Q_1 \): Queue length at node 1
- Both bounded: \( 0 \leq Q_i \leq Q_{\max} \) (default: 50)

### Action Space
**8 discrete actions** encoding three decisions:
- \( a \in \{0, 1, 2, 3, 4, 5, 6, 7\} \)

Decoded as:
```
route_A = (a // 4) % 2        # Route arrival A to node 0 or 1
route_B = (a // 2) % 2        # Route arrival B to node 0 or 1  
serve_node = a % 2            # Serve node 0 or 1
```

### Exogenous Variables (Noise)
The **exogenous variables** \( U_t = (A_A, A_B, F_0, F_1) \) represent external randomness:

1. **\( A_A \sim \text{Bernoulli}(\lambda_A) \)**: Arrival at source A (default: λ_A = 0.3)
2. **\( A_B \sim \text{Bernoulli}(\lambda_B) \)**: Arrival at source B (default: λ_B = 0.3)
3. **\( F_0 \sim \text{Bernoulli}(\mu_0) \)**: Service completion at node 0 (default: μ_0 = 0.9)
4. **\( F_1 \sim \text{Bernoulli}(\mu_1) \)**: Service completion at node 1 (default: μ_1 = 0.9)

These are **independent** of the agent's policy and represent environmental stochasticity.

### Endogenous Variables
The **endogenous variables** are determined by the agent's actions and the exogenous noise:

**Next State**: \( s_{t+1} = f(s_t, a_t, U_t) \)

**Transition Function** (Structural Causal Model):

```python
def transition(state, action, noise):
    Q0, Q1 = state
    A_A, A_B, F_0, F_1 = noise
    route_A, route_B, serve_node = decode_action(action)
    
    # 1. Service (endogenous - depends on action)
    if serve_node == 0 and Q0 > 0 and F_0 == 1:
        Q0 -= 1
    elif serve_node == 1 and Q1 > 0 and F_1 == 1:
        Q1 -= 1
    
    # 2. Arrivals and routing (endogenous - depends on action)
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
    
    # 3. Bounds
    Q0 = min(Q0, max_queue)
    Q1 = min(Q1, max_queue)
    
    return (Q0, Q1)
```

### Reward Function
**Cost-based reward** (negative queue lengths):
```
r_t = -(w_0 · Q_0 + w_1 · Q_1)
```

Where:
- \( w_0, w_1 \): Weights for each queue (default: 1.0 each)
- **Goal**: Minimize queue lengths → Maximize reward

### Key Properties

1. **Markovian**: Next state depends only on current state, action, and noise
2. **Continuing Task**: Episodes truncated after 200 steps (max_episode_steps=200)
3. **Stochastic**: Randomness from arrivals and service completions
4. **Controllable**: Agent influences routing and scheduling
5. **SCM Structure**: Clear separation of exogenous (U) and endogenous variables

---

## 2. Counterfactual Computation

### What are Counterfactuals?

**Counterfactuals** answer the question: *"What would have happened if I had taken a different action, given the same environmental randomness?"*

In our setting:
- **Factual**: What actually happened with action \( a_t \) and noise \( U_t \)
- **Counterfactual**: What would happen with action \( a' \neq a_t \) but **same** noise \( U_t \)

### Why Use Counterfactuals?

Traditional RL baselines (like value functions) estimate:
```
V(s) = E_π[G_t | s_t = s]
```

This averages over:
1. Future actions from policy π
2. **All possible noise realizations**

**Counterfactual baseline** conditions on the observed noise:
```
b_cf(s, U) = E_π[r(s, a, U) | s, U]
           = Σ_a π(a|s) · r(s, a, U)
```

**Variance reduction**: By conditioning on U, we remove environmental stochasticity from the baseline!

### How to Compute Counterfactuals

**Algorithm**: Given state \( s_t \), action \( a_t \), and noise \( U_t \):

```python
def compute_counterfactual_baseline(s_t, U_t, policy):
    # 1. Get policy probabilities for current state
    π = policy(s_t)  # π(a|s) for all actions a
    
    # 2. Simulate ALL possible actions with SAME noise
    counterfactual_rewards = []
    for a in all_actions:
        s_next, r = transition(s_t, a, U_t)  # Use same U_t!
        counterfactual_rewards.append(r)
    
    # 3. Compute weighted average baseline
    baseline = Σ_a π(a|s_t) · r(s_t, a, U_t)
    
    return baseline
```

**Key insight**: We use the **stored noise** \( U_t \) from the actual environment step to evaluate all counterfactual actions.

### Mathematical Formulation

Given trajectory \( \tau = (s_0, a_0, r_0, U_0, s_1, a_1, r_1, U_1, ...) \):

For each timestep \( t \):

1. **Counterfactual rewards** for all actions:
   ```
   r_cf(a) = R(s_t, a, U_t)  for all a ∈ A
   ```

2. **Policy probabilities**:
   ```
   π(a|s_t)  for all a ∈ A
   ```

3. **Counterfactual baseline**:
   ```
   b_cf(s_t, U_t) = Σ_a π(a|s_t) · r_cf(a)
   ```

4. **Causal advantage**:
   ```
   A_causal = r_t - b_cf(s_t, U_t)
   ```

This advantage has **lower variance** than standard \( r_t - V(s_t) \) because it conditions on the realized noise.

### Implementation Details

```python
# From utils/counterfactual.py
def compute_counterfactual_rewards(state, noise, policy, w0, w1, max_queue):
    # Get policy probabilities
    state_tensor = torch.tensor(state).unsqueeze(0)
    logits = policy(state_tensor)
    probs = torch.softmax(logits, dim=-1).numpy()
    
    # Compute counterfactual rewards for all 8 actions
    r_cf = []
    for action in range(8):
        next_state, reward = TandemQueueEnv.transition(
            state, action, noise,  # Same noise for all actions!
            w0=w0, w1=w1, max_queue=max_queue
        )
        r_cf.append(reward)
    
    r_cf = np.array(r_cf)
    
    # Compute baseline: E_π[r | s, U]
    baseline = (probs * r_cf).sum()
    
    return r_cf, probs, baseline
```

---

## 3. The Four Algorithms

### 3.1 REINFORCE (Vanilla Policy Gradient)

**Type**: On-policy policy gradient with learned value baseline

**Policy**: \( \pi_\theta(a|s) \) - Categorical distribution over 8 actions
```
π_θ(a|s) = softmax(MLP_θ(s))
```

**Value Function**: \( V_\phi(s) \) - Scalar estimate of state value
```
V_φ(s) = MLP_φ(s)
```

**Objective**: Maximize expected return
```
J(θ) = E_τ~π_θ [Σ_t γ^t r_t]
```

**Policy Gradient**:
```
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) · A_t]
```

**Advantage Estimation**:
```
A_t = G_t - V_φ(s_t)
```

Where:
- \( G_t = \sum_{k=0}^∞ γ^k r_{t+k} \) (Monte Carlo return)
- \( V_\phi(s_t) \) is the learned baseline

**Update Equations**:

1. **Policy Loss**:
   ```
   L_π = -E_t [log π_θ(a_t|s_t) · (G_t - V_φ(s_t))]
   ```

2. **Value Loss**:
   ```
   L_V = E_t [(V_φ(s_t) - G_t)²]
   ```

3. **Total Loss**:
   ```
   L = L_π + 0.5 · L_V
   ```

**Pros**: 
- Simple and stable
- Value function reduces variance

**Cons**: 
- High variance (doesn't exploit causal structure)
- Slow convergence

---

### 3.2 Causal-REINFORCE

**Type**: Policy gradient with counterfactual baseline

**Policy**: \( \pi_\theta(a|s) \) - Same as REINFORCE
```
π_θ(a|s) = softmax(MLP_θ(s))
```

**No value function** - baseline comes from counterfactuals!

**Objective**: Same as REINFORCE
```
J(θ) = E_τ~π_θ [Σ_t γ^t r_t]
```

**Causal Policy Gradient**:
```
∇_θ J(θ) = E_τ [Σ_t ∇_θ log π_θ(a_t|s_t) · A_causal,t]
```

**Causal Advantage**:
```
A_causal,t = r_t - b_cf(s_t, U_t)
```

Where:
```
b_cf(s_t, U_t) = Σ_a π_θ(a|s_t) · R(s_t, a, U_t)
```

**Update Equations**:

1. **Compute counterfactual baseline** for each step:
   ```
   For each (s_t, a_t, r_t, U_t):
       b_t = Σ_a π_θ(a|s_t) · R(s_t, a, U_t)
   ```

2. **Policy Loss**:
   ```
   L = -E_t [log π_θ(a_t|s_t) · (r_t - b_t)]
   ```

**Key Difference from REINFORCE**:
- **REINFORCE**: Uses learned value function \( V_\phi(s) \) that averages over all noise
- **Causal-REINFORCE**: Uses counterfactual baseline \( b_{cf}(s, U) \) conditioned on observed noise

**Pros**: 
- Lower variance than vanilla REINFORCE
- Exploits causal structure of environment

**Cons**: 
- Still higher variance than actor-critic methods
- Requires noise storage and counterfactual computation

---

### 3.3 PPO (Proximal Policy Optimization)

**Type**: On-policy actor-critic with clipped objective

**Policy**: \( \pi_\theta(a|s) \) - Categorical distribution
```
π_θ(a|s) = softmax(MLP_θ(s))
```

**Value Function**: \( V_\phi(s) \) - State value estimate
```
V_φ(s) = MLP_φ(s)
```

**Objective**: Clipped surrogate objective
```
J(θ) = E_t [min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

Where:
- \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \) (probability ratio)
- \( \epsilon \) = 0.2 (clipping parameter)

**Advantage Estimation** (using GAE):
```
A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

Where:
- \( \delta_t = r_t + γV_\phi(s_{t+1}) - V_\phi(s_t) \) (TD error)
- \( \lambda \) = 0.95 (GAE parameter)

**Update Equations**:

1. **Collect rollout**: \( n_{steps} = 200 \) steps

2. **Compute advantages** using GAE

3. **Multiple epochs** (K=10) of mini-batch updates:
   
   **Policy Loss**:
   ```
   L_π = -E [min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t)]
   ```
   
   **Value Loss**:
   ```
   L_V = E [(V_φ(s_t) - G_t)²]
   ```
   
   **Entropy Bonus**:
   ```
   H = E [Σ_a π_θ(a|s) log π_θ(a|s)]
   ```
   
   **Total Loss**:
   ```
   L = L_π + c_1 · L_V - c_2 · H
   ```
   
   Where: \( c_1 = 0.5, c_2 = 0.0 \)

**Pros**: 
- State-of-the-art performance
- Sample efficient
- Stable training (clipping prevents large updates)

**Cons**: 
- Doesn't exploit causal structure
- Value function averages over all noise

---

### 3.4 Causal-PPO (Novel Contribution)

**Type**: PPO with counterfactual-augmented advantages

**Policy**: \( \pi_\theta(a|s) \) - Same as PPO
```
π_θ(a|s) = softmax(MLP_θ(s))
```

**Dual Baseline System**:
1. **Value Function**: \( V_\phi(s) \) - Long-term state value
2. **Counterfactual Baseline**: \( b_{cf}(s, U) \) - Conditioned on noise

**Objective**: PPO objective with causal advantages
```
J(θ) = E_t [min(r_t(θ) · A_causal,t, clip(r_t(θ), 1-ε, 1+ε) · A_causal,t)]
```

**Causal Advantage Combination**:
```
A_causal = (1 - α) · A_standard + α · A_cf
```

Where:
- \( A_{standard} = G_t - V_\phi(s_t) \) (standard PPO advantage)
- \( A_{cf} = G_t - b_{cf}(s_t, U_t) \) (counterfactual advantage)
- \( \alpha \in [0,1] \) is the counterfactual weight (default: 0.5)

**Update Equations**:

1. **Collect rollout** with noise storage: \( n_{steps} = 200 \)

2. **Compute standard GAE advantages**:
   ```
   A_standard,t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
   ```

3. **Compute counterfactual baselines**:
   ```
   For each (s_t, U_t):
       b_cf,t = Σ_a π_θ(a|s_t) · R(s_t, a, U_t)
   ```

4. **Compute counterfactual advantages**:
   ```
   A_cf,t = G_t - b_cf,t
   ```

5. **Combine advantages**:
   ```
   A_t = (1 - α) · A_standard,t + α · A_cf,t
   ```

6. **Normalize advantages**:
   ```
   Â_t = (A_t - mean(A)) / (std(A) + 1e-8)
   ```

7. **Standard PPO update** with combined advantages:
   
   **Policy Loss**:
   ```
   L_π = -E [min(r_t(θ) · Â_t, clip(r_t(θ), 1-ε, 1+ε) · Â_t)]
   ```
   
   **Value Loss** (unchanged):
   ```
   L_V = E [(V_φ(s_t) - G_t)²]
   ```
   
   **Total Loss**:
   ```
   L = L_π + 0.5 · L_V
   ```

**Hyperparameters**:
- \( \alpha = 0.5 \) (counterfactual weight)
- \( n_{steps} = 200 \)
- \( batch\_size = 50 \)
- \( n_{epochs} = 10 \)
- \( \gamma = 0.99 \)
- \( \lambda = 0.95 \)
- \( \epsilon = 0.2 \)

**Key Innovation**: Combines two complementary sources of information:
1. **Value function** \( V_\phi(s) \): Captures long-term value, averaged over all noise
2. **Counterfactual baseline** \( b_{cf}(s, U) \): Captures immediate effect, conditioned on observed noise

**Pros**: 
- **Best performance** among all 4 algorithms
- Lower variance than PPO (counterfactual conditioning)
- Stable training (inherits PPO's clipping)
- Sample efficient (exploits causal structure)

**Cons**: 
- More complex implementation
- Requires noise storage and counterfactual computation
- Additional hyperparameter (α)

---

## 4. Performance Comparison

### Experimental Results (10,000 steps)

| Algorithm | Avg Regret/Episode | Cumulative Regret | Steps to -250 |
|-----------|-------------------|-------------------|---------------|
| **Causal-PPO** | **118.12** ✓ | **5,906** ✓ | **5,800** ✓ |
| PPO | 130.50 | 6,525 | 6,800 |
| Causal-REINFORCE | 327.72 | 16,386 | Never |
| REINFORCE | 397.34 | 19,867 | Never |

### Key Findings

1. **Causal-PPO outperforms all baselines**:
   - 9.5% better than PPO
   - 64% better than Causal-REINFORCE
   - 70% better than REINFORCE

2. **Sample Complexity**: Causal-PPO reaches near-optimal performance (-250) fastest

3. **Variance Reduction**: Counterfactual conditioning significantly reduces gradient variance

4. **Scalability**: All PPO-based methods scale better than REINFORCE variants

---

## 5. Mathematical Summary

### Policy Gradient Theorem

All algorithms optimize:
```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) · Ψ_t]
```

Where \( \Psi_t \) differs by algorithm:

| Algorithm | \( \Psi_t \) (Advantage) | Baseline Type |
|-----------|------------------------|---------------|
| REINFORCE | \( G_t - V_\phi(s_t) \) | Learned value function |
| Causal-REINFORCE | \( r_t - b_{cf}(s_t, U_t) \) | Counterfactual (conditioned) |
| PPO | \( \hat{A}_t^{GAE} \) with clipping | GAE with value function |
| Causal-PPO | \( (1-\alpha)\hat{A}_t^{GAE} + \alpha(G_t - b_{cf}) \) | Hybrid: Value + Counterfactual |

### Variance Analysis

**Variance decomposition**:
```
Var[∇J] = Var[Ψ_t]
```

For standard baseline:
```
Var[G_t - V(s_t)] = E_U[Var_{A,U'}[G_t - V(s_t)]]
```

For counterfactual baseline:
```
Var[r_t - b_cf(s_t, U_t)] = E_U[Var_A[r_t - b_cf(s_t, U_t) | U_t]]
```

**Key insight**: Conditioning on \( U_t \) removes variance from environmental randomness!

---

## 6. Implementation Architecture

### Training Loop (All Algorithms)

```python
for episode in range(num_episodes):
    # Rollout phase
    for step in range(n_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Store transition (including noise for causal methods)
        buffer.add(state, action, reward, done, log_prob, info["noise"])
        
        state = next_state
        
        if done:
            state = env.reset()
    
    # Update phase
    stats = agent.update(buffer)
    
    # Logging
    log_metrics(stats)
```

### Network Architectures

**Policy Network** (all algorithms):
```
MLP: [obs_dim] → [128] → [128] → [n_actions]
Activation: ReLU
Output: Logits (categorical distribution)
```

**Value Network** (REINFORCE, PPO, Causal-PPO):
```
MLP: [obs_dim] → [128] → [128] → [1]
Activation: ReLU
Output: Scalar value estimate
```

---

## 7. Key Takeaways

1. **Causal Structure Matters**: Exploiting the SCM structure through counterfactuals leads to significant performance gains

2. **Complementary Baselines**: Combining value functions (long-term) with counterfactual baselines (short-term, conditioned) gives best results

3. **Actor-Critic Dominance**: PPO-based methods outperform REINFORCE variants due to better credit assignment and sample efficiency

4. **Novel Contribution**: Causal-PPO successfully integrates counterfactual reasoning into modern RL algorithms

---

## References

- **REINFORCE**: Williams (1992) - "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
- **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **Structural Causal Models**: Pearl (2009) - "Causality: Models, Reasoning and Inference"
- **Counterfactual RL**: Buesing et al. (2018) - "Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search"

---

**Implementation**: All code available in this repository
**Framework**: PyTorch + Gymnasium + Stable-Baselines3
**Environment**: Custom tandem queue with SCM structure

