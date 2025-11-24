import argparse
from envs.tandem_queue_env import TandemQueueEnv
from algos.reinforce import REINFORCEAgent
from algos.causal_reinforce import CausalREINFORCEAgent
from utils.storage import RolloutBuffer
from utils.logging_utils import create_writer
from utils.seeding import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["reinforce", "causal"], default="reinforce")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--update-steps", type=int, default=2048)
    parser.add_argument("--log-interval", type=int, default=1, 
                       help="Log every N episodes (default: 1)")
    parser.add_argument("--print-interval", type=int, default=10,
                       help="Print every N episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs_custom")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    env = TandemQueueEnv(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if args.algo == "reinforce":
        agent = REINFORCEAgent(obs_dim, n_actions, device=args.device)
    else:
        agent = CausalREINFORCEAgent(
            obs_dim, n_actions,
            device=args.device,
            w0=env.w0, w1=env.w1, max_queue=env.max_queue
        )

    writer, log_dir = create_writer(args.log_dir, args.algo)
    print(f"Logging to {log_dir}")

    buffer = RolloutBuffer(args.update_steps, obs_dim)

    obs, info = env.reset()
    ep_return = 0.0
    ep_len = 0
    global_step = 0
    episode_idx = 0
    
    # Regret tracking (optimal reward for this env is ~-200 based on near-optimal policy)
    optimal_episode_reward = -200.0  # Approximate optimal performance
    cumulative_regret = 0.0
    
    # Sample complexity tracking
    milestones = {-600: None, -500: None, -400: None, -350: None, -300: None, -250: None}
    recent_returns = []  # Rolling window for milestone detection

    while global_step < args.total_steps:
        buffer.reset()
        for _ in range(args.update_steps):
            action, log_prob = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, done, log_prob, info.get("noise"))
            ep_return += reward
            ep_len += 1
            global_step += 1

            obs = next_obs
            if done:
                # Update regret
                instantaneous_regret = optimal_episode_reward - ep_return
                cumulative_regret += instantaneous_regret
                
                # Update rolling window for milestones
                recent_returns.append(ep_return)
                if len(recent_returns) > 10:
                    recent_returns.pop(0)
                
                # Check milestones (average of last 10 episodes)
                if len(recent_returns) >= 10:
                    avg_recent = sum(recent_returns) / len(recent_returns)
                    for threshold in sorted(milestones.keys(), reverse=True):
                        if milestones[threshold] is None and avg_recent >= threshold:
                            milestones[threshold] = global_step
                            print(f"[{args.algo}] MILESTONE: Avg return >= {threshold} at step {global_step}")
                            writer.add_scalar("milestones/steps_to_threshold", global_step, threshold)
                
                # Log every episode or at specified interval
                if episode_idx % args.log_interval == 0:
                    writer.add_scalar("charts/episode_return", ep_return, global_step)
                    writer.add_scalar("charts/episode_length", ep_len, global_step)
                    writer.add_scalar("regret/instantaneous", instantaneous_regret, global_step)
                    writer.add_scalar("regret/cumulative", cumulative_regret, global_step)
                    writer.add_scalar("regret/average", cumulative_regret / (episode_idx + 1), global_step)
                
                # Print at specified interval
                if episode_idx % args.print_interval == 0:
                    print(f"[{args.algo}] Step {global_step} | Ep {episode_idx} | Return {ep_return:.2f} | "
                          f"Regret {instantaneous_regret:.2f} | CumRegret {cumulative_regret:.2f}")
                
                ep_return = 0.0
                ep_len = 0
                episode_idx += 1
                obs, info = env.reset()

            if global_step >= args.total_steps:
                break

        stats = agent.update(buffer)
        for k, v in stats.items():
            writer.add_scalar(f"losses/{k}", v, global_step)
        print(f"[{args.algo}] Update at step {global_step} | " +
              " | ".join(f"{k}: {v:.3f}" for k, v in stats.items()))

    # Save final metrics
    import json
    metrics = {
        "algorithm": args.algo,
        "total_steps": global_step,
        "total_episodes": episode_idx,
        "cumulative_regret": float(cumulative_regret),
        "average_regret": float(cumulative_regret / max(episode_idx, 1)),
        "milestones": {k: v for k, v in milestones.items() if v is not None}
    }
    
    with open(f"{log_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[{args.algo}] Final Metrics:")
    print(f"  Cumulative Regret: {cumulative_regret:.2f}")
    print(f"  Average Regret: {cumulative_regret / max(episode_idx, 1):.2f}")
    print(f"  Milestones reached: {sum(1 for v in milestones.values() if v is not None)}/{len(milestones)}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
