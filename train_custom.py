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
                writer.add_scalar("charts/episode_return", ep_return, global_step)
                writer.add_scalar("charts/episode_length", ep_len, global_step)
                if episode_idx % 10 == 0:
                    print(f"[{args.algo}] Step {global_step} | Ep {episode_idx} | Return {ep_return:.2f} | Len {ep_len}")
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

    env.close()
    writer.close()

if __name__ == "__main__":
    main()
