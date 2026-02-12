import argparse
import csv
import json
import os
import multiprocessing as mp
from collections import deque
from multiprocessing.connection import Connection
from typing import List

import torch

from snake_dqn_8_multi.agent import Agent
from snake_dqn_8_multi.game import SnakeGame


DEFAULT_NUM_ENVS = 8
DEFAULT_BLOCK_SIZE = 60
DEFAULT_RENDER_COLS = 4


def _set_window_position(env_id: int, grid_size: int) -> None:
    cols_raw = os.environ.get("SNAKE_RENDER_COLS")
    block_raw = os.environ.get("SNAKE_RENDER_BLOCK_SIZE")

    cols = DEFAULT_RENDER_COLS
    if cols_raw:
        try:
            cols = int(cols_raw)
        except ValueError:
            cols = DEFAULT_RENDER_COLS
    if cols < 1:
        cols = DEFAULT_RENDER_COLS

    block_size = DEFAULT_BLOCK_SIZE
    if block_raw:
        try:
            block_size = int(block_raw)
        except ValueError:
            block_size = DEFAULT_BLOCK_SIZE
    if block_size < 1:
        block_size = DEFAULT_BLOCK_SIZE

    window_px = grid_size * block_size
    x = (env_id % cols) * window_px
    y = (env_id // cols) * window_px
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"


def _worker(remote: Connection, parent_remote: Connection, grid_size: int, fps: int, render_every: int, env_id: int) -> None:
    parent_remote.close()
    render_requested = render_every > 0
    if render_requested:
        _set_window_position(env_id, grid_size)
    game = SnakeGame(render=render_requested, fps=fps, grid_size=grid_size)
    episode_idx = 0
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "reset":
                episode_idx += 1
                if render_requested:
                    game.render_enabled = episode_idx % render_every == 0
                remote.send(game.reset())
            elif cmd == "step":
                state, reward, done, score = game.step(int(data))
                remote.send((state, reward, done, score, game.last_death_reason))
            elif cmd == "close":
                remote.send(None)
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    finally:
        game.close()
        remote.close()


class VecSnakeEnv:
    def __init__(self, num_envs: int, grid_size: int, fps: int, render_every: int = 0):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.remotes = list(self.remotes)
        self.work_remotes = list(self.work_remotes)
        self.ps: List[mp.Process] = []

        for env_id, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, grid_size, fps, render_every, env_id),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.ps.append(proc)

    def reset(self) -> List[tuple]:
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def reset_at(self, index: int) -> tuple:
        self.remotes[index].send(("reset", None))
        return self.remotes[index].recv()

    def step(self, actions: List[int]) -> tuple[List[tuple], List[float], List[bool], List[int], List[str]]:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, dones, scores, deaths = zip(*results)
        return list(next_states), list(rewards), list(dones), list(scores), list(deaths)

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for remote in self.remotes:
            try:
                remote.recv()
            except Exception:
                pass
        for proc in self.ps:
            proc.join(timeout=1.0)


def train(
    episodes: int = 100,
    render_every: int = 0,
    fps: int = 8,
    grid_size: int = 8,
    num_envs: int = DEFAULT_NUM_ENVS,
    gamma: float = 0.99,
    lr: float = 1e-4,
    batch_size: int = 512,
    epsilon_start: float = 0.01,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 1.0,
    save_every: int = 0,
    save_dir: str = "checkpoints_8_multi",
    load_from: str | None = None,
    session_id: int = 1,
) -> None:
    agent = Agent(
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    os.makedirs(save_dir, exist_ok=True)
    session_ckpt = os.path.join(save_dir, f"session_{session_id}.pth")
    session_meta = os.path.join(save_dir, f"session_{session_id}.json")
    session_csv = os.path.join(save_dir, f"session_{session_id}.csv")
    episodes_completed = 0

    if load_from:
        try:
            meta = agent.load_checkpoint(load_from)
            episodes_completed = int(meta.get("episodes_trained", 0))
            print(f"Loaded checkpoint from: {load_from} (episodes: {episodes_completed})")
        except Exception as exc:
            print(f"Failed to load weights ({exc}). Starting from scratch.")
    elif os.path.exists(session_ckpt):
        try:
            meta = agent.load_checkpoint(session_ckpt)
            print(f"Loaded session {session_id} from: {session_ckpt}")
            episodes_completed = int(meta.get("episodes_trained", 0))
            print(f"  -> episodes trained so far: {episodes_completed}")
            print(f"  -> epsilon at resume: {agent.epsilon:.3f}")
        except Exception as exc:
            print(f"Failed to load existing session ({exc}). Starting from scratch.")
            episodes_completed = 0

    scores: List[int] = []
    mean_window_20 = deque(maxlen=20)
    mean_window_100 = deque(maxlen=100)

    csv_needs_header = not os.path.exists(session_csv) or os.path.getsize(session_csv) == 0
    if num_envs > 1:
        vec_env = VecSnakeEnv(num_envs=num_envs, grid_size=grid_size, fps=fps, render_every=render_every)
        states = vec_env.reset()
        episodes_done = 0
        try:
            with open(session_csv, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "total_episode",
                        "episode_in_run",
                        "env_id",
                        "score",
                        "mean_20",
                        "mean_100",
                        "epsilon",
                        "loss",
                        "death_reason",
                    ],
                )
                if csv_needs_header:
                    writer.writeheader()

                while episodes_done < episodes:
                    actions = agent.act_batch(states)
                    next_states, rewards, dones, scores_step, deaths = vec_env.step(actions)

                    for env_id in range(num_envs):
                        agent.remember(
                            states[env_id],
                            actions[env_id],
                            rewards[env_id],
                            next_states[env_id],
                            dones[env_id],
                        )
                    states = next_states

                    loss = agent.replay()

                    for env_id, done in enumerate(dones):
                        if not done:
                            continue
                        episodes_done += 1
                        score = scores_step[env_id]
                        scores.append(score)
                        mean_window_20.append(score)
                        mean_window_100.append(score)

                        mean_20 = sum(mean_window_20) / len(mean_window_20)
                        mean_100 = sum(mean_window_100) / len(mean_window_100)
                        loss_display = f", loss={loss:.4f}" if loss is not None else ""
                        total_ep = episodes_completed + episodes_done
                        death = deaths[env_id] or "-"
                        agent.update_epsilon()
                        print(
                            f"[Ep {total_ep:04d}] env={env_id} score={score} mean(20)={mean_20:.2f} "
                            f"mean(100)={mean_100:.2f} epsilon={agent.epsilon:.3f}{loss_display} death={death}"
                        )

                        writer.writerow(
                            {
                                "total_episode": total_ep,
                                "episode_in_run": episodes_done,
                                "env_id": env_id,
                                "score": score,
                                "mean_20": f"{mean_20:.4f}",
                                "mean_100": f"{mean_100:.4f}",
                                "epsilon": f"{agent.epsilon:.6f}",
                                "loss": f"{loss:.6f}" if loss is not None else "",
                                "death_reason": death,
                            }
                        )
                        csv_file.flush()

                        if save_every > 0 and total_ep % save_every == 0:
                            meta = {
                                "session_id": session_id,
                                "episodes_trained": total_ep,
                                "epsilon": agent.epsilon,
                                "num_envs": num_envs,
                            }
                            agent.save_checkpoint(session_ckpt, meta=meta)
                            with open(session_meta, "w", encoding="utf-8") as f:
                                json.dump(meta, f)
                            print(f"  -> saved model: {session_ckpt} (episodes in session: {total_ep})")

                        if episodes_done >= episodes:
                            break
                        states[env_id] = vec_env.reset_at(env_id)
        finally:
            vec_env.close()
    else:
        render_requested = render_every > 0
        game = SnakeGame(render=render_requested, fps=fps, grid_size=grid_size)
        game.reset()
        try:
            with open(session_csv, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "total_episode",
                        "episode_in_run",
                        "env_id",
                        "score",
                        "mean_20",
                        "mean_100",
                        "epsilon",
                        "loss",
                        "death_reason",
                    ],
                )
                if csv_needs_header:
                    writer.writeheader()

                for episode in range(1, episodes + 1):
                    # render only every N episodes (or never)
                    if render_requested:
                        game.render_enabled = episode % render_every == 0
                    else:
                        game.render_enabled = False

                    state = game.reset()
                    done = False
                    loss = None

                    while not done:
                        action = agent.act(state)
                        next_state, reward, done, score = game.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        loss = agent.replay()
                        state = next_state

                    agent.update_epsilon()
                    scores.append(score)
                    mean_window_20.append(score)
                    mean_window_100.append(score)

                    mean_20 = sum(mean_window_20) / len(mean_window_20)
                    mean_100 = sum(mean_window_100) / len(mean_window_100)
                    loss_display = f", loss={loss:.4f}" if loss is not None else ""
                    total_ep = episodes_completed + episode
                    death = game.last_death_reason or "-"
                    print(
                        f"[Ep {total_ep:04d}] score={score} mean(20)={mean_20:.2f} mean(100)={mean_100:.2f} "
                        f"epsilon={agent.epsilon:.3f}{loss_display} death={death}"
                    )

                    writer.writerow(
                        {
                            "total_episode": total_ep,
                            "episode_in_run": episode,
                            "env_id": 0,
                            "score": score,
                            "mean_20": f"{mean_20:.4f}",
                            "mean_100": f"{mean_100:.4f}",
                            "epsilon": f"{agent.epsilon:.6f}",
                            "loss": f"{loss:.6f}" if loss is not None else "",
                            "death_reason": death,
                        }
                    )
                    csv_file.flush()

                    if save_every > 0 and episode % save_every == 0:
                        total_ep = episodes_completed + episode
                        meta = {
                            "session_id": session_id,
                            "episodes_trained": total_ep,
                            "epsilon": agent.epsilon,
                        }
                        agent.save_checkpoint(session_ckpt, meta=meta)
                        with open(session_meta, "w", encoding="utf-8") as f:
                            json.dump(meta, f)
                        print(f"  -> saved model: {session_ckpt} (episodes in session: {total_ep})")
        finally:
            game.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN CNN training (default grid 8x8)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render-every", type=int, default=0, help="0 disables, 1 shows every episode, N shows every Nth")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epsilon-start", type=float, default=0.01)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=0, help="save weights every N episodes (0=disabled)")
    parser.add_argument("--save-dir", type=str, default="checkpoints_8_multi", help="checkpoint directory")
    parser.add_argument("--load-from", type=str, default=None, help="path to .pth checkpoint to load")
    parser.add_argument("--session-id", type=int, default=1, help="session number (saved as session_<id>.pth)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes=args.episodes,
        render_every=args.render_every,
        fps=args.fps,
        num_envs=args.num_envs,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        save_every=args.save_every,
        save_dir=args.save_dir,
        load_from=args.load_from,
        session_id=args.session_id,
        grid_size=args.grid_size,
    )
