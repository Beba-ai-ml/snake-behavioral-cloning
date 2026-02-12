import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from snake_dqn_8_multi.snake_dqn_8_multi.agent import Agent  # noqa: E402
from snake_dqn_8_multi.snake_dqn_8_multi.game import SnakeGame  # noqa: E402
from snake_dqn_8_multi.snake_dqn_8_multi.train import DEFAULT_NUM_ENVS, VecSnakeEnv  # noqa: E402


ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}

BG_COLOR = (30, 30, 30)
GRID_COLOR = (60, 60, 60)
SNAKE_COLOR = (80, 160, 80)
HEAD_COLOR = (50, 200, 50)
FOOD_COLOR = (220, 50, 50)


@dataclass
class EpisodeBuffer:
    frames: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    last_score: int = 0
    steps: int = 0

    def reset(self) -> None:
        self.frames.clear()
        self.actions.clear()
        self.last_score = 0
        self.steps = 0


def _neighbor_mask(snake: np.ndarray, dx: int, dy: int) -> np.ndarray:
    neighbor = np.zeros_like(snake, dtype=bool)
    if dx == 1:
        neighbor[:, :-1] = snake[:, 1:]
    elif dx == -1:
        neighbor[:, 1:] = snake[:, :-1]
    elif dy == 1:
        neighbor[:-1, :] = snake[1:, :]
    elif dy == -1:
        neighbor[1:, :] = snake[:-1, :]
    return neighbor


def infer_head_mask(state: np.ndarray, prev_state: np.ndarray | None) -> np.ndarray | None:
    if state.ndim != 3 or state.shape[0] < 9:
        return None
    snake = state[0] > 0.5
    if prev_state is not None and prev_state.ndim == 3:
        prev_snake = prev_state[0] > 0.5
        diff = snake & ~prev_snake
        if int(diff.sum()) == 1:
            return diff

    direction = int(np.argmax(state[5:9]))
    deltas = {
        0: (0, -1),  # UP
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (1, 0),   # RIGHT
    }
    dx, dy = deltas.get(direction, (0, 0))
    if dx == 0 and dy == 0:
        return None
    neighbor_back = _neighbor_mask(snake, -dx, -dy)
    neighbor_front = _neighbor_mask(snake, dx, dy)
    candidates = snake & neighbor_back & ~neighbor_front
    if int(candidates.sum()) == 1:
        return candidates
    if candidates.any():
        head_mask = np.zeros_like(snake, dtype=bool)
        y, x = np.argwhere(candidates)[0]
        head_mask[y, x] = True
        return head_mask
    return None


def render_state(
    state: np.ndarray,
    block_size: int,
    draw_grid: bool,
    head_mask: np.ndarray | None = None,
) -> np.ndarray:
    if state.ndim != 3:
        raise ValueError("state must have shape (C, H, W)")
    _, grid_h, grid_w = state.shape
    cell = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    cell[:] = BG_COLOR
    snake_mask = state[0] > 0.5
    food_mask = state[1] > 0.5
    cell[snake_mask] = SNAKE_COLOR
    if head_mask is not None:
        cell[head_mask] = HEAD_COLOR
    cell[food_mask] = FOOD_COLOR
    img = np.repeat(np.repeat(cell, block_size, axis=0), block_size, axis=1)
    if draw_grid:
        for i in range(grid_h + 1):
            y = i * block_size
            img[y:y + 1, :, :] = GRID_COLOR
        for i in range(grid_w + 1):
            x = i * block_size
            img[:, x:x + 1, :] = GRID_COLOR
    return img


def save_episode(
    output_dir: str,
    episode_id: int,
    frames: List[np.ndarray],
    actions: List[int],
    score: int,
    steps: int,
    grid_size: int,
    save_on: str,
) -> None:
    episode_dir = os.path.join(output_dir, f"episode_{episode_id:05d}")
    frames_dir = os.path.join(episode_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_path = os.path.join(frames_dir, f"{idx:05d}.png")
        Image.fromarray(frame).save(frame_path)

    with open(os.path.join(episode_dir, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions, f)

    meta = {
        "score": score,
        "steps": steps,
        "frames": len(frames),
        "grid_size": grid_size,
        "win": True,
        "save_on": save_on,
        "action_names": ACTION_NAMES,
    }
    with open(os.path.join(episode_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def collect_single(
    agent: Agent,
    output_dir: str,
    wins_target: int,
    grid_size: int,
    block_size: int,
    save_on: str,
    draw_grid: bool,
) -> None:
    game = SnakeGame(render=False, fps=0, grid_size=grid_size)
    max_score = grid_size * grid_size - 3
    wins = 0
    episode_id = 0
    buffer = EpisodeBuffer()
    prev_state: np.ndarray | None = None
    state = game.reset()
    try:
        while wins < wins_target:
            action = agent.act(state)
            head_mask = infer_head_mask(state, prev_state)
            frame = render_state(state, block_size, draw_grid, head_mask=head_mask)
            next_state, _reward, done, score = game.step(action)
            buffer.steps += 1

            if save_on == "all" or (save_on == "eat" and score > buffer.last_score):
                buffer.frames.append(frame)
                buffer.actions.append(action)

            buffer.last_score = score

            prev_state = state

            if score >= max_score:
                episode_id += 1
                save_episode(
                    output_dir,
                    episode_id,
                    buffer.frames,
                    buffer.actions,
                    score,
                    buffer.steps,
                    grid_size,
                    save_on,
                )
                wins += 1
                print(f"[WIN {wins}/{wins_target}] saved episode {episode_id} (frames={len(buffer.frames)})")
                buffer.reset()
                state = game.reset()
                prev_state = None
                continue

            if done:
                buffer.reset()
                state = game.reset()
                prev_state = None
            else:
                state = next_state
    finally:
        game.close()


def collect_multi(
    agent: Agent,
    output_dir: str,
    wins_target: int,
    grid_size: int,
    num_envs: int,
    block_size: int,
    save_on: str,
    draw_grid: bool,
) -> None:
    vec_env = VecSnakeEnv(num_envs=num_envs, grid_size=grid_size, fps=0)
    max_score = grid_size * grid_size - 3
    wins = 0
    episode_id = 0
    buffers = [EpisodeBuffer() for _ in range(num_envs)]
    prev_states: List[np.ndarray | None] = [None for _ in range(num_envs)]
    states = vec_env.reset()
    try:
        while wins < wins_target:
            actions = agent.act_batch(states)
            frames = [
                render_state(
                    state,
                    block_size,
                    draw_grid,
                    head_mask=infer_head_mask(state, prev_states[env_id]),
                )
                for env_id, state in enumerate(states)
            ]
            next_states, _rewards, dones, scores, _deaths = vec_env.step(actions)

            for env_id in range(num_envs):
                buffers[env_id].steps += 1
                if save_on == "all" or (save_on == "eat" and scores[env_id] > buffers[env_id].last_score):
                    buffers[env_id].frames.append(frames[env_id])
                    buffers[env_id].actions.append(actions[env_id])
                buffers[env_id].last_score = scores[env_id]
                prev_states[env_id] = states[env_id]

                if scores[env_id] >= max_score:
                    episode_id += 1
                    save_episode(
                        output_dir,
                        episode_id,
                        buffers[env_id].frames,
                        buffers[env_id].actions,
                        scores[env_id],
                        buffers[env_id].steps,
                        grid_size,
                        save_on,
                    )
                    wins += 1
                    print(f"[WIN {wins}/{wins_target}] saved episode {episode_id} (frames={len(buffers[env_id].frames)})")
                    buffers[env_id].reset()
                    states[env_id] = vec_env.reset_at(env_id)
                    prev_states[env_id] = None
                    continue

                if dones[env_id]:
                    buffers[env_id].reset()
                    states[env_id] = vec_env.reset_at(env_id)
                    prev_states[env_id] = None
                else:
                    states[env_id] = next_states[env_id]
    finally:
        vec_env.close()


def parse_args() -> argparse.Namespace:
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_bc")
    parser = argparse.ArgumentParser(description="Collect behavioral cloning dataset from DQN expert.")
    parser.add_argument("--wins", type=int, default=500, help="number of winning episodes to save")
    parser.add_argument("--output-dir", type=str, default=default_output, help="output dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to .pth checkpoint")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--block-size", type=int, default=16, help="pixel size of each grid cell")
    parser.add_argument(
        "--save-on",
        type=str,
        choices=["all", "eat"],
        default="eat",
        help="all=save every step, eat=save only when apple is eaten",
    )
    parser.add_argument("--draw-grid", action="store_true", help="draw grid lines on images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    agent = Agent()
    agent.load_checkpoint(args.checkpoint)
    agent.model.eval()
    agent.target_model.eval()

    if args.num_envs <= 1:
        collect_single(
            agent=agent,
            output_dir=args.output_dir,
            wins_target=args.wins,
            grid_size=args.grid_size,
            block_size=args.block_size,
            save_on=args.save_on,
            draw_grid=args.draw_grid,
        )
    else:
        collect_multi(
            agent=agent,
            output_dir=args.output_dir,
            wins_target=args.wins,
            grid_size=args.grid_size,
            num_envs=args.num_envs,
            block_size=args.block_size,
            save_on=args.save_on,
            draw_grid=args.draw_grid,
        )


if __name__ == "__main__":
    main()
