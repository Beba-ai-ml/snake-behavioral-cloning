#!/usr/bin/env python3
"""Generate a gameplay GIF of the trained BC agent playing Snake."""

import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# We need the SnakeGame from the DQN module included in this repo
sys.path.insert(0, str(REPO_ROOT))
# The BC train_and_play.py imports from snake_dqn_8_multi.snake_dqn_8_multi.game
# but we can also use the local snake_dqn_88 or snake_dqn_8_multi copies
from snake_dqn_8_multi.snake_dqn_8_multi.game import SnakeGame

# Import BC model and helpers from train_and_play
sys.path.insert(0, str(REPO_ROOT / "snake_bc_rgb"))
from train_and_play import CNNPolicy, render_state, infer_head_mask, preprocess_frame

# --- Config ---
GRID_SIZE = 8
MODEL_PATH = str(REPO_ROOT / "pretrained" / "best.pt")
OUTPUT_PATH = str(REPO_ROOT / "gameplay.gif")
MAX_EPISODES = 100  # try many episodes since BC has ~67% win rate
CELL_SIZE = 30  # pixels per cell for display GIF
OBS_BLOCK_SIZE = 16  # block size for observation frames (matching training)
DRAW_GRID = True  # draw grid lines in observation (matching training)
FPS = 64
FRAME_DURATION = 1000 // FPS  # ms per frame (~16ms)

# Colors (matching train_and_play.py)
BG_COLOR = (30, 30, 30)
GRID_LINE_COLOR = (50, 50, 50)
HEAD_COLOR = (50, 200, 50)
BODY_COLOR = (80, 160, 80)
FOOD_COLOR = (220, 50, 50)


def render_display_frame(game: SnakeGame) -> Image.Image:
    """Render the game state as a PIL image for the GIF display."""
    size = GRID_SIZE * CELL_SIZE
    img = Image.new("RGB", (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        pos = i * CELL_SIZE
        draw.line([(pos, 0), (pos, size - 1)], fill=GRID_LINE_COLOR, width=1)
        draw.line([(0, pos), (size - 1, pos)], fill=GRID_LINE_COLOR, width=1)

    # Draw snake body (skip head)
    for idx, (x, y) in enumerate(game.snake):
        if idx == 0:
            continue
        x0, y0 = x * CELL_SIZE + 1, y * CELL_SIZE + 1
        x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
        draw.rectangle([x0, y0, x1, y1], fill=BODY_COLOR)

    # Draw head
    hx, hy = game.snake[0]
    x0, y0 = hx * CELL_SIZE + 1, hy * CELL_SIZE + 1
    x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
    draw.rectangle([x0, y0, x1, y1], fill=HEAD_COLOR)

    # Draw food
    if game.food:
        fx, fy = game.food
        x0, y0 = fx * CELL_SIZE + 1, fy * CELL_SIZE + 1
        x1, y1 = x0 + CELL_SIZE - 2, y0 + CELL_SIZE - 2
        draw.rectangle([x0, y0, x1, y1], fill=FOOD_COLOR)

    return img


def select_action_bc(model, state, prev_state, device):
    """Select action using the BC model (from RGB rendered observation)."""
    head_mask = infer_head_mask(state, prev_state)
    # Render state to RGB observation (same as during BC training)
    frame = render_state(state, OBS_BLOCK_SIZE, DRAW_GRID, head_mask=head_mask)
    # Preprocess: normalize and convert to tensor
    model.eval()
    with torch.no_grad():
        arr = frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(device)
        logits = model(tensor)
        return int(logits.argmax(dim=1).item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BC model
    model = CNNPolicy().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("BC model loaded successfully.")

    game = SnakeGame(grid_size=GRID_SIZE, render=False)
    max_score = GRID_SIZE * GRID_SIZE - 3  # win = 61 for 8x8

    best_score = 0
    best_frames = []

    for ep in range(1, MAX_EPISODES + 1):
        state = game.reset()
        prev_state = None
        frames = [render_display_frame(game)]
        done = False
        score = 0

        while not done:
            action = select_action_bc(model, state, prev_state, device)
            next_state, reward, done, score = game.step(action)
            frames.append(render_display_frame(game))
            prev_state = state
            state = next_state

        win = score >= max_score
        print(f"Episode {ep}: score={score} win={win} death={game.last_death_reason} frames={len(frames)}")

        if score > best_score:
            best_score = score
            best_frames = frames
            print(f"  -> New best! score={best_score}")

        if win:
            print(f"WIN found at episode {ep}! Score: {score}")
            best_frames = frames
            best_score = score
            break

    if not best_frames:
        print("No episodes completed. Something is wrong.")
        return

    # For very long episodes, subsample to keep GIF small
    total_frames = len(best_frames)
    if total_frames > 500:
        best_frames = best_frames[::2]
        frame_duration = FRAME_DURATION * 2
    else:
        frame_duration = FRAME_DURATION

    print(f"Saving GIF with {len(best_frames)} frames, score={best_score}...")

    # Save as GIF
    best_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=best_frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )

    import os
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"GIF saved to {OUTPUT_PATH} ({size_mb:.2f} MB, {len(best_frames)} frames)")

    if size_mb > 5:
        print("WARNING: GIF is over 5MB. Reducing...")
        reduced = best_frames[::3]
        reduced[0].save(
            OUTPUT_PATH,
            save_all=True,
            append_images=reduced[1:],
            duration=frame_duration * 3,
            loop=0,
            optimize=True,
        )
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"Reduced GIF: {size_mb:.2f} MB, {len(reduced)} frames")


if __name__ == "__main__":
    main()
