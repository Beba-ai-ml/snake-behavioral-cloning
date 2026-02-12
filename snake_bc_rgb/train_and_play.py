#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from snake_dqn_8_multi.snake_dqn_8_multi.game import SnakeGame  # noqa: E402


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
CHECKPOINT_NAME = "checkpoint.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_spec(dataset_dir: Path) -> tuple[int, int, bool]:
    episodes = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {dataset_dir}")
    meta = read_json(episodes[0] / "meta.json")
    grid_size = int(meta.get("grid_size", 8))
    save_on = meta.get("save_on")
    if save_on != "all":
        raise ValueError(f"Expected save_on=all, got {save_on}")

    frames = sorted((episodes[0] / "frames").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {episodes[0] / 'frames'}")
    img = Image.open(frames[0]).convert("RGB")
    width, height = img.size
    if width != height or width % grid_size != 0:
        raise ValueError("Frame size is not compatible with grid_size")
    block_size = width // grid_size
    arr = np.asarray(img)
    draw_grid = bool(np.any(np.all(arr == np.array(GRID_COLOR, dtype=arr.dtype), axis=-1)))
    return grid_size, block_size, draw_grid


def split_episodes(episodes: list[Path], seed: int, train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    episodes = episodes[:]
    rng.shuffle(episodes)
    n_total = len(episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train = episodes[:n_train]
    val = episodes[n_train:n_train + n_val]
    test = episodes[n_train + n_val:]
    return train, val, test


def build_index(episodes: list[Path]) -> tuple[list[tuple[str, int]], np.ndarray]:
    index: list[tuple[str, int]] = []
    counts = np.zeros(4, dtype=np.int64)
    for ep_dir in episodes:
        actions = read_json(ep_dir / "actions.json")
        frames = sorted((ep_dir / "frames").glob("*.png"))
        if len(frames) != len(actions):
            raise ValueError(f"Frame/action mismatch in {ep_dir}: {len(frames)} vs {len(actions)}")
        for frame_path, action in zip(frames, actions):
            action_int = int(action)
            index.append((str(frame_path), action_int))
            counts[action_int] += 1
    return index, counts


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_val: float,
    bad_epochs: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "bad_epochs": bad_epochs,
        },
        path,
    )


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, int]:
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model" in data:
        model.load_state_dict(data["model"])
        if "optimizer" in data:
            optimizer.load_state_dict(data["optimizer"])
            _move_optimizer_state(optimizer, device)
        if "scheduler" in data:
            scheduler.load_state_dict(data["scheduler"])
        start_epoch = int(data.get("epoch", 0))
        best_val = float(data.get("best_val", float("inf")))
        bad_epochs = int(data.get("bad_epochs", 0))
        return start_epoch, best_val, bad_epochs
    raise ValueError(f"Checkpoint at {path} is not in the expected format.")


def load_or_create_splits(
    run_dir: Path,
    dataset_dir: Path,
    episodes: list[Path],
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    splits_path = run_dir / "splits.json"
    if splits_path.exists():
        splits = read_json(splits_path)
        train_eps = [dataset_dir / name for name in splits.get("train", [])]
        val_eps = [dataset_dir / name for name in splits.get("val", [])]
        test_eps = [dataset_dir / name for name in splits.get("test", [])]
        missing = [p for p in train_eps + val_eps + test_eps if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing episodes referenced in {splits_path}: {missing[:3]}")
        return train_eps, val_eps, test_eps

    train_eps, val_eps, test_eps = split_episodes(episodes, seed, 0.8, 0.1)
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": [p.name for p in train_eps],
                "val": [p.name for p in val_eps],
                "test": [p.name for p in test_eps],
            },
            f,
            indent=2,
        )
    return train_eps, val_eps, test_eps


class FrameActionDataset(Dataset):
    def __init__(self, index: list[tuple[str, int]]) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        frame_path, action = self.index[idx]
        img = Image.open(frame_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = np.ascontiguousarray(arr)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, action


class CNNPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, 4)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_every: int = 0,
    label: str = "train",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)
    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * batch_y.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += batch_y.size(0)
        if is_train and log_every > 0 and batch_idx % log_every == 0:
            batch_acc = (logits.argmax(dim=1) == batch_y).float().mean().item()
            print(
                f"[{label}] batch {batch_idx}/{num_batches} loss={loss.item():.4f} acc={batch_acc:.3f}",
                flush=True,
            )
    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)


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
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
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


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    arr = frame.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).permute(2, 0, 1)


def select_action(model: nn.Module, frame: np.ndarray, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        x = preprocess_frame(frame).unsqueeze(0).to(device)
        logits = model(x)
        return int(logits.argmax(dim=1).item())


def play_live(
    model: nn.Module,
    device: torch.device,
    grid_size: int,
    obs_block_size: int,
    draw_grid: bool,
    display_block_size: int,
    fps: int,
    episodes: int,
    log_path: Path,
    render: bool,
) -> None:
    game = SnakeGame(render=render, fps=fps, grid_size=grid_size, block_size=display_block_size)
    max_score = grid_size * grid_size - 3
    scores_20 = deque(maxlen=20)
    scores_100 = deque(maxlen=100)
    wins_20 = deque(maxlen=20)
    wins_100 = deque(maxlen=100)

    state = game.reset()
    prev_state = None
    with log_path.open("w", encoding="utf-8") as log_file:
        for episode in range(1, episodes + 1):
            done = False
            steps = 0
            score = 0
            while not done:
                head_mask = infer_head_mask(state, prev_state)
                frame = render_state(state, obs_block_size, draw_grid, head_mask=head_mask)
                action = select_action(model, frame, device)
                next_state, _reward, done, score = game.step(action)
                steps += 1
                prev_state = state
                state = next_state

            win = score >= max_score
            scores_20.append(score)
            scores_100.append(score)
            wins_20.append(1 if win else 0)
            wins_100.append(1 if win else 0)
            mean20 = float(np.mean(scores_20))
            mean100 = float(np.mean(scores_100))
            win20 = float(np.mean(wins_20))
            win100 = float(np.mean(wins_100))
            log_entry = {
                "episode": episode,
                "score": score,
                "steps": steps,
                "win": win,
                "death_reason": game.last_death_reason,
                "mean_score_20": mean20,
                "mean_score_100": mean100,
                "win_rate_20": win20,
                "win_rate_100": win100,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            print(
                f"[LIVE {episode}/{episodes}] score={score} win={win} "
                f"mean20={mean20:.2f} mean100={mean100:.2f} "
                f"win20={win20:.2f} win100={win100:.2f} death={game.last_death_reason}"
            )
            state = game.reset()
            prev_state = None
    game.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral cloning on RGB Snake frames, then play live.")
    parser.add_argument("--dataset-dir", type=str, default=str(REPO_ROOT / "snake_bc_8_multi" / "dataset_bc"))
    parser.add_argument("--runs-dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--session-id", type=int, default=None, help="save outputs to runs/session_<id>")
    parser.add_argument("--overwrite", action="store_true", help="allow overwriting an existing session directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=200, help="log progress every N batches (train only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--live-episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--display-block-size", type=int, default=32)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="skip test evaluation before live play")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    resume = False
    if args.session_id is not None:
        run_dir = runs_dir / f"session_{args.session_id}"
        if run_dir.exists():
            if args.overwrite:
                shutil.rmtree(run_dir)
            else:
                resume = True
    else:
        run_dir = runs_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Starting BC training...", flush=True)
    grid_size, obs_block_size, draw_grid = infer_dataset_spec(dataset_dir)
    print(f"Dataset spec: grid_size={grid_size} block_size={obs_block_size} draw_grid={draw_grid}", flush=True)

    episodes = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    train_eps, val_eps, test_eps = load_or_create_splits(run_dir, dataset_dir, episodes, args.seed)

    train_index, train_counts = build_index(train_eps)
    val_index, _ = build_index(val_eps)
    test_index, _ = build_index(test_eps)

    class_weights = train_counts.sum() / np.maximum(train_counts, 1)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Action counts (train): {train_counts.tolist()}")

    train_ds = FrameActionDataset(train_index)
    val_ds = FrameActionDataset(val_index)
    test_ds = FrameActionDataset(test_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNPolicy().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    checkpoint_path = run_dir / CHECKPOINT_NAME
    best_path = run_dir / "best.pt"
    best_val = float("inf")
    bad_epochs = 0
    start_epoch = 0
    if resume:
        if checkpoint_path.exists():
            start_epoch, best_val, bad_epochs = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            print(f"Resuming session from epoch {start_epoch}.", flush=True)
        elif best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print("Checkpoint missing: loaded best.pt and restarting optimizer state.", flush=True)
        else:
            raise FileNotFoundError(
                f"{checkpoint_path} not found. Use --overwrite or a new --session-id."
            )

    if start_epoch >= args.epochs:
        print(f"Training already completed at epoch {start_epoch}.", flush=True)
    else:
        current_epoch = start_epoch
        try:
            for epoch in range(start_epoch + 1, args.epochs + 1):
                current_epoch = epoch
                train_loss, train_acc = run_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    log_every=args.log_every,
                    label="train",
                )
                val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
                scheduler.step(val_loss)
                print(
                    f"[Epoch {epoch:02d}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                )
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    bad_epochs = 0
                    torch.save(model.state_dict(), run_dir / "best.pt")
                else:
                    bad_epochs += 1
                    if bad_epochs >= args.patience:
                        print("Early stopping.")
                        save_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            best_val,
                            bad_epochs,
                        )
                        break
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    bad_epochs,
                )
        except KeyboardInterrupt:
            print("Interrupted: saving checkpoint...", flush=True)
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                current_epoch,
                best_val,
                bad_epochs,
            )
            print("Checkpoint saved. Resume with the same --session-id.", flush=True)
            return

    if (run_dir / "best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    if args.skip_test:
        print("Skipping test evaluation.", flush=True)
    else:
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        print(f"[Test] loss={test_loss:.4f} acc={test_acc:.3f}")

    config = {
        "dataset_dir": str(dataset_dir),
        "grid_size": grid_size,
        "obs_block_size": obs_block_size,
        "draw_grid": draw_grid,
        "normalization": "rgb/255",
        "model": "cnn_4lvl_gap_fc256_4",
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    render = not args.no_render
    log_path = run_dir / "live_results.jsonl"
    play_live(
        model=model,
        device=device,
        grid_size=grid_size,
        obs_block_size=obs_block_size,
        draw_grid=draw_grid,
        display_block_size=args.display_block_size,
        fps=args.fps,
        episodes=args.live_episodes,
        log_path=log_path,
        render=render,
    )


if __name__ == "__main__":
    main()
