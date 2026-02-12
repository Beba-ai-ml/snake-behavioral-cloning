# Snake Behavioral Cloning

A complete pipeline for training a Snake AI via **Behavioral Cloning (BC)**: first
train a DQN expert that wins consistently, then distill its behavior into a simpler
CNN policy that learns purely from image-action pairs.

## Overview

```
DQN Expert Training  -->  Dataset Collection  -->  BC Training  -->  Live Play
  (snake_dqn_88 /          (collect_bc.py)      (train_and_play.py)
   snake_dqn_8_multi)
```

1. **DQN Expert** -- A Double DQN agent with Dueling architecture and NoisyNet
   exploration learns to play Snake on an 8x8 grid. After sufficient training it
   wins (fills the board) virtually every game.

2. **Dataset Collection** -- The expert plays thousands of games; only winning
   episodes are saved as RGB frame sequences paired with the expert's actions.

3. **BC Training** -- A 4-layer CNN is trained via supervised learning (cross-entropy)
   on the collected image-action pairs. The model learns to predict the expert's
   action from a single RGB frame of the board state.

4. **Live Play** -- The trained BC model plays Snake in real time, evaluated over
   hundreds of episodes with win-rate tracking.

## Results

| Agent | Win Rate | Notes |
|-------|----------|-------|
| DQN Expert | ~100% | Fills the entire 8x8 board consistently |
| BC Clone | ~67% | Trained on 500 winning episodes, eat-only frames |

The BC model achieves a respectable ~67% win rate, demonstrating that a purely
supervised approach can capture most of the expert's strategy from pixel
observations alone.

## Project Structure

```
.
├── snake_dqn_88/                   # DQN expert (single-env variant)
│   ├── agent.py                    #   DQN agent with replay buffer
│   ├── game.py                     #   Snake environment (8x8, 9-channel state)
│   ├── model.py                    #   CNN + Dueling + NoisyNet
│   └── train.py                    #   Training loop
│
├── snake_dqn_8_multi/              # DQN expert (multi-env variant, faster)
│   └── snake_dqn_8_multi/
│       ├── agent.py                #   Same architecture + batch inference
│       ├── game.py                 #   Same environment
│       ├── model.py                #   Same model
│       └── train.py                #   Vectorized training with multiprocessing
│
├── snake_bc_8_multi/               # BC dataset collection
│   ├── collect_bc.py               #   Runs DQN expert, saves winning episodes
│   └── dataset_bc/                 #   [gitignored] 500 episodes, ~1.1 GB
│       └── README.md               #   Dataset format documentation
│
├── snake_bc_rgb/                   # BC training & evaluation
│   └── train_and_play.py           #   Train CNN on frames, then play live
│
├── pretrained/                     # Pre-trained BC model
│   └── best.pt                     #   Best CNN policy checkpoint (4.5 MB)
│
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Beba-ai-ml/snake-behavioral-cloning.git
cd snake-behavioral-cloning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA separately:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Train the DQN Expert

```bash
python -m snake_dqn_8_multi.snake_dqn_8_multi.train \
    --episodes 5000 \
    --num-envs 8 \
    --save-every 500 \
    --session-id 1
```

Training produces checkpoints in `checkpoints_8_multi/`. The expert should reach
near-100% win rate after several thousand episodes.

To watch the expert play:
```bash
python -m snake_dqn_8_multi.snake_dqn_8_multi.train \
    --episodes 20 \
    --render-every 1 \
    --fps 8 \
    --load-from checkpoints_8_multi/session_1.pth
```

### 3. Collect the BC Dataset

```bash
python -m snake_bc_8_multi.collect_bc \
    --checkpoint snake_dqn_8_multi/checkpoints_8_multi/session_1.pth \
    --wins 500 \
    --save-on all \
    --block-size 16
```

This runs the expert and saves 500 winning episodes to `snake_bc_8_multi/dataset_bc/`.
Each episode contains RGB frames and action labels. With `--save-on all` every step
is saved; with `--save-on eat` only apple-eating steps are saved (smaller dataset).

### 4. Train the BC Model

```bash
python snake_bc_rgb/train_and_play.py \
    --dataset-dir snake_bc_8_multi/dataset_bc \
    --epochs 40 \
    --batch-size 64 \
    --session-id 1 \
    --live-episodes 100
```

After training, the script automatically runs live evaluation. The best model is
saved to `snake_bc_rgb/runs/session_1/best.pt`.

### 5. Play with the Pretrained Model

A pretrained BC model is included at `pretrained/best.pt`. To evaluate it:

```bash
python snake_bc_rgb/train_and_play.py \
    --dataset-dir snake_bc_8_multi/dataset_bc \
    --epochs 0 \
    --session-id 99 \
    --live-episodes 200 \
    --fps 12
```

(With `--epochs 0` the script skips training and proceeds to live play using the
best checkpoint.)

## Dataset Format

Each episode folder contains:

```
episode_00001/
  frames/          # RGB PNG images (128x128 with block_size=16)
    00000.png
    00001.png
    ...
  actions.json     # List of integer actions aligned 1:1 with frames
  meta.json        # Episode metadata (score, steps, grid_size, save_on)
```

Action mapping: `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT`

A game is won when `score >= grid_size^2 - 3` (61 for 8x8).

See `snake_bc_8_multi/dataset_bc/README.md` for the complete specification.

## How to Regenerate the Dataset

The dataset (~1.1 GB) is not included in the repository. To regenerate it:

1. Train a DQN expert (step 2 above) until it wins consistently
2. Run the collector (step 3 above) with the trained checkpoint
3. The collector will play games until it has the requested number of wins

With 8 parallel environments, collecting 500 winning episodes typically takes
a few minutes.

## Architecture Details

### DQN Expert
- **Input:** 9-channel 8x8 grid (body, food, empty, food_xy, direction one-hot)
- **Network:** Conv(9->64) -> Conv(64->128) -> AvgPool -> FC(2048->512) -> Dueling(V+A)
- **Exploration:** NoisyNet (no epsilon-greedy needed)
- **Algorithm:** Double DQN with soft target updates (tau=0.01)

### BC Policy (CNN)
- **Input:** RGB image (128x128 with default block_size=16)
- **Network:** 4x [Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> MaxPool2x2] -> AdaptiveAvgPool -> FC(256->4)
- **Training:** Cross-entropy loss with class weighting, AdamW, ReduceLROnPlateau
- **Regularization:** Early stopping (patience=6)

## Requirements

- Python 3.10+
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pygame >= 2.1.0 (for rendering)
- Pillow >= 9.0.0 (for BC training on RGB frames)

## License

MIT License. See [LICENSE](LICENSE).
