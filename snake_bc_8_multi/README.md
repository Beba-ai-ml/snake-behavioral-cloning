# Snake BC Dataset Collector (8x8)

Collects a behavioral cloning dataset from a trained DQN expert agent. Runs the
expert on an 8x8 Snake environment, saves winning episodes as RGB frame sequences
with corresponding action labels.

## Quick Start

```bash
python -m snake_bc_8_multi.collect_bc \
    --checkpoint ../snake_dqn_8_multi/checkpoints_8_multi/session_5.pth \
    --wins 500 \
    --save-on eat \
    --block-size 16
```

## Key Parameters

- `--checkpoint` - path to a trained DQN `.pth` file (required)
- `--wins N` - number of winning episodes to collect (default: 500)
- `--save-on eat|all` - save only eat-steps or every step (default: eat)
- `--block-size N` - pixel size per grid cell in saved frames (default: 16)
- `--num-envs N` - parallel environments for faster collection (default: 8)
- `--draw-grid` - include grid lines in saved images

## Output

Dataset is saved to `dataset_bc/` with one folder per episode. See
`dataset_bc/README.md` for the full format specification.
