# Snake DQN (8x8, Multi-Env)

DQN expert agent on an 8x8 grid with CNN input (9x8x8 channels). Uses Double DQN
with Dueling architecture and NoisyNet layers for exploration. Supports parallel
environments via multiprocessing for faster training.

## Quick Start

```bash
python -m snake_dqn_8_multi.train --episodes 200 --render-every 0 --save-every 100 --session-id 1
```

## Key Parameters

- `--num-envs N` - number of parallel environments (default: 8)
- `--render-every N` - render every Nth episode (0 = disabled)
- `--save-every N` - checkpoint every N episodes
- `--session-id K` - session number for checkpoint naming
- `--load-from path.pth` - resume from a specific checkpoint
