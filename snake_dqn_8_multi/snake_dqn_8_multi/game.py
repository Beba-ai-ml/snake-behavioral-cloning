import random
from collections import deque
from typing import List, Tuple

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - optional when running headless
    pygame = None


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class SnakeGame:
    """Snake environment (default 8x8) with CNN-compatible state representation."""

    def __init__(self, grid_size: int = 8, render: bool = False, block_size: int = 60, fps: int = 8):
        self.grid_size = grid_size
        self.block_size = block_size
        self.fps = fps
        self.render_enabled = render and pygame is not None

        self.snake: deque[Tuple[int, int]] = deque()
        self.direction: int = RIGHT
        self.food: Tuple[int, int] | None = None
        self.score: int = 0
        self.steps_since_food: int = 0
        self.base_max_steps = 150  # minimum step limit; rest depends on snake length
        self.no_progress_steps = 0
        self.last_death_reason: str = ""

        # Pygame objects are only created when rendering is enabled.
        self.screen = None
        self.clock = None
        if self.render_enabled:
            pygame.init()
            window = self.block_size * self.grid_size
            self.screen = pygame.display.set_mode((window, window))
            pygame.display.set_caption(f"Snake DQN {self.grid_size}x{self.grid_size}")
            self.clock = pygame.time.Clock()

    def reset(self) -> np.ndarray:
        center = self.grid_size // 2
        self.direction = RIGHT
        self.snake = deque([(center, center), (center - 1, center), (center - 2, center)])
        self.score = 0
        self.steps_since_food = 0
        self.no_progress_steps = 0
        self.last_death_reason = ""
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        snake_set = set(self.snake)
        free = list(positions - snake_set)
        self.food = random.choice(free) if free else None

    def _opposite(self, action: int) -> bool:
        return (self.direction == UP and action == DOWN) or \
            (self.direction == DOWN and action == UP) or \
            (self.direction == LEFT and action == RIGHT) or \
            (self.direction == RIGHT and action == LEFT)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, int]:
        """Execute a move. Returns (next_state, reward, done, score)."""
        if self._opposite(action):
            action = self.direction  # prevent 180-degree turn in a single frame
        self.last_death_reason = ""

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food if self.food else (head_x, head_y)
        prev_dist = abs(head_x - food_x) + abs(head_y - food_y)

        self.direction = action
        new_head = self._next_head(action)
        reward = -0.005  # small penalty for each idle step
        done = False
        self.steps_since_food += 1

        new_dist = abs(new_head[0] - food_x) + abs(new_head[1] - food_y)
        reward += 0.05 * (prev_dist - new_dist)  # mild distance shaping
        max_dist = (self.grid_size - 1) * 2
        reward -= 0.02 * (new_dist / max_dist)  # penalty for being far from food

        if new_dist < prev_dist:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        if self.no_progress_steps >= 20:
            reward -= 1.0
            self.no_progress_steps = 0

        max_steps = max(self.base_max_steps, len(self.snake) * 60)
        out_of_bounds = new_head[0] < 0 or new_head[0] >= self.grid_size or new_head[1] < 0 or new_head[1] >= self.grid_size
        hit_self = new_head in list(self.snake)
        timeout = self.steps_since_food > max_steps
        if out_of_bounds or hit_self or timeout:
            reward = -25.0
            done = True
            if out_of_bounds:
                self.last_death_reason = "wall"
            elif hit_self:
                self.last_death_reason = "body"
            else:
                self.last_death_reason = "timeout"
            return self.get_state(), reward, done, self.score

        self.snake.appendleft(new_head)

        if new_head == self.food:
            reward = 20.0 + 0.5 * self.score  # increasing reward for successive apples
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.render_enabled:
            self._render()

        return self.get_state(), reward, done, self.score

    def _next_head(self, action: int) -> Tuple[int, int]:
        x, y = self.snake[0]
        if action == UP:
            y -= 1
        elif action == DOWN:
            y += 1
        elif action == LEFT:
            x -= 1
        elif action == RIGHT:
            x += 1
        return x, y

    def _collision(self, point: Tuple[int, int]) -> bool:
        x, y = point
        out_of_bounds = x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size
        hit_self = point in list(self.snake)
        return out_of_bounds or hit_self

    def _danger_in_direction(self, direction: int) -> float:
        next_point = self._next_head(direction)
        return 1.0 if self._collision(next_point) else 0.0

    def get_state(self) -> np.ndarray:
        """
        Returns an 8x8 map with 9 channels:
        [0] body/head, [1] food, [2] empty cells,
        [3] food_x (constant plane), [4] food_y,
        [5..8] direction one-hot (up/down/left/right) as constant planes.
        """
        grid = np.zeros((9, self.grid_size, self.grid_size), dtype=np.float32)

        # body (channel 0)
        for x, y in self.snake:
            grid[0, y, x] = 1.0

        # food (channel 1)
        if self.food:
            fx, fy = self.food
            grid[1, fy, fx] = 1.0

        # empty cells (channel 2)
        occupied = grid[0] + grid[1]
        grid[2] = (occupied == 0).astype(np.float32)

        # global features as constant-value planes
        norm = float(self.grid_size - 1)
        fx_norm, fy_norm = (self.food if self.food else (0, 0))
        fx_norm /= norm
        fy_norm /= norm
        grid[3] = fx_norm
        grid[4] = fy_norm
        grid[5] = 1.0 if self.direction == UP else 0.0
        grid[6] = 1.0 if self.direction == DOWN else 0.0
        grid[7] = 1.0 if self.direction == LEFT else 0.0
        grid[8] = 1.0 if self.direction == RIGHT else 0.0

        return grid

    def _render(self) -> None:
        assert self.screen is not None and self.clock is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((30, 30, 30))

        # Grid lines (visual aid)
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (i * self.block_size, 0),
                (i * self.block_size, self.block_size * self.grid_size),
                1,
            )
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (0, i * self.block_size),
                (self.block_size * self.grid_size, i * self.block_size),
                1,
            )

        # Snake
        for idx, (x, y) in enumerate(self.snake):
            color = (50, 200, 50) if idx == 0 else (80, 160, 80)
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(
                    x * self.block_size,
                    y * self.block_size,
                    self.block_size,
                    self.block_size,
                ),
            )

        # Food
        if self.food:
            fx, fy = self.food
            pygame.draw.rect(
                self.screen,
                (220, 50, 50),
                pygame.Rect(
                    fx * self.block_size,
                    fy * self.block_size,
                    self.block_size,
                    self.block_size,
                ),
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self.render_enabled and pygame:
            pygame.quit()
