import pygame
import numpy as np
import sys
import random
from typing import Tuple, Dict, Optional

# Game config
GRID_SIZE = 20
CELL_SIZE = 24
FPS = 10
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

# Actions: 0=left, 1=right, 2=up, 3=down
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Colors (R, G, B)
COLOR_BG = (30, 30, 30)
COLOR_SNAKE = (0, 200, 0)
COLOR_FOOD = (200, 0, 0)
COLOR_HEAD = (0, 255, 0)

np_random = np.random


class SnakeEnv:
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.custom_reward_fn = None
        self.reset()

    def reset(self) -> np.ndarray:
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # start moving down
        self.spawn_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def spawn_food(self) -> None:
        empty = set((x, y) for x in range(self.grid_size)
                    for y in range(self.grid_size))
        empty -= set(self.snake)
        self.food = random.choice(list(empty))

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Dict]:
        if self.done:
            return self._get_state(), 0, True, {'score': self.score}
        # Update direction
        dx, dy = ACTIONS[action]
        # Prevent reverse
        if (dx, dy) == tuple(-i for i in self.direction):
            dx, dy = self.direction
        self.direction = (dx, dy)
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        # Check collision
        if (not 0 <= new_head[0] < self.grid_size or
                not 0 <= new_head[1] < self.grid_size or
                new_head in self.snake):
            self.done = True
            reward = self.get_reward(collision=True, ate_food=False)
            return self._get_state(), reward, True, {'score': self.score}
        # Move snake
        self.snake.insert(0, new_head)
        ate_food = new_head == self.food
        if ate_food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()
        reward = self.get_reward(collision=False, ate_food=ate_food)
        return self._get_state(), reward, self.done, {'score': self.score}

    def get_reward(self, collision: bool, ate_food: bool) -> int:
        """Reward function: override or modify for custom RL reward shaping."""
        if self.custom_reward_fn is not None:
            return self.custom_reward_fn(collision=collision, ate_food=ate_food, env=self)
        if collision:
            return -1
        if ate_food:
            return 1
        return 0

    def _get_state(self) -> np.ndarray:
        # State: (grid, grid, 3) uint8
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for x, y in self.snake[1:]:
            state[y, x] = COLOR_SNAKE
        head_x, head_y = self.snake[0]
        state[head_y, head_x] = COLOR_HEAD
        fx, fy = self.food
        state[fy, fx] = COLOR_FOOD
        return state


# Module-level API for RL usage
_env: Optional[SnakeEnv] = None


def reset() -> np.ndarray:
    global _env
    if _env is None:
        _env = SnakeEnv()
    return _env.reset()


def step(action: int) -> Tuple[np.ndarray, int, bool, Dict]:
    global _env
    if _env is None:
        raise RuntimeError('Call reset() before step()')
    return _env.step(action)


def np_random():
    """Expose numpy.random for RL example in README."""
    return np.random


def set_reward_fn(reward_fn):
    global _env
    if _env is None:
        _env = SnakeEnv()
    _env.custom_reward_fn = reward_fn


# Human play demo
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    env = SnakeEnv()
    state = env.reset()
    key_to_action = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_DOWN: 3,
    }
    action = 3  # start moving down
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
        state, reward, done, info = env.step(action)
        # Draw
        screen.fill(COLOR_BG)
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                color = tuple(state[y, x])
                if color != (0, 0, 0):
                    pygame.draw.rect(
                        screen,
                        color,
                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    )
        # Draw score
        score_surf = font.render(
            f"Score: {info['score']}", True, (255, 255, 255))
        screen.blit(score_surf, (8, 8))
        pygame.display.flip()
        clock.tick(FPS)
        if done:
            pygame.time.wait(1000)
            state = env.reset()
            action = 3
    pygame.quit()
    sys.exit()
