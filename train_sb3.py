import snake_rl
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import pygame


class SnakeGymWrapper(gym.Env):
    """Minimal Gym wrapper for snake_rl."""

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(20, 20, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

    def reset(self):
        state = snake_rl.reset()
        return state

    def step(self, action):
        state, reward, done, info = snake_rl.step(action)
        return state, reward, done, info

    def render(self, mode='ai'):
        # Visualize using pygame window
        if mode == 'human':
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((480, 480))
            state = snake_rl._env._get_state()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            self.screen.fill((30, 30, 30))
            cell_size = 24
            for y in range(20):
                for x in range(20):
                    color = tuple(state[y, x])
                    if color != (0, 0, 0):
                        pygame.draw.rect(
                            self.screen,
                            color,
                            (x * cell_size, y * cell_size, cell_size, cell_size)
                        )
            pygame.display.flip()
        elif mode == 'ai':
            pass
        else:
            pass

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()


env = SnakeGymWrapper()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)
print('Training complete.')

# Visualize agent performance
obs = env.reset()
done = False
score = 0
for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    score = info.get('score', 0)
    print(f'Step {step}: reward={reward}, score={score}')
    env.render()
    pygame.time.wait(100)
    if done:
        print('Episode finished.')
        break
env.close()
