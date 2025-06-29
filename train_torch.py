import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import snake_rl

# --- Custom Reward Function Example ---


def custom_reward_fn(collision, ate_food, env):
    # Example: encourage longer survival
    if collision:
        return -10
    if ate_food:
        return 2
    # Small living reward
    return -0.01


# Set the custom reward function
snake_rl.set_reward_fn(custom_reward_fn)

# --- DQN Network ---


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)


# --- Hyperparameters ---
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-3
MEM_SIZE = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 1000
NUM_EPISODES = 300
MAX_STEPS = 500

# --- Replay Buffer ---


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --- Preprocessing ---


def preprocess(state):
    # Convert (H, W, 3) uint8 to (3, H, W) float32 in [0,1]
    state = np.transpose(state, (2, 0, 1))
    return state.astype(np.float32) / 255.0

# --- Main Training Loop ---


def train():
    env = snake_rl
    n_actions = 4
    obs = env.reset()
    state_shape = preprocess(obs).shape
    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    policy_net = DQN(state_shape, n_actions).to(device)
    target_net = DQN(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(MEM_SIZE)
    steps_done = 0
    eps = EPS_START

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        state = preprocess(obs)
        total_reward = 0
        for t in range(MAX_STEPS):
            eps = max(EPS_END, EPS_START - steps_done / EPS_DECAY)
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, device=device).unsqueeze(0)
                    q = policy_net(s)
                    action = q.argmax(1).item()
            next_obs, reward, done, info = env.step(action)
            next_state = preprocess(next_obs)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1
            # Learn
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch
                states = torch.tensor(states, device=device)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(
                    rewards, device=device, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, device=device)
                dones = torch.tensor(dones, device=device,
                                     dtype=torch.float32).unsqueeze(1)
                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(
                        next_states).max(1, keepdim=True)[0]
                    target = rewards + GAMMA * max_next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update target
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
        print(f"Episode {episode+1}: total_reward={total_reward}")
    print("Training complete.")
    torch.save(policy_net.state_dict(), "dqn_snake.pth")


if __name__ == "__main__":
    train()
