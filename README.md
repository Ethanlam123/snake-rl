# Snake RL

A minimal, clean Snake game in Pygame for both human play and RL research.

## Features
- 2D grid (default 20×20)
- Human play (arrow keys)
- RL-style API: `reset()`, `step(action)`
- Rewards: +1 (food), -1 (collision), 0 (otherwise)
- Fast, idiomatic, single-file code

## Setup

1. **Install [uv](https://github.com/astral-sh/uv):**
   ```sh
   pip install uv
   ```
2. **Create and activate the environment:**
   ```sh
   uv venv snake-rl
   source snake-rl/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   uv pip install -r requirements.txt
   ```

## Running the Game (Human Play)

```sh
python snake_rl.py
```

## RL Usage Example

```python
import snake_rl as env
state = env.reset()
done = False
while not done:
    action = env.np_random().choice([0, 1, 2, 3])  # random agent
    next_state, reward, done, info = env.step(action)
    state = next_state
```

- `state`: NumPy array of shape (grid_height, grid_width, 3)
- `action`: int, one of [0: left, 1: right, 2: up, 3: down]
- `reward`: +1 (food), -1 (collision), 0 (otherwise)
- `done`: True if game over
- `info`: dict (may include score, etc.)

## Code Quality
- Passes `flake8` (except line length ≤100)
- Idiomatic, type-annotated, well-commented

## License
MIT 