import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.humanoid_env import HumanoidImitationEnv
import time

env = HumanoidImitationEnv("../data/processed_motion/walk.npy", render=True)
model = PPO.load("../models/humanoid_walk_final")

print("Visualizing humanoid — press Ctrl+C to stop\n")

obs, info = env.reset()
total_reward = 0
steps = 0

while steps < 1000:
    action, _ = model.predict(obs, deterministic=False)  # less rigid
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    total_reward += reward
    time.sleep(0.05)  # 30fps — more natural speed
    action, _ = model.predict(obs, deterministic=True)


    if steps % 100 == 0:
        print(f"Step {steps} | Reward: {total_reward:.1f} | Pose Error: {info['pose_error']:.3f}")

    if terminated or truncated:
        print(f"Fell at step {steps} — resetting...")
        time.sleep(2)
        obs, info = env.reset()

env.close()