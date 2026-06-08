import cv2
import mediapipe as mp
import mujoco
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
import imageio

env = gym.make("Humanoid-v5")

print("VS CODE READY")
env.close()

import gymnasium as gym

env = gym.make("Humanoid-v5", render_mode="human")
obs, info = env.reset()

for i in range(500):
    action = env.action_space.sample()  # random actions
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()