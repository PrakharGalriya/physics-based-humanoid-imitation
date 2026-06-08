import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from env.humanoid_env import HumanoidImitationEnv
import numpy as np

def make_env():
    return HumanoidImitationEnv("../data/processed_motion/walk.npy")

if __name__ == "__main__":
    print("Setting up training...")

    # vectorized env — uses all CPU cores
    env = make_vec_env(make_env, n_envs=4)

    # checkpoint callback — saves every 10k steps
    checkpoint = CheckpointCallback(
        save_freq=10_000,
        save_path="../models/checkpoints/",
        name_prefix="humanoid_walk"
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.0,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256])
    )

    
    print("Checkpoints saved every 10k steps in models/checkpoints/\n")

    model.learn(
        total_timesteps=5_000_000,
        callback=checkpoint
    )

    # save final model
    os.makedirs("../models", exist_ok=True)
    model.save("../models/humanoid_walk_final")
    print("\nTraining complete! Model saved to models/humanoid_walk_final")