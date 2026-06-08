# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from stable_baselines3 import PPO
# from env.humanoid_env import HumanoidImitationEnv
# import numpy as np

# def evaluate(model_path, motion_path, episodes=5):
#     env = HumanoidImitationEnv(motion_path)
#     model = PPO.load(model_path)

#     print(f"Evaluating: {model_path}\n")

#     all_rewards = []
#     all_errors = []

#     for ep in range(episodes):
#         obs, info = env.reset()
#         ep_reward = 0
#         ep_errors = []
#         steps = 0

#         while True:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             ep_reward += reward
#             ep_errors.append(info.get("pose_error", 0))
#             # ep_errors.append(info["pose_error"])
#             steps += 1
#             if terminated or truncated:
#                 break

#         all_rewards.append(ep_reward)
#         all_errors.append(np.mean(ep_errors))
#         print(f"Episode {ep+1} | Steps: {steps} | Reward: {ep_reward:.1f} | Avg Error: {np.mean(ep_errors):.3f}")

#     print(f"\nAverage reward: {np.mean(all_rewards):.1f}")
#     print(f"Average pose error: {np.mean(all_errors):.3f}")
#     env.close()

# if __name__ == "__main__":
#     evaluate(
#         model_path="../models/humanoid_walk_final",
#         motion_path="../data/processed_motion/walk.npy",
#         episodes=5
#     )
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.humanoid_env import HumanoidImitationEnv
import numpy as np

def evaluate(model_path, motion_path, episodes=5):
    # use env directly, not vecenv
    env   = HumanoidImitationEnv(motion_path)
    model = PPO.load(model_path)
    print(f"Evaluating: {model_path}\n")

    all_rewards = []
    all_errors  = []
    all_steps   = []

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_errors = []
        steps     = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_errors.append(info["pose_error"])
            steps += 1
            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_errors.append(np.mean(ep_errors))
        all_steps.append(steps)
        print(f"Episode {ep+1} | Steps: {steps} | Reward: {ep_reward:.1f} | Avg Error: {np.mean(ep_errors):.3f}")

    print(f"\nAverage reward:     {np.mean(all_rewards):.1f}")
    print(f"Average steps:      {np.mean(all_steps):.1f}")
    print(f"Average pose error: {np.mean(all_errors):.3f}")
    env.close()

if __name__ == "__main__":
    evaluate(
        model_path="../models/humanoid_walk_final",
        motion_path="../data/processed_motion/walk.npy",
        episodes=5
    )