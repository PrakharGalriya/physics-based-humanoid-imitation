# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# print(f"Total obs size: {obs.shape}")
# print("\n--- RESET obs ---")
# print(f"obs[0]:     {obs[0]:.3f}  ← torso z-position")
# print(f"obs[1]:     {obs[1]:.3f}  ← torso x-velocity")
# print(f"obs[2]:     {obs[2]:.3f}  ← torso y-velocity")
# print(f"obs[3]:     {obs[3]:.3f}  ← torso z-velocity")
# print(f"obs[4:8]:   {np.round(obs[4:8], 3)}  ← torso orientation")
# print(f"obs[8:16]:  {np.round(obs[8:16], 3)}  ← ?")
# print(f"obs[16:24]: {np.round(obs[16:24], 3)}  ← ?")
# print(f"obs[24:32]: {np.round(obs[24:32], 3)}  ← ?")
# print(f"obs[32:40]: {np.round(obs[32:40], 3)}  ← ?")
# print(f"obs[40:48]: {np.round(obs[40:48], 3)}  ← ?")

# # take a step forward
# action = np.zeros(env.action_space.shape)  # zero action — no movement
# obs2, _, _, _, _ = env.step(action)

# print("\n--- AFTER STEP ---")
# print(f"obs[0]:     {obs2[0]:.3f}  ← torso z-position")
# print(f"obs[1]:     {obs2[1]:.3f}  ← torso x-velocity")
# print(f"obs[2]:     {obs2[2]:.3f}  ← torso y-velocity")
# print(f"obs[3]:     {obs2[3]:.3f}  ← torso z-velocity")
# print(f"obs[8:16]:  {np.round(obs2[8:16], 3)}")
# print(f"obs[16:24]: {np.round(obs2[16:24], 3)}")
# print(f"obs[24:32]: {np.round(obs2[24:32], 3)}")
# print(f"obs[32:40]: {np.round(obs2[32:40], 3)}")

# print("\n--- DIFFERENCE (what changed) ---")
# diff = obs2 - obs
# for i in range(50):
#     if abs(diff[i]) > 0.001:
#         print(f"obs[{i}]: {obs[i]:.3f} → {obs2[i]:.3f}  (changed by {diff[i]:.3f})")

# env.close()

# import numpy as np
# import gymnasium as gym

# # check reference shape
# ref = np.load("../data/processed_motion/walk.npy")
# print(f"Reference shape: {ref.shape}")
# print(f"Reference frame 0: {ref[0]}")
# print(f"Reference min: {ref.min():.3f}  max: {ref.max():.3f}")

# # check obs joint range
# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()
# print(f"\nobs[8:16]: {np.round(obs[8:16], 3)}")
# print(f"obs range: min {obs[8:16].min():.3f}  max {obs[8:16].max():.3f}")

# env.close()

# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# print("Checking all ranges for joint-like values (0.1 to 3.14):")
# for i in range(0, 100):
#     if abs(obs[i]) > 0.1:
#         print(f"obs[{i}]: {obs[i]:.4f}")

# env.close()

# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# print("Joint limits:")
# print("Lower:", env.observation_space.low[8:16])
# print("Upper:", env.observation_space.high[8:16])
# env.close()

# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# # get actual joint limits from MuJoCo model
# model = env.unwrapped.model
# print("Joint names and limits:")
# for i in range(model.njnt):
#     name = model.joint(i).name
#     limit = model.joint(i).range
#     print(f"Joint {i}: {name:20s} range: {limit}")

# env.close()
# Add this to humanoid_env.py __init__ temporarily
# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# raw_joints = obs[8:16]

# joint_min = np.array([-0.436, -1.047, -1.920, -2.793,
#                        -0.436, -1.047, -1.920, -2.793])
# joint_max = np.array([ 0.087,  0.611,  0.349, -0.035,
#                         0.087,  0.611,  0.349, -0.035])

# current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
# current_joints = np.clip(current_joints, 0, 1)

# ref = np.load("../data/processed_motion/walk.npy")
# ref_frame = ref[0]

# print(f"current_joints: {np.round(current_joints, 3)}")
# print(f"ref_frame:      {np.round(ref_frame, 3)}")
# print(f"pose_error:     {np.sum((current_joints - ref_frame)**2):.3f}")
# print(f"Both in 0-1?  current: {current_joints.min():.2f}-{current_joints.max():.2f}  ref: {ref_frame.min():.2f}-{ref_frame.max():.2f}")

# env.close()
# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# model = env.unwrapped.model

# print("Joint name | Min limit | Max limit")
# print("-" * 45)
# for i in range(model.njnt):
#     name = model.joint(i).name
#     limit = model.joint(i).range
#     print(f"Joint {i:2d}: {name:20s} | {limit[0]:8.4f} | {limit[1]:8.4f}")

# # also check which obs indices correspond to joints
# obs, _ = env.reset()
# print(f"\nObs shape: {obs.shape}")
# print(f"\nChecking joint position range in obs:")
# print(f"obs[5:22] = {np.round(obs[5:22], 3)}")

# env.close()

# in tests/test.py
# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# joint_min = np.array([
#     -0.7854, -1.3090, -0.6109,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -1.4835, -1.4835, -1.5708,
#     -1.0472, -1.0472, -1.5708
# ])
# joint_max = np.array([
#      0.7854,  0.5236,  0.6109,
#      0.0873,  0.6109,  0.3491, -0.0349,
#      0.0873,  0.6109,  0.3491, -0.0349,
#      1.0472,  1.0472,  0.8727,
#      1.4835,  1.4835,  0.8727
# ])

# raw_joints = obs[5:22]
# current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
# current_joints = np.clip(current_joints, 0, 1)

# ref = np.load("../data/processed_motion/walk.npy")
# ref_frame = ref[0]

# print(f"current_joints shape: {current_joints.shape}")
# print(f"ref_frame shape:      {ref_frame.shape}")
# print(f"current_joints: {np.round(current_joints, 3)}")
# print(f"ref_frame:      {np.round(ref_frame, 3)}")
# print(f"pose_error: {np.sum((current_joints - ref_frame)**2):.3f}")
# print(f"Both 0-1? current: {current_joints.min():.2f}-{current_joints.max():.2f}  ref: {ref_frame.min():.2f}-{ref_frame.max():.2f}")

# env.close()

# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# joint_min = np.array([
#     -0.7854, -1.3090, -0.6109,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -1.4835, -1.4835, -1.5708,
#     -1.0472, -1.0472, -1.5708
# ])
# joint_max = np.array([
#      0.7854,  0.5236,  0.6109,
#      0.0873,  0.6109,  0.3491, -0.0349,
#      0.0873,  0.6109,  0.3491, -0.0349,
#      1.0472,  1.0472,  0.8727,
#      1.4835,  1.4835,  0.8727
# ])

# joint_names = [
#     'abdomen_z', 'abdomen_y', 'abdomen_x',
#     'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee',
#     'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee',
#     'right_shoulder1', 'right_shoulder2', 'right_elbow',
#     'left_shoulder1', 'left_shoulder2', 'left_elbow'
# ]

# raw_joints = obs[5:22]
# current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
# current_joints = np.clip(current_joints, 0, 1)

# ref = np.load("../data/processed_motion/walk.npy")
# ref_frame = ref[0]

# print("Joint by joint comparison:")
# for i, name in enumerate(joint_names):
#     diff = abs(current_joints[i] - ref_frame[i])
#     print(f"{name:20s} | cur: {current_joints[i]:.3f} | ref: {ref_frame[i]:.3f} | diff: {diff:.3f}")

# env.close()
# 
# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# # get joint positions directly from MuJoCo
# qpos = env.unwrapped.data.qpos
# qvel = env.unwrapped.data.qvel

# print(f"qpos shape: {qpos.shape}")
# print(f"qvel shape: {qvel.shape}")
# print(f"\nqpos (joint positions):")
# for i in range(len(qpos)):
#     print(f"qpos[{i}]: {qpos[i]:.4f}")

# env.close()
# import gymnasium as gym
# import numpy as np

# env = gym.make("Humanoid-v5")
# obs, _ = env.reset()

# qpos = env.unwrapped.data.qpos
# joint_angles = qpos[7:24]  # 17 actual joint angles

# print(f"Joint angles shape: {joint_angles.shape}")

# joint_names = [
#     'abdomen_z', 'abdomen_y', 'abdomen_x',
#     'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee',
#     'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee',
#     'right_shoulder1', 'right_shoulder2', 'right_elbow',
#     'left_shoulder1', 'left_shoulder2', 'left_elbow'
# ]

# joint_min = np.array([
#     -0.7854, -1.3090, -0.6109,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -0.4363, -1.0472, -1.9199, -2.7925,
#     -1.4835, -1.4835, -1.5708,
#     -1.0472, -1.0472, -1.5708
# ])
# joint_max = np.array([
#      0.7854,  0.5236,  0.6109,
#      0.0873,  0.6109,  0.3491,  0.1,
#      0.0873,  0.6109,  0.3491,  0.1,
#      1.0472,  1.0472,  0.8727,
#      1.4835,  1.4835,  0.8727
# ])

# normalized = (joint_angles - joint_min) / (joint_max - joint_min + 1e-8)
# normalized = np.clip(normalized, 0, 1)

# ref = np.load("../data/processed_motion/walk.npy")
# ref_frame = ref[0]

# print("\nJoint by joint comparison:")
# for i, name in enumerate(joint_names):
#     diff = abs(normalized[i] - ref_frame[i])
#     clip = "CLIPPED" if normalized[i] == 1.0 or normalized[i] == 0.0 else ""
#     print(f"{name:20s} | raw: {joint_angles[i]:7.4f} | cur: {normalized[i]:.3f} | ref: {ref_frame[i]:.3f} | diff: {diff:.3f} {clip}")

# print(f"\nPose error: {np.sum((normalized - ref_frame)**2):.3f}")
# env.close()

import numpy as np
ref = np.load("../data/processed_motion/walk.npy")
print(ref.shape)  # should be (188, 17)