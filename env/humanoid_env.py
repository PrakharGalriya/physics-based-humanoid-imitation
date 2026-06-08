import gymnasium as gym
import numpy as np

class HumanoidImitationEnv(gym.Wrapper):
    def __init__(self, reference_motion_path,render=False):
        if render:
            env = gym.make("Humanoid-v5",render_mode="human")
        else:
            env = gym.make("Humanoid-v5")
        super().__init__(env)

        # load reference motion
        self.reference = np.load(reference_motion_path)
        self.num_frames = len(self.reference)
        self.step_idx = 0

        print(f"Loaded reference: {reference_motion_path}")
        print(f"Frames: {self.num_frames}")

    def reset(self, **kwargs):
        self.step_idx = 0
        return self.env.reset(**kwargs)


    # def step(self, action):
    #     obs, _, terminated, truncated, info = self.env.step(action)

    #     current_joints = obs[2:10]
    #     ref_joints = self.reference[self.step_idx % self.num_frames]

    #     # pose reward
    #     pose_error = np.sum((current_joints - ref_joints) ** 2)
    #     pose_reward = np.exp(-0.5 * pose_error)  # exponential — smoother gradient

    #     # alive bonus — reward just for not falling
    #     torso_height = obs[0]
    #     alive_bonus = 2.0 if torso_height > 0.3 else 0.0

    #     # fall penalty — very lenient now
    #     if torso_height < 0.2:
    #         terminated = True

    #     reward = pose_reward + alive_bonus
    #     self.step_idx += 1
    #     info["pose_error"] = pose_error

    #     return obs, reward, terminated, truncated, info
    # def step(self, action):
    #     obs, _, terminated, truncated, info = self.env.step(action)
        

    #     # raw_joints = obs[8:16]
    #     # joint_min = np.array([-0.436, -1.047, -1.920, -2.793,   # right hip x,z,y + knee
    #     #                    -0.436, -1.047, -1.920, -2.793])  # left hip x,z,y + knee
    #     # joint_max = np.array([ 0.087,  0.611,  0.349, -0.035,   # right hip x,z,y + knee
    #     #                         0.087,  0.611,  0.349, -0.035])  # left
    #     # current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
    #     # current_joints = np.clip(current_joints, 0, 1)
    #     # exact limits from MuJoCo output
    #     joint_min = np.array([
    #         -0.7854, -1.3090, -0.6109,  # abdomen z,y,x
    #         -0.4363, -1.0472, -1.9199, -2.7925,  # right hip x,z,y + knee
    #         -0.4363, -1.0472, -1.9199, -2.7925,  # left hip x,z,y + knee
    #         -1.4835, -1.4835, -1.5708,  # right shoulder1,2 + elbow
    #         -1.0472, -1.0472, -1.5708   # left shoulder1,2 + elbow
    #     ])

    #     joint_max = np.array([
    #         0.7854,  0.5236,  0.6109,  # abdomen z,y,x
    #         0.0873,  0.6109,  0.3491, 0.1,  # right hip x,z,y + knee
    #         0.0873,  0.6109,  0.3491, 0.1,  # left hip x,z,y + knee
    #         1.0472,  1.0472,  0.8727,  # right shoulder1,2 + elbow
    #         1.4835,  1.4835,  0.8727   # left shoulder1,2 + elbow
    #     ])

    #     # use obs[5:22] for all 17 joints
    #     raw_joints = obs[5:22]
    #     current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
    #     current_joints = np.clip(current_joints, 0, 1)

    #     ref_joints = self.reference[self.step_idx % self.num_frames]

    #     # 1. pose reward
    #     pose_error = np.sum((current_joints - ref_joints) ** 2)
    #     pose_reward = np.exp(-0.5 * pose_error)

    #     # 2. velocity matching
    #     current_vel = obs[24:32]
    #     vel_error = np.sum(current_vel ** 2)
    #     vel_reward = np.exp(-0.1 * vel_error)

    #     # 3. center of mass reward
    #     torso_height = obs[0]
    #     com_reward = 1.0 if torso_height > 0.8 else 0.0

    #     # 4. energy penalty
    #     energy_penalty = 0.001 * np.sum(action ** 2)

    #     # 5. alive bonus
    #     alive_bonus = 2.0 if torso_height > 0.3 else 0.0

    #     # fall
    #     if torso_height < 0.2:
    #         terminated = True

    #     forward_vel = obs[1]
    #     forward_reward = 2.0 * forward_vel
    #     reward = pose_reward + alive_bonus + com_reward + vel_reward + forward_reward - energy_penalty
    #     self.step_idx += 1
    #     info["pose_error"] = pose_error

    #     return obs, reward, terminated, truncated, info

#     # def step(self, action):
#     obs, _, terminated, truncated, info = self.env.step(action)

#     current_joints = obs[2:10]
#     ref_joints = self.reference[self.step_idx % self.num_frames]

#     # pose reward
#     pose_error = np.sum((current_joints - ref_joints) ** 2)
#     pose_reward = np.exp(-0.5 * pose_error)  # exponential — smoother gradient

#     # alive bonus — reward just for not falling
#     torso_height = obs[0]
#     alive_bonus = 2.0 if torso_height > 0.3 else 0.0

#     # fall penalty — very lenient now
#     if torso_height < 0.2:
#         terminated = True

#     reward = pose_reward + alive_bonus
#     self.step_idx += 1
#     info["pose_error"] = pose_error

#     return obs, reward, terminated, truncated, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # use actual joint angles from qpos
        qpos = self.env.unwrapped.data.qpos
        raw_joints = qpos[7:24]  # 17 actual joint angles

        joint_min = np.array([
            -0.7854, -1.3090, -0.6109,
            -0.4363, -1.0472, -1.9199, -2.7925,
            -0.4363, -1.0472, -1.9199, -2.7925,
            -1.4835, -1.4835, -1.5708,
            -1.0472, -1.0472, -1.5708
        ])
        joint_max = np.array([
            0.7854,  0.5236,  0.6109,
            0.0873,  0.6109,  0.3491,  0.1,
            0.0873,  0.6109,  0.3491,  0.1,
            1.0472,  1.0472,  0.8727,
            1.4835,  1.4835,  0.8727
        ])

        current_joints = (raw_joints - joint_min) / (joint_max - joint_min + 1e-8)
        current_joints = np.clip(current_joints, 0, 1)

        ref_joints = self.reference[self.step_idx % self.num_frames]

        pose_error = np.sum((current_joints - ref_joints) ** 2)
        pose_reward = np.exp(-0.5 * pose_error)

        current_vel = self.env.unwrapped.data.qvel[6:23]  # actual joint velocities
        vel_error = np.sum(current_vel ** 2)
        vel_reward = np.exp(-0.3 * vel_error)
        torso_height = obs[0]
        upright_reward = 2.0 * min(torso_height / 1.0, 1.0)
        alive_bonus = 3.0
        forward_vel = obs[1]
        forward_reward = 1.0 * forward_vel
        energy_penalty = 0.002 * np.sum(action ** 2)


        # torso_height = obs[0]
        # com_reward = 1.0 if torso_height > 0.8 else 0.0
        # energy_penalty = 0.002 * np.sum(action ** 2)
        # alive_bonus = 3.0 if torso_height > 0.3 else 0.0

        # if torso_height < 0.2:
        #     terminated = True

        # forward_vel = obs[1]
        # forward_reward = 1.0 * forward_vel
        reward = 2.0*pose_reward + upright_reward  + vel_reward + alive_bonus + forward_reward - energy_penalty
        if torso_height < 0.2:
            reward -= 100.0  
            terminated = True

        self.step_idx += 1
        info["pose_error"] = float(pose_error)
        info["forward_vel"] = float(forward_vel)
        # info["pose_error"] = pose_error

        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    env = HumanoidImitationEnv("../data/processed_motion/walk.npy")
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")

    # test 5 steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1} | reward: {reward:.3f} | pose_error: {info['pose_error']:.3f}")

    env.close()
    print("\nWeek 4 Complete! Environment working correctly.")