import numpy as np
import os

# MediaPipe landmark indices
# LANDMARKS = {
#     "left_shoulder": 11,  "right_shoulder": 12,
#     "left_elbow": 13,     "right_elbow": 14,
#     "left_wrist": 15,     "right_wrist": 16,
#     "left_hip": 23,       "right_hip": 24,
#     "left_knee": 25,      "right_knee": 26,
#     "left_ankle": 27,     "right_ankle": 28,
# }

# def angle_between(a, b, c):
#     """Angle at point b, between points a-b-c"""
#     ba = a - b
#     bc = c - b
#     cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
#     return np.arccos(np.clip(cosine, -1.0, 1.0))

# def extract_joint_angles(keypoints):
#     """
#     Convert (T, 33, 4) keypoints to (T, 8) joint angles
#     matching MuJoCo humanoid joints
#     """
#     T = keypoints.shape[0]
#     angles = np.zeros((T, 8))

#     for t in range(T):
#         kp = keypoints[t, :, :3]  # x, y, z only

#         # left elbow
#         angles[t, 0] = angle_between(kp[11], kp[13], kp[15])
#         # right elbow
#         angles[t, 1] = angle_between(kp[12], kp[14], kp[16])
#         # left shoulder
#         angles[t, 2] = angle_between(kp[23], kp[11], kp[13])
#         # right shoulder
#         angles[t, 3] = angle_between(kp[24], kp[12], kp[14])
#         # left hip
#         angles[t, 4] = angle_between(kp[11], kp[23], kp[25])
#         # right hip
#         angles[t, 5] = angle_between(kp[12], kp[24], kp[26])
#         # left knee
#         angles[t, 6] = angle_between(kp[23], kp[25], kp[27])
#         # right knee
#         angles[t, 7] = angle_between(kp[24], kp[26], kp[28])

#     return angles
# def extract_joint_angles(keypoints):
#     T = keypoints.shape[0]
#     angles = np.zeros((T, 8))

#     for t in range(T):
#         kp = keypoints[t, :, :3]

#         # match MuJoCo obs[8:16] order exactly:
#         # right_hip_x
#         angles[t, 0] = angle_between(kp[12], kp[24], kp[26])
#         # right_hip_z
#         angles[t, 1] = angle_between(kp[12], kp[24], kp[26])
#         # right_hip_y
#         angles[t, 2] = angle_between(kp[12], kp[24], kp[26])
#         # right_knee
#         angles[t, 3] = angle_between(kp[24], kp[26], kp[28])
#         # left_hip_x
#         angles[t, 4] = angle_between(kp[11], kp[23], kp[25])
#         # left_hip_z
#         angles[t, 5] = angle_between(kp[11], kp[23], kp[25])
#         # left_hip_y
#         angles[t, 6] = angle_between(kp[11], kp[23], kp[25])
#         # left_knee
#         angles[t, 7] = angle_between(kp[23], kp[25], kp[27])

#     return angles

# if __name__ == "__main__":
#     motions = ["walk", "jump"]
#     os.makedirs("../data/processed_motion", exist_ok=True)

#     for name in motions:
#         path = f"../data/extracted_keypoints/{name}_smoothed.npy"
#         print(f"\nConverting {name}...")

#         kp = np.load(path)
#         angles = extract_joint_angles(kp)

#         # normalize to 0-1
#         angles = (angles - angles.min()) / (angles.max() - angles.min() + 1e-8)

#         save_path = f"../data/processed_motion/{name}.npy"
#         np.save(save_path, angles)
#         print(f"  Shape: {angles.shape}")
#         print(f"  Min: {angles.min():.3f}  Max: {angles.max():.3f}")
#         print(f"  Saved → {save_path}")

#     print("\nWeek 3 Complete! walk.npy and jump.npy ready in processed_motion/")
import numpy as np
import os

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def extract_joint_angles(keypoints):
    T = keypoints.shape[0]
    angles = np.zeros((T, 17))

    for t in range(T):
        kp = keypoints[t, :, :3]

        # abdomen (joints 1,2,3)
        angles[t, 0] = angle_between(kp[11], kp[23], kp[24])  # abdomen_z
        angles[t, 1] = np.pi - angle_between(kp[12], kp[11], kp[23])  # abdomen_y
        angles[t, 2] = angle_between(kp[11], kp[12], kp[24])  # abdomen_x

        # right leg (joints 4,5,6,7)
        angles[t, 3] = angle_between(kp[12], kp[24], kp[26])  # right_hip_x
        angles[t, 4] = angle_between(kp[12], kp[24], kp[26])  # right_hip_z
        angles[t, 5] = angle_between(kp[12], kp[24], kp[26])  # right_hip_y
        angles[t, 6] = angle_between(kp[24], kp[26], kp[28])  # right_knee

        # left leg (joints 8,9,10,11)
        angles[t, 7]  = angle_between(kp[11], kp[23], kp[25]) # left_hip_x
        angles[t, 8]  = angle_between(kp[11], kp[23], kp[25]) # left_hip_z
        angles[t, 9]  = angle_between(kp[11], kp[23], kp[25]) # left_hip_y
        angles[t, 10] = angle_between(kp[23], kp[25], kp[27]) # left_knee

        # right arm (joints 12,13,14)
        angles[t, 11] = np.pi - angle_between(kp[24], kp[12], kp[14]) # right_shoulder1
        angles[t, 12] = np.pi - angle_between(kp[24], kp[12], kp[14]) # right_shoulder2
        angles[t, 13] = angle_between(kp[12], kp[14], kp[16]) # right_elbow

        # left arm (joints 15,16,17)
        angles[t, 14] = np.pi - angle_between(kp[23], kp[11], kp[13]) # left_shoulder1
        angles[t, 15] = np.pi - angle_between(kp[23], kp[11], kp[13]) # left_shoulder2
        angles[t, 16] = angle_between(kp[11], kp[13], kp[15]) # left_elbow

    return angles

if __name__ == "__main__":
    motions = ["walk", "jump"]
    os.makedirs("../data/processed_motion", exist_ok=True)

    for name in motions:
        path = f"../data/extracted_keypoints/{name}_smoothed.npy"
        print(f"\nConverting {name}...")
        kp = np.load(path)
        angles = extract_joint_angles(kp)
        angles = (angles - angles.min()) / (angles.max() - angles.min() + 1e-8)
        save_path = f"../data/processed_motion/{name}.npy"
        np.save(save_path, angles)
        print(f"  Shape: {angles.shape}")
        print(f"  Min: {angles.min():.3f}  Max: {angles.max():.3f}")
        print(f"  Saved → {save_path}")