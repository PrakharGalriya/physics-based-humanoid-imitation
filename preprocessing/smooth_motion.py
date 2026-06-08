import cv2
import numpy as np
import os

def smooth_keypoints(keypoints,window=5):
    smoothed = np.copy(keypoints)
    for joint in range(keypoints.shape[1]):
        for coord in range(3):
            signal = keypoints[:,joint,coord]

            kernel = np.ones(window)/window
            smoothed[:, joint, coord] = np.convolve(signal, kernel, mode='same')
    return smoothed

if __name__ == "__main__":
    motions = ["walk", "jump"]

    os.makedirs("../data/processed_motion", exist_ok=True)

    for name in motions:
        path = f"../data/extracted_keypoints/{name}_keypoints.npy"
        print(f"\nSmoothing {name}...")

        kp = np.load(path)
        print(f"  Loaded shape: {kp.shape}")

        smoothed = smooth_keypoints(kp, window=5)
        save_path = f"../data/extracted_keypoints/{name}_smoothed.npy"
        np.save(save_path, smoothed)
        print(f"  Saved smoothed → {save_path}")

    print("\nDone! Ready for joint angle conversion.")



