import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_pose(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:  # skip every 2nd frame low compute
                continue

            # resize for speed
            frame = cv2.resize(frame, (480, 270))
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                keypoints.append(landmarks)

    cap.release()

    keypoints = np.array(keypoints)
    np.save(save_path, keypoints)
    print(f"Saved {len(keypoints)} frames → {save_path}")
    print(f"Shape: {keypoints.shape}")
    return keypoints

if __name__ == "__main__":
    videos = {
        "walk": "../data/raw_videos/walk.mp4",
        "jump": "../data/raw_videos/jump.mp4",
    }

    os.makedirs("../data/extracted_keypoints", exist_ok=True)

    for name, path in videos.items():
        print(f"\nProcessing {name}...")
        extract_pose(path, f"../data/extracted_keypoints/{name}_keypoints.npy")

    print("\nDone! Check data/extracted_keypoints/")