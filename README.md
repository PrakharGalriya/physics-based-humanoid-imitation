# Robotics Project: Physics-Based Humanoid Motion Imitation

## Overview

This project focuses on humanoid motion learning using PPO-based reinforcement learning. The objective is to train a simulated MuJoCo humanoid to imitate human motion extracted from standard monocular video, reducing dependence on expensive Motion Capture (MoCap) systems.

The pipeline combines computer vision, motion preprocessing, kinematic retargeting, and reinforcement learning to generate stable humanoid locomotion inside a physics-based simulation environment.

---

## Features

* **Pose Extraction:** Extracts human body keypoints from monocular videos using MediaPipe.
* **Motion Smoothing:** Applies temporal smoothing to reduce pose jitter and improve motion consistency.
* **Joint Angle Conversion:** Converts extracted spatial coordinates into MuJoCo-compatible joint angles across multiple humanoid degrees of freedom.
* **PPO Training Pipeline:** Uses Stable-Baselines3 with Proximal Policy Optimization (PPO) and a custom imitation reward function for humanoid training.
* **Evaluation & Visualization:** Includes tools for benchmarking pose error, evaluating policy stability, and rendering learned motions in simulation.

---

## Tech Stack

* Python
* MuJoCo
* Gymnasium
* Stable-Baselines3
* PyTorch
* MediaPipe
* NumPy

---

## Project Structure

```text
Robotics-project/
│
├── data/
│   ├── extracted_keypoints/
│   ├── processed_motion/
│   └── raw_videos/
│
├── env/
│   └── humanoid_env.py
│
├── models/
│   └── checkpoints/
│
├── preprocessing/
│   ├── extract_pose.py
│   ├── smooth_motion.py
│   └── convert_to_joint_angles.py
│
├── tests/
│   ├── test_setup.py
│   └── test.py
│
├── train/
│   ├── train_ppo.py
│   ├── evaluate.py
│   └── visualise.py
│
└── README.md
```

---

## Module Description

### `data/`

Contains all raw and processed motion assets.

* `extracted_keypoints/`
  Stores intermediate NumPy arrays generated from MediaPipe pose extraction and smoothing pipelines.

* `processed_motion/`
  Contains finalized reference trajectories mapped to MuJoCo-compatible joint representations.

* `raw_videos/`
  Source monocular videos used for motion extraction.

---

### `env/`

* `humanoid_env.py`
  Custom Gymnasium wrapper implementing MuJoCo humanoid environment logic, observation spaces, reward computation, and episode handling.

---

### `models/`

* `checkpoints/`
  Stores intermediate policy checkpoints generated during training.

---

### `preprocessing/`

* `extract_pose.py`
  Extracts pose landmarks from input videos using MediaPipe.

* `smooth_motion.py`
  Applies temporal smoothing filters to reduce noise and instability in motion trajectories.

* `convert_to_joint_angles.py`
  Maps processed pose coordinates into humanoid joint-space representations compatible with MuJoCo.

---

### `tests/`

* `test_setup.py` & `test.py`
  Utility scripts used to validate simulation setup, dependencies, and environment initialization.

---

### `train/`

* `train_ppo.py`
  Main PPO training script for humanoid imitation learning.

* `evaluate.py`
  Evaluates trained policies using metrics such as episode stability and pose accuracy.

* `visualise.py`
  Renders the learned policy within the MuJoCo simulation environment for qualitative inspection.

---

## Installation

Ensure Python 3.11+ is installed. Using a virtual environment or Conda environment is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Extract Pose Data

```bash
python preprocessing/extract_pose.py
```

### Smooth Motion

```bash
python preprocessing/smooth_motion.py
```

### Convert to Joint Angles

```bash
python preprocessing/convert_to_joint_angles.py
```

### Train PPO Policy

```bash
python train/train_ppo.py
```

### Evaluate Trained Policy

```bash
python train/evaluate.py
```

### Visualize Learned Motion

```bash
python train/visualise.py
```

---

## Results

* Learned stable humanoid walking behavior inside MuJoCo simulation.
* Reduced motion jitter using temporal smoothing techniques.
* Successfully integrated monocular video pose extraction with reinforcement learning-based imitation.

Future improvements may include:

* Multi-motion imitation
* Real-time pose retargeting
* Transformer-based motion priors
* Sim-to-real transfer

---

## Contributors

Developed collaboratively by:

* Prakhar Galriya
* Baibhaw Kumar
