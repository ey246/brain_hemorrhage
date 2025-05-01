# 🧠 Brain Hemorrhage Detection using Deep Q-Learning

This repository contains a Deep Reinforcement Learning (DQN) implementation for brain hemorrhage localization in CT scan images. The model uses a CNN-based feature extractor (VGG16) and is trained to iteratively move a bounding box toward the hemorrhage region using reward signals. Unlike traditional CNNs, this approach is well-suited for small datasets and provides interpretable predictions through visible agent trajectories and tunable reward structures.

---

## 📌 Project Overview

- **Goal**: Use DQN to localize brain hemorrhages in CT scans via reinforcement learning.
- **Approach**:
  - Extract features using a pretrained **VGG16** network.
  - Agent learns to move a bounding box toward hemorrhage regions using custom reward functions.
  - Rewards include binary correctness, IoU-based difference, coordinate, and aspect/area penalties.

---

## Directory
- `benchmark_unet.ipynb` contains the benchmark U-Net taken from [Kaggle](https://www.kaggle.com/code/ranjithkumarat/brain-stroke-detection-using-ct-images)
- `RL_eps1.py` contains the most up-to-date version of the DQN agent, it's a script that can be easily tuned to output a saved model along with loss and reward plots
- `evaluation.ipynb` contains methods used to easily evaluate the .keras model outputted by RL_eps1.py and visualize them
- `DQN-GPU-Debug.ipynb` contains the second-most up-to-date version of the DQN agent; an updated version of its non-debugged counterpart

Older versions:
- `DQN-GPU.ipynb` contains an updated version of DQN.ipynb compatible with GPUs
- `DQN.ipynb` contains an old version of the DQN before adding GPU-compatibility, has all functions within it such as visualization and evaluation code

## Dataset
- `hemorrhage_CT` contains all CTs with hemorrhages from the [Kaggle RSNA Intracranial Hemorrhage Detection Competition](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images/data)
- `mini_dataset` is a small (30 image) split off of hemorrhage_CT with 18 pre-split training and 12 testing images
