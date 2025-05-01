# 🧠 Brain Hemorrhage Detection using Deep Q-Learning

This repository contains a Deep Reinforcement Learning (DQN) implementation for brain hemorrhage localization in CT scan images. The model uses a CNN-based feature extractor (VGG16) and is trained to iteratively move a bounding box toward the hemorrhage region using reward signals.

---

## 📌 Project Overview

- **Goal**: Use DQN to localize brain hemorrhages in CT scans via reinforcement learning.
- **Approach**:
  - Extract features using a pretrained **VGG16** network.
  - Agent learns to move a bounding box toward hemorrhage regions using custom reward functions.
  - Rewards include binary correctness, IoU-based difference, coordinate, and aspect/area penalties.

---

