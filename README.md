# Deep Q-Network for Brain Hemorrhage Localization

A comparative study implementing Deep Q-Network (DQN) reinforcement learning and U-Net segmentation approaches for automated localization of brain hemorrhages in CT scans.

## Overview

This project explores the application of reinforcement learning to medical image analysis, specifically focusing on the automated detection and localization of intracranial hemorrhages. We compare a novel DQN-based approach with traditional U-Net segmentation to analyze trade-offs in data requirements, training stability, and model interpretability.

## Authors
- Eric Yang
- Suraj Godithi

## Key Features

- **DQN Agent**: 9-action bounding box manipulation (move, zoom, squash, trigger)
- **State Representation**: VGG16 feature extraction + action history encoding
- **Multiple Reward Functions**: IoU-based, Manhattan distance, coordinate-based feedback systems
- **Comparative Baseline**: U-Net segmentation model for performance comparison
- **Comprehensive Evaluation**: Systematic ablation studies across architectures and hyperparameters

## Technical Architecture

### Deep Q-Network Implementation
- **Feature Extraction**: Pre-trained VGG16 (fc1 layer, 4096-dim features)
- **Network Architecture**: Two hidden layers (1024 units each) with ReLU activation
- **Training**: Experience replay buffer, epsilon-greedy exploration, Adam optimizer
- **Action Space**: 9 discrete actions for bounding box manipulation

### U-Net Baseline
- Standard U-Net architecture for medical image segmentation
- Dice loss optimization with Jaccard coefficient evaluation
- Fine-tuned on hemorrhage-specific dataset

## Results & Key Findings

- **Data Efficiency**: DQN requires less labeled data but demands extensive computational resources and reward engineering
- **Training Stability**: U-Net shows more predictable convergence; DQN exhibits high variance in learning dynamics
- **Interpretability**: DQN provides observable decision trajectories; U-Net offers traditional segmentation masks
- **Performance Trade-offs**: "No free lunch" - choice between data curation (supervised) vs computational complexity (RL)

## Technical Report

For detailed methodology, experimental results, and analysis, see the full technical paper:
📄 **[Technical Report](docs/DQN_Brain_Hemorrhage_Localization.pdf)**

The paper includes:
- Comprehensive literature review and motivation
- Detailed experimental methodology
- Systematic ablation studies across reward functions
- Performance comparison and trade-off analysis
- Future work recommendations

## Installation & Usage

### Requirements
```bash
pip install tensorflow
pip install numpy pandas matplotlib
pip install opencv-python
pip install scikit-learn
```

### Running the Models

To run the models, you'll need to modify the hyperparameters directly in the Python files:

**DQN Training:**
- Edit the `.py` file to set epochs, batch_size, and reward_type parameters
- Common configurations used: epochs=50, batch_size=300, reward types include IoU-based, Manhattan distance, and coordinate-based
- Run: `python [dqn_training_file].py`

**U-Net Training:**
- Modify parameters in the U-Net script as needed
- Run: `python [unet_training_file].py`

**Note**: Check the individual Python files for specific parameter settings and file names.

## Dataset

[Dataset information to be updated based on actual dataset used - see paper for details]

## Hardware Requirements

- **Development Environment**: High-end GPU cluster with enterprise-grade hardware
- **Recommended**: High-end GPU with 16GB+ VRAM for efficient training  
- **Minimum**: CUDA-compatible GPU with 8GB+ VRAM
- Multi-GPU support implemented for large-scale experiments

**Note**: Training times can be extensive - DQN experiments often required days of computation even on high-end hardware.

## Key Insights for Production ML

1. **Reward Engineering is Critical**: Simple, interpretable reward functions outperformed complex geometric penalties
2. **Stability vs Flexibility**: Supervised methods offer stability; RL provides adaptability with higher computational cost
3. **Medical AI Considerations**: Interpretability and expert validation essential for healthcare applications

## Future Work

- Implementation of imitation learning with expert demonstrations
- Double Q-learning to mitigate overestimation bias
- Extension to 3D volumetric analysis
- Integration with clinical workflow systems

## Citation

If you use this work in your research, please cite:
```
Yang, E., & Godithi, S. (2025). DQN for Detecting Brain Hemorrhage: 
A Comparative Study of Reinforcement Learning and Supervised Learning 
Approaches in Medical Image Analysis.
```

---

*This project was developed as part of DS340 coursework, demonstrating advanced machine learning techniques in medical imaging applications.*
