# SAC-N with Q-Ensemble for Offline RL on JAX
**One-sentence Summary:**  
This project is a high-performance, single-file implementation of the SAC-N algorithm for offline reinforcement learning on JAX, featuring a Q-Ensemble and utilizing both Flax and Equinox frameworks to achieve substantial speedups compared to similar PyTorch implementations.

---

## Overview

This repository provides a streamlined, research-oriented implementation of Soft Actor-Critic with Q-Ensemble (SAC-N) for offline reinforcement learning using JAX. The code is written in a single file for each framework variant and demonstrates a compact, efficient, and easy-to-understand approach to offline RL. It leverages JAX’s just-in-time compilation and functional programming capabilities to optimize performance—reportedly achieving up to 10× speed improvements over comparable PyTorch implementations from CORL.

---

## Key Features

- **Offline RL with Q-Ensemble:**  
  Implements SAC-N using a Q-Ensemble, where multiple critics are maintained to better estimate the value function and stabilize learning in the offline setting.

- **Dual Framework Implementation:**  
  Two implementations are provided:
  - **Flax Variant:** `sac_n_jax_flax.py`
  - **Equinox Variant:** `sac_n_jax_eqx.py`  
  Both versions share the same underlying logic while catering to different user preferences.

- **Performance Optimization:**  
  The project emphasizes efficient computation by jitting not only individual network updates but also the entire epoch loop (using constructs such as `jax.lax.fori_loop` or `jax.lax.scan`). This holistic jitting strategy significantly speeds up the training process compared to updating only the networks.

- **Configurable and Flexible:**  
  Users can easily adjust hyperparameters such as environment name, number of critics, batch size, learning rates, and more via command-line arguments or YAML configuration files (using pyrallis).

- **Comprehensive RL Pipeline:**  
  Includes components for:
  - Building and updating actor and critic networks.
  - Maintaining a replay buffer sourced from D4RL datasets.
  - Applying gradient updates with Optax.
  - Logging training progress and evaluation metrics with Weights & Biases (wandb).

- **Research and Experimentation:**  
  Designed with extensibility in mind, making it suitable for experimentation with different network architectures, optimization techniques, and hyperparameter tuning. The codebase is clear and modular to support further enhancements and research exploration.
