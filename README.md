# RadDQN-E
RadDQN-E: An Energy-Efficient Exploration Algorithm for Radiated Environments

This repository contains the source code for the RadDQN-E algorithm, developed as part of the MSc dissertation at The University of Manchester. The algorithm extends the work of RadDQN by introducing an energy consumption model to optimise path planning under three conflicting objectives: mission time, radiation exposure, and energy efficiency.

Description:

The core of this project is a Deep Q-Network agent trained in a simulated grid-world environment with radiation sources and a finite energy budget. The key innovation is the integration of a kinematic-based energy penalty into the reward function, enabling the agent to learn dynamic, state-aware policies.

This repository includes:

The Python implementation of the RadDQN-E algorithm.

The environment simulation code.

Scripts to reproduce the experiments presented in the dissertation.

The baseline implementations of DQN and RadDQN used for comparison.

Prerequisites:

Python 3.8+

PyTorch

NumPy

Matplotlib

Install an Anaconda will be helpful

Clone:

git clone [https://github.com/](https://github.com/)[MyUsername]/[MyRepositoryName].git

Usage:

To run the main training script for the simple task scenario, execute the following command:

python main.py --config_file configs/[Choose A .yaml File] --logdir [Any folder you want to place the training results] --seed 3007

For more details on the different arguments and experiments, please refer to the dissertation.

Citation:

If you find this work useful in your research, please consider citing the dissertation:

Wentian Zhou

RadDQN-E: An Energy-Efficient Exploration Algorithm for Radiated Environments

License:

This project is licensed under the MIT License - see the LICENSE.md file for details.
