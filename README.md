# Multi-Agent Q-Learning in a 2D Grid Environment

This repository contains a multi-agent reinforcement learning project based on **Q-learning**.  
The objective is to train multiple agents to reach a goal in a 2D grid environment while avoiding obstacles, other agents, and environment boundaries.

## Project Overview

- Environment: 2D grid world
- Learning method: Q-learning
- Multi-agent setting
- Agents learn through trial and error
- Dynamic difficulty: obstacle density increases based on agents’ success rate
- Visualization implemented using Pygame

The environment becomes more challenging when agents achieve a predefined success threshold, encouraging continuous learning and adaptation.

## Files Description

- `main.py` – Entry point of the simulation
- `agent.py` – Q-learning agent implementation
- `environment.py` – Grid environment and obstacle management
- `config.py` – Configuration parameters
- `utils.py` – Helper functions (logging and plotting)
- `rewards_plot.png` – Reward progression visualization
- `training_log_multi.csv` – Training performance logs

## Requirements

- Python 3.x
- pygame
- numpy
- matplotlib

## How to Run

Install the required libraries:
```bash
pip install pygame numpy matplotlib

