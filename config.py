import numpy as np
import torch

config = {
    'env': 'CartPole-v0',
    'total_timesteps': 100000,
    'replay_buffer_size': 40000,
    'minimum_replay_before_updates': 1000,
    'target_update_steps': 1000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.5,
    'epsilon': 0.9,
    'hidden': 128,
    'batch_size': 8,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'print_steps': 1000

}