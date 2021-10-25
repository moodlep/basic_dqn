import numpy as np
import torch

config = {
    'env': 'cartpole-v2',
    'total_timesteps': 100000,
    'replay_buffer_size': 40000,
    'minimum_replay_before_updates': 10000,
    'target_update_steps': 5000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.5,
    'hidden': 128,
    'batch_size': 64,
    'learning_rate': 0.001,
    'gamma': 0.99

}