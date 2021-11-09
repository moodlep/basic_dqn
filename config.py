import numpy as np
import torch

config = {
    'env': 'CartPole-v0',
    'total_timesteps': 150000,
    'replay_buffer_size': 5000,
    'minimum_replay_before_updates': 1000,
    'target_update_steps': 2000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.5,
    'epsilon': 0.4,
    'hidden': 128,
    'batch_size': 64,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'print_steps': 1000,
    'tensorboard_folder': '/Users/perusha/tensorboard/november_2021/'

}