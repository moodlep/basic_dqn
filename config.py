import numpy as np
import torch

config = {
    'env': 'CartPole-v0',
    'total_timesteps': 500000,
    'replay_buffer_size': 10000,
    'minimum_replay_before_updates': 1000,
    'target_update_steps': 5000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.3,
    # 'epsilon': 0.4,
    'hidden': 128,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'print_steps': 10000,
    'tensorboard_folder': '/Users/perusha/tensorboard/november_2021/',
    'checkpointing_steps': 10000,
}