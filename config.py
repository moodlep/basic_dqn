import numpy as np
import torch

from frenv.envs.env_wrapper_v7 import FourRoomsEnvExtv7

config = {
    'env': 'FourRoomsEnvExtv7-v0',
    'total_timesteps': 300000,
    'replay_buffer_size': 5000,
    'minimum_replay_before_updates': 1000,
    'target_update_steps': 1000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.3,
    'hidden': 128,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'print_steps': 10000,
    'tensorboard_folder': '/Users/perusha/tensorboard/nov_2021_frenv/',
    'checkpointing_steps': 10000,
}

config_success_400episodes = {
    'env': 'FourRoomsEnvExtv7-v0',
    'total_timesteps': 300000,
    'replay_buffer_size': 5000,
    'minimum_replay_before_updates': 1000,
    'target_update_steps': 1000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'exploration_percentage': 0.3,
    'hidden': 128,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'print_steps': 10000,
    'tensorboard_folder': '/Users/perusha/tensorboard/nov_2021_frenv/',
    'checkpointing_steps': 10000,
}

sb3_config = {
    'replay_size': 5000,  # buffer size
    'replay_initial': 500,  # learning_starts
    'target_net_sync': 2000,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'batch_size': 64,
    'total_timesteps': 400000,  # 2000* max_no of episodes
    'exploration_fraction': 0.5,  # explore over half the total steps

}

config_cartpole_v0 = {
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