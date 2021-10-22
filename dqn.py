from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gym


class Replay:
    def __init__(self, buffer_size):
        # TODO: look into using a deque

        self.buffer_size = buffer_size
        # self.states = torch.zeros((buffer_size,env.observation_space.shape[0]), dtype=torch.float64)
        # self.next_states = torch.zeros((buffer_size,env.observation_space.shape[0]), dtype=torch.float64)
        # self.actions = torch.zeros((buffer_size,env.action_space.shape[0]), dtype=torch.int64)
        # self.rewards = torch.zeros((buffer_size,1), dtype=torch.float64)
        # self.dones = torch.zeros((buffer_size,1), dtype=torch.bool)

        self.states = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.buffer_size-1,batch_size)
        # TODO: extract these sampled indices from each matrix
        states = [self.states[i] for i in indices ]
        next_states = [self.next_states[i] for i in indices ]
        actions = [self.actions[i] for i in indices ]
        rewards = [self.rewards[i] for i in indices ]
        dones = [self.dones[i] for i in indices ]

        return states, next_states, actions, rewards, dones

    def insert_datapoint(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)



class DQN(nn.Module):

    def __init__(self, config):
        super(DQN).__init__()

        self.config = config

        # init the env
        self.env = gym.make(config['env'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # init the replay buffer
        self.buffer = Replay(self.config['replay_buffer_size'])

        # create the dqn and target networks
        network = nn.Sequential(
            nn.Linear(self.state_dim, config['hidden']),
            F.leaky_relu(),
            nn.Linear(config['hidden'], config['hidden']),
            F.leaky_relu(),
            nn.Linear(config['hidden'], self.action_dim)  # logits for q-values for each action
        )

    def forward(self, states):
        # TODO: convert states to tensors and check shape
        q_values = self.network(states)
        return q_values

    def select_action(self, state):
        # TODO: implement epsilon greedy
        action = self.env.action_space.sample()

        return action


    def train(self):
        total_steps = 0
        state = self.env.reset()

        while total_steps < self.config['total_timesteps']:

            # Collect data until the buffer has reached the minimum size
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.buffer.insert_datapoint(state, action, next_state, reward, done)




            # set state to next_state
            state = next_state






