from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gym


class Replay:
    def __init__(self, buffer_size):
        # TODO: look into using a deque vs matrices

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

class DQN_Network(nn.Module):
    def __init__(self, config):
        super(DQN_Network).__init__()

        # create the dqn and target network architecture
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, config['hidden']),
            F.leaky_relu(),
            nn.Linear(config['hidden'], config['hidden']),
            F.leaky_relu(),
            nn.Linear(config['hidden'], self.action_dim)  # logits for q-values for each action
        )

    def forward(self, input):
        # convert states to tensors and check shape
        states_t = torch.tensor(input, dtype=torch.float64)
        q_values = self.network(states_t)
        return q_values


class DQN():

    def __init__(self, config):
        super(DQN).__init__()

        self.config = config

        # init the env
        self.env = gym.make(config['env'])
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # init the replay buffer
        self.buffer = Replay(self.config['replay_buffer_size'])

        # Setup the DQN and Target networks
        self.dqn = DQN_Network(config)
        self.target = DQN_Network(config)
        # Sync target and dqn networks
        self.update_target_network()

        # setup the optimiser
        self.opt = torch.optim.Adam(lr=config['learning_rate'])

    def select_action(self, state):
        # implement epsilon greedy
        if np.random.random() < self.config['epsilon']:
            # Explore
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.int64)
        else:
            # Exploit
            with torch.no_grad():
                action = self.dqn(state).argmax(axis=1)  # n_batch x n_actions

        return action.item()

    def update_target_network(self ):
        # copy network params from DQN to Target NN
        self.target.load_state_dict(self.dqn.state_dict())

    def calculate_targets(self, next_states, rewards, dones):
        idones= [int(i) for i in dones]  # convert bool to int

        with torch.no_grad():
            targets = rewards + self.config['gamma'] * idones * (self.target(next_states).max) # TODO

        return targets

    def mse_loss(self, states, actions, targets, dones):

        # get q-values - forward predictions
        q_values = self.forward(states)
        q_values = torch.gather(q_values, 1, torch.tensor(actions, dtype=torch.int64))

        # MSE loss
        loss = F.mse_loss(q_values, targets)

        self.opt.zero_grad()
        # backwards??
        self.opt.step()

        return loss

    def train(self):
        total_steps = 0
        state = self.env.reset()

        # Create the dqn and target networks

        while total_steps < self.config['total_timesteps']:

            # Collect data until the buffer has reached the minimum size
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.buffer.insert_datapoint(state, action, next_state, reward, done)

            if len(self.buffer) < self.config['minimum_replay_before_updates']:
                continue

            if total_steps % self.config['target_update_steps'] ==0:
                self.update_target_network()

            # Update DQN:
            # 1. sample a batch from the buffer
            states, next_states, actions, rewards, dones = self.buffer.sample(self.config['batch_size'])

            # 2. calculate the target q-values
            targets = self.calculate_targets(next_states, rewards)

            # 3. Calculate MSE loss
            loss = self.mse_loss(states, actions, targets, dones)


            # set state to next_state
            state = next_state






