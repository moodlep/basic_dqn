from collections import deque

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gym
from torch.utils.tensorboard import SummaryWriter


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
        states_t = torch.tensor(input, dtype=torch.float)
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

        # Setup the DQN and Target networks and the syncing mechanism
        self.dqn = DQN_Network(config)
        self.target = DQN_Network(config)
        # Sync target and dqn networks
        self.update_target_network()

        # setup the optimiser: what's important here is attaching the parameters to the optimiser
        self.opt = torch.optim.Adam(self.dqn.parameters(),lr=config['learning_rate'])

        # Setup the tensorboard writer
        self.summary_writer = SummaryWriter(log_dir = '/Users/perusha/tensorboard/october_2021/', flush_secs=30)

    def select_action(self, state):
        # Input state: 1 x dim_state
        # Output action: 1 x dim_action
        # implement epsilon greedy
        if np.random.random() < self.config['epsilon']:
            # Explore
            action = torch.tensor(self.env.action_space.sample(), dtype=torch.int64)
        else:
            # Exploit
            with torch.no_grad():
                action = self.dqn(state).argmax(axis=1)  # n_batch (= 1) x n_actions

        return action.item()

    def update_target_network(self ):
        # copy network params from DQN to Target NN
        self.target.load_state_dict(self.dqn.state_dict())

    def calculate_targets(self, next_states, rewards, dones):
        idones= [int(i) for i in dones]  # convert bool to int

        with torch.no_grad():
            next_actions = self.target(next_states).argmax(axis=1)
            targets = rewards + self.config['gamma'] * idones * (self.target(torch.tensor(next_states,
                                                                                          dtype=torch.float)).max(1))

        return targets

    def get_batch_from_buffer(self):
        # sample a batch and return as tensors
        states, next_states, actions, rewards, dones = self.buffer.sample(self.config['batch_size'])
        states_t = torch.tensor(states, dtype=torch.float)
        next_states_t = torch.tensor(states, dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.int)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t = torch.tensor(dones, dtype=torch.bool)

        return states_t, actions_t, next_states_t, rewards_t, dones_t

    def mse_loss(self, states, actions, targets):

        # get q-values - forward predictions
        q_values = self.forward(states)
        q_values = torch.gather(q_values, 1, torch.tensor(actions, dtype=torch.int64))

        # MSE loss
        loss = F.mse_loss(q_values, targets)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def train(self):
        total_steps = 0
        episode_counter = 0
        state = self.env.reset()

        ep_rewards = []
        ep_len = 0

        # With a max step count limit, run episodes, collect data and update the DQN.
        # Outside loop counts max steps for DQN in total. Could also set this to a max episode count
        # When do we perform the update: once per step (not waiting for episode completion or anything)

        while total_steps < self.config['total_timesteps']:

            # Collect data until the buffer has reached the minimum size
            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            # collect data
            self.buffer.insert_datapoint(state, action, next_state, reward, done)
            ep_rewards.append(reward)
            ep_len +=1

            if len(self.buffer) < self.config['minimum_replay_before_updates']:
                continue

            # Update the target network from the dqn's latest weights:
            if total_steps % self.config['target_update_steps'] ==0:
                self.update_target_network()

            # If we have enough data in the replay buffer, start training a batch after every step is collected from
            # the end, i.e. update the DQN
            # 1. sample a batch from the buffer
            states, next_states, actions, rewards, dones = self.buffer.sample(self.config['batch_size'])

            # 2. calculate the target q-values
            targets = self.calculate_targets(next_states, rewards)

            # 3. Calculate MSE loss
            loss = self.mse_loss(states, actions, targets, dones)
            self.summary_writer.add_scalar("loss", loss, total_steps)

            # set state to next_state
            state = next_state

            # update step counter
            total_steps+=1

            # process episode
            if done:
                state = self.env.reset()
                self.summary_writer.add_scalar("episode_reward", sum(ep_rewards), episode_counter)
                self.summary_writer.add_scalar("episode_length", sum(ep_len), episode_counter)
                episode_counter +=1
                ep_rewards = []
                ep_len = 0






