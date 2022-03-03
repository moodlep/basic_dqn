from collections import deque
import time, datetime,os
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.nn.functional import one_hot
import gym
from torch.utils.tensorboard import SummaryWriter
from config import config

from action_selection_functions import Action_selection

# Set seeds - list from cleanrl
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class Replay:
    def __init__(self, buffer_size):
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
        indices = np.random.randint(0, self.len()-1,batch_size)
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

    def len(self):
        return len(self.states)

class DQN_Network(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(DQN_Network, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # create the dqn and target network architecture
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, config['hidden']),
            nn.LeakyReLU(),
            nn.Linear(config['hidden'], config['hidden']),
            nn.LeakyReLU(),
            nn.Linear(config['hidden'], self.action_dim)  # logits for q-values for each action
        )

    def forward(self, input):
        # convert states to tensors and check shape
        states_t = torch.tensor(torch.nn.functional.one_hot(torch.tensor(input), self.state_dim), dtype=torch.float)
        # states_t = torch.tensor(input, dtype=torch.float)
        q_values = self.network(states_t)
        return q_values

    def stats(self):
        for param in self.network.parameters():
            print(param, param.grad)

class DQN():

    def __init__(self, config):
        super(DQN).__init__()

        self.config = config

        # init the env
        self.env = gym.make(config['env'], env_config={'task_config': None})
        self.env.render()
        self.action_dim = self.env.action_space.n
        if type(self.env.observation_space) is gym.spaces.discrete.Discrete:
            self.state_dim = self.env.observation_space.n
        else:
            self.state_dim = self.env.observation_space.shape[0]

        # init the replay buffer
        self.buffer = Replay(self.config['replay_buffer_size'])

        # Setup the DQN and Target networks and the syncing mechanism
        self.dqn = DQN_Network(self.state_dim, self.action_dim, config)
        self.target = DQN_Network(self.state_dim, self.action_dim, config)
        # Sync target and dqn networks
        self.update_target_network()

        # setup the optimiser: what's important here is attaching the parameters to the optimiser ;) Doh!
        self.opt = torch.optim.Adam(self.dqn.parameters(),lr=config['learning_rate'])

        # Setup the tensorboard writer
        self.setup_logging()
        # Adding graph is causing some problem in tensorboard and not showing any data... remove for now...
        # self.summary_writer.add_graph(self.dqn, [self.env.observation_space.sample()])
        # self.summary_writer.add_graph(self.dqn, torch.tensor(torch.nn.functional.one_hot(torch.tensor(
        #     self.env.observation_space.sample()) , self.state_dim), dtype=torch.float))

        self.epsilon_setup()

        # add biased action selection
        self.action_selection = Action_selection()

    def setup_logging(self):
        ts = time.time()
        self.timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')

        log_dir = self.config['tensorboard_folder']+self.timestamp
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.summary_writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def epsilon_setup(self):
        self.duration = self.config['total_timesteps'] * self.config['exploration_percentage']
        self.start_e = self.config['epsilon_start']
        self.end_e = self.config['epsilon_end']
        self.slope =  (self.end_e - self.start_e) / self.duration
        self.epsilon = self.start_e

    def epsilon_linear_schedule(self, t: int):
        # borrowed from cleanrl just cause it was neater...
        return max(self.slope * t + self.start_e, self.end_e)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            if self.config['biased_exploration']:
                action = torch.tensor(self.action_selection.biased_explore_flat(), dtype=torch.int64)
            else:
                action = torch.tensor(self.env.action_space.sample(), dtype=torch.int64)
        else:
            with torch.no_grad():
                action = self.dqn(state).argmax()  # n_batch (= 1) x n_actions

        return action.item()

    def update_target_network(self ):
        # copy network params from DQN to Target NN
        self.target.load_state_dict(self.dqn.state_dict())

    def calculate_targets(self, next_states, rewards, dones):
        idones= [int(not i) for i in dones]  # invert bool and convert to int
        idones_t = torch.tensor(idones, dtype=torch.int)

        with torch.no_grad():
            # next_action_values = self.target(next_states).max(1)[0]
            # next_states_t = torch.nn.functional.one_hot(torch.tensor(next_states), self.state_dim)
            next_action_values = self.target(next_states)
            targets = torch.tensor(rewards, dtype=torch.float) + self.config['gamma'] * idones_t * (
                torch.max(next_action_values,dim=1).values)

        return targets.unsqueeze(-1)

    def get_batch_from_buffer(self):
        # sample a batch and return as tensors
        states, next_states, actions, rewards, dones = self.buffer.sample(self.config['batch_size'])
        states_t = torch.tensor(states, dtype=torch.float)
        next_states_t = torch.tensor(states, dtype=torch.float)
        actions_t = torch.tensor(actions, dtype=torch.int)
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        dones_t = torch.tensor(dones, dtype=torch.bool)

        return states_t, actions_t, next_states_t, rewards_t, dones_t

    def log_network_stats(self):
        # capture the grad norm to tensorboard
        pass

    def mse_loss(self, states, actions, targets):
        # get q-values - forward predictions
        current_q_values = self.dqn.forward(states)

        # print("all q_values: ", current_q_values.detach().shape)
        # print("actions: ", len(actions), actions)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        q_values = torch.gather(current_q_values, -1, actions_t)
        # MSE loss
        loss = F.mse_loss(q_values, targets)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # perform multiple gradient updates during high exploration phase to make the most of the biased
        # exploration.
        if self.config['multi_grad_updates'] and self.epsilon > self.config['multi_grad_threshold']:
            self.opt.step()

        self.log_network_stats()

        return loss.item()

    def save(self, filename):
        torch.save(self.dqn.state_dict(), filename+"_"+str(self.timestamp))

    def train(self):
        total_steps = 0
        episode_counter = 0
        state = self.env.reset()

        ep_rewards = []
        ep_len = 0

        running_loss = 0.0

        # With a max step count limit, run episodes, collect data and update the DQN.
        # Outside loop counts max steps for DQN in total. Could also set this to a max episode count
        # When do we perform the update: once per step (not waiting for episode completion or anything)

        while total_steps < self.config['total_timesteps']:

            # Collect data until the buffer has reached the minimum size
            self.epsilon = self.epsilon_linear_schedule(total_steps)
            self.summary_writer.add_scalar("epsilon", self.epsilon, total_steps)

            action = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            # collect data
            self.buffer.insert_datapoint(state, action, next_state, reward, done)

            if self.buffer.len() < self.config['minimum_replay_before_updates']:
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
                continue

            # When do we start logging the rewards and lengths? After we have min buffer info?
            ep_rewards.append(reward)
            ep_len +=1

            # Update the target network from the dqn's latest weights:
            if total_steps % self.config['target_update_steps'] ==0:
                self.update_target_network()

            # If we have enough data in the replay buffer, start training a batch after every step is collected from
            # the end, i.e. update the DQN
            # 1. sample a batch from the buffer
            states, next_states, actions, rewards, dones = self.buffer.sample(self.config['batch_size'])

            # 2. calculate the target q-values
            targets = self.calculate_targets(next_states, rewards, dones)

            # 3. Calculate MSE loss
            loss = self.mse_loss(states, actions, targets)  # returns loss.item()
            running_loss += loss
            self.summary_writer.add_scalar("loss", loss, total_steps-self.config['minimum_replay_before_updates'])
            self.summary_writer.add_scalar("running_loss", running_loss, total_steps-self.config[
                'minimum_replay_before_updates'])

            # set state to next_state
            state = next_state

            # update step counter
            total_steps+=1

            # Print some stats to screen
            if total_steps % self.config['print_steps'] == 0:
                print("Loss at ", total_steps, " steps is ", loss)

            if total_steps % self.config['checkpointing_steps'] == 0:
                self.save("chkpts/chkpt_dqn_"+str(total_steps/self.config['checkpointing_steps']))

            # process episode
            if done:
                state = self.env.reset()
                self.summary_writer.add_scalar("episode_reward", sum(ep_rewards), episode_counter)
                self.summary_writer.add_scalar("episode_length", ep_len, episode_counter)
                episode_counter +=1
                ep_rewards = []
                ep_len = 0

            self.summary_writer.flush()

        self.summary_writer.close()
        self.save("chkpts/final_chkpt_dqn")


# agent = DQN(config)
# agent.train()

