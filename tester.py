from dqn_frenv import DQN
from config import config

agent = DQN(config)
agent.train()