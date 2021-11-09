from dqn import DQN
from config import config

agent = DQN(config)
agent.train()