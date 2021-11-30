from dqn_frenv import DQN
from config import config
import torch
import numpy as np
import random

# Set seeds - list from cleanrl
seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = args.torch_deterministic
# env.seed(args.seed)
# env.action_space.seed(args.seed)
# env.observation_space.seed(args.seed)


agent = DQN(config)
agent.train()