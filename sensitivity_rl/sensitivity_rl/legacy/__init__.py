import sys
import os

# 1. Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to sys.path
sys.path.append(parent_dir)

import envs as envs
import buffers as buffers
import networks as networks
import utils as utils