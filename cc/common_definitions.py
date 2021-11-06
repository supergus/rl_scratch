"""
Common definitions of variables that can be used across files
"""

from tensorflow.keras.initializers import glorot_normal
from pathlib import Path

# general parameters
CHECKPOINTS_PATH = Path.cwd() / "checkpoints/DDPG_"
TF_LOG_DIR = Path.cwd() / './logs/DDPG/'

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()
# KERNEL_INITIALIZER = tf.random_uniform_initializer(-1.5e-3, 1.5e-3)

# buffer parameters
UNBALANCE_P = 0.8  # newer entries are prioritized   # TODO: reduce a bit? might have "strange" seqs in recent states; pick up old good ones
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 200
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 2000  # orig 10000
# TOTAL_EPISODES = 3  # orig 10000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
WARM_UP = 20  # num of warm up epochs
MAX_TRAJECTORY_LEN = 20    # orig 2000?

# Other params
RENDER_ENV = False
LEARN = True
USE_NOISE = True
SAVE_WEIGHTS = True
EPS_GREEDY = 0.95
AVG_REW_WINDOW = 100   # Was 40. But we have many wonky sequences...
DTYPE = 'float32'
