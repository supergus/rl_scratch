"""
Main file
"""
import argparse
import logging
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import gym
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path

from utils import Tensorboard
from common_definitions import CHECKPOINTS_PATH, TOTAL_EPISODES, TF_LOG_DIR, UNBALANCE_P, MAX_TRAJECTORY_LEN
from model import Brain
from env_builder import Environment
from controls.core.experiments import ControlsExperimentPackage
from controls.core.models import CMPTaxonomy
from fun import fancy_print

# Convert DTYPE to Tensorflow dtype
DTYPE = 'float32'
TF_DTYPE = tf.dtypes.as_dtype(DTYPE)   # We cast tensors as we construct training examples

taxonomy = CMPTaxonomy()
taxonomy.nudge_mode = 'absolute'
required_config = dict(
    description='I like models',
    model_name='demo',
    model_type=1,
)
taxonomy.update(required_config)
options = {
    'model_tf_name': 'demo'
}


# RL_TASK = args.env
RENDER_ENV = False
LEARN = True
USE_NOISE = True
WARM_UP = True
SAVE_WEIGHTS = True
EPS_GREEDY = 0.95
AVG_REW_WINDOW = 40

# Create the gym environment
env = Environment()

# Testing: Manually create controls epk and LCP.
#  Normally we'd have a saved (and fully trained) LCP, and load directly into env. SPOOF it here.
tgt = r'D:\chris\Documents\Programming\other_peoples_repos\DDPG-tf2-master\DDPG-tf2-master\cc\2021-09-24--191345-UTC--Spartanburg--run-3882.lpp'
epk = ControlsExperimentPackage()
epk.load_lpp(tgt, password='Akira2019!')
# epk.autoconfigure_all(ctrl_inputs='RPM', alarm_percent=0.1)
epk.autoconfigure_all(ctrl_inputs='RPM', alarm_basis='iqr', alarm_percent=1.0)

# Register epk to environment
env.register_experiment(epk)

# TODO: add autoconfig
env.reward_configuration.step = 1
env.reward_configuration.error = -1
env.reward_configuration.l2_norm = -0.25

# Read out dims
action_space_high = env.action_space.high

# Build Brain w/models
brain = Brain(env, taxonomy)
# tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

# Load weights if available
# logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
# brain.load_weights(CHECKPOINTS_PATH)

# Define metrics
accumulated_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

# Define working lists to capture episode rewards and average over last few episodes
ep_reward_list = []
avg_reward_list = []

# Optional: Messages
env.set_verbosity(False)
# noinspection PyProtectedMember
brain._print_msg(f'Launching training...')
brain.set_verbosity(False)

# TODO: PRIORITIES: (4) mpk refactor incl baseline model; (5) Gym construction (relocate Brain)
# TODO: CHECK... are we running pipeline AFTER applying action? NO! but we MUST.... this is gonna slow things WAAAAAY down.... :=(

# TODO: exploration - keep to minimum - init act as 0-sentered normal with smallish spread - 0 == "avg setpoint" which will be CLOSE


# Train
rc = env.reward_configuration
msg = f'STRUCTURE: STEP {rc.step} / ERROR {rc.error} / L2_NORM {rc.l2_norm}, MAX TRAJ {MAX_TRAJECTORY_LEN}'
fancy_print(msg, fg='chartreuse', header=True)

for ep in range(TOTAL_EPISODES):

    prev_state = env.reset()
    accumulated_reward.reset_states()
    actions_squared.reset_states()
    Q_loss.reset_states()
    A_loss.reset_states()
    brain.noise.reset()

    msg = f'Episode {ep} from playhead {env.playhead.current_index}:'.ljust(40)
    color = 'yellow' if ep >= WARM_UP else 'light_red'
    fancy_print(msg, fg=color, end='')

    for i in range(MAX_TRAJECTORY_LEN):
        # TODO: parameterize max trajectory length (currently 2000)
        #  The other way to "end" a trajectory is if there is no "next seq" available...

        fancy_print(f'{i} ', fg='light_cerulean', end='')

        if RENDER_ENV:  # render the environment into GUI
            env.render()

        # Receive state and reward from environment.
        # TODO: Batch these and distribute the computations.
        #  E.g. get 64 actions at once? Compute 64 env steps at once? WHOLE DPK at once...???
        cur_act = brain.act(
            state=prev_state,
            not_random=(ep >= WARM_UP) and (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES),
            noise=USE_NOISE,
            action_space=env.action_space,
        )
        state, reward, done, _ = env.step(cur_act)
        brain.remember(prev_state, reward, state, done)

        # Update network weights and capture losses
        if LEARN:
            experience = brain.buffer.get_batch(unbalance_p=UNBALANCE_P)
            c, a = brain.learn(experience)
            Q_loss(c)
            A_loss(a)

        # Update metrics and prepare for next step
        accumulated_reward(reward)
        actions_squared(np.square(cur_act/action_space_high))
        prev_state = state

        if done:
            fancy_print(f'done ', fg='light_cerulean', end='')
            break

    fancy_print(f'/ r {round(float(accumulated_reward.result().numpy()), 3)}', fg='light_cerulean')

    ep_reward_list.append(accumulated_reward.result().numpy())
    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-AVG_REW_WINDOW:])
    avg_reward_list.append(avg_reward)

    # print the average reward
    # t.set_postfix(r=avg_reward)
    # tensorboard(ep, accumulated_reward, actions_squared, Q_loss, A_loss)

    # save weights
    if ep % 5 == 0 and SAVE_WEIGHTS:
        brain.save_weights(CHECKPOINTS_PATH)
        with open(Path(TF_LOG_DIR) / 'ep_rew.txt', 'w') as f:
            f.write(str(ep_reward_list)[1:-1])
        with open(Path(TF_LOG_DIR) / 'avg_ep_rew.txt', 'w') as f:
            f.write(str(avg_reward_list)[1:-1])

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
c = env.reward_configuration
plt.title(f'Reward: {c.step} / {c.error} / {c.l2_norm}, Max traj: {MAX_TRAJECTORY_LEN}')
plt.show()

# env.close()
brain.save_weights(CHECKPOINTS_PATH)

logging.info("Training done...")

