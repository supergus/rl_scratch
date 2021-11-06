"""
Main file
"""
import argparse
import logging
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from controls import rl  # <--- want to set a seed for repeatability
from controls.core.experiments import ControlsExperimentPackage

# import gym
# from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
from pathlib import Path

from utils import Tensorboard
from common_definitions import CHECKPOINTS_PATH, TOTAL_EPISODES, TF_LOG_DIR, UNBALANCE_P, MAX_TRAJECTORY_LEN
from common_definitions import RENDER_ENV, LEARN, USE_NOISE, SAVE_WEIGHTS, EPS_GREEDY, AVG_REW_WINDOW, DTYPE, WARM_UP
from model import Brain
from env_builder import Environment
from fun import fancy_print, viz

rl.smash_old_project_files()

# Create the gym environment
env = Environment()

# Testing: Manually create controls epk
epk = ControlsExperimentPackage()
tgt = r'D:\chris\Documents\Programming\liveline_repos\ll_ml_physics_tf2_beta\physics\assets\sample_data\livonia_run140\livonia_run140_lstm.lpp'
# tgt = Path.cwd() / '2021-09-24--191345-UTC--Spartanburg--run-3882.lpp'
epk.load_lpp(tgt, password='Akira2019!')
# epk.autoconfigure_all(ctrl_inputs='RPM', alarm_basis='iqr', alarm_decimal=1.0)
epk.autoconfigure_all(ctrl_inputs='RPM', alarm_basis='iqr', alarm_decimal=1.33)
# epk.autoconfigure_all(ctrl_inputs='RPM', alarm_basis='iqr', alarm_decimal=1.5)   # <--- Can be TIGHTER than obs outputs during "imitate" phase; will give short trajectories

# # What do ctrl_inputs look like, un-nudged?
# dpk = epk.datapackages[0]
# dpk.transform_data(epk.lpp.experiment.pipeline, force_fit=True)
# dpk.plot_data('transformed')

# TODO: retrain physics examples, update sample_data in physics. remove from controls? redundant
# TODO: MovingAvg... do we need to enforce a 2x lookback? we are already handling edges...

# Register epk to environment
env.register_experiment(epk)

# Build Brain w/models
brain = Brain(env)

# Define Tensorflow metrics for Tensorboard (optional)
TF_DTYPE = tf.dtypes.as_dtype(DTYPE)   # We cast tensors as we construct training examples
accumulated_reward = tf.keras.metrics.Sum('reward', dtype=TF_DTYPE)
actions_squared = tf.keras.metrics.Mean('actions', dtype=TF_DTYPE)
Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=TF_DTYPE)
A_loss = tf.keras.metrics.Mean('A_loss', dtype=TF_DTYPE)

# Define working lists to capture episode rewards and averages over last few episodes
ep_reward_list = []
avg_reward_list = []

# Optional: Messages
env.set_verbosity(False)
brain._print_msg(f'Launching training...')
brain.set_verbosity(False)

# TODO: PRIORITIES:
#  (5) mpk refactor incl baseline model;
#  (6) Gym construction (relocate Brain)
#  (7) track dtype and ensure we're running all tf ops w float32

# TODO: exploration - keep to minimum - init act as 0-centered normal with smallish spread - 0 == "avg setpoint" which will be CLOSE

# Train
rc = env.reward_configuration
msg = f'STRUCTURE: STEP {rc.step} / ERROR {rc.error} / L2_NORM {rc.l2_norm}, MAX TRAJ {MAX_TRAJECTORY_LEN}'
fancy_print(msg, fg='chartreuse', header=True)
touches = 0

for ep in range(TOTAL_EPISODES):

    prev_state = env.reset()
    accumulated_reward.reset_states()
    actions_squared.reset_states()
    Q_loss.reset_states()
    A_loss.reset_states()
    brain.noise.reset()

    imitate = True if ep < 100 else False

    msg = f'Episode {ep} from playhead {env.playhead.current_index}:'.ljust(40)
    color = 'yellow' if (ep >= WARM_UP and not imitate) else 'light_red'
    fancy_print(msg, fg=color, end='')

    for i in range(MAX_TRAJECTORY_LEN):

        touches += 1

        # Console output
        fancy_print(f'{i} ', fg='light_cerulean', end='')
        if RENDER_ENV:  # render the environment into GUI
            env.render()

        # Receive state and reward from environment.
        # TODO: Batch these and distribute the computations.
        #  E.g. get 64 actions at once? Compute 64 env steps at once? WHOLE DPK at once...???
        c1 = ep >= WARM_UP
        c2 = random.random() < EPS_GREEDY + ((1-EPS_GREEDY) * (ep / TOTAL_EPISODES))
        random_exploration = not(c1 and c2)

        # TODO: Mesh imitate / random_exploration options. First imitate for n epochs, then explore for m?

        cur_act = brain.act(
            state=prev_state,
            random=random_exploration,
            noise=USE_NOISE,
            action_space=env.action_space,
            imitate=imitate,
            control_delay=env.control_delay,
        )
        state, reward, done, _ = env.step(cur_act)
        brain.remember(prev_state, reward, state, done)

        # Update network weights and capture losses
        if LEARN:
            # TODO: Is it really OK to "learn" on imitated actions? Actor network did not produce them...
            experience = brain.buffer.get_batch(unbalance_p=UNBALANCE_P)
            c, a = brain.learn(experience)
            Q_loss(c)
            A_loss(a)

        # Update metrics and prepare for next step
        accumulated_reward(reward)
        actions_squared(np.square(cur_act/env.action_space.high))
        prev_state = state

        if done:
            fancy_print(f'done ', fg='light_cerulean', end='')
            break

    fancy_print(f'/ r {round(float(accumulated_reward.result().numpy()), 3)}', fg='light_cerulean')

    ep_reward_list.append(accumulated_reward.result().numpy())
    avg_reward = np.mean(ep_reward_list[-AVG_REW_WINDOW:])
    avg_reward_list.append(avg_reward)

    # save weights
    if ep % 5 == 0 and SAVE_WEIGHTS:
        brain.save_weights(CHECKPOINTS_PATH)
        if not TF_LOG_DIR.exists():
            Path.mkdir(TF_LOG_DIR, parents=True)
        with open(TF_LOG_DIR / 'ep_rew.txt', 'w') as f:
            f.write(str(ep_reward_list)[1:-1])
        with open(TF_LOG_DIR / 'avg_ep_rew.txt', 'w') as f:
            f.write(str(avg_reward_list)[1:-1])

# Plot episodes versus avg rewards
r = env.reward_configuration
viz.line(avg_reward_list,
         suptitle='AVERAGE EPISODIC REWARD',
         title=f'step {r.step}   |   error {r.error}   |   l2_norm {r.l2_norm}   |   touches {touches}',
         ylabel=f'{AVG_REW_WINDOW}-Episode Moving Average',
         xlabel='Episode')
# TODO: Add detail to plot. alarm basis & setting (decimal, value, sigma)
# env.close()
brain.save_weights(CHECKPOINTS_PATH)

logging.info("Training done...")

