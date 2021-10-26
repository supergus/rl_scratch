"""
Buffer system for the RL
"""

import numpy as np

from common_definitions import BUFFER_UNBALANCE_GAP
import random
from collections import deque


class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the attributes.

        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [BUFFER_UNBALANCE_GAP / 2]
        return

    def append(self, prev_state, action, reward, state, done):
        """Appends SARSD tuple to the Buffer deque.

        Arguments:
            prev_state (dict): The state prior to action. Keys = 5 standard sequences.
            action (np.ndarray): The action. Shape = (number of controllable input signals, ).
            reward (np.ndarray): The reward after action. Shape = (1, ).
            state (dict): The next state after action. Keys = 5 standard sequences.
            done (bool): Done (whether one loop is done or not).

        Returns:
            No returns.
        """
        # Map to SARSD convenience variables
        s = prev_state
        a = action
        r = reward
        sn = state
        d = int(done)

        self.buffer.append([s, a, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])
        return

    def get_batch(self, unbalance_p=True):
        """Gets a randomly-selected batch from the buffer.

        Arguments:
            unbalance_p (bool): If True, recent events are prioritized, and have a higher probability of
                being drawn from the buffer.
                Default is True.

        Returns:
            The resulting batch, a list of SARSD tuples.
        """
        # Init
        probability_of_indices = None

        # Extend list with relative probabilities of drawing each index
        if random.random() < unbalance_p:
            self.p_indices.extend(
                (np.arange(len(self.buffer) - len(self.p_indices)) + 1) * BUFFER_UNBALANCE_GAP + self.p_indices[-1]
            )
            # Normalize the probabilities
            probability_of_indices = self.p_indices / np.sum(self.p_indices)

        # Randomly select target indices
        target_indices = np.random.choice(
            a=len(self.buffer),
            size=min(self.batch_size, len(self.buffer)),
            replace=False,
            p=probability_of_indices,
        )

        return [self.buffer[i] for i in target_indices]
