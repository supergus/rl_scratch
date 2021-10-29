import pandas as pd
import numpy as np
import bisect
import gc

from controls.core.lcp import LivelineControlsPackage
from controls.core.experiments import ControlsExperimentPackage

from physics.core.base import VerboseObjectABC, VersionedObjectABC
from physics.core.lpp import LivelinePhysicsPackage
from pathlib import Path
from physics.utils import misc as misc_physics
from physics.core import validation


ACTION_SPACE_HIGH = 1
ACTION_SPACE_LOW = -1
OBSERVATION_SPACE_HIGH = np.inf
OBSERVATION_SPACE_LOW = np.NINF
DTYPE = 'float32'

ENV_MSG_COLOR = 'yellow'
ENV_WARN_COLOR = 'light_red'


class Environment(VerboseObjectABC):

    """
    Inspired by the Environment class in OpenAI's Gym.
    Adheres to the Gym API, but does not fully replicate all Gym functionality.
    See: https://gym.openai.com/
    """

    def __init__(self, nickname=None):
        super().__init__(msg_color=ENV_MSG_COLOR, warn_color=ENV_WARN_COLOR, name='environment builder')
        self.experiment = None
        self.action_space = Space()
        self.current_shard = None  # A Shard object from the epk.shard_server.
        self.msg = None   # Stores tmp messages that can be used in self._construct_info()
        self.nickname = 'Generic Environment' if nickname is None else nickname
        self.observation_space = Space()
        self.playhead = Playhead()
        self.reward_configuration = RewardConfiguration()
        self.state = None  # A tuple with (sample index, ctrl_input values)

        # Hidden class definitions
        self._action_space_item_class = Space
        self._experiment_item_class = ControlsExperimentPackage
        self._observation_space_item_class = Space
        self._reward_configuration_item_class = RewardConfiguration

        return

    def __repr__(self):
        return f'RL environment patterned after Open AI Gym'

    def close(self):
        self.__init__()
        gc.collect()
        return

    def load_experiment(self, target):
        """Loads a ControlsExperimentPackage from target file and registers to the Environment.

        Arguments:
            target (str, Path): A Path-like object describing an absolute path to the saved file.
                Also accepts a *ControlsExperimentPackage* object, which will be auto-upgraded if needed.

        Returns
            No returns.
        """
        epk = ControlsExperimentPackage()
        epk.load(target)
        self.register_experiment(epk)
        return

    # noinspection PyProtectedMember
    def register_experiment(self, epk):
        """Registers a ControlsExperimentPackage to the Environment.

        This is equivalent to *env.make()* in Open AI Gym.

        **NOTES:**
            - *ControlsExperimentPackage* must contain trained models for controls and physics (LPP).

        Arguments:
            epk (ControlsExperimentPackage): A valid *ControlsExperimentPackage* object.

        Returns:
            No returns.

        Raises:
            TypeError: If LCP is not a *ControlsExperimentPackage*.
            RuntimeError: If *ControlsExperimentPackage* is not ready to train
            RuntimeError: If LPP (inside the *ControlsExperimentPackage*) is not ready to run
        """
        self.__init__()

        self._validate_epk_registration(epk)

        # Action space: Default values
        # TODO: Get high/low values using median values +/- ACTION_SIGMA, SCALED, from LPP DataPackage?
        #  Or... from controller limits as in LCP...? NOTE: ctrl limits are currently +/- infinity...
        #  Check. if not inf, use. if inf, use median +/- (or min/max?)
        num_sigs = len(epk.signal_selections.post_pipeline.ctrl_inputs)
        high = self.action_space.build_values(num_sigs, ACTION_SPACE_HIGH)
        low = self.action_space.build_values(num_sigs, ACTION_SPACE_LOW)
        self.action_space.high = high
        self.action_space.low = low

        # Observation space: Default values
        # NOTE: Values are not important, only the dimensionality
        num_sigs = len(epk.signal_selections.post_pipeline.outputs)
        high = self.observation_space.build_values(num_sigs, OBSERVATION_SPACE_HIGH)
        low = self.observation_space.build_values(num_sigs, OBSERVATION_SPACE_LOW)
        self.observation_space.high = high
        self.observation_space.low = low

        # Reward configuration
        self.reward_configuration.step = epk.modelpackage.taxonomy.reward_step
        self.reward_configuration.error = epk.modelpackage.taxonomy.reward_error
        self.reward_configuration.l2_norm = epk.modelpackage.taxonomy.reward_l2_norm

        # Assign
        self.experiment = epk

        # Initialize Playhead
        self.experiment._prep_sequences()   # Builds shards using raw (not transformed) data
        self.playhead.initialize(self.experiment)

        # Safety check.
        # Environment uses Numpy, but need to ensure signal order matches LPP predictions.
        self._validate_lpp_pred_signal_order_vs_lcp_config()

        return

    def render(self):
        # Not required
        raise NotImplementedError(f'FUTURE: May plot responses, etc.')

    def reset(self):
        """Resets environment and returns a preliminary observation of the state.

        The state observation is a dict consisting of four sequences:
            - 'ctrl_inputs': Controllable inputs
            - 'unctrl_inputs': Uncontrollable inputs
            - 'obs_outputs': Observed outputs
            - 'nudged_outputs': Observed outputs with nudges applied to controllable inputs
        """

        # Must return a value within OBSERVATION SPACE.
        # It "restarts" the environment.
        # Let this be a sequence starting at a particular timestamp from DataPackage.

        # NOTE: Could get "unlucky" and get a seq near the end of the available data.
        # In that case, the trajectory may terminate very quickly.
        # On average, this should not have a major impact...
        feature_df = self.playhead.random()

        # Transform
        feature_df = self.experiment.lpp.experiment.pipeline.run(feature_df)

        # Predict the output sequence without any nudges
        pred_outputs = self.experiment.lpp.run(
            feature_df,
            data_is_transformed=True,
            descale=False,
            check_prerequisites=False,
        )

        # NOTE:
        # Why use 'pred_outputs' instead of simply using 'obs_outputs'?
        # Because the predicted outputs w/nudges contain LPP model bias, while the observed output data do not.
        # This effectively gives us two separate universes of outputs.
        # The LCP model, while training, would "chase" the LPP model bias.
        # Instead, we want to train an LCP that give the best control policy for a given LPP model.
        # To do that, we need 'pred_outputs' vs 'pred_outputs_nudged'.

        ctrl_inputs, unctrl_inputs, obs_outputs = self._decompose_feature_df(feature_df)
        state = self._assemble_state(
            ctrl_inputs, unctrl_inputs, obs_outputs, pred_outputs
        )

        return state

    def seed(self):
        # Sets random seeds
        raise NotImplementedError()

    def step(self, action, n=1):
        """Takes a step in the environment using an action, and returns a 4-tuple.

        The 4-tuple follows the Open AI Gym API, and consists of:
            - state observation (dict, see below)
            - reward (float)
            - done (boolean)
            - info (string)

        The state observation is a dict consisting of five sequences:
            - 'ctrl_inputs': Controllable inputs
            - 'unctrl_inputs': Uncontrollable inputs
            - 'obs_outputs': Observed outputs
            - 'pred_outputs': Predicted outputs without nudges
            - 'pred_nudged_outputs': Predicted outputs with nudges applied to controllable inputs
        """

        feature_df = self.playhead.next(n)
        ctrl_inputs = self._get_ctrl_input_df_from_feature_df(feature_df)
        action_df = self._build_action_df(action, ctrl_inputs)
        # TODO: Change sig. pass feature_df, get ctrl_inputs inside. And action samples should be equal to all feature_df, not just input subseq

        # TODO: Scale and translate action.
        #  self.current_action = (self.current_action * self.action_space.scale) + self.action_space.translate

        # Update dfs with action: ctrl_inputs (for return in state) and feature_df (for feeding lpp model)
        ctrl_inputs.update(action_df)
        feature_df.update(action_df)
        # TODO: WHY do we not have new vals for RPMS here?? We do, but ONLY for input sub-seq... a PROBLEM.
        #  What does pipeline expect? what does physics model need? overwrite ALL samples w action?

        # Run pipeline on updated feature_df
        feature_df = self.experiment.lpp.experiment.pipeline.run(feature_df)

        # Predict the output sequence, replacing 'ctrl_inputs' with nudged values (the "action")
        pred_outputs_nudged = self.experiment.lpp.run(
            feature_df,
            data_is_transformed=True,
            descale=False,
            check_prerequisites=False,
        )

        # NOTE:
        # By "state" we mean the time-evolutionary dynamics of the system.
        # Observed dynamics are interpreted as an "encoding" of hidden physical state variables.
        ctrl_inputs, unctrl_inputs, obs_outputs = self._decompose_feature_df(feature_df)
        state = self._assemble_state(
            ctrl_inputs, unctrl_inputs, obs_outputs, pred_outputs_nudged
        )

        reward = self._compute_reward(state, action)
        done = self._compute_done(state, action)
        info = self._compute_info(state, action)

        return state, reward, done, info

    @staticmethod
    def _assemble_state(ctrl_inputs, unctrl_inputs, obs_outputs, pred_outputs):
        """Returns a dict containing the 4 standard state elements.

        Dict keys must superset the signature expected by the actor & critic models.
        """
        state = dict(
            observed_ctrl_inputs=np.expand_dims(np.array(ctrl_inputs), axis=0),
            observed_unctrl_inputs=np.expand_dims(np.array(unctrl_inputs), axis=0),
            observed_outputs=np.expand_dims(np.array(obs_outputs), axis=0),
            predicted_outputs=np.expand_dims(np.array(pred_outputs), axis=0),
        )
        return state

    def _compute_reward(self, state, action):
        """Returns a float as scalar reward for a given state and action."""

        # "Participation trophy" for surviving to take another step, without the
        # trajectory triggering "done".
        step_val = 1

        # Prediction accuracy: Normalized RMSE
        c = self.experiment.configuration
        pred = state['predicted_outputs'].squeeze()
        tgt = np.tile(np.array(list(c.output_targets.values())), (pred.shape[0], 1))  # shape = (seq_len, signals)
        iqr = np.array(list(c.output_interquartile_ranges.values()))   # shape = (signals)
        err = tgt - pred
        nrmse_signalwise = np.sqrt(np.mean(np.square(err), axis=0)) / iqr   # shape = (signals)
        nrmse_total = np.sqrt(np.mean(np.square(nrmse_signalwise)))

        # Action L2 norm
        # TODO: Reward structure assumes "zero" is the preferred action... generalize. Need a config setting?
        action_level = action - np.zeros(action.shape)
        action_l2_norm = np.sqrt(np.sum(np.square(np.abs(action_level))))

        r = self.reward_configuration.compute(
            step=step_val,
            error=nrmse_total,
            l2_norm=action_l2_norm,
        )

        return r

    def _compute_done(self, state, action):
        """Returns a boolean denoting whether the trajectory is done or not."""

        done, m = False, str()

        # Get values
        c = self.experiment.configuration
        lims = np.array(list(c.output_alarm_limits.values()))  # shape = (signals, 2 (low, high))
        lims_max = lims[:, 1]
        lims_min = lims[:, 0]
        preds = state['predicted_outputs'].squeeze()
        preds_max = preds.max(axis=0)
        preds_min = preds.min(axis=0)

        # Done if ANY prediction in the output sequence exceeds limits, for ANY signal.
        # FUTURE: Could "discount" predictions deep in the sequence, prioritize near-term predictions?
        if np.any(preds_max > lims_max):
            done = True
            m += f'Predicted outputs exceeded upper alarm limits\n'
        if np.any(preds_min < lims_min):
            done = True
            m += f'Predicted outputs exceeded lower alarm limits\n'

        if len(m) > 0:
            self.msg = m if self.msg is None else self.msg + m

        return done

    def _compute_info(self, state, action):
        """Returns a string with (possibly!) useful info."""
        # m = f'Env: {self.nickname}, state elements: {len(state)}\n'
        m = str()
        final_msg = m + self.msg if isinstance(self.msg, str) else m
        # Reinitialize self.msg
        self.msg = None
        return final_msg

    @staticmethod
    def _build_action_df(action, ctrl_inputs_df):
        """Converts action to a DataFrame.

         The resulting DataFrame will have:
            - Columns = Controllable input signal names
            - Data = Constant action value for each signal
            - Index = Pandas Datetime as per *ctrl_inputs_df*

        Arguments:
            action (object): A Pandas DataFrame or array-like object.
              Should have outer dimension = number of controllable input signals.
              If inner dimension is less than *input_seq_len*, the index from *ctrl_inputs_df* will be used.
            ctrl_inputs_df (pd.DataFrame): A Pandas DataFrame with controllable input signal data.
              Dimension should be: *(input_seq_len, number of controllable input signals)*.
         """

        action_seq = None

        if isinstance(action, pd.DataFrame):
            if action.shape == ctrl_inputs_df.shape:
                if set(action.columns.tolist()) == set(ctrl_inputs_df.columns.tolist()):
                    # Force index to match
                    action.index = ctrl_inputs_df.index
                    action_seq = action

        if action_seq is None:
            data = ctrl_inputs_df.copy().values
            columns = ctrl_inputs_df.columns
            index = ctrl_inputs_df.index
            for i, c in enumerate(columns):
                data[:, i] = action[i]
            action_seq = pd.DataFrame(data=data, columns=columns, index=index)

        return action_seq

    def _build_unnudged_responses(self):
        """Builds TFRecords using un-nudged responses.

        Can be used to accelerate RL training by pre-computing these responses in batch mode.
        """
        # noinspection PyProtectedMember
        self.lcp.experiment._build_unnudged_responses()
        return

    def _decompose_feature_df(self, feature_df):
        """Separates feature_df into 3 dfs: ctrl_inputs, unctrl_inputs, and outputs.
        Assumed feature_df is one sequence from the Playhead; we will harvest in & out sub-seqs from the "end".
        """
        ss_post = self.experiment.signal_selections.post_pipeline
        t = self.experiment.lpp.experiment.modelpackage.taxonomy
        in_ts = feature_df.index[-(t.input_seq_len + t.output_seq_len): -t.output_seq_len]
        out_ts = feature_df.index[-t.output_seq_len:]
        ctrl_inputs = feature_df.loc[in_ts, ss_post.ctrl_inputs]
        unctrl_inputs = feature_df.loc[in_ts, ss_post.unctrl_inputs]
        obs_outputs = feature_df.loc[out_ts, ss_post.outputs]
        return ctrl_inputs, unctrl_inputs, obs_outputs

    def _get_ctrl_input_df_from_feature_df(self, feature_df):
        """Returns DataFrame with ctrl_input signals.
        Assumed feature_df is one sequence from the Playhead; we will harvest in & out sub-seqs from the "end".
        """
        ss_post = self.experiment.signal_selections.post_pipeline
        t = self.experiment.lpp.experiment.modelpackage.taxonomy
        in_ts = feature_df.index[-(t.input_seq_len + t.output_seq_len): -t.output_seq_len]
        ctrl_inputs = feature_df.loc[in_ts, ss_post.ctrl_inputs]
        return ctrl_inputs

    @staticmethod
    def _validate_epk_registration(epk):
        """Raises exception if ControlsExperimentPackage does not contain all necessary elements."""
        if not isinstance(epk, ControlsExperimentPackage):
            msg = f'Expected a ControlsExperimentPackage but got type: {type(epk)}'
            raise TypeError(msg)
        if not epk.is_ready_to_train():
            msg = f'ControlsExperimentPackage is not ready to train; check warnings in console for details'
            raise RuntimeError(msg)
        return

    def _validate_lpp_pred_signal_order_vs_lcp_config(self):
        """Raises a RuntimeError if signal order in LPP physics predictions does not match LCP configuration."""

        og_verbosity = self.verbose
        self.set_verbosity(False)

        feature_df = self.playhead.current()
        pred_outputs_df = self.experiment.lpp.run(
            feature_df,
            data_is_transformed=False,
            descale=False,
            check_prerequisites=False,
        )
        pred_sigs_list = list(pred_outputs_df)

        # Get output sig list from config, and check that it's internally consistent
        config_sigs_list = list(self.experiment.configuration.output_targets)
        for attr_name in ('output_alarm_limits', 'output_interquartile_ranges'):
            attr_dict = getattr(self.experiment.configuration, attr_name)
            attr_sig_list = list(attr_dict)
            if not misc_physics.check_list_equality(config_sigs_list, attr_sig_list):
                msg = f'Could not resolve output values in LCP configuration; inconsistent signal names'
                raise RuntimeError(msg)

        # Compare
        if not misc_physics.check_list_equality(pred_sigs_list, config_sigs_list):
            msg = f'Output signal order in LPP physics predictions does not match LCP configuration'
            raise RuntimeError(msg)

        # Restore verbosity
        self.set_verbosity(og_verbosity)
        return


class Space(VerboseObjectABC):

    def __init__(self):
        super().__init__(msg_color='yellow', warn_color='red', name='action space')
        self.high = None
        self.low = None

    def __repr__(self):
        n_high, n_low = None, None
        if self.high is not None:
            flat_array = self.high.flatten()
            n_high = len(flat_array)
        if self.low is not None:
            flat_array = self.high.flatten()
            n_low = len(flat_array)
        if n_high != n_low:
            msg = f'Dimensionality is not consistent between high and low attributes'
            raise RuntimeError(msg)
        n = n_high if n_high is not None else 'n'
        return f'Multidimensional real space (R{n})'

    @property
    def dimensions(self):
        try:
            self._validate_consistency()   # Requires values to be set (not None)
            return self.high.flatten().shape[0]
        except RuntimeError:
            return None

    @property
    def high(self):
        return self._high

    @high.setter
    def high(self, values):
        self._high = self._parse_values(values, 'high')
        return

    @high.deleter
    def high(self):
        self._high = None
        return

    @property
    def low(self):
        return self._low

    @low.setter
    def low(self, values):
        self._low = self._parse_values(values, 'low')
        return

    @low.deleter
    def low(self):
        self._low = None
        return

    @staticmethod
    def build_values(num_signals, values):
        """Returns an array of space limit values.

        Arguments:
            num_signals (int): Number of signals, a.k.a. dimensionality of space.
            values (object): A numeric value or array-like object of numeric values.
                If it's a single numeric value, it will be applied to all signals / dimensions.
                If it's an array, the number of elements must equal *num_signals*.

        Returns:
            Numpy array with parsed values.
        """

        num_signals = int(num_signals)

        if values is None:
            return None

        try:
            values = np.array([values], dtype=DTYPE).flatten()
        except:
            raise RuntimeError(f'Unable to convert argument of type {type(values)} to Numpy array')

        if values.shape[0] == 0:
            # Got no values
            values = None

        elif values.shape[0] == 1:
            # Got a single numeric value; apply to all signals
            values = np.full(num_signals, values)

        else:
            # Got an array of values
            if values.shape[0] != num_signals:
                msg = f'Dimensionality of values ({values.shape}) does not match num_signals ({num_signals})'
                raise ValueError(msg)

        return values

    def random(self):
        """Returns a Numpy array with random values for each dimension.

        Values are drawn from a uniform distribution with low / high values based on the space's attributes.
        """
        return np.random.uniform(self.low, self.high, self.dimensions)

    def _enforce_consistency_with_other(self, this_value, this_name):
        """Deletes values in "that" limit type if inconsistent with "this" one. Permits None values."""

        if this_name == 'high':
            that_name = 'low'
        elif this_name == 'low':
            that_name = 'high'
        else:
            msg = f'Could not parse \'this_name\'; ' \
                  f'expected one of [\'high\', \'low\'] but got: \'{this_name}\''
            raise ValueError(msg)

        that_value = getattr(self, that_name)

        different = False
        if this_value is not None and that_value is not None:
            if this_value.shape != that_value.shape:
                different = True

        if different:
            msg = f'Dimensionality of {this_name} values does not match current {that_name} values. ' \
                  f'Clearing {that_name} values. Set them again.'
            self._print_warning(msg)
            setattr(self, that_name, None)

        return

    def _parse_values(self, values, this_name):
        """Returns an array of values."""

        if values is None:
            return None

        try:
            values = np.array([values], dtype=DTYPE).flatten()
        except:
            raise RuntimeError(f'Unable to convert argument of type {type(values)} to Numpy array')

        if values.shape[0] == 0:
            # Got no values
            values = None

        elif values.shape[0] == 1:
            # Got a single numeric value
            pass

        else:
            # Got an array of values
            pass

        self._enforce_consistency_with_other(values, this_name)
        return values

    def _validate_consistency(self):
        """Raises exception if high and low do not share same dimensionality, or if either is None."""

        if self.high is None:
            msg = f'High attribute has not been set'
            raise RuntimeError(msg)

        if self.low is None:
            msg = f'Low attribute has not been set'
            raise RuntimeError(msg)

        d_high = self.high.flatten().shape[0]
        d_low = self.low.flatten().shape[0]

        if d_high != d_low:
            msg = f'Dimensionality is not consistent between high and low attributes'
            raise RuntimeError(msg)

        return

# TODO: Default action limits should be +/- 3 sigma, scaled?


class Playhead(VerboseObjectABC):

    current_index = validation.ValidateNumber(minvalue=0, allow_none=True)
    directory = validation.ValidateDict(allow_none=True)
    index_low = validation.ValidateNumber(minvalue=0, allow_none=True)
    index_high = validation.ValidateNumber(minvalue=1, allow_none=True)

    def __init__(self):
        super().__init__(msg_color='yellow', warn_color='red', name='playhead')
        self.current_index = None
        self.current_shard = None   # A Shard object
        self.directory = None
        self.index_low = None
        self.index_high = None
        self.shard_server = None   # Assign epk.shard_server on self.initialize(epk)

    def __repr__(self):
        return f'Controls playback of DataPackage contents'

    def current(self):
        """Returns the DataFrame starting on the current sequence index."""
        self._swap_shard()
        return self.current_shard.seq_dfs[self.current_index]

    def goto(self, n):
        """Returns the sequence with nth index.

        Wraps around to first sequence if all sequences exhausted.

        Arguments:
            n (int): Index of desired sequence.

        Returns:
            A Sequence object.
        """
        self.current_index = self._wrap(n)
        return self.current()

    def initialize(self, epk):
        """Initializes Playhead using a ControlsExperimentPackage."

        Arguments:
            epk (ControlsExperimentPackage): A valid *ControlsExperimentPackage* object.

        Returns:
            No returns.

        Raises:
            TypeError: If epk is not a *ControlsExperimentPackage*.
            RuntimeError: If no *DataPackages* registered to epk, or to its embedded LPP (physics experiment).
        """

        # Validation of epk and its LPP is done when registering epk to the environment.
        # We don't validate again here, as Playhead.initialize() is called by the environment after val.

        if not epk.sequence_directory.splits_available():
            msg = f'Splits are not available in CEPSequenceDirectory'
            raise RuntimeError(msg)

        # Build dictionary of sequences using TRAIN split.
        # Test and validation strategy: Run `epk.simulate()` using additional DataPackages.
        self.directory = {seq.index: seq for seq in epk.sequence_directory.train}
        self.index_low = min(self.directory)
        self.index_high = max(self.directory)

        # Validate
        if self.index_high <= self.index_low:
            msg = f'Highest sequence index ({self.index_high}) is less than or ' \
                  f'equal to lowest sequence index ({self.index_low})'
            raise ValueError(msg)

        # # Set 'current_index' to random value
        # _ = self.random()

        # Assign shard server and onboard first shard;
        # We are exploiting temporal order from sequence_directory / shard_server
        self.shard_server = epk.shard_server
        self.shard_server.reset()

        # Set 'current_index' to FIRST value
        self.current_index = self.index_low
        self._swap_shard()

        return

    def next(self, n=1):
        """Returns the sequence on the nth index after current sequence index.

        Wraps around to first sequence if all sequences exhausted.

        Arguments:
            n (int): Number of sequence indices to increment.

        Returns:
            A Sequence object.
        """
        self.current_index = self._wrap(self.current_index + n)
        return self.current()

    def random(self):
        """Returns the sequence starting on a randomly selected index."""
        self._print_msg('Randomizing playhead location')
        # `np.random.randint()` is INCLUSIVE of interval start, EXCLUSIVE of interval end.
        self.current_index = np.random.randint(low=self.index_low, high=self.index_high)
        return self.current()

    def _swap_shard(self):
        """Offloads current shard and onboards necessary shard according to targeted sequence index."""

        split = 'train'

        # Ensure we have a shard server
        if self.shard_server is None:
            msg = f'Shard server not available in Playhead'
            raise RuntimeError(msg)

        # Ensure we have a shard
        if self.current_shard is None:
            self.shard_server.reset()
            self.current_shard = self.shard_server.next(processed=False, split=split)
            if self.current_shard is None:
                msg = f'Unable to get next shard'
                raise RuntimeError(msg)

        if self.current_index is not None:
            if self.current_index in self.current_shard.seq_ids:
                # We already have the shard that contains our targeted seq index
                return
            else:
                # Need to find the correct shard
                bingo_shard = None
                split_dict = getattr(self.shard_server.directory, split)
                for shard_idx, shard in split_dict.items():
                    if shard.uuid != self.current_shard.uuid:

                        # Use bisect to check where we'd insert into shard.seq_ids list, if inserting (we're not)
                        i = bisect.bisect_left(shard.seq_ids, self.current_index)

                        # Bingo if this is a valid (existing) index
                        if i != len(shard.seq_ids) and shard.seq_ids[i] == self.current_index:
                            bingo_shard = shard
                            bingo_shard.onboard()
                            break

                        # if self.current_index in shard.seq_ids:
                        #     bingo_shard = shard
                        #     bingo_shard.onboard()
                        #     break

                if bingo_shard is None:
                    msg = f'Unable to locate shard with targeted sequence ID ({self.current_index}) ' \
                          f'in {split} split'
                    raise RuntimeError(msg)
                else:
                    self.current_shard = bingo_shard

        # Else there is no 'self.current_index', so return silently

        return

    def _wrap(self, n):
        """Given a proposed sequence index n, wraps on low / high indices as needed."""

        n = int(n)

        #  High wrap
        if n > self.index_high:
            n = (n % self.index_high) + self.index_low

        # Low wrap
        if n < self.index_low:
            offset = abs(self.index_low - n)
            delta = self.index_high - self.index_low
            if offset >= delta:
                offset = offset % delta
            n = self.index_high - offset

        return n


class RewardConfiguration(VerboseObjectABC):

    def __init__(self):
        super().__init__(msg_color=ENV_MSG_COLOR, warn_color=ENV_WARN_COLOR, name='environment builder')

        # These attributes will contain weights
        self.error = None
        self.l2_norm = None
        self.step = None

        # NOTE:
        # If you add any attributes, be sure to update list 'include' in self.compute().

    def compute(self, **kwargs):
        """Returns reward value.

        Keyword Arguments:
            accuracy (float): Score for accuracy of outputs attained.
            l2_norm (float): L2 norm statistic. For example, the "amount" of action.
            step (float): Score for completing another step in the environment.

        Returns:
            Float.
        """
        include = ['error', 'l2_norm', 'step']
        r = 0
        for k, v in self.__dict__.items():
            if k in include:
                v = 0 if v is None else v
                r += kwargs.get(k, 0) * v

        return r





