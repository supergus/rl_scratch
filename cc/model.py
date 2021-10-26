"""
The main model declaration
"""
import logging
import os

import numpy as np
import tensorflow as tf

from common_definitions import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, BUFFER_SIZE, BATCH_SIZE,
                                CRITIC_LR, ACTOR_LR)
from buffer import ReplayBuffer
from utils import OUActionNoise

from controls.core.models import CMPTaxonomy
from controls.config import DTYPE
from physics.core.base import VerboseObjectABC

from tensorflow.keras.initializers import glorot_normal, glorot_uniform

from env_builder import Space, Environment

# Convert DTYPE to Tensorflow dtype
TF_DTYPE = tf.dtypes.as_dtype(DTYPE)

BRAIN_MSG_COLOR = 'light_green'
BRAIN_WARN_COLOR = 'light_red'


def ActorNetwork(env, taxonomy, **options):
    """Builds Actor network.

    Arguments:
        env (Environment): An Environment object.
        taxonomy (CMPTaxonomy): An CEPTaxonomy object.

    Keyword Arguments:
        model_tf_name (str): Optional. Name to assign model in TF graph.
            Default is to allow TF to assign the name.

    Returns:
        A Keras model.
    """

    taxonomy.update({
        'ctrl_input_dim': len(env.experiment.signal_selections.post_pipeline.ctrl_inputs),
        'unctrl_input_dim': len(env.experiment.signal_selections.post_pipeline.unctrl_inputs),
        'output_dim': len(env.experiment.signal_selections.post_pipeline.outputs),
        'memory_units_hidden_nn': 600,
        'activation_hidden_nn': tf.nn.leaky_relu,  # Update w leaky ReLU; don't want to import tf in config file
        'init_hidden_nn': glorot_normal(),  # Update w glorot; don't want to import tf in config file
        'memory_units_exposed_nn': 300,
        'activation_exposed_nn': 'tanh',  # tanh important; need (-1, 1) range; will scale by 'action_high'
        'init_exposed_nn': tf.random_normal_initializer(stddev=0.0005),  # normal w/small dev important; 0 = "avg ctrl"
        'memory_units_lstm': 128,
    })

    # Get standard input layers for actor
    obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs = _standard_input_layers_actor(taxonomy)

    # =======================================================
    # Encoder, not stateful
    # =======================================================

    # An LSTM for each input layer; Using default activations & init
    m = taxonomy.memory_units_lstm
    # TODO: Enhance taxonomy to allow separate init & activations for kernel and recurrent; harvest here for kwargs
    kwargs = {'return_state': False,
              'return_sequences': False,
              'stateful': False,
              'dropout': taxonomy.dropout_input,
              'recurrent_dropout': taxonomy.dropout_recurrent,
              }
    kwargs.update({'name': 'encoder_lstm_0'})
    encoder_lstm_0 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_1'})
    encoder_lstm_1 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_2'})
    encoder_lstm_2 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_3'})
    encoder_lstm_3 = tf.keras.layers.LSTM(m, **kwargs)

    # =======================================================
    # Neural Net
    # =======================================================

    units_high = taxonomy.memory_units_hidden_nn
    units_med = int(taxonomy.memory_units_hidden_nn / 2)

    nn_hidden_0 = tf.keras.layers.Dense(
        units=units_high,
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_hidden_0',
        dtype=TF_DTYPE,
    )

    nn_hidden_1 = tf.keras.layers.Dense(
        units=units_med,
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_hidden_1',
        dtype=TF_DTYPE,
    )

    nn_exposed = tf.keras.layers.Dense(
        units=taxonomy.ctrl_input_dim,
        activation=taxonomy.activation_exposed_nn,
        kernel_initializer=taxonomy.init_exposed_nn,
        name='actions',
        dtype=TF_DTYPE,
    )

    # =======================================================
    # Connections
    # =======================================================

    x_0 = encoder_lstm_0(obs_ctrl_ins)
    x_1 = encoder_lstm_1(obs_unctrl_ins)
    x_2 = encoder_lstm_2(obs_outs)
    x_3 = encoder_lstm_3(pred_outs)

    lstm_features = _lstm_connection_manager(taxonomy, x_0, x_1, x_2, x_3)
    x = tf.keras.layers.BatchNormalization()(lstm_features)
    x = nn_hidden_0(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = nn_hidden_1(x)
    x = tf.keras.layers.BatchNormalization()(x)
    actions = nn_exposed(x) * env.action_space.high   # TODO: NO. we will scale & translate in Brain.act().

    model = _assemble_keras_model(
        obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs, actions,
        **options,
    )

    return model


def CriticNetwork(env, taxonomy, **options):
    """Builds Critic network.

    Arguments:
        env (Environment): An Environment object.
        taxonomy (CMPTaxonomy): An CEPTaxonomy object.

    Keyword Arguments:
        model_tf_name (str): Optional. Name to assign model in TF graph.
            Default is to allow TF to assign the name.

    Returns:
        A Keras model.
    """

    taxonomy.update({
        'ctrl_input_dim': len(env.experiment.signal_selections.post_pipeline.ctrl_inputs),
        'unctrl_input_dim': len(env.experiment.signal_selections.post_pipeline.unctrl_inputs),
        'output_dim': len(env.experiment.signal_selections.post_pipeline.outputs),
        'memory_units_hidden_nn': 600,
        'activation_hidden_nn': tf.nn.leaky_relu,  # Update w leaky ReLU; don't want to import tf in config file
        'init_hidden_nn': glorot_normal(),  # Update w glorot; don't want to import tf in config file
        'memory_units_exposed_nn': 300,
        'activation_exposed_nn': 'tanh',
        'init_exposed_nn': tf.random_normal_initializer(stddev=0.0005),
        'memory_units_lstm': 128,
    })

    # Get standard input layers for critic
    obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs, action = _standard_input_layers_critic(taxonomy)
    # nudged_ctrl_ins, unctrl_ins, pred_outs, actions = _standard_input_layers_critic(taxonomy)

    # =======================================================
    # Encoder, not stateful - for STATES
    # =======================================================

    # An LSTM for each input layer
    m = taxonomy.memory_units_lstm
    # TODO: Enhance taxonomy to allow separate init & activations for kernel and recurrent; harvest here for kwargs
    kwargs = {'return_state': False,
              'return_sequences': False,
              'stateful': False,
              'dropout': taxonomy.dropout_input,
              'recurrent_dropout': taxonomy.dropout_recurrent,
              }
    kwargs.update({'name': 'encoder_lstm_0'})
    encoder_lstm_0 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_1'})
    encoder_lstm_1 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_2'})
    encoder_lstm_2 = tf.keras.layers.LSTM(m, **kwargs)

    kwargs.update({'name': 'encoder_lstm_3'})
    encoder_lstm_3 = tf.keras.layers.LSTM(m, **kwargs)

    # =======================================================
    # NN - for ACTIONS and CRITIC
    # =======================================================

    units_high = taxonomy.memory_units_hidden_nn
    units_med = int(taxonomy.memory_units_hidden_nn / 2)
    units_low = int(taxonomy.memory_units_hidden_nn / 4)

    nn_action_hidden = tf.keras.layers.Dense(
        units=units_med,  # e.g. 300
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_action_hidden',
        dtype=TF_DTYPE,
    )
    nn_state_hidden_0 = tf.keras.layers.Dense(
        units=units_high,  # e.g. 600
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_state_hidden_0',
        dtype=TF_DTYPE,
    )
    nn_state_hidden_1 = tf.keras.layers.Dense(
        units=units_med,  # e.g. 300
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_state_hidden_1',
        dtype=TF_DTYPE,
    )
    nn_critic_hidden = tf.keras.layers.Dense(
        units=units_low,  # e.g. 150
        activation=taxonomy.activation_hidden_nn,
        kernel_initializer=taxonomy.init_hidden_nn,
        name='nn_critic_hidden',
        dtype=TF_DTYPE,
    )
    nn_critic_exposed = tf.keras.layers.Dense(
        units=1,  # Output a single value from critic
        activation='linear',  # Default for Dense() is 'linear'
        kernel_initializer=taxonomy.init_exposed_nn,
        name='nn_critic_exposed',
        dtype=TF_DTYPE,
    )

    # =======================================================
    # Connections
    # =======================================================

    x_0 = encoder_lstm_0(obs_ctrl_ins)
    x_1 = encoder_lstm_1(obs_unctrl_ins)
    x_2 = encoder_lstm_2(obs_outs)
    x_3 = encoder_lstm_3(pred_outs)

    # State trunk
    state_inputs = _lstm_connection_manager(taxonomy, x_0, x_1, x_2, x_3)
    s = tf.keras.layers.BatchNormalization()(state_inputs)
    s = nn_state_hidden_0(s)
    s = tf.keras.layers.BatchNormalization()(s)
    s = nn_state_hidden_1(s)

    # Action trunk
    a = nn_action_hidden(action / env.action_space.high)

    # Merge trunks with equal shapes using Add()
    merged = tf.keras.layers.Add()([s, a])
    merged = tf.keras.layers.BatchNormalization()(merged)

    # Critic: Outputs a single Q-value for given [state, action]
    c = nn_critic_hidden(merged)
    c = tf.keras.layers.BatchNormalization()(c)
    critic_output = nn_critic_exposed(c)

    model = _assemble_keras_model(
        obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs, action, critic_output,
        **options,
    )

    return model


def _standard_input_layers_actor(taxonomy):
    """Creates Keras Input layers according to the standard LCG API.

    We allow *taxonomy.unctrl_input_dim* to be zero or None.
    All other dimensions must be integers greater than zero.

    Arguments:
        taxonomy (CMPTaxonomy): A CMPTaxonomy object.

    Returns:
        4 Keras Input layers: obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs.
    """
    # Following the standard API:

    # Inputs:   All observable signals (controllable inputs, uncontrollable inputs, outputs)
    #           AND ALSO the "next" batch of predicted outputs using UNALTERED (not nudged) inputs.
    #           Each must be an array with shape = (1, samples, signals).
    #           The 1 denotes a single batch that we feed to the model.

    # Check `Environment._assemble_state()`;
    # state dict keys MUST be consistent with the layers defined here.

    # Multiple input layers:
    obs_ctrl_ins = tf.keras.Input(
        shape=(None, taxonomy.ctrl_input_dim),
        name='observed_ctrl_inputs',
        dtype=TF_DTYPE,
    )
    obs_unctrl_ins = None
    if taxonomy.unctrl_input_dim is not None:
        if taxonomy.unctrl_input_dim > 0:
            obs_unctrl_ins = tf.keras.Input(
                shape=(None, taxonomy.unctrl_input_dim),
                name='observed_unctrl_inputs',
                dtype=TF_DTYPE,
            )
    obs_outs = tf.keras.Input(
        shape=(None, taxonomy.output_dim),
        name='observed_outputs',
        dtype=TF_DTYPE,
    )
    pred_outs = tf.keras.Input(
        shape=(None, taxonomy.output_dim),
        name='predicted_outputs',  # WITHOUT nudge; actor hasn't predicted the nudge yet!
        dtype=TF_DTYPE,
    )

    return obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs


def _standard_input_layers_critic(taxonomy):
    """Creates Keras Input layers according to the standard LCG API.

    We allow *taxonomy.unctrl_input_dim* to be zero or None.
    All other dimensions must be integers greater than zero.

    Arguments:
        taxonomy (CMPTaxonomy): A CMPTaxonomy object.

    Returns:
        5 Keras Input layers: obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs, actions.
    """
    # Following the standard API:

    # Inputs:   All observable signals (controllable inputs, uncontrollable inputs, outputs)
    #           AND ALSO the "next" batch of predicted outputs using UNALTERED (not nudged) inputs.
    #           Each must be an array with shape = (1, samples, signals).
    #           The 1 denotes a single batch that we feed to the model.

    # Check `Environment._assemble_state()`;
    # state dict keys MUST be consistent with the layers defined here.

    # STATE: LSTM input layers: Sequences as inputs
    obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs = _standard_input_layers_actor(taxonomy)

    # ACTION: NN input layer: 1D vector (NOT a sequence)
    action = tf.keras.layers.Input(
        shape=taxonomy.ctrl_input_dim,
        name='action',
        dtype=TF_DTYPE,
    )

    return obs_ctrl_ins, obs_unctrl_ins, obs_outs, pred_outs, action


def _lstm_connection_manager(taxonomy, x_0, x_1, x_2, x_3):
    """Selects LSTMs to include in final model and returns a concatenated feature tensor.

    Features are the LSTM outputs, not LSTM states or sequences.

    **ASSUMPTIONS:**
        - LSTMs do not return states, just the actual LSTM output (`return_states=False`)
        - LSTMs do not return sequences (`return_sequences=False`)

    Arguments:
        taxonomy (CMPTaxonomy): A CMPTaxonomy object.
        x_0 (obj): Output from LSTM 0 or None.
        x_1 (obj): Output from LSTM 1 or None.
        x_2 (obj): Output from LSTM 2 or None.
        x_3 (obj): Output from LSTM 3 or None.

    Returns:
        Concatenated feature vector.

    """
    lstm_list = list()

    if taxonomy.use_obs_ctrl_inputs and x_0 is not None:
        lstm_list.append(x_0)
    if taxonomy.use_obs_unctrl_inputs and x_1 is not None:
        lstm_list.append(x_1)
    if taxonomy.use_obs_outputs and x_2 is not None:
        lstm_list.append(x_2)
    if taxonomy.use_pred_outputs and x_3 is not None:
        lstm_list.append(x_3)

    # TODO: For concatenation of input layers in _lstm_connection_manager(), consider Add()? MUST HAVE SAME SHAPE...
    return_list = tf.keras.layers.concatenate(lstm_list)
    # return_list = tf.keras.layers.Add()(lstm_list)

    return return_list


def _assemble_keras_model(*args, **kwargs):
    """Makes final connections and returns a Keras model.

    Arguments:
        input_layer (tf.keras.layers.InputLayer, None): An arbitrary number of input layers.
        output_tensor (object):  A Tensorflow tensor. Last element in args will be treated as
            the output_tensor.

    Keyword Arguments:
        model_tf_name (str): Optional. Name to assign model in TF graph.
            Default is to allow TF to assign the name.

    Returns:
        A Keras model.

    Raises:
        RuntimeError: If no InputLayers are provided (all arguments are None).
        RuntimeError: If out not provided (argument is None).
    """
    # Unpack and parse arguments
    output_tensor = args[-1]
    candidate_inputs = args[:-1]
    input_list = [x for x in candidate_inputs if x is not None]

    # Parse kwargs
    model_tf_name = kwargs.get('model_tf_name', 'critic_model_default_tf_name')

    # Validate
    if len(input_list) == 0:
        msg = f'No InputLayers selected'
        raise RuntimeError(msg)
    if not tf.keras.backend.is_keras_tensor(output_tensor):
        msg = f'No model output selected'
        raise RuntimeError(msg)

    # Build
    model = tf.keras.Model(
        inputs=input_list,
        outputs=[output_tensor],
        name=model_tf_name
    )

    return model


# noinspection PyUnresolvedReferences
def update_target(model_target, model_ref, rho=0):
    """Updates target network weights using the given model reference.

    Arguments:
        model_target (object): Target model to be changed
        model_ref (object): Reference model
        rho (float): Ratio of new and old weights
    """
    model_target.set_weights(
        [rho * ref_weight + (1 - rho) * target_weight
         for (target_weight, ref_weight) in
         list(zip(model_target.get_weights(), model_ref.get_weights()))]
    )
    return


class Brain(VerboseObjectABC):

    def __init__(self, env, taxonomy):
        super().__init__(msg_color=BRAIN_MSG_COLOR, warn_color=BRAIN_WARN_COLOR, name='brain')

        # Assign action space from Environment
        self.action_space = env.action_space

        # Initialize networks
        self._print_msg(f'Building actor and critic networks')
        self.actor_network = ActorNetwork(env, taxonomy)
        self.critic_network = CriticNetwork(env, taxonomy)
        self.actor_target = ActorNetwork(env, taxonomy)
        self.critic_target = CriticNetwork(env, taxonomy)

        # Equalize weights with target networks
        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        # Setup experience buffer
        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        # Set update parameters for target networks
        self.gamma = tf.constant(GAMMA)
        self.rho = RHO

        # OU Noise generator
        # noinspection PyTypeChecker
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(STD_DEV) * np.ones(1))

        # Instantiate optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR, amsgrad=True)

        # Tmp variable to capture current action
        self.current_action = None

        # define update weights with tf.function for improved performance

        action_dim = env.action_space.dimensions
        out_dim = env.observation_space.dimensions
        unctrl_in_dim = len(env.experiment.signal_selections.post_pipeline.unctrl_inputs)
        in_seq_len = env.experiment.lpp.experiment.modelpackage.taxonomy.input_seq_len
        out_seq_len = env.experiment.lpp.experiment.modelpackage.taxonomy.output_seq_len

        input_signature = dict(
            s_observed_ctrl_inputs=tf.TensorSpec(shape=(None, in_seq_len, action_dim), dtype=TF_DTYPE),
            s_observed_unctrl_inputs=tf.TensorSpec(shape=(None, in_seq_len, unctrl_in_dim), dtype=TF_DTYPE),
            s_observed_outputs=tf.TensorSpec(shape=(None, out_seq_len, out_dim), dtype=TF_DTYPE),
            s_predicted_outputs=tf.TensorSpec(shape=(None, out_seq_len, out_dim), dtype=TF_DTYPE),
            a=tf.TensorSpec(shape=(None, action_dim), dtype=TF_DTYPE),
            r=tf.TensorSpec(shape=(None, 1), dtype=TF_DTYPE),
            sn_observed_ctrl_inputs=tf.TensorSpec(shape=(None, in_seq_len, action_dim), dtype=TF_DTYPE),
            sn_observed_unctrl_inputs=tf.TensorSpec(shape=(None, in_seq_len, unctrl_in_dim), dtype=TF_DTYPE),
            sn_observed_outputs=tf.TensorSpec(shape=(None, out_seq_len, out_dim), dtype=TF_DTYPE),
            sn_predicted_outputs=tf.TensorSpec(shape=(None, out_seq_len, out_dim), dtype=TF_DTYPE),
            d=tf.TensorSpec(shape=(None, 1), dtype=TF_DTYPE),
        )

        @tf.function(input_signature=[input_signature])
        def update_weights(features):
            """Function to update model weights."""

            state_elems = ('observed_ctrl_inputs', 'observed_unctrl_inputs', 'observed_outputs', 'predicted_outputs')

            s = {k[2:]: v for k, v in features.items() if k[2:] in state_elems}
            sn = {k[3:]: v for k, v in features.items() if k[3:] in state_elems}
            a = features['a']
            r = features['r']
            d = features['d']

            # First the critic model
            with tf.GradientTape() as tape:
                # Target
                a_next = self.actor_target(sn)
                critic_feed_next = {k: v for k, v in sn.items()}
                critic_feed_next.update({'action': a_next})
                y = r + self.gamma * (1 - d) * self.critic_target(critic_feed_next)

                # Delta Q
                critic_feed = {k: v for k, v in s.items()}
                critic_feed.update({'action': a})
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network(critic_feed)))

            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_network.trainable_variables))

            # Then the actor model
            with tf.GradientTape() as tape:
                # Delta mu
                critic_feed = {k: v for k, v in s.items()}
                new_a = self.actor_network(s)
                critic_feed.update({'action': new_a})

                actor_loss = -tf.math.reduce_mean(self.critic_network(critic_feed))

            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_network.trainable_variables))

            return critic_loss, actor_loss

        # Assign function to attribute
        self.update_weights = update_weights

    def act(self, state, not_random=True, noise=True, action_space=None):
        """Given the state, get an action from the Actor network.

        Arguments:
            state (dict): The current state. Contains 4 standard keys.
            not_random (bool): If True, we use greedy.
                Default is True.
            noise (bool): If True, noise is added to the resulting action.
                This can improve exploration during training.
                Default is True.
            action_space (Space): Optional. An n-dimensional Space object.
                If provided, random actions will be drawn from action_space.random().
                If omitted, random actions will be drawn from a uniform distribution.

        Returns:
            A Numpy array with one float value for each dimension in action space.
        """
        # Optional noise
        n = self.noise() if noise else 0

        if not_random:
            # Get action from network
            self.current_action = self.actor_network(state)[0].numpy() + n
        else:
            # Get action from random sample of space
            if action_space is not None:
                self.current_action = action_space.random() + n
            else:
                self.current_action = np.random.uniform(
                    self.action_space.low,
                    self.action_space.high,
                    self.action_space.dimensions,
                ) + n

        # TODO: Do not clip here... do after scale/translate?
        self.current_action = np.clip(self.current_action, self.action_space.low, self.action_space.high)

        return self.current_action

    def remember(self, prev_state, reward, state, done):
        """Store states, reward, and done value to the buffer.

        Actions are harvested from *Brain.current_action* and are also stored in the buffer.

        Arguments:
            prev_state (dict): The state prior to action. Keys = 5 standard sequences.
            reward (np.ndarray): The reward after action. Shape = (1, ).
            state (dict): The next state after action. Keys = 5 standard sequences.
            done (bool): Done (whether one loop is done or not).

        Returns:
            No returns.
        """
        self.buffer.append(prev_state, self.current_action, reward, state, done)
        return

    def learn(self, experience):
        """Run update for all networks (for training).

        Arguments:
            experience (list): A list of SARSD tuples from the experience buffer.

        Returns:
            Updated losses for critic and actor networks.
        """
        # Build feed
        features = self._build_features_for_update_weights(experience)

        # Update weights and get losses for actor & critic networks
        critic_loss, actor_loss = self.update_weights(features)

        # Update target networks
        update_target(self.actor_target, self.actor_network, self.rho)
        update_target(self.critic_target, self.critic_network, self.rho)

        return critic_loss, actor_loss

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_network.save_weights(path + "cn.h5")
        self.critic_target.save_weights(path + "ct.h5")
        self.actor_target.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target.load_weights(path + "ct.h5")
            self.actor_target.load_weights(path + "at.h5")
        except OSError as err:
            # logging.warning("Weights files cannot be found, %s", err)
            pass

    @staticmethod
    def _build_features_for_update_weights(experience):
        """Builds features for feeding update_weights()."""

        # Convert list elements to tuples for each SARSD component
        s, a, r, sn, d = zip(*experience)

        # ==============================================================
        # Build state tensors
        # ==============================================================

        s_dict = dict(
            observed_ctrl_inputs=list(),
            observed_unctrl_inputs=list(),
            observed_outputs=list(),
            predicted_outputs=list(),
        )
        sn_dict = {k: list() for k, v in s_dict.items()}

        # Build lists of 2D Numpy arrays, one array for each experience.
        # Later we will convert lists to 3D Numpy arrays; this is faster than v-stacking directly in Numpy.
        for exp in s:
            for k, v in s_dict.items():
                if k in exp:
                    v.append(exp[k].squeeze())
                    s_dict.update({k: v})
        for exp in sn:
            for k, v in sn_dict.items():
                if k in exp:
                    v.append(exp[k].squeeze())
                    sn_dict.update({k: v})

        # Convert lists to 3D Numpy arrays and then TF Tensors
        for k, v in s_dict.items():
            v = np.array(v)
            v = tf.convert_to_tensor(v, dtype=TF_DTYPE)
            s_dict.update({k: v})
        for k, v in sn_dict.items():
            v = np.array(v)
            v = tf.convert_to_tensor(v, dtype=TF_DTYPE)
            sn_dict.update({k: v})

        # ==============================================================
        # Build action, reward, and done tensors
        # ==============================================================

        a_tensor = tf.convert_to_tensor([action for action in a], dtype=TF_DTYPE)
        r_tensor = tf.convert_to_tensor([reward for reward in r], dtype=TF_DTYPE)
        d_tensor = tf.convert_to_tensor([done for done in d], dtype=TF_DTYPE)

        # ==============================================================
        # Assemble return as per signature for update_weights()
        # ==============================================================

        feed_dict = dict(
            s_observed_ctrl_inputs=s_dict['observed_ctrl_inputs'],
            s_observed_unctrl_inputs=s_dict['observed_unctrl_inputs'],
            s_observed_outputs=s_dict['observed_outputs'],
            s_predicted_outputs=s_dict['predicted_outputs'],
            a=a_tensor,
            r=r_tensor,
            sn_observed_ctrl_inputs=sn_dict['observed_ctrl_inputs'],
            sn_observed_unctrl_inputs=sn_dict['observed_unctrl_inputs'],
            sn_observed_outputs=sn_dict['observed_outputs'],
            sn_predicted_outputs=sn_dict['predicted_outputs'],
            d=d_tensor,
        )

        return feed_dict
