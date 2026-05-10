# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""D4PG agent implementation."""

import copy
import dataclasses
import functools
from typing import Iterator, List, Optional, Tuple, Union, Sequence, Dict

import numpy as np

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as reverb_adders
from acme.agents import agent
from Agents.MAD4PG_GNN_DynENV.actors import FeedForwardActor    # 第215行用到，创建了actor：FeedForwardActor(...
from Agents.MAD4PG_GNN_DynENV.learning import D4PGLearner
from Agents.MAD4PG_GNN_DynENV.dynamic_gnn import DynGNNModule   # EDIT
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]


@dataclasses.dataclass
class D4PGConfig:
    """Configuration options for the D4PG agent."""

    accelerator: Optional[str] = None
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    variable_update_period: int = 1000
    policy_optimizer: Optional[List[snt.Optimizer]] = None
    critic_optimizer: Optional[List[snt.Optimizer]] = None
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 5
    sigma: float = 0.3
    clipping: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    gnn_optimizer: Optional[snt.Optimizer] = None # EDIT: gnn optimizer


@dataclasses.dataclass
class D4PGNetworks:
    """Structure containing the networks for D4PG."""

    policy_network: snt.Module
    critic_network: snt.Module
    observation_network: snt.Module

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.observation_network = utils.to_sonnet_module(observation_network)

    # EDIT init: modified the input specification of policy and critic
    def init(
        self,
        environment_spec: specs.EnvironmentSpec,
        gnn_embedding_dim: int = 0,
    ):
        """Initialize the networks given an environment spec.

        GNN version:
        - observation_network takes edge_observations
        - policy_network is initialized with fused input [z_e ; g_e]
        - critic_network is also initialized with fused input [z_e ; g_e]
        """
        # ----- observation network -----
        obs_spec = environment_spec.edge_observations
        dummy_obs = tf.zeros([1, *obs_spec.shape], dtype=tf.float32)

        z = self.observation_network(dummy_obs)
        z = tf.cast(z, tf.float32)

        # ----- fused representation -----
        if gnn_embedding_dim > 0:
            dummy_g = tf.zeros([1, gnn_embedding_dim], dtype=tf.float32)
            fused = tf.concat([z, dummy_g], axis=-1)
            fused = tf.cast(fused, tf.float32)
        else:
            fused = z

        # ----- policy network -----
        _ = self.policy_network(fused)

        # ----- critic network -----
        act_spec = environment_spec.critic_actions
        dummy_action = tf.zeros([1, *act_spec.shape], dtype=tf.float32)

        try:
            _ = self.critic_network(fused, dummy_action)
        except TypeError:
            # Fallback for critics that expect a single concatenated input.
            critic_input = tf.concat(
                [tf.reshape(fused, [1, -1]),
                tf.reshape(dummy_action, [1, -1])],
                axis=-1
            )
            _ = self.critic_network(critic_input)

    # EDIT：新链路不需要
    # def make_policy(
    #     self,
    #     environment_spec: specs.EnvironmentSpec,
    #     sigma: float = 0.0,
    # ) -> snt.Module:
    #     """Create a single network which evaluates the policy."""
    #     # Stack the observation and policy networks.
    #     stack = [
    #         self.observation_network,
    #         self.policy_network,
    #     ]

    #     # If a stochastic/non-greedy policy is requested, add Gaussian noise on
    #     # top to enable a simple form of exploration.
    #     # TODO(mwhoffman): Refactor this to remove it from the class.
    #     if sigma > 0.0:
    #         stack += [
    #             network_utils.ClippedGaussian(sigma),
    #             network_utils.ClipToSpec(environment_spec.edge_actions),
    #         ]

    #     # Return a network which sequentially evaluates everything in the stack.
    #     return snt.Sequential(stack)

# helper function
def make_gnn_replay_environment_spec(
    base_environment_spec: specs.EnvironmentSpec,
    sample_snapshot: Dict[str, np.ndarray],
    gnn_num_layers: int,
    gnn_hidden_dim: int,
) -> specs.EnvironmentSpec:
    import numpy as np
    from acme import specs as acme_specs

    node_feature = sample_snapshot["node_feature"]
    node_type = sample_snapshot["node_type"]
    edge_index = sample_snapshot["edge_index"]
    edge_feature = sample_snapshot["edge_feature"]
    edge_type = sample_snapshot["edge_type"]
    edge_node_indices = sample_snapshot["edge_node_indices"]
    # DynENV-EDIT
    edge_mask = sample_snapshot["edge_mask"]

    if isinstance(edge_node_indices, dict):
        edge_node_indices = np.array(
            [edge_node_indices[k] for k in sorted(edge_node_indices.keys())],
            dtype=np.int32,
        )

    num_nodes = node_feature.shape[0]

    observation_spec = {
        "local_observation": base_environment_spec.observations,
        "snapshot": {
            "node_feature": acme_specs.Array(node_feature.shape, np.float32),
            "node_type": acme_specs.Array(node_type.shape, np.int32),
            "edge_index": acme_specs.Array(edge_index.shape, np.int32),
            "edge_feature": acme_specs.Array(edge_feature.shape, np.float32),
            "edge_type": acme_specs.Array(edge_type.shape, np.int32),
            "edge_node_indices": acme_specs.Array(edge_node_indices.shape, np.int32),
            "time_index": acme_specs.Array((), np.int32),
            "edge_mask": acme_specs.Array(edge_mask.shape, np.float32),     # DynENV-EDIT
        },
        "prev_state": acme_specs.Array(
            (gnn_num_layers, num_nodes, gnn_hidden_dim), np.float32
        ),
    }

    return specs.EnvironmentSpec(
        observations=observation_spec,
        actions=base_environment_spec.actions,
        rewards=base_environment_spec.rewards,
        discounts=base_environment_spec.discounts,
    )


class D4PGBuilder:
    """Builder for D4PG which constructs individual components of the agent."""

    def __init__(self, config: D4PGConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))

        return [replay_table]

    def make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size)

        replicator = get_replicator(self._config.accelerator)
        dataset = replicator.experimental_distribute_dataset(dataset)

        # TODO(b/155086959): Fix type stubs and remove.
        return iter(dataset)  # pytype: disable=wrong-arg-types

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)

    # EDIT: modified the signature and return, due to the change of the actor.FeedForwardActor
    def make_actor(
        self,
        agent_number: int,
        agent_action_size: int,
        observation_networks: List[snt.Module],
        policy_networks: List[snt.Module],
        dyn_gnn_module: snt.Module,
        gnn_embedding_dim: int,
        gnn_is_dynmic: bool,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
        sigma: Optional[float] = 0.2,
    ):
        """Create an actor instance."""
        if variable_source:
            # EDIT: gnn varibles added
            # variables = {'policy_%d' % i: policy_network.variables
            #             for i, policy_network in enumerate(policy_networks)}
            # variables['dyn_gnn'] = dyn_gnn_module.variables
            # EDIT : add observation variable
            variables = {}

            for i, observation_network in enumerate(observation_networks):
                variables[f'observation_{i}'] = observation_network.variables

            for i, policy_network in enumerate(policy_networks):
                variables[f'policy_{i}'] = policy_network.variables

            variables['dyn_gnn'] = dyn_gnn_module.variables

            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables=variables,
                update_period=self._config.variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        else:
            variable_client = None

        # Create the actor which defines how we take actions.
        return FeedForwardActor(
            agent_number=agent_number,
            agent_action_size=agent_action_size,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            dyn_gnn_module=dyn_gnn_module,
            gnn_embedding_dim=gnn_embedding_dim,
            gnn_is_dynmic=gnn_is_dynmic,
            adder=adder,
            variable_client=variable_client,
            sigma=sigma,
        )

    def make_learner(
        self,
        agent_number: int,
        agent_action_size: int,
        networks: Tuple[List[D4PGNetworks], List[D4PGNetworks]],
        dyn_gnn_modules: Tuple[snt.Module, snt.Module],
        gnn_embedding_dim: int,
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = False,
    ):
        """Creates an instance of the learner."""
        online_networks, target_networks = networks
        dyn_gnn_module, target_dyn_gnn_module = dyn_gnn_modules

        # The learner updates the parameters (and initializes them).
        return D4PGLearner(
            agent_number=agent_number,
            agent_action_size=agent_action_size,
            
            online_networks=online_networks,
            target_networks=target_networks,

            # GNN
            dyn_gnn_module=dyn_gnn_module,
            target_dyn_gnn_module=target_dyn_gnn_module,
            gnn_embedding_dim=gnn_embedding_dim,

            # policy_network=online_networks.policy_network,
            # critic_network=online_networks.critic_network,
            # observation_network=online_networks.observation_network,
            # target_policy_network=target_networks.policy_network,
            # target_critic_network=target_networks.critic_network,
            # target_observation_network=target_networks.observation_network,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            gnn_optimizer=self._config.gnn_optimizer, # EDIT: gnn optimizer
            clipping=self._config.clipping,
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            dataset_iterator=dataset,
            replicator=get_replicator(self._config.accelerator),
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )


def _ensure_accelerator(accelerator: str) -> str:
    """Checks for the existence of the expected accelerator type.

    Args:
        accelerator: 'CPU', 'GPU' or 'TPU'.

    Returns:
        The validated `accelerator` argument.

    Raises:
        RuntimeError: Thrown if the expected accelerator isn't found.
    """
    devices = tf.config.get_visible_devices(device_type=accelerator)

    if devices:
        return accelerator
    else:
        error_messages = [f'Couldn\'t find any {accelerator} devices.',
                        'tf.config.get_visible_devices() returned:']
        error_messages.extend([str(d) for d in devices])
        raise RuntimeError('\n'.join(error_messages))


def _get_first_available_accelerator_type(
    wishlist: Sequence[str] = ('TPU', 'GPU', 'CPU')) -> str:
    """Returns the first available accelerator type listed in a wishlist.

    Args:
        wishlist: A sequence of elements from {'CPU', 'GPU', 'TPU'}, listed in
        order of descending preference.

    Returns:
        The first available accelerator type from `wishlist`.

    Raises:
        RuntimeError: Thrown if no accelerators from the `wishlist` are found.
    """
    get_visible_devices = tf.config.get_visible_devices

    for wishlist_device in wishlist:
        devices = get_visible_devices(device_type=wishlist_device)
        if devices:
            return wishlist_device

    available = ', '.join(
        sorted(frozenset([d.type for d in get_visible_devices()])))
    raise RuntimeError(
        'Couldn\'t find any devices from {wishlist}.' +
        f'Only the following types are available: {available}.')


# Only instantiate one replicator per (process, accelerator type), in case
# a replicator stores state that needs to be carried between its method calls.
@functools.lru_cache()
def get_replicator(accelerator: Optional[str]) -> Replicator:
    """Returns a replicator instance appropriate for the given accelerator.

    This caches the instance using functools.cache, so that only one replicator
    is instantiated per process and argument value.

    Args:
        accelerator: None, 'TPU', 'GPU', or 'CPU'. If None, the first available
        accelerator type will be chosen from ('TPU', 'GPU', 'CPU').

    Returns:
        A replicator, for replciating weights, datasets, and updates across
        one or more accelerators.
    """
    if accelerator:
        accelerator = _ensure_accelerator(accelerator)
    else:
        accelerator = _get_first_available_accelerator_type()

    if accelerator == 'TPU':
        tf.tpu.experimental.initialize_tpu_system()
        return snt.distribute.TpuReplicator()
    else:
        return snt.distribute.Replicator()