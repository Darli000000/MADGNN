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

"""Defines the D4PG agent class."""

import copy
from typing import Callable, Dict, Optional, List

import acme
from acme import specs
from Agents.MAD4PG_GNN_DynENV.agent import D4PGNetworks, D4PGConfig, D4PGBuilder, get_replicator
from Agents.MAD4PG_GNN_DynENV.dynamic_gnn import DynGNNModule # EDIT
from Agents.MAD4PG_GNN_DynENV.agent import make_gnn_replay_environment_spec # EDIT
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf
from Utilities.FileOperator import load_obj
from dyn_environment_loop_gnn import EnvironmentLoop # EDIT

# EDIT NOTICE: 注意，这里没有引用Agents.MAD4PG.networks的原因是，networks是在run_mad4pg.py中提前创建好通过参数传给DistributedD4PG的

# Valid values of the "accelerator" argument.
_ACCELERATORS = ('CPU', 'GPU', 'TPU')


class DistributedD4PG:
    """Program definition for D4PG."""

    def __init__(
        self,
        agent_number: int,
        agent_action_size: int,
        environment_file: str,
        networks: List[D4PGNetworks],
        accelerator: Optional[str] = None,
        num_actors: int = 1,
        num_caches: int = 0,
        environment_spec: Optional[specs.EnvironmentSpec] = None,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        sigma: float = 0.3,
        clipping: bool = True,
        discount: float = 0.99,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        target_update_period: int = 100,
        variable_update_period: int = 1000,
        max_actor_steps: Optional[int] = None,
        log_every: float = 10.0,

        # EDIT: add gnn
        # TODO: gnn structure , 在这里可以修改gnn的构造
        gnn_hidden_dim: int = 32,
        gnn_num_layers: int = 2,
        gnn_updater_type: str = 'gru',
        gnn_is_dynmic: bool = True,
    ):
        self._agent_number = agent_number
        self._agent_action_size = agent_action_size
        
        if accelerator is not None and accelerator not in _ACCELERATORS:
            raise ValueError(f'Accelerator must be one of {_ACCELERATORS}, '
                        f'not "{accelerator}".')

        environment = load_obj(environment_file)
        if not environment_spec:
            environment_spec = specs.make_environment_spec(environment)

        # EDIT: add gnn
        # GNN DEBUG
        print(f"[GNN DEBUG]gnn_hidden_dim: {gnn_hidden_dim}")
        print(f"[GNN DEBUG]gnn_num_layers: {gnn_num_layers}")
        print(f"[GNN DEBUG]gnn_updater_type: {gnn_updater_type}")
        print(f"[GNN DEBUG]gnn_is_dynmic: {gnn_is_dynmic}")
        self._gnn_hidden_dim = gnn_hidden_dim
        self._gnn_num_layers = gnn_num_layers
        self._gnn_updater_type = gnn_updater_type
        self._gnn_is_dynmic = gnn_is_dynmic

        # Build a sample snapshot for replay spec.
        environment.reset()
        sample_snapshot = environment.get_current_snapshot()

        self._replay_environment_spec = make_gnn_replay_environment_spec(
            base_environment_spec=environment_spec,
            sample_snapshot=sample_snapshot,
            gnn_num_layers=self._gnn_num_layers,
            gnn_hidden_dim=self._gnn_hidden_dim,
        )

        self._environment_file = environment_file
        self._networks = networks  # NOTICE：Form Agents.MAD4PG.networks: make_default_networks
        self._environment_spec = environment_spec
        self._sigma = sigma
        self._num_actors = num_actors
        self._num_caches = num_caches
        self._max_actor_steps = max_actor_steps
        self._log_every = log_every
        self._accelerator = accelerator
        self._variable_update_period = variable_update_period

        self._builder = D4PGBuilder(
            # TODO(mwhoffman): pass the config dataclass in directly.
            # TODO(mwhoffman): use the limiter rather than the workaround below.
            D4PGConfig(
                accelerator=accelerator,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                variable_update_period=variable_update_period,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
            ))

    def replay(self):
        """The replay storage."""
        # EDIT: to _replay_environment_spec
        # return self._builder.make_replay_tables(self._environment_spec)
        return self._builder.make_replay_tables(self._replay_environment_spec)

    def counter(self):
        return tf2_savers.CheckpointingRunner(counting.Counter(),
                                            time_delta_minutes=1,
                                            subdirectory='counter')

    def coordinator(self, counter: counting.Counter):
        return lp_utils.StepsLimiter(counter, self._max_actor_steps)

    def learner(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        # If we are running on multiple accelerator devices, this replicates
        # weights and updates across devices.
        replicator = get_replicator(self._accelerator)

        with replicator.scope():
            # Create the networks to optimize (online) and target networks.
            online_networks = self._networks # NOTICE：这里和actor一样，用的都是D4PGNetworks。待办：为什么？好像是里面有不同函数如make_policy
            target_networks = [copy.deepcopy(online_network) for online_network in online_networks] # target 网络最初直接deepcopyonline网络
            

            # GNN Init
            online_dyn_gnn = DynGNNModule(
                node_feat_dim=12,
                edge_feat_dim=4,
                hidden_dim=self._gnn_hidden_dim,
                num_layers=self._gnn_num_layers,
                updater_type=self._gnn_updater_type,
            )
            target_dyn_gnn = DynGNNModule(
                node_feat_dim=12,
                edge_feat_dim=4,
                hidden_dim=self._gnn_hidden_dim,
                num_layers=self._gnn_num_layers,
                updater_type=self._gnn_updater_type,
            )
            environment = load_obj(self._environment_file)
            environment.reset()
            sample_snapshot = environment.get_current_snapshot()

            _ = online_dyn_gnn(snapshot=sample_snapshot, prev_state_list=None)
            _ = target_dyn_gnn(snapshot=sample_snapshot, prev_state_list=None)


            # Initialize the networks.
            for online_network, target_network in zip(online_networks, target_networks):
                # EDIT: adapted to gnn version
                online_network.init(
                    self._environment_spec,
                    gnn_embedding_dim=self._gnn_hidden_dim,
                )
                target_network.init(
                    self._environment_spec,
                    gnn_embedding_dim=self._gnn_hidden_dim,
                )

        dataset = self._builder.make_dataset_iterator(replay)

        counter = counting.Counter(counter, 'learner')
        logger = loggers.make_default_logger(
            'learner', time_delta=self._log_every, steps_key='learner_steps')

        # NOTICE: 这里的_builder指的是D4PGBuilder
        return self._builder.make_learner(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            networks=(online_networks, target_networks),
            dyn_gnn_modules=(online_dyn_gnn, target_dyn_gnn),
            gnn_embedding_dim=self._gnn_hidden_dim,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=True,
        )

    def actor(
        self,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> EnvironmentLoop:
        """The actor process."""

        # Create the environment first.
        environment = load_obj(self._environment_file)

        # Raw networks: do NOT use make_policy() anymore.
        networks = self._networks

        # Keep observation_network and policy_network separate.
        observation_networks = [network.observation_network for network in networks]
        policy_networks = [network.policy_network for network in networks]

        # Shared actor-side DynGNN.
        # NOTICE: node和edge的feat_dim在这里写死了
        actor_dyn_gnn = DynGNNModule(
            node_feat_dim=12,
            edge_feat_dim=4,
            hidden_dim=self._gnn_hidden_dim,
            num_layers=self._gnn_num_layers,
            updater_type=self._gnn_updater_type,
        )

        # Build modules once before VariableClient.update_and_wait().
        timestep = environment.reset()
        snapshot = environment.get_current_snapshot()

        # Build DynGNN variables.
        _ = actor_dyn_gnn(snapshot=snapshot, prev_state_list=None)

        # Build observation/policy variables with fused input.
        sample_obs = tf.convert_to_tensor(timestep.observation[0:1], dtype=tf.float32)  # [1, obs_dim]
        for obs_net, pol_net in zip(observation_networks, policy_networks):
            z = obs_net(sample_obs)  # [1, z_dim]
            g = tf.zeros([1, self._gnn_hidden_dim], dtype=z.dtype)
            fused = tf.concat([z, g], axis=-1)
            _ = pol_net(fused)

        # Create the actor.
        actor = self._builder.make_actor(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            dyn_gnn_module=actor_dyn_gnn,
            gnn_embedding_dim=self._gnn_hidden_dim,
            gnn_is_dynmic=self._gnn_is_dynmic,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
            sigma=self._sigma,
        )

        # Create logger and counter.
        counter = counting.Counter(counter, 'actor')
        logger = loggers.make_default_logger(
            'actor',
            save_data=False,
            time_delta=self._log_every,
            steps_key='actor_steps')

        # Create the loop and return it.
        return EnvironmentLoop(environment, actor, counter, logger)

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        logger: Optional[loggers.Logger] = None,
    ):
        """The evaluation process."""

        # Create the environment first.
        environment = load_obj(self._environment_file)

        # Raw networks: do NOT use make_policy() anymore.
        networks = self._networks
        observation_networks = [network.observation_network for network in networks]
        policy_networks = [network.policy_network for network in networks]

        # Shared evaluator-side DynGNN.
        evaluator_dyn_gnn = DynGNNModule(
            node_feat_dim=12,
            edge_feat_dim=4,
            hidden_dim=self._gnn_hidden_dim,
            num_layers=self._gnn_num_layers,
            updater_type=self._gnn_updater_type,
        )

        # Build modules once.
        timestep = environment.reset()
        snapshot = environment.get_current_snapshot()

        _ = evaluator_dyn_gnn(snapshot=snapshot, prev_state_list=None)

        sample_obs = tf.convert_to_tensor(timestep.observation[0:1], dtype=tf.float32)
        for obs_net, pol_net in zip(observation_networks, policy_networks):
            z = obs_net(sample_obs)
            g = tf.zeros([1, self._gnn_hidden_dim], dtype=z.dtype)

            fused = tf.concat([z, g], axis=-1)

            # print("sample_obs dtype:", sample_obs.dtype)
            # print("z dtype:", z.dtype)
            # print("g dtype:", g.dtype)
            # print("fused dtype:", fused.dtype)

            _ = pol_net(fused)

        # Create the actor (evaluator has no adder).
        # NOTICE: evaluator 没有adder
        actor = self._builder.make_actor(
            agent_number=self._agent_number,
            agent_action_size=self._agent_action_size,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            dyn_gnn_module=evaluator_dyn_gnn,
            gnn_embedding_dim=self._gnn_hidden_dim,
            gnn_is_dynmic=self._gnn_is_dynmic,
            variable_source=variable_source,
        )

        counter = counting.Counter(counter, 'evaluator')
        logger = logger or loggers.make_default_logger(
            'evaluator',
            time_delta=self._log_every,
            steps_key='evaluator_steps',
        )

        return EnvironmentLoop(environment, actor, counter, logger)


    def build(self, name='d4pg'):
        """Build the distributed agent topology."""
        program = lp.Program(name=name)

        with program.group('replay'):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group('counter'):
            counter = program.add_node(lp.CourierNode(self.counter))

        # NOTICE：'coordinator'用于监控 max_actor_steps
        if self._max_actor_steps:
            with program.group('coordinator'):
                _ = program.add_node(lp.CourierNode(self.coordinator, counter))

        with program.group('learner'):
            learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

        with program.group('evaluator'):
            program.add_node(lp.CourierNode(self.evaluator, learner, counter))

        if not self._num_caches:
            # Use our learner as a single variable source.
            sources = [learner]
        else:
            with program.group('cacher'):
                # Create a set of learner caches.
                sources = []
                for _ in range(self._num_caches):
                    cacher = program.add_node(
                        lp.CacherNode(
                            learner, refresh_interval_ms=2000, stale_after_ms=4000))
                sources.append(cacher)

        with program.group('actor'):
            # Add actors which pull round-robin from our variable sources.
            for actor_id in range(self._num_actors):
                source = sources[actor_id % len(sources)]
                program.add_node(lp.CourierNode(self.actor, replay, source, counter))

        return program