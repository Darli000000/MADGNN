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

"""D4PG learner implementation."""

import time
from typing import Dict, Iterator, List, Optional, Union, Sequence

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks as acme_nets
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree

from Agents.MAD4PG_GNN.dynamic_gnn import run_dyn_gnn_batched   # EDIT

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]


class D4PGLearner(acme.Learner):
    """D4PG learner.

    This is the learning component of a D4PG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """
    def __init__(
        self,
        agent_number: int,
        agent_action_size: int,
        
        online_networks,
        target_networks,

        # GNN
        dyn_gnn_module: snt.Module,
        target_dyn_gnn_module: snt.Module,
        gnn_embedding_dim: int,

        # policy_network: snt.Module,
        # critic_network: snt.Module,
        # target_policy_network: snt.Module,
        # target_critic_network: snt.Module,
        discount: float,
        target_update_period: int,
        dataset_iterator: Iterator[reverb.ReplaySample],
        replicator: Optional[Replicator] = None,
        # observation_network: types.TensorTransformation = lambda x: x,
        # target_observation_network: types.TensorTransformation = lambda x: x,
        policy_optimizer: Optional[List[snt.Optimizer]] = None,
        critic_optimizer: Optional[List[snt.Optimizer]] = None,
        gnn_optimizer: Optional[snt.Optimizer] = None, # EDIT: gnn optimizer
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
    ):
        """Initializes the learner.

        Args:
        policy_network: the online (optimized) policy.
        critic_network: the online critic.
        target_policy_network: the target policy (which lags behind the online
            policy).
        target_critic_network: the target critic.
        discount: discount to use for TD updates.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_reverb_dataset` documentation).
        replicator: Replicates variables and their update methods over multiple
            accelerators, such as the multiple chips in a TPU.
        observation_network: an optional online network to process observations
            before the policy and the critic.
        target_observation_network: the target observation network.
        policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
        critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
        clipping: whether to clip gradients by global norm.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        """

        # Store online and target networks.
        self._policy_networks = [online_network.policy_network for online_network in online_networks]
        self._critic_networks = [online_network.critic_network for online_network in online_networks]
        self._target_policy_networks = [target_network.policy_network for target_network in target_networks]
        self._target_critic_networks = [target_network.critic_network for target_network in target_networks]

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_networks = [tf2_utils.to_sonnet_module(online_network.observation_network) for online_network in online_networks]
        self._target_observation_networks = [tf2_utils.to_sonnet_module(
            target_network.observation_network) for target_network in target_networks]

        # EDIT
        # 新增： GNN
        self._dyn_gnn = dyn_gnn_module
        self._target_dyn_gnn = target_dyn_gnn_module
        self._gnn_embedding_dim = gnn_embedding_dim

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping
        
        self._agent_number = agent_number
        self._agent_action_size = agent_action_size

        # Replicates Variables across multiple accelerators
        if not replicator:
            accelerator = _get_first_available_accelerator_type()
            if accelerator == 'TPU':
                replicator = snt.distribute.TpuReplicator()
            else:
                replicator = snt.distribute.Replicator()

        self._replicator = replicator

        with replicator.scope():
            # Necessary to track when to update target networks.
            self._num_steps = tf.Variable(0, dtype=tf.int32)
            self._target_update_period = target_update_period

            # Create optimizers if they aren't given.
            self._critic_optimizer = critic_optimizer or [snt.optimizers.Adam(1e-4) for _ in range(self._agent_number)]
            self._policy_optimizer = policy_optimizer or [snt.optimizers.Adam(1e-4) for _ in range(self._agent_number)]
            self._gnn_optimizer = gnn_optimizer or snt.optimizers.Adam(1e-4) # EDIT: gnn optimizer

        # Batch dataset and create iterator.
        self._iterator = dataset_iterator

        # Expose the variables.
        self._variables = {}
        # for i in range(self._agent_number):
        #     policy_network_to_expose = snt.Sequential(
        #         [self._target_observation_networks[i], self._target_policy_networks[i]])
        #     self._variables['policy_' + str(i)] = policy_network_to_expose.variables
        #     self._variables['critic_' + str(i)] = self._target_critic_networks[i].variables
        # EDIT: expose obs variables explicitly
        for i in range(self._agent_number):
            self._variables[f'observation_{i}'] = self._target_observation_networks[i].variables
            self._variables[f'policy_{i}'] = self._target_policy_networks[i].variables
            self._variables[f'critic_{i}'] = self._target_critic_networks[i].variables
        # EDIT: GNN, expose gnn variables
        self._variables['dyn_gnn'] = self._target_dyn_gnn.variables
        

        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        # if checkpoint:
        #     self._checkpointer = tf2_savers.Checkpointer(
        #         subdirectory='d4pg_learner',
        #         objects_to_save={
        #             'counter': self._counter,
        #             'policy': self._policy_network,
        #             'critic': self._critic_network,
        #             'observation': self._observation_network,
        #             'target_policy': self._target_policy_network,
        #             'target_critic': self._target_critic_network,
        #             'target_observation': self._target_observation_network,
        #             'policy_optimizer': self._policy_optimizer,
        #             'critic_optimizer': self._critic_optimizer,
        #             'num_steps': self._num_steps,
        #         })
        #     critic_mean = snt.Sequential(
        #         [self._critic_network, acme_nets.StochasticMeanHead()])
        #     self._snapshotter = tf2_savers.Snapshotter(
        #         objects_to_save={
        #             'policy': self._policy_network,
        #             'critic': critic_mean,
        #         })

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

        # EDIT DEBUG
        self._debug_count = 0


    @tf.function
    def _step(self, sample) -> Dict[str, tf.Tensor]:
        transitions: types.Transition = sample.data

        # NOTICE: cast f64 to f32, still being CONTROVERSIAL
        # ====== numeric tensors cast ======
        actions = tf.cast(transitions.action, tf.float32)
        rewards = tf.cast(transitions.reward, tf.float32)
        discounts = tf.cast(transitions.discount, tf.float32)
        discount = tf.cast(self._discount, tf.float32)

        # ====== unpack dict obs ======
        obs = transitions.observation
        next_obs = transitions.next_observation

        local_observation = tf.cast(obs["local_observation"], tf.float32)
        local_next_observation = tf.cast(next_obs["local_observation"], tf.float32)

        prev_state = tf.cast(obs["prev_state"], tf.float32)
        next_prev_state = tf.cast(next_obs["prev_state"], tf.float32)

        snapshot = {
            "node_feature": tf.cast(obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        next_snapshot = {
            "node_feature": tf.cast(next_obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(next_obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(next_obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(next_obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        batch_size = tf.shape(local_observation)[0]

        # DEBUG EDIT: Batched call DEBUG
        # if self._debug_count < 2:
        #     self._debug_count += 1    # 下次把这个计数逻辑移到如step()的非tf.function里面，不然会卡bug

        #     old_out = self._run_batched_dyn_gnn_full_old(snapshot, prev_state, self._dyn_gnn, True)
        #     new_out = self._run_batched_dyn_gnn_full(snapshot, prev_state, self._dyn_gnn, True)

        #     def _print_diff(name, x, y):
        #         diff = tf.abs(x - y)
        #         max_diff = tf.reduce_max(diff)
        #         mean_diff = tf.reduce_mean(diff)

        #         tf.print(f"\n[DEBUG]{self._debug_count}", name)
        #         tf.print("shape old =", tf.shape(x), ", shape new =", tf.shape(y))
        #         tf.print("max abs diff =", max_diff)
        #         tf.print("mean abs diff =", mean_diff)

        #         # 打印一个小切片，方便肉眼比较
        #         tf.print("old sample =", tf.reshape(x, [-1])[:8])
        #         tf.print("new sample =", tf.reshape(y, [-1])[:8])
        #         tf.print("diff sample =", tf.reshape(diff, [-1])[:8])

        #     _print_diff("all_node_embedding", old_out["all_node_embedding"], new_out["all_node_embedding"])
        #     _print_diff("edge_embedding", old_out["edge_embedding"], new_out["edge_embedding"])
        #     _print_diff("new_state_list", old_out["new_state_list"], new_out["new_state_list"])


        # ====== target gnn & actions ======
        # target dyn gnn outside tape
        dyn_out_t = self._run_batched_dyn_gnn_full(
            snapshot=next_snapshot,
            prev_state=next_prev_state,
            dyn_gnn_module=self._target_dyn_gnn,
            training=False,
        )
        g_t_all = dyn_out_t["edge_embedding"]

        # target actions
        a_t_list = []
        for i in range(self._agent_number):
            z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
            z_t = tf.cast(z_t, tf.float32)
            z_t = tf.stop_gradient(z_t)

            g_t_i = tf.stop_gradient(g_t_all[:, i, :])
            fused_t = tf.concat([z_t, g_t_i], axis=-1)
            fused_t = tf.cast(fused_t, tf.float32)

            a_t = self._target_policy_networks[i](fused_t)
            a_t_list.append(a_t)

        agent_next_a_t = tf.concat(a_t_list, axis=1)
        agent_next_a_t = tf.reshape(
            agent_next_a_t,
            [batch_size, self._agent_number, self._agent_action_size]
        )

        # ====== outer tape, recording all gradients ======
        with tf.GradientTape(persistent=True) as tape:
            # IMPORTANT: online dyn gnn must be inside tape
            dyn_out_tm1 = self._run_batched_dyn_gnn_full(
                snapshot=snapshot,
                prev_state=prev_state,
                dyn_gnn_module=self._dyn_gnn,
                training=True,
            )

            g_tm1_all = dyn_out_tm1["edge_embedding"]

            critic_losses = []
            policy_losses = []

            for i in range(self._agent_number):
                # current fused state
                z_tm1 = self._observation_networks[i](local_observation[:, i, :])
                z_tm1 = tf.cast(z_tm1, tf.float32)

                g_tm1_i = g_tm1_all[:, i, :]
                fused_tm1 = tf.concat([z_tm1, g_tm1_i], axis=-1)
                fused_tm1 = tf.cast(fused_tm1, tf.float32)

                # target fused state
                z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
                z_t = tf.cast(z_t, tf.float32)
                z_t = tf.stop_gradient(z_t)

                g_t_i = tf.stop_gradient(g_t_all[:, i, :])
                fused_t = tf.concat([z_t, g_t_i], axis=-1)
                fused_t = tf.cast(fused_t, tf.float32)

                # critic loss
                critic_actions_tm1 = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    actions[:, i, :],
                ])
                q_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_tm1, [batch_size, -1])
                )

                target_a_i = a_t_list[i]
                critic_actions_t = tf2_utils.batch_concat([
                    agent_next_a_t[:, :i, :],
                    agent_next_a_t[:, i + 1:, :],
                    target_a_i,
                ])
                q_t = self._target_critic_networks[i](
                    fused_t,
                    tf.reshape(critic_actions_t, [batch_size, -1])
                )

                critic_loss_i = losses.categorical(
                    q_tm1,
                    rewards[:, i],
                    discount * discounts,
                    q_t,
                )
                critic_loss_i = tf.reduce_mean(critic_loss_i, axis=[0])
                critic_losses.append(critic_loss_i)

                # actor loss
                # dpg_a_t = self._policy_networks[i](fused_t)
                # critic_actions_policy = tf2_utils.batch_concat([
                #     actions[:, :i, :],
                #     actions[:, i + 1:, :],
                #     dpg_a_t,
                # ])
                # dpg_z_t = self._critic_networks[i](
                #     fused_t,
                #     tf.reshape(critic_actions_policy, [batch_size, -1])
                # )
                # dpg_q_t = dpg_z_t.mean()
                # 
                # dqda_clipping = 1.0 if self._clipping else None
                # policy_loss_i = losses.dpg(
                #     dpg_q_t,
                #     critic_actions_policy,
                #     tape=tape,
                #     dqda_clipping=dqda_clipping,
                #     clip_norm=self._clipping,
                # )
                # policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
                # policy_losses.append(policy_loss_i)
                dpg_a_tm1 = self._policy_networks[i](fused_tm1)

                critic_actions_policy = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    dpg_a_tm1,
                ])

                dpg_z_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_policy, [batch_size, -1])
                )
                dpg_q_tm1 = dpg_z_tm1.mean()

                dqda_clipping = 1.0 if self._clipping else None
                policy_loss_i = losses.dpg(
                    dpg_q_tm1,
                    critic_actions_policy,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping,
                )
                policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
                policy_losses.append(policy_loss_i)

            # NOTICE ：区别于旧版，求两套loss，sum和mean，一个用于policy与critic梯度下降，一个用于最后的log与gnn
            sum_critic_loss = tf.add_n(critic_losses)
            sum_policy_loss = tf.add_n(policy_losses)

            # mean_critic_loss = sum_critic_loss / float(self._agent_number)
            # mean_policy_loss = sum_policy_loss / float(self._agent_number)

        # ====== optimization ======
        replica_context = tf.distribute.get_replica_context()

        # 1) collect all variables first
        all_policy_vars = []
        policy_var_sizes = []

        all_critic_vars = []
        critic_var_sizes = []

        per_agent_policy_vars = []
        per_agent_critic_vars = []

        for i in range(self._agent_number):
            policy_variables = list(self._policy_networks[i].trainable_variables)
            critic_variables = (
                list(self._observation_networks[i].trainable_variables) +
                list(self._critic_networks[i].trainable_variables)
            )

            per_agent_policy_vars.append(policy_variables)
            per_agent_critic_vars.append(critic_variables)

            all_policy_vars.extend(policy_variables)
            policy_var_sizes.append(len(policy_variables))

            all_critic_vars.extend(critic_variables)
            critic_var_sizes.append(len(critic_variables))

        gnn_variables = list(self._dyn_gnn.trainable_variables)

        # 2) only two major backward passes
        policy_gradients_all = _average_gradients_across_replicas(
            replica_context,
            tape.gradient(sum_policy_loss, all_policy_vars)
        )

        critic_and_gnn_gradients_all = _average_gradients_across_replicas(
            replica_context,
            tape.gradient(sum_critic_loss, all_critic_vars + gnn_variables)
        )

        # split critic and gnn gradients
        num_critic_vars = len(all_critic_vars)
        critic_gradients_all = critic_and_gnn_gradients_all[:num_critic_vars]
        gnn_gradients = critic_and_gnn_gradients_all[num_critic_vars:]

        # NOTICE: 处理gnn gradient为g/agent num, 保持原状，后续可以决定需不需要这一步
        gnn_gradients = [
            g / float(self._agent_number) if g is not None else None
            for g in gnn_gradients
        ]

        # 3) slice gradients back to each agent, keep independent optimizers
        p_start = 0
        c_start = 0

        for i in range(self._agent_number):
            policy_variables = per_agent_policy_vars[i]
            critic_variables = per_agent_critic_vars[i]

            p_len = policy_var_sizes[i]
            c_len = critic_var_sizes[i]

            policy_gradients = policy_gradients_all[p_start:p_start + p_len]
            critic_gradients = critic_gradients_all[c_start:c_start + c_len]

            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

            self._policy_optimizer[i].apply(policy_gradients, policy_variables)
            self._critic_optimizer[i].apply(critic_gradients, critic_variables)

            p_start += p_len
            c_start += c_len

        # 4) apply GNN gradients once
        if self._clipping:
            gnn_gradients = tf.clip_by_global_norm(gnn_gradients, 40.)[0]

        # DEBUG #
        # 正式训练时先删了
        # total_gnn_grad_norm = tf.add_n([
        #     tf.norm(g) for g in gnn_gradients if g is not None
        # ])
        # tf.print("[LEARNER DEBUG] total_gnn_grad_norm =", total_gnn_grad_norm)
        # DEBUG #

        self._gnn_optimizer.apply(gnn_gradients, gnn_variables)

        # ====== optimization end ======


        # # DEBUG #
        # tf.print("[LEARNER DEBUG] learner_gnn_mean =", tf.reduce_mean(self._target_dyn_gnn.variables[0]))
        # tf.print("[LEARNER DEBUG] learner_policy_mean =", tf.reduce_mean(self._target_policy_networks[0].variables[0]))
        # # learner_gnn_mean = tf.reduce_mean(self._target_dyn_gnn.variables[0])
        # # learner_policy_mean = tf.reduce_mean(self._target_policy_networks[0].variables[0])
        # # DEBUG #

        # ====== return ======
        del tape

        new_critic_losses = tf.reduce_mean(tf.stack(critic_losses, axis=0))
        new_policy_losses = tf.reduce_mean(tf.stack(policy_losses, axis=0))

        return {
            "policy_loss": new_policy_losses,
            "critic_loss": new_critic_losses,
        }
    
    # @tf.function
    # def _step(self, sample) -> Dict[str, tf.Tensor]:
    #     transitions: types.Transition = sample.data

    #     # NOTICE: cast f64 to f32, still being CONTROVERSIAL
    #     # ====== numeric tensors cast ======
    #     actions = tf.cast(transitions.action, tf.float32)
    #     rewards = tf.cast(transitions.reward, tf.float32)
    #     discounts = tf.cast(transitions.discount, tf.float32)
    #     discount = tf.cast(self._discount, tf.float32)

    #     # ====== unpack dict obs ======
    #     obs = transitions.observation
    #     next_obs = transitions.next_observation

    #     local_observation = tf.cast(obs["local_observation"], tf.float32)
    #     local_next_observation = tf.cast(next_obs["local_observation"], tf.float32)

    #     prev_state = tf.cast(obs["prev_state"], tf.float32)
    #     next_prev_state = tf.cast(next_obs["prev_state"], tf.float32)

    #     snapshot = {
    #         "node_feature": tf.cast(obs["snapshot"]["node_feature"], tf.float32),
    #         "edge_index": tf.cast(obs["snapshot"]["edge_index"], tf.int32),
    #         "edge_feature": tf.cast(obs["snapshot"]["edge_feature"], tf.float32),
    #         "edge_node_indices": tf.cast(obs["snapshot"]["edge_node_indices"], tf.int32),
    #         "edge_mask": tf.cast(obs["snapshot"]["edge_mask"], tf.float32),         # DynENV-EDIT
    #     }

    #     next_snapshot = {
    #         "node_feature": tf.cast(next_obs["snapshot"]["node_feature"], tf.float32),
    #         "edge_index": tf.cast(next_obs["snapshot"]["edge_index"], tf.int32),
    #         "edge_feature": tf.cast(next_obs["snapshot"]["edge_feature"], tf.float32),
    #         "edge_node_indices": tf.cast(next_obs["snapshot"]["edge_node_indices"], tf.int32),
    #         "edge_mask": tf.cast(next_obs["snapshot"]["edge_mask"], tf.float32),        # DynENV-EDIT
    #     }

    #     batch_size = tf.shape(local_observation)[0]

    #     # DEBUG EDIT: Batched call DEBUG
    #     # if self._debug_count < 2:
    #     #     self._debug_count += 1    # 下次把这个计数逻辑移到如step()的非tf.function里面，不然会卡bug

    #     #     old_out = self._run_batched_dyn_gnn_full_old(snapshot, prev_state, self._dyn_gnn, True)
    #     #     new_out = self._run_batched_dyn_gnn_full(snapshot, prev_state, self._dyn_gnn, True)

    #     #     def _print_diff(name, x, y):
    #     #         diff = tf.abs(x - y)
    #     #         max_diff = tf.reduce_max(diff)
    #     #         mean_diff = tf.reduce_mean(diff)

    #     #         tf.print(f"\n[DEBUG]{self._debug_count}", name)
    #     #         tf.print("shape old =", tf.shape(x), ", shape new =", tf.shape(y))
    #     #         tf.print("max abs diff =", max_diff)
    #     #         tf.print("mean abs diff =", mean_diff)

    #     #         # 打印一个小切片，方便肉眼比较
    #     #         tf.print("old sample =", tf.reshape(x, [-1])[:8])
    #     #         tf.print("new sample =", tf.reshape(y, [-1])[:8])
    #     #         tf.print("diff sample =", tf.reshape(diff, [-1])[:8])

    #     #     _print_diff("all_node_embedding", old_out["all_node_embedding"], new_out["all_node_embedding"])
    #     #     _print_diff("edge_embedding", old_out["edge_embedding"], new_out["edge_embedding"])
    #     #     _print_diff("new_state_list", old_out["new_state_list"], new_out["new_state_list"])


    #     # ====== target gnn & actions ======
    #     # target dyn gnn outside tape
    #     dyn_out_t = self._run_batched_dyn_gnn_full(
    #         snapshot=next_snapshot,
    #         prev_state=next_prev_state,
    #         dyn_gnn_module=self._target_dyn_gnn,
    #         training=False,
    #     )
    #     g_t_all = dyn_out_t["edge_embedding"]

    #     # target actions
    #     a_t_list = []
    #     for i in range(self._agent_number):
    #         z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
    #         z_t = tf.cast(z_t, tf.float32)
    #         z_t = tf.stop_gradient(z_t)

    #         g_t_i = tf.stop_gradient(g_t_all[:, i, :])
    #         fused_t = tf.concat([z_t, g_t_i], axis=-1)
    #         fused_t = tf.cast(fused_t, tf.float32)

    #         a_t = self._target_policy_networks[i](fused_t)
    #         a_t_list.append(a_t)

    #     agent_next_a_t = tf.concat(a_t_list, axis=1)
    #     agent_next_a_t = tf.reshape(
    #         agent_next_a_t,
    #         [batch_size, self._agent_number, self._agent_action_size]
    #     )

    #     # ====== outer tape, recording all gradients ======
    #     with tf.GradientTape(persistent=True) as tape:
    #         # IMPORTANT: online dyn gnn must be inside tape
    #         dyn_out_tm1 = self._run_batched_dyn_gnn_full(
    #             snapshot=snapshot,
    #             prev_state=prev_state,
    #             dyn_gnn_module=self._dyn_gnn,
    #             training=True,
    #         )

    #         g_tm1_all = dyn_out_tm1["edge_embedding"]

    #         critic_losses = []
    #         policy_losses = []

    #         for i in range(self._agent_number):
    #             # current fused state
    #             z_tm1 = self._observation_networks[i](local_observation[:, i, :])
    #             z_tm1 = tf.cast(z_tm1, tf.float32)

    #             g_tm1_i = g_tm1_all[:, i, :]
    #             fused_tm1 = tf.concat([z_tm1, g_tm1_i], axis=-1)
    #             fused_tm1 = tf.cast(fused_tm1, tf.float32)

    #             # target fused state
    #             z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
    #             z_t = tf.cast(z_t, tf.float32)
    #             z_t = tf.stop_gradient(z_t)

    #             g_t_i = tf.stop_gradient(g_t_all[:, i, :])
    #             fused_t = tf.concat([z_t, g_t_i], axis=-1)
    #             fused_t = tf.cast(fused_t, tf.float32)

    #             # critic loss
    #             critic_actions_tm1 = tf2_utils.batch_concat([
    #                 actions[:, :i, :],
    #                 actions[:, i + 1:, :],
    #                 actions[:, i, :],
    #             ])
    #             q_tm1 = self._critic_networks[i](
    #                 fused_tm1,
    #                 tf.reshape(critic_actions_tm1, [batch_size, -1])
    #             )

    #             target_a_i = a_t_list[i]
    #             critic_actions_t = tf2_utils.batch_concat([
    #                 agent_next_a_t[:, :i, :],
    #                 agent_next_a_t[:, i + 1:, :],
    #                 target_a_i,
    #             ])
    #             q_t = self._target_critic_networks[i](
    #                 fused_t,
    #                 tf.reshape(critic_actions_t, [batch_size, -1])
    #             )

    #             critic_loss_i = losses.categorical(
    #                 q_tm1,
    #                 rewards[:, i],
    #                 discount * discounts,
    #                 q_t,
    #             )
    #             critic_loss_i = tf.reduce_mean(critic_loss_i, axis=[0])
    #             critic_losses.append(critic_loss_i)

    #             # actor loss
    #             # dpg_a_t = self._policy_networks[i](fused_t)
    #             # critic_actions_policy = tf2_utils.batch_concat([
    #             #     actions[:, :i, :],
    #             #     actions[:, i + 1:, :],
    #             #     dpg_a_t,
    #             # ])
    #             # dpg_z_t = self._critic_networks[i](
    #             #     fused_t,
    #             #     tf.reshape(critic_actions_policy, [batch_size, -1])
    #             # )
    #             # dpg_q_t = dpg_z_t.mean()
    #             # 
    #             # dqda_clipping = 1.0 if self._clipping else None
    #             # policy_loss_i = losses.dpg(
    #             #     dpg_q_t,
    #             #     critic_actions_policy,
    #             #     tape=tape,
    #             #     dqda_clipping=dqda_clipping,
    #             #     clip_norm=self._clipping,
    #             # )
    #             # policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
    #             # policy_losses.append(policy_loss_i)
    #             dpg_a_tm1 = self._policy_networks[i](fused_tm1)

    #             critic_actions_policy = tf2_utils.batch_concat([
    #                 actions[:, :i, :],
    #                 actions[:, i + 1:, :],
    #                 dpg_a_tm1,
    #             ])

    #             dpg_z_tm1 = self._critic_networks[i](
    #                 fused_tm1,
    #                 tf.reshape(critic_actions_policy, [batch_size, -1])
    #             )
    #             dpg_q_tm1 = dpg_z_tm1.mean()

    #             dqda_clipping = 1.0 if self._clipping else None
    #             policy_loss_i = losses.dpg(
    #                 dpg_q_tm1,
    #                 critic_actions_policy,
    #                 tape=tape,
    #                 dqda_clipping=dqda_clipping,
    #                 clip_norm=self._clipping,
    #             )
    #             policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
    #             policy_losses.append(policy_loss_i)

    #         # NOTICE ：区别于旧版，求两套loss，sum和mean，一个用于policy与critic梯度下降，一个用于最后的log与gnn
    #         sum_critic_loss = tf.add_n(critic_losses)
    #         sum_policy_loss = tf.add_n(policy_losses)

    #         # mean_critic_loss = sum_critic_loss / float(self._agent_number)
    #         # mean_policy_loss = sum_policy_loss / float(self._agent_number)

    #     # ====== optimization ======
    #     replica_context = tf.distribute.get_replica_context()

    #     # 1) collect all variables first
    #     all_policy_vars = []
    #     policy_var_sizes = []

    #     all_critic_vars = []
    #     critic_var_sizes = []

    #     per_agent_policy_vars = []
    #     per_agent_critic_vars = []

    #     for i in range(self._agent_number):
    #         policy_variables = list(self._policy_networks[i].trainable_variables)
    #         critic_variables = (
    #             list(self._observation_networks[i].trainable_variables) +
    #             list(self._critic_networks[i].trainable_variables)
    #         )

    #         per_agent_policy_vars.append(policy_variables)
    #         per_agent_critic_vars.append(critic_variables)

    #         all_policy_vars.extend(policy_variables)
    #         policy_var_sizes.append(len(policy_variables))

    #         all_critic_vars.extend(critic_variables)
    #         critic_var_sizes.append(len(critic_variables))

    #     gnn_variables = list(self._dyn_gnn.trainable_variables)

    #     # 2) only two major backward passes
    #     policy_gradients_all = _average_gradients_across_replicas(
    #         replica_context,
    #         tape.gradient(sum_policy_loss, all_policy_vars)
    #     )

    #     critic_and_gnn_gradients_all = _average_gradients_across_replicas(
    #         replica_context,
    #         tape.gradient(sum_critic_loss, all_critic_vars + gnn_variables)
    #     )

    #     # split critic and gnn gradients
    #     num_critic_vars = len(all_critic_vars)
    #     critic_gradients_all = critic_and_gnn_gradients_all[:num_critic_vars]
    #     gnn_gradients = critic_and_gnn_gradients_all[num_critic_vars:]

    #     # NOTICE: 处理gnn gradient为g/agent num, 保持原状，后续可以决定需不需要这一步
    #     gnn_gradients = [
    #         g / float(self._agent_number) if g is not None else None
    #         for g in gnn_gradients
    #     ]

    #     # 3) slice gradients back to each agent, keep independent optimizers
    #     p_start = 0
    #     c_start = 0

    #     for i in range(self._agent_number):
    #         policy_variables = per_agent_policy_vars[i]
    #         critic_variables = per_agent_critic_vars[i]

    #         p_len = policy_var_sizes[i]
    #         c_len = critic_var_sizes[i]

    #         policy_gradients = policy_gradients_all[p_start:p_start + p_len]
    #         critic_gradients = critic_gradients_all[c_start:c_start + c_len]

    #         if self._clipping:
    #             policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
    #             critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

    #         self._policy_optimizer[i].apply(policy_gradients, policy_variables)
    #         self._critic_optimizer[i].apply(critic_gradients, critic_variables)

    #         p_start += p_len
    #         c_start += c_len

    #     # 4) apply GNN gradients once
    #     if self._clipping:
    #         gnn_gradients = tf.clip_by_global_norm(gnn_gradients, 40.)[0]

    #     # DEBUG #
    #     # 正式训练时先删了
    #     # total_gnn_grad_norm = tf.add_n([
    #     #     tf.norm(g) for g in gnn_gradients if g is not None
    #     # ])
    #     # tf.print("[LEARNER DEBUG] total_gnn_grad_norm =", total_gnn_grad_norm)
    #     # DEBUG #

    #     self._gnn_optimizer.apply(gnn_gradients, gnn_variables)

    #     # ====== optimization end ======


    #     # # DEBUG #
    #     # tf.print("[LEARNER DEBUG] learner_gnn_mean =", tf.reduce_mean(self._target_dyn_gnn.variables[0]))
    #     # tf.print("[LEARNER DEBUG] learner_policy_mean =", tf.reduce_mean(self._target_policy_networks[0].variables[0]))
    #     # # learner_gnn_mean = tf.reduce_mean(self._target_dyn_gnn.variables[0])
    #     # # learner_policy_mean = tf.reduce_mean(self._target_policy_networks[0].variables[0])
    #     # # DEBUG #

    #     # ====== return ======
    #     del tape

    #     new_critic_losses = tf.reduce_mean(tf.stack(critic_losses, axis=0))
    #     new_policy_losses = tf.reduce_mean(tf.stack(policy_losses, axis=0))

    #     return {
    #         "policy_loss": new_policy_losses,
    #         "critic_loss": new_critic_losses,
    #     }

    @tf.function
    def _step_joint_gnn(self, sample) -> Dict[str, tf.Tensor]:
        transitions: types.Transition = sample.data

        # ====== numeric tensors cast ======
        actions = tf.cast(transitions.action, tf.float32)
        rewards = tf.cast(transitions.reward, tf.float32)
        discounts = tf.cast(transitions.discount, tf.float32)
        discount = tf.cast(self._discount, tf.float32)

        # ====== unpack dict obs ======
        obs = transitions.observation
        next_obs = transitions.next_observation

        local_observation = tf.cast(obs["local_observation"], tf.float32)
        local_next_observation = tf.cast(next_obs["local_observation"], tf.float32)

        prev_state = tf.cast(obs["prev_state"], tf.float32)
        next_prev_state = tf.cast(next_obs["prev_state"], tf.float32)

        snapshot = {
            "node_feature": tf.cast(obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        next_snapshot = {
            "node_feature": tf.cast(next_obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(next_obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(next_obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(next_obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        batch_size = tf.shape(local_observation)[0]

        # ====== target gnn & target actions ======
        dyn_out_t = self._run_batched_dyn_gnn_full(
            snapshot=next_snapshot,
            prev_state=next_prev_state,
            dyn_gnn_module=self._target_dyn_gnn,
            training=False,
        )
        g_t_all = dyn_out_t["edge_embedding"]   # [B, E, d_g]

        a_t_list = []
        for i in range(self._agent_number):
            z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
            z_t = tf.cast(z_t, tf.float32)
            z_t = tf.stop_gradient(z_t)

            g_t_i = tf.stop_gradient(g_t_all[:, i, :])
            fused_t = tf.concat([z_t, g_t_i], axis=-1)
            fused_t = tf.cast(fused_t, tf.float32)

            a_t = self._target_policy_networks[i](fused_t)
            a_t_list.append(a_t)

        agent_next_a_t = tf.concat(a_t_list, axis=1)
        agent_next_a_t = tf.reshape(
            agent_next_a_t,
            [batch_size, self._agent_number, self._agent_action_size]
        )

        # ====== forward inside tape ======
        with tf.GradientTape(persistent=True) as tape:
            dyn_out_tm1 = self._run_batched_dyn_gnn_full(
                snapshot=snapshot,
                prev_state=prev_state,
                dyn_gnn_module=self._dyn_gnn,
                training=True,
            )
            g_tm1_all = dyn_out_tm1["edge_embedding"]  # [B, E, d_g]

            critic_losses = []
            policy_losses = []
            # gnn_policy_losses = []
            
            # joint path
            lambda_gnn_policy = tf.constant(0.3, dtype=tf.float32)
            # sum_gnn_policy_loss = tf.add_n(gnn_policy_losses)

            for i in range(self._agent_number):
                # ----- current fused state -----
                z_tm1 = self._observation_networks[i](local_observation[:, i, :])
                z_tm1 = tf.cast(z_tm1, tf.float32)

                g_tm1_i = g_tm1_all[:, i, :]
                fused_tm1 = tf.concat([z_tm1, g_tm1_i], axis=-1)
                fused_tm1 = tf.cast(fused_tm1, tf.float32)

                # ----- target fused state -----
                z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
                z_t = tf.cast(z_t, tf.float32)
                z_t = tf.stop_gradient(z_t)

                g_t_i = tf.stop_gradient(g_t_all[:, i, :])
                fused_t = tf.concat([z_t, g_t_i], axis=-1)
                fused_t = tf.cast(fused_t, tf.float32)

                # ===== critic loss =====
                critic_actions_tm1 = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    actions[:, i, :],
                ])
                q_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_tm1, [batch_size, -1])
                )

                target_a_i = a_t_list[i]
                critic_actions_t = tf2_utils.batch_concat([
                    agent_next_a_t[:, :i, :],
                    agent_next_a_t[:, i + 1:, :],
                    target_a_i,
                ])
                q_t = self._target_critic_networks[i](
                    fused_t,
                    tf.reshape(critic_actions_t, [batch_size, -1])
                )

                critic_loss_i = losses.categorical(
                    q_tm1,
                    rewards[:, i],
                    discount * discounts,
                    q_t,
                )
                critic_loss_i = tf.reduce_mean(critic_loss_i, axis=[0])
                critic_losses.append(critic_loss_i)

                # ===== actor loss: use CURRENT fused state =====
                dpg_a_tm1 = self._policy_networks[i](fused_tm1)

                critic_actions_policy = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    dpg_a_tm1,
                ])

                dpg_z_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_policy, [batch_size, -1])
                )
                dpg_q_tm1 = dpg_z_tm1.mean()

                # gnn_policy_losses.append(-dpg_q_tm1)

                dqda_clipping = 1.0 if self._clipping else None
                policy_loss_i = losses.dpg(
                    dpg_q_tm1,
                    critic_actions_policy,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping,
                )
                policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
                policy_losses.append(policy_loss_i)

            sum_critic_loss = tf.add_n(critic_losses)
            sum_policy_loss = tf.add_n(policy_losses)
            gnn_total_loss = sum_critic_loss + lambda_gnn_policy * sum_policy_loss

        # ====== optimization ======
        replica_context = tf.distribute.get_replica_context()

        all_policy_vars = []
        policy_var_sizes = []

        all_critic_vars = []
        critic_var_sizes = []

        per_agent_policy_vars = []
        per_agent_critic_vars = []

        for i in range(self._agent_number):
            policy_variables = list(self._policy_networks[i].trainable_variables)
            critic_variables = (
                list(self._observation_networks[i].trainable_variables) +
                list(self._critic_networks[i].trainable_variables)
            )

            per_agent_policy_vars.append(policy_variables)
            per_agent_critic_vars.append(critic_variables)

            all_policy_vars.extend(policy_variables)
            policy_var_sizes.append(len(policy_variables))

            all_critic_vars.extend(critic_variables)
            critic_var_sizes.append(len(critic_variables))

        gnn_variables = list(self._dyn_gnn.trainable_variables)

        # 1) policy gradients: policy-only
        policy_gradients_all = _average_gradients_across_replicas(
            replica_context,
            tape.gradient(sum_policy_loss, all_policy_vars)
        )

        # 2) critic gradients: critic-only
        critic_gradients_all = _average_gradients_across_replicas(
            replica_context,
            tape.gradient(sum_critic_loss, all_critic_vars)
        )

        # 3) gnn gradients: critic + lambda * policy
        # NOTICE: first diagnose, then choose a safe path
        gnn_variables = list(self._dyn_gnn.trainable_variables)

        # critic-only path (known baseline path)
        raw_gnn_grads_critic = tape.gradient(sum_critic_loss, gnn_variables)

        # joint path
        raw_gnn_grads_joint = tape.gradient(gnn_total_loss, gnn_variables)

        # Python-side diagnostics (works fine here because gradients list is a Python list)
        critic_non_none = sum(g is not None for g in raw_gnn_grads_critic)
        joint_non_none = sum(g is not None for g in raw_gnn_grads_joint)

        # tf.print("[JOINT GNN DEBUG] len(gnn_variables) =", len(gnn_variables))
        # tf.print("[JOINT GNN DEBUG] critic_non_none =", critic_non_none,
        #         ", joint_non_none =", joint_non_none)

        # Prefer joint gradients if they exist; otherwise safely fall back to critic-only.
        raw_gnn_grads = raw_gnn_grads_joint if joint_non_none > 0 else raw_gnn_grads_critic

        gnn_gradients = _average_gradients_across_replicas(
            replica_context,
            raw_gnn_grads
        )
        
        gnn_gradients = [
            g / float(self._agent_number) if g is not None else None
            for g in gnn_gradients
        ]

        # Filter out None gradients before apply
        non_none_pairs = [(g, v) for g, v in zip(gnn_gradients, gnn_variables) if g is not None]

        if non_none_pairs:
            apply_grads, apply_vars = zip(*non_none_pairs)
            apply_grads = list(apply_grads)
            apply_vars = list(apply_vars)

            if self._clipping:
                apply_grads = tf.clip_by_global_norm(apply_grads, 40.)[0]

            self._gnn_optimizer.apply(apply_grads, apply_vars)
        else:
            tf.print("[JOINT GNN DEBUG] skip gnn apply: all gradients are None")

        # 4) slice gradients back to each agent
        p_start = 0
        c_start = 0

        for i in range(self._agent_number):
            policy_variables = per_agent_policy_vars[i]
            critic_variables = per_agent_critic_vars[i]

            p_len = policy_var_sizes[i]
            c_len = critic_var_sizes[i]

            policy_gradients = policy_gradients_all[p_start:p_start + p_len]
            critic_gradients = critic_gradients_all[c_start:c_start + c_len]

            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

            self._policy_optimizer[i].apply(policy_gradients, policy_variables)
            self._critic_optimizer[i].apply(critic_gradients, critic_variables)

            p_start += p_len
            c_start += c_len


        del tape

        new_critic_losses = tf.reduce_mean(tf.stack(critic_losses, axis=0))
        new_policy_losses = tf.reduce_mean(tf.stack(policy_losses, axis=0))

        return {
            "policy_loss": new_policy_losses,
            "critic_loss": new_critic_losses,
        }


    @tf.function
    def _step_old(self, sample) -> Dict[str, tf.Tensor]:
        transitions: types.Transition = sample.data

        # NOTICE: cast f64 to f32, still being CONTROVERSIAL
        # ====== numeric tensors cast ======
        actions = tf.cast(transitions.action, tf.float32)
        rewards = tf.cast(transitions.reward, tf.float32)
        discounts = tf.cast(transitions.discount, tf.float32)
        discount = tf.cast(self._discount, tf.float32)

        # ====== unpack dict obs ======
        obs = transitions.observation
        next_obs = transitions.next_observation

        local_observation = tf.cast(obs["local_observation"], tf.float32)
        local_next_observation = tf.cast(next_obs["local_observation"], tf.float32)

        prev_state = tf.cast(obs["prev_state"], tf.float32)
        next_prev_state = tf.cast(next_obs["prev_state"], tf.float32)

        snapshot = {
            "node_feature": tf.cast(obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        next_snapshot = {
            "node_feature": tf.cast(next_obs["snapshot"]["node_feature"], tf.float32),
            "edge_index": tf.cast(next_obs["snapshot"]["edge_index"], tf.int32),
            "edge_feature": tf.cast(next_obs["snapshot"]["edge_feature"], tf.float32),
            "edge_node_indices": tf.cast(next_obs["snapshot"]["edge_node_indices"], tf.int32),
        }

        batch_size = tf.shape(local_observation)[0]

        # DEBUG EDIT: Batched call DEBUG
        # if self._debug_count < 2:
        #     self._debug_count += 1    # 下次把这个计数逻辑移到如step()的非tf.function里面，不然会卡bug

        #     old_out = self._run_batched_dyn_gnn_full_old(snapshot, prev_state, self._dyn_gnn, True)
        #     new_out = self._run_batched_dyn_gnn_full(snapshot, prev_state, self._dyn_gnn, True)

        #     def _print_diff(name, x, y):
        #         diff = tf.abs(x - y)
        #         max_diff = tf.reduce_max(diff)
        #         mean_diff = tf.reduce_mean(diff)

        #         tf.print(f"\n[DEBUG]{self._debug_count}", name)
        #         tf.print("shape old =", tf.shape(x), ", shape new =", tf.shape(y))
        #         tf.print("max abs diff =", max_diff)
        #         tf.print("mean abs diff =", mean_diff)

        #         # 打印一个小切片，方便肉眼比较
        #         tf.print("old sample =", tf.reshape(x, [-1])[:8])
        #         tf.print("new sample =", tf.reshape(y, [-1])[:8])
        #         tf.print("diff sample =", tf.reshape(diff, [-1])[:8])

        #     _print_diff("all_node_embedding", old_out["all_node_embedding"], new_out["all_node_embedding"])
        #     _print_diff("edge_embedding", old_out["edge_embedding"], new_out["edge_embedding"])
        #     _print_diff("new_state_list", old_out["new_state_list"], new_out["new_state_list"])


        # ====== target gnn & actions ======
        # target dyn gnn outside tape
        dyn_out_t = self._run_batched_dyn_gnn_full( # EDIT: 该为old版本做一个实验
            snapshot=next_snapshot,
            prev_state=next_prev_state,
            dyn_gnn_module=self._target_dyn_gnn,
            training=False,
        )
        g_t_all = dyn_out_t["edge_embedding"]

        # target actions
        a_t_list = []
        for i in range(self._agent_number):
            z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
            z_t = tf.cast(z_t, tf.float32)
            z_t = tf.stop_gradient(z_t)

            g_t_i = tf.stop_gradient(g_t_all[:, i, :])
            fused_t = tf.concat([z_t, g_t_i], axis=-1)
            fused_t = tf.cast(fused_t, tf.float32)

            a_t = self._target_policy_networks[i](fused_t)
            a_t_list.append(a_t)

        agent_next_a_t = tf.concat(a_t_list, axis=1)
        agent_next_a_t = tf.reshape(
            agent_next_a_t,
            [batch_size, self._agent_number, self._agent_action_size]
        )

        # ====== outer tape, recording all gradients ======
        with tf.GradientTape(persistent=True) as tape:
            # IMPORTANT: online dyn gnn must be inside tape
            dyn_out_tm1 = self._run_batched_dyn_gnn_full(
                snapshot=snapshot,
                prev_state=prev_state,
                dyn_gnn_module=self._dyn_gnn,
                training=True,
            )

            g_tm1_all = dyn_out_tm1["edge_embedding"]

            critic_losses = []
            policy_losses = []

            for i in range(self._agent_number):
                # current fused state
                z_tm1 = self._observation_networks[i](local_observation[:, i, :])
                z_tm1 = tf.cast(z_tm1, tf.float32)

                g_tm1_i = g_tm1_all[:, i, :]
                fused_tm1 = tf.concat([z_tm1, g_tm1_i], axis=-1)
                fused_tm1 = tf.cast(fused_tm1, tf.float32)

                # target fused state
                z_t = self._target_observation_networks[i](local_next_observation[:, i, :])
                z_t = tf.cast(z_t, tf.float32)
                z_t = tf.stop_gradient(z_t)

                g_t_i = tf.stop_gradient(g_t_all[:, i, :])
                fused_t = tf.concat([z_t, g_t_i], axis=-1)
                fused_t = tf.cast(fused_t, tf.float32)

                # critic loss
                critic_actions_tm1 = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    actions[:, i, :],
                ])
                q_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_tm1, [batch_size, -1])
                )

                target_a_i = a_t_list[i]
                critic_actions_t = tf2_utils.batch_concat([
                    agent_next_a_t[:, :i, :],
                    agent_next_a_t[:, i + 1:, :],
                    target_a_i,
                ])
                q_t = self._target_critic_networks[i](
                    fused_t,
                    tf.reshape(critic_actions_t, [batch_size, -1])
                )

                critic_loss_i = losses.categorical(
                    q_tm1,
                    rewards[:, i],
                    discount * discounts,
                    q_t,
                )
                critic_loss_i = tf.reduce_mean(critic_loss_i, axis=[0])
                critic_losses.append(critic_loss_i)

                # actor loss
                dpg_a_tm1 = self._policy_networks[i](fused_tm1)

                critic_actions_policy = tf2_utils.batch_concat([
                    actions[:, :i, :],
                    actions[:, i + 1:, :],
                    dpg_a_tm1,
                ])

                dpg_z_tm1 = self._critic_networks[i](
                    fused_tm1,
                    tf.reshape(critic_actions_policy, [batch_size, -1])
                )
                dpg_q_tm1 = dpg_z_tm1.mean()

                dqda_clipping = 1.0 if self._clipping else None
                policy_loss_i = losses.dpg(
                    dpg_q_tm1,
                    critic_actions_policy,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping,
                )
                policy_loss_i = tf.reduce_mean(policy_loss_i, axis=[0])
                policy_losses.append(policy_loss_i)

            total_critic_loss = tf.add_n(critic_losses) / float(self._agent_number)
            total_policy_loss = tf.add_n(policy_losses) / float(self._agent_number)

        # ====== optimization ======
        replica_context = tf.distribute.get_replica_context()

        # policy/critic optimization for agent i
        for i in range(self._agent_number):
            policy_variables = self._policy_networks[i].trainable_variables
            critic_variables = (
                self._observation_networks[i].trainable_variables +
                self._critic_networks[i].trainable_variables
            )

            # IMPORTANT:
            # 用各自的 loss，而不是 total loss，更贴近你原来的独立优化逻辑
            policy_gradients = _average_gradients_across_replicas(
                replica_context,
                tape.gradient(policy_losses[i], policy_variables)
            )
            critic_gradients = _average_gradients_across_replicas(
                replica_context,
                tape.gradient(critic_losses[i], critic_variables)
            )

            if self._clipping:
                policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
                critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

            self._policy_optimizer[i].apply(policy_gradients, policy_variables)
            self._critic_optimizer[i].apply(critic_gradients, critic_variables)

        # DynGNN optimization
        gnn_variables = self._dyn_gnn.trainable_variables
        gnn_gradients = _average_gradients_across_replicas(
            replica_context,
            tape.gradient(total_critic_loss, gnn_variables)
        )

        if self._clipping:
            gnn_gradients = tf.clip_by_global_norm(gnn_gradients, 40.)[0]

        # DEBUG #
        # # 正式训练时先删了
        # total_gnn_grad_norm = tf.add_n([
        #     tf.norm(g) for g in gnn_gradients if g is not None
        # ])
        # tf.print("[LEARNER DEBUG] total_gnn_grad_norm =", total_gnn_grad_norm)
        # DEBUG #

        self._gnn_optimizer.apply(gnn_gradients, gnn_variables)

        # # DEBUG #
        # tf.print("[LEARNER DEBUG] learner_gnn_mean =", tf.reduce_mean(self._target_dyn_gnn.variables[0]))
        # tf.print("[LEARNER DEBUG] learner_policy_mean =", tf.reduce_mean(self._target_policy_networks[0].variables[0]))
        # # learner_gnn_mean = tf.reduce_mean(self._target_dyn_gnn.variables[0])
        # # learner_policy_mean = tf.reduce_mean(self._target_policy_networks[0].variables[0])
        # # DEBUG #

        # ====== return ======
        del tape

        new_critic_losses = tf.reduce_mean(tf.stack(critic_losses, axis=0))
        new_policy_losses = tf.reduce_mean(tf.stack(policy_losses, axis=0))

        return {
            "policy_loss": new_policy_losses,
            "critic_loss": new_critic_losses,
        }


    @tf.function
    def _replicated_step(self):
        # EDIT: GNN, update target dyn_gnn module
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(self._dyn_gnn.variables, self._target_dyn_gnn.variables):
                dest.assign(src)

        # Update target network
        for i in range(self._agent_number):
            online_variables = (
                *self._observation_networks[i].variables,
                *self._critic_networks[i].variables,
                *self._policy_networks[i].variables,
            )
            target_variables = (
                *self._target_observation_networks[i].variables,
                *self._target_critic_networks[i].variables,
                *self._target_policy_networks[i].variables,
            )

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)

        # This mirrors the structure of the fetches returned by self._step(),
        # but the Tensors are replaced with replicated Tensors, one per accelerator.
        replicated_fetches = self._replicator.run(self._step_joint_gnn, args=(sample,))    # NOTICE: 调用最核心训练函数_step(), 试验一下，改成old版

        def reduce_mean_over_replicas(replicated_value):
            """Averages a replicated_value across replicas."""
            # The "axis=None" arg means reduce across replicas, not internal axes.
            return self._replicator.reduce(
                reduce_op=tf.distribute.ReduceOp.MEAN,
                value=replicated_value,
            axis=None)

        fetches = tree.map_structure(reduce_mean_over_replicas, replicated_fetches)

        return fetches

    def step(self):
        # Run the learning step.
        fetches = self._replicated_step() # NOTICE：中间层训练函数调用位置

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
    
    # ============= Helper Function ============= #
    def _unstack_prev_state(self, prev_state_single: tf.Tensor):
        """[L, N, d] -> List[[N, d]]"""
        num_layers = prev_state_single.shape[0]
        return [prev_state_single[i] for i in range(num_layers)]


    def _run_single_dyn_gnn_full(
        self,
        snapshot_single: Dict[str, tf.Tensor],
        prev_state_single: tf.Tensor,
        dyn_gnn_module: snt.Module,
        training: bool,
    ):
        """Run DynGNN on a single graph sample.

        Args:
        snapshot_single:
            node_feature [N, d_node]
            edge_index [2, M]
            edge_feature [M, d_edge]
            edge_node_indices [E]
        prev_state_single:
            [L, N, d]
        Returns:
        dict with:
            new_state_list: [L, N, d]
            all_node_embedding: [N, d]
            edge_embedding: [E, d]
        """
        prev_state_list = self._unstack_prev_state(prev_state_single)

        out = dyn_gnn_module(
            snapshot=snapshot_single,
            prev_state_list=prev_state_list,
            training=training,
        )

        # stack new_state_list back to [L, N, d]
        stacked_new_state = tf.stack(out["new_state_list"], axis=0)

        return {
            "new_state_list": stacked_new_state,
            "all_node_embedding": out["all_node_embedding"],
            "edge_embedding": out["edge_embedding"],
        }

    
    # EDIT: Performance improvement
    def _run_batched_dyn_gnn_full_old(
        self,
        snapshot: Dict[str, tf.Tensor],
        prev_state: tf.Tensor,
        dyn_gnn_module: snt.Module,
        training: bool,
    ):
        """Run DynGNN on a batch of graph samples.

        snapshot:
        node_feature: [B, N, d_node]
        edge_index: [B, 2, M]
        edge_feature: [B, M, d_edge]
        edge_node_indices: [B, E]
        prev_state:
        [B, L, N, d]

        Returns:
        dict with:
            new_state_list: [B, L, N, d]
            all_node_embedding: [B, N, d]
            edge_embedding: [B, E, d]
        """

        def _single_fn(inputs):
            node_feature, edge_index, edge_feature, edge_node_indices, prev_state_single = inputs

            snapshot_single = {
                "node_feature": node_feature,
                "edge_index": edge_index,
                "edge_feature": edge_feature,
                "edge_node_indices": edge_node_indices,
            }

            out = self._run_single_dyn_gnn_full(
                snapshot_single=snapshot_single,
                prev_state_single=prev_state_single,
                dyn_gnn_module=dyn_gnn_module,
                training=training,
            )

            return (
                out["new_state_list"],      # [L, N, d]
                out["all_node_embedding"],  # [N, d]
                out["edge_embedding"],      # [E, d]
            )

        elems = (
            snapshot["node_feature"],
            snapshot["edge_index"],
            snapshot["edge_feature"],
            snapshot["edge_node_indices"],
            prev_state,
        )

        out_signature = (
            tf.TensorSpec(
                shape=(self._dyn_gnn.num_layers, None, self._gnn_embedding_dim),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(None, self._gnn_embedding_dim),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(self._agent_number, self._gnn_embedding_dim),
                dtype=tf.float32,
            ),
        )

        new_state_b, all_node_b, edge_b = tf.map_fn(
            _single_fn,
            elems,
            fn_output_signature=out_signature,
        )

        return {
            "new_state_list": new_state_b,         # [B, L, N, d]
            "all_node_embedding": all_node_b,      # [B, N, d]
            "edge_embedding": edge_b,              # [B, E, d]
        }

    
    def _run_batched_dyn_gnn_full(
        self,
        snapshot,
        prev_state,
        dyn_gnn_module,
        training: bool,
    ):
        return run_dyn_gnn_batched(
            dyn_gnn_module=dyn_gnn_module,
            snapshot=snapshot,
            prev_state=prev_state,
            training=training,
        )



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


# def _average_gradients_across_replicas(replica_context, gradients):
#     """Computes the average gradient across replicas.

#     This computes the gradient locally on this device, then copies over the
#     gradients computed on the other replicas, and takes the average across
#     replicas.

#     This is faster than copying the gradients from TPU to CPU, and averaging
#     them on the CPU (which is what we do for the losses/fetches).

#     Args:
#         replica_context: the return value of `tf.distribute.get_replica_context()`.
#         gradients: The output of tape.gradients(loss, variables)

#     Returns:
#         A list of (d_loss/d_varabiable)s.
#     """

#     # We must remove any Nones from gradients before passing them to all_reduce.
#     # Nones occur when you call tape.gradient(loss, variables) with some
#     # variables that don't affect the loss.
#     # See: https://github.com/tensorflow/tensorflow/issues/783
#     gradients_without_nones = [g for g in gradients if g is not None]
#     original_indices = [i for i, g in enumerate(gradients) if g is not None]

#     results_without_nones = replica_context.all_reduce('mean',
#                                                         gradients_without_nones)
#     results = [None] * len(gradients)
#     for ii, result in zip(original_indices, results_without_nones):
#         results[ii] = result

#     return results

def _average_gradients_across_replicas(replica_context, gradients):
    """Average gradients across replicas when multi-replica training is active.

    In single-replica training, or when no valid replica reduction context exists,
    just return gradients unchanged.
    """
    gradients_without_nones = [g for g in gradients if g is not None]
    original_indices = [i for i, g in enumerate(gradients) if g is not None]

    results = [None] * len(gradients)

    # No gradients at all.
    if not gradients_without_nones:
        return results

    # Robust single-replica detection.
    strategy = tf.distribute.get_strategy()
    num_replicas = getattr(strategy, "num_replicas_in_sync", 1)

    # If there is no real multi-replica setup, just return gradients unchanged.
    if replica_context is None or num_replicas <= 1:
        for ii, g in zip(original_indices, gradients_without_nones):
            results[ii] = g
        return results

    # True multi-replica path.
    results_without_nones = replica_context.all_reduce('mean', gradients_without_nones)
    for ii, result in zip(original_indices, results_without_nones):
        results[ii] = result

    return results
