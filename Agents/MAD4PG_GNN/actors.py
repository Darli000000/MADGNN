# python3
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

# EDIT 3/13/26: Added dynamic gnn module

"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional, List, Dict, Any

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

tfd = tfp.distributions


def _ensure_tensor_edge_node_indices(snapshot: Dict[str, Any]) -> tf.Tensor:
    """snapshot['edge_node_indices'] 可以是 dict 或 array，这里统一成 [E] int32 tensor."""
    edge_node_indices = snapshot["edge_node_indices"]

    if isinstance(edge_node_indices, dict):
        keys = sorted(edge_node_indices.keys())
        vals = [edge_node_indices[k] for k in keys]
        return tf.convert_to_tensor(vals, dtype=tf.int32)

    return tf.convert_to_tensor(edge_node_indices, dtype=tf.int32)


class FeedForwardActor(core.Actor):
    """Actor with shared global DynGNN + per-agent local policy."""

    def __init__(
        self,
        agent_number: int,
        agent_action_size: int,
        observation_networks: List[snt.Module],
        policy_networks: List[snt.Module],
        dyn_gnn_module: snt.Module,
        gnn_embedding_dim: int,
        gnn_is_dynmic: bool,
        adder: Optional[adders.Adder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        sigma: Optional[float] = 0.2,
    ):
        self._adder = adder
        self._variable_client = variable_client

        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._dyn_gnn_module = dyn_gnn_module

        self._agent_number = agent_number
        self._agent_action_size = agent_action_size
        self._gnn_embedding_dim = gnn_embedding_dim

        # episode-level global graph memory
        self._prev_state_list = None
        self._sigma = sigma

        self._gnn_is_dynmic = gnn_is_dynmic

        # # EDIT DEBUG Parameter
        # self._sync_debug_count = 0
        # # DEBUG V0.2
        # self._debug_gnn_step = 0
        # self._debug_episode_index = 0
        

    def _compute_graph_context(self, snapshot: Dict[str, Any]) -> tf.Tensor:
        """Run DynGNN once globally, update prev_state_list, return edge embeddings [E, d_g]."""
        # snapshot 中 edge_node_indices 统一成 tensor
        snapshot = dict(snapshot)
        snapshot["edge_node_indices"] = _ensure_tensor_edge_node_indices(snapshot)

        out = self._dyn_gnn_module(
            snapshot=snapshot,
            prev_state_list=self._prev_state_list,
            training=False,
        )

        # ============ V0.2 DEBUG ============ #
        # if self._debug_episode_index <= 2 and self._debug_gnn_step < 3:
        #     if self._prev_state_list is None:
        #         print(f"[ROLAND DEBUG] ep={self._debug_episode_index} step={self._debug_gnn_step} prev_state = None")
        #     else:
        #         prev0 = self._prev_state_list[0].numpy()
        #         print(f"[ROLAND DEBUG] ep={self._debug_episode_index} step={self._debug_gnn_step} prev_state[0] mean={prev0.mean():.6f}, std={prev0.std():.6f}")

        # out = self._dyn_gnn_module(
        #     snapshot=snapshot,
        #     prev_state_list=self._prev_state_list,
        #     training=False,
        # )

        # if self._debug_episode_index <= 2 and self._debug_gnn_step < 3:
        #     new0 = out["new_state_list"][0].numpy()
        #     print(f"[ROLAND DEBUG] ep={self._debug_episode_index} step={self._debug_gnn_step} new_state[0] mean={new0.mean():.6f}, std={new0.std():.6f}")

        #     if self._prev_state_list is not None:
        #         diff = np.mean(np.abs(new0 - prev0))
        #         print(f"[ROLAND DEBUG] ep={self._debug_episode_index} step={self._debug_gnn_step} mean_abs(new-prev)={diff:.6f}")

        # # READOUT DEBUG
        # if self._debug_episode_index <= 2 and self._debug_gnn_step < 3:
        #     H = out["all_node_embedding"].numpy()
        #     G = out["edge_embedding"].numpy()
        #     edge_node_indices = snapshot["edge_node_indices"]

        #     if isinstance(edge_node_indices, dict):
        #         edge_node_indices = np.array(
        #             [edge_node_indices[k] for k in sorted(edge_node_indices.keys())],
        #             dtype=np.int32
        #         )
        #     else:
        #         edge_node_indices = np.asarray(edge_node_indices, dtype=np.int32)

        #     e = 0  # 先检查第0个edge agent
        #     idx = edge_node_indices[e]
        #     readout_diff = np.max(np.abs(G[e] - H[idx]))
        #     print(f"[ROLAND READOUT DEBUG] ep={self._debug_episode_index} step={self._debug_gnn_step} edge={e} node_idx={idx} max_abs(g-H[idx])={readout_diff:.8f}")

        # # self._prev_state_list = out["new_state_list"]
        # self._debug_gnn_step += 1
        # ============ V0.2 DEBUG ============ #

        if self._gnn_is_dynmic:
            # print("GNN STATIC DEBUG: Dynamic")
            out = self._dyn_gnn_module(
                snapshot=snapshot,
                prev_state_list=self._prev_state_list,
                training=False,
            )
            self._prev_state_list = out["new_state_list"]
        else:
            # print("GNN STATIC DEBUG: Static")
            out = self._dyn_gnn_module(
                snapshot=snapshot,
                prev_state_list=None,
                training=False,
            )
            self._prev_state_list = None

        edge_embedding = out["edge_embedding"]   # [E, d_g]

        return edge_embedding

    def _select_action_from_bundle(self, observation_bundle: Dict[str, Any]) -> tf.Tensor:
        """Core actor logic:
        snapshot -> DynGNN -> g_edge
        local_observation -> z_edge
        [z_edge ; g_edge] -> policy
        """
        local_observation = tf.convert_to_tensor(
            observation_bundle["local_observation"], dtype=tf.float32
        )  # [E, obs_dim]

        snapshot = observation_bundle["snapshot"]
        edge_embedding = self._compute_graph_context(snapshot)  # [E, d_g]

        agent_actions = []
        for i in range(self._agent_number):
            # local obs of agent i
            agent_observation = local_observation[i, :]  # [obs_dim]
            agent_batched_observation = tf2_utils.add_batch_dim(agent_observation)  # [1, obs_dim]

            # old observation network: o_e -> z_e
            z_e = self._observation_networks[i](agent_batched_observation)  # [1, z_dim] usually

            # graph context: g_e
            g_e = edge_embedding[i, :]                       # [d_g]
            g_e = tf2_utils.add_batch_dim(g_e)              # [1, d_g]
            g_e = tf.cast(g_e, z_e.dtype)

            # fuse
            fused = tf.concat([z_e, g_e], axis=-1)          # [1, z_dim + d_g]

            # per-agent policy
            agent_policy = self._policy_networks[i](fused)

            # agent_action = (
            #     agent_policy.sample()
            #     if isinstance(agent_policy, tfd.Distribution)
            #     else agent_policy
            # )
            # EDIT : Add evaluator action logic
            agent_action = None
            if isinstance(agent_policy, tfd.Distribution):
                if self._adder is not None:    # actor
                    agent_action = agent_policy.sample()
                else:                          # evaluator
                    agent_action = agent_policy.mean()
            else:
                agent_action = agent_policy

            # ============ V0.1 DEBUG ============ #
            # if not hasattr(self, "_debug_action_compare_step"):
            #     self._debug_action_compare_step = 0

            # fused = tf.concat([z_e, g_e], axis=-1)
            # fused = tf.cast(fused, tf.float32)

            # agent_policy = self._policy_networks[i](fused)
            # agent_action = agent_policy.sample() if isinstance(agent_policy, tfd.Distribution) else agent_policy

            # if self._debug_action_compare_step < 2:
            #     zero_g = tf.zeros_like(g_e)
            #     fused_zero = tf.concat([z_e, zero_g], axis=-1)
            #     fused_zero = tf.cast(fused_zero, tf.float32)

            #     zero_policy = self._policy_networks[i](fused_zero)
            #     zero_action = zero_policy.sample() if isinstance(zero_policy, tfd.Distribution) else zero_policy

            #     diff = tf.reduce_mean(tf.abs(tf.cast(agent_action, tf.float32) - tf.cast(zero_action, tf.float32)))
            #     print(f"[ACTION DEBUG] agent={i} step={self._debug_action_compare_step} mean_abs(action_with_g - action_zero_g)={diff.numpy():.6f}")

            # if i == self._agent_number - 1:
            #     self._debug_action_compare_step += 1
            # ============ V0.1 DEBUG ============ #

            # remove batch dim if present
            if len(agent_action.shape) >= 2 and agent_action.shape[0] == 1:
                agent_action = tf.squeeze(agent_action, axis=0)

            agent_actions.append(agent_action)

        action = tf.stack(agent_actions, axis=0)  # [E, action_dim]
        action = tf.cast(action, tf.float64)      # keep same style as old code
        
        # EDIT Add noise to only actors
        if self._adder is not None:   # actor only, evaluator no noise
            noise = tf.random.normal(
                shape=tf.shape(action),
                stddev=self._sigma,
                dtype=tf.float64
            )
            action = action + noise

        # NOTICE: Hard code clip operation with 0.0 and 1.0
        # TODO: adapted to environment_spec.minimum/maximum
        # 原来写错了：agent_action = tf.clip_by_value(agent_action, 0.0, 1.0)
        action = tf.clip_by_value(action, 0.0, 1.0)

        # EDIT DEBUG
        # self._sync_debug_count += 1
        # if self._sync_debug_count % 500 == 0:
        #     # DEBUG #
        #     tf.print("[ACTOR UPDATE DEBUG] actor_gnn_mean =", tf.reduce_mean(self._dyn_gnn_module.variables[0]))
        #     tf.print("[ACTOR UPDATE DEBUG] actor_policy_mean =", tf.reduce_mean(self._policy_networks[0].variables[0]))
        #     # actor_gnn_mean = tf.reduce_mean(self._dyn_gnn_module.variables[0])
        #     # actor_policy_mean = tf.reduce_mean(self._policy_networks[0].variables[0])
        #     # DEBUG #

        action = tf.reshape(action, [self._agent_number, self._agent_action_size])
        return action

    # NOTICE: 此处输入的observation必须是dict形式，包含"local_observation"和"snapshot"，然后直接跳到select action bundle分支
    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        """Supports either:
        1) old raw observation array: [E, obs_dim]
        2) new dict:
           {
             "local_observation": ...,
             "snapshot": ...
           }
        """
        # backward-compatible fallback
        if not isinstance(observation, dict):
            # no snapshot provided -> zero graph embedding fallback
            local_observation = tf.convert_to_tensor(observation, dtype=tf.float32)
            zero_edge_embedding = tf.zeros(
                [self._agent_number, self._gnn_embedding_dim], dtype=tf.float32
            )

            agent_actions = []
            for i in range(self._agent_number):
                agent_observation = local_observation[i, :]
                agent_batched_observation = tf2_utils.add_batch_dim(agent_observation)

                z_e = self._observation_networks[i](agent_batched_observation)
                g_e = tf2_utils.add_batch_dim(zero_edge_embedding[i, :])
                g_e = tf.cast(g_e, z_e.dtype)

                fused = tf.concat([z_e, g_e], axis=-1)
                agent_policy = self._policy_networks[i](fused)
                agent_action = (
                    agent_policy.sample()
                    if isinstance(agent_policy, tfd.Distribution)
                    else agent_policy
                )
                if len(agent_action.shape) >= 2 and agent_action.shape[0] == 1:
                    agent_action = tf.squeeze(agent_action, axis=0)
                agent_actions.append(agent_action)

            action = tf.stack(agent_actions, axis=0)
            action = tf.cast(action, tf.float64)

            # EDIT TODO Add noise
            # EDIT Add noise
            if self._adder is not None:   # actor only, evaluator no noise
                noise = tf.random.normal(
                    shape=tf.shape(action),
                    stddev=self._sigma,
                    dtype=tf.float64
                )
                action = action + noise

            return tf.reshape(action, [self._agent_number, self._agent_action_size])

        # new path
        return self._select_action_from_bundle(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        # new episode -> clear graph memory
        self._prev_state_list = None

        # ====== V0.2 Debug param ======
        # if self._debug_episode_index <= 3:
        #     print(f"[ROLAND RESET DEBUG] episode={self._debug_episode_index} reset prev_state to None")
        # self._debug_gnn_step = 0
        # self._debug_episode_index += 1
        # ====== V0.2 Debug param ======

        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)

    # optional debug helper
    def get_prev_state_shapes(self):
        if self._prev_state_list is None:
            return None
        return [tuple(x.shape) for x in self._prev_state_list]

    
    def get_prev_state_for_replay(self, snapshot: Optional[Dict[str, Any]] = None):
        """Return prev_state as a stacked numpy array:
        [num_layers, num_nodes, gnn_hidden_dim].

        If prev_state_list is None, return zeros (requires snapshot to infer num_nodes).
        """
        if self._prev_state_list is None:
            if snapshot is None:
                return None
            num_nodes = int(snapshot["node_feature"].shape[0])
            return np.zeros(
                (self._dyn_gnn_module.num_layers, num_nodes, self._gnn_embedding_dim),
                dtype=np.float32,
            )

        state_list = []
        for s in self._prev_state_list:
            if isinstance(s, tf.Tensor):
                s = s.numpy()
            state_list.append(np.asarray(s, dtype=np.float32))

        return np.stack(state_list, axis=0)

