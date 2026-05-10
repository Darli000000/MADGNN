# dynamic_gnn.py

from typing import Any, Dict, List, Optional, Tuple

import sonnet as snt
import tensorflow as tf


def _to_tensor(x, dtype):
    if x is None:
        return None
    return tf.convert_to_tensor(x, dtype=dtype)


def _build_edge_node_indices_tensor(snapshot: Dict[str, Any]) -> tf.Tensor:
    """将 snapshot 中的 edge_node_indices 统一转成 [E] int32 Tensor."""
    edge_node_indices = snapshot["edge_node_indices"]

    if isinstance(edge_node_indices, dict):
        # 假设 key 是 0,1,2,...,E-1
        keys = sorted(edge_node_indices.keys())
        values = [edge_node_indices[k] for k in keys]
        return tf.convert_to_tensor(values, dtype=tf.int32)

    return tf.convert_to_tensor(edge_node_indices, dtype=tf.int32)


class EdgeMessagePassing(snt.Module):
    """一个简单的 edge-aware message passing 层。

    对每条边 (src -> dst)：
        msg = MLP([x_src, e_feat])
    然后按 dst 做 sum 聚合，再和 self projection 相加。
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        name: str = "edge_message_passing",
    ):
        super().__init__(name=name)
        self._hidden_dim = hidden_dim
        self._edge_feat_dim = edge_feat_dim

        self._msg_mlp = snt.nets.MLP(
            [hidden_dim, hidden_dim],
            activate_final=True,
            name="msg_mlp",
        )
        self._self_linear = snt.Linear(hidden_dim, name="self_linear")
        self._out_ln = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="out_ln")

    def __call__(
        self,
        node_x: tf.Tensor,       # [N, d]
        edge_index: tf.Tensor,   # [2, M]
        edge_feature: tf.Tensor, # [M, d_e]
        edge_mask=None           # DynENV-EDIT
    ) -> tf.Tensor:
        src = tf.cast(edge_index[0], tf.int32)   # [M]
        dst = tf.cast(edge_index[1], tf.int32)   # [M]

        x_src = tf.gather(node_x, src)  # [M, d]
        msg_input = tf.concat([x_src, edge_feature], axis=-1)  # [M, d + d_e]
        msg = self._msg_mlp(msg_input)  # [M, hidden_dim]

        # DynENV-EDIT
        if edge_mask is not None:
            edge_mask = tf.cast(edge_mask, msg.dtype)           # [M]
            msg = msg * tf.expand_dims(edge_mask, axis=-1)      # [M, hidden_dim]

        num_nodes = tf.shape(node_x)[0]
        agg = tf.math.unsorted_segment_sum(msg, dst, num_segments=num_nodes)  # [N, hidden_dim]

        out = self._self_linear(node_x) + agg
        out = self._out_ln(out)
        out = tf.nn.relu(out)
        return out


class GRUUpdater(snt.Module):
    """GRU 风格 updater:
       H_t = GRU(x_msg, H_{t-1})
    """

    def __init__(self, hidden_dim: int, name: str = "gru_updater"):
        super().__init__(name=name)
        self._gru = snt.GRU(hidden_size=hidden_dim)

    def __call__(self, x_msg: tf.Tensor, prev_state: tf.Tensor) -> tf.Tensor:
        prev_state = tf.stop_gradient(prev_state)
        _, new_state = self._gru(x_msg, prev_state)
        return new_state


class MLPUpdater(snt.Module):
    """MLP 风格 updater:
       H_t = MLP([x_msg, H_{t-1}])
    """

    def __init__(self, hidden_dim: int, name: str = "mlp_updater"):
        super().__init__(name=name)
        self._mlp = snt.nets.MLP(
            [hidden_dim, hidden_dim],
            activate_final=False,
            name="mlp",
        )
        self._ln = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln")

    def __call__(self, x_msg: tf.Tensor, prev_state: tf.Tensor) -> tf.Tensor:
        prev_state = tf.stop_gradient(prev_state)
        h = tf.concat([x_msg, prev_state], axis=-1)
        h = self._mlp(h)
        h = self._ln(h)
        h = tf.nn.relu(h)
        return h


class MovingAverageUpdater(snt.Module):
    """Moving average 风格 updater:
       H_t = alpha * H_{t-1} + (1-alpha) * x_msg
    """

    def __init__(self, alpha: float = 0.8, name: str = "moving_average_updater"):
        super().__init__(name=name)
        self._alpha = alpha

    def __call__(self, x_msg: tf.Tensor, prev_state: tf.Tensor) -> tf.Tensor:
        prev_state = tf.stop_gradient(prev_state)
        return self._alpha * prev_state + (1.0 - self._alpha) * x_msg


class DynGNNLayer(snt.Module):
    """一层 DynGNN：
       current_x --(message passing)--> x_msg
       x_msg + H_prev^(l) --(updater)--> H_t^(l)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        updater_type: str = "gru",
        moving_average_alpha: float = 0.8,
        name: str = "dyn_gnn_layer",
    ):
        super().__init__(name=name)
        self._hidden_dim = hidden_dim
        self._mp = EdgeMessagePassing(hidden_dim=hidden_dim, edge_feat_dim=edge_feat_dim, name="mp")

        if updater_type == "gru":
            self._updater = GRUUpdater(hidden_dim=hidden_dim, name="updater")
        elif updater_type == "mlp":
            self._updater = MLPUpdater(hidden_dim=hidden_dim, name="updater")
        elif updater_type == "moving_average":
            self._updater = MovingAverageUpdater(alpha=moving_average_alpha, name="updater")
        else:
            raise ValueError(f"Unknown updater_type: {updater_type}")

    def __call__(
        self,
        current_x: tf.Tensor,     # [N, d]
        edge_index: tf.Tensor,    # [2, M]
        edge_feature: tf.Tensor,  # [M, d_e]
        prev_state: tf.Tensor,    # [N, d]
        edge_mask=None,           # DynENV-EDIT : ADD edge_mask=edge_mask
    ) -> tf.Tensor:
        x_msg = self._mp(current_x, edge_index, edge_feature, edge_mask=edge_mask)    # DynENV-EDIT : ADD edge_mask=edge_mask
        new_state = self._updater(x_msg, prev_state)
        return new_state


class DynGNNModule(snt.Module):
    """全局 DynGNN 编码模块。

    输入:
        snapshot:
            {
                "node_feature": [N, d_node],
                "edge_index": [2, M],
                "edge_feature": [M, d_edge],
                "edge_node_indices": [E] 或 dict
            }
        prev_state_list:
            list of [N, hidden_dim], 长度 = num_layers
            若为 None，则自动初始化为 0

    输出:
        {
            "new_state_list": List[[N, hidden_dim]],
            "all_node_embedding": [N, hidden_dim],
            "edge_embedding": [E, hidden_dim],
        }
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        updater_type: str = "gru",
        moving_average_alpha: float = 0.8,
        name: str = "dyn_gnn_module",
    ):
        super().__init__(name=name)
        self._node_feat_dim = node_feat_dim
        self._edge_feat_dim = edge_feat_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._updater_type = updater_type

        self._input_proj = snt.nets.MLP(
            [hidden_dim],
            activate_final=True,
            name="input_proj",
        )

        self._layers = []
        for i in range(num_layers):
            self._layers.append(
                DynGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_feat_dim,
                    updater_type=updater_type,
                    moving_average_alpha=moving_average_alpha,
                    name=f"dyn_layer_{i}",
                )
            )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def initial_state(
        self,
        num_nodes: int,
        dtype: tf.dtypes.DType = tf.float32,
    ) -> List[tf.Tensor]:
        return [
            tf.zeros([num_nodes, self._hidden_dim], dtype=dtype)
            for _ in range(self._num_layers)
        ]

    def __call__(
        self,
        snapshot: Dict[str, Any],
        prev_state_list: Optional[List[tf.Tensor]] = None,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        del training  # 这版先不用 training flag

        node_feature = _to_tensor(snapshot["node_feature"], tf.float32)   # [N, d_node]
        edge_index = _to_tensor(snapshot["edge_index"], tf.int32)         # [2, M]
        edge_feature = _to_tensor(snapshot["edge_feature"], tf.float32)   # [M, d_edge]
        edge_node_indices = _build_edge_node_indices_tensor(snapshot)     # [E]

        # DynENV-EDIT
        edge_mask = snapshot.get("edge_mask", None)
        if edge_mask is not None:
            edge_mask = _to_tensor(edge_mask, tf.float32)

        num_nodes = tf.shape(node_feature)[0]

        if prev_state_list is None:
            prev_state_list = self.initial_state(num_nodes=num_nodes)
        else:
            if len(prev_state_list) != self._num_layers:
                raise ValueError(
                    f"prev_state_list length {len(prev_state_list)} != num_layers {self._num_layers}"
                )
            prev_state_list = [tf.convert_to_tensor(x, dtype=tf.float32) for x in prev_state_list]

        x = self._input_proj(node_feature)  # [N, hidden_dim]

        # =============================== Debug Info =============================== #
        # tf.print("\n[DynGNN] node_feature shape =", tf.shape(node_feature))
        # tf.print("[DynGNN] edge_index shape =", tf.shape(edge_index))
        # tf.print("[DynGNN] edge_feature shape =", tf.shape(edge_feature))
        # tf.print("[DynGNN] edge_node_indices shape =", tf.shape(edge_node_indices))
        # =============================== Debug Info =============================== #

        new_state_list = []
        for layer_id in range(self._num_layers):
            prev_state = prev_state_list[layer_id]
            new_state = self._layers[layer_id](
                current_x=x,
                edge_index=edge_index,
                edge_feature=edge_feature,
                prev_state=prev_state,
                edge_mask=edge_mask,    # DynENV-EDIT
            )
            new_state_list.append(new_state)
            x = new_state  # 下一层输入就是本层更新后的节点表示

            # ================================ Debug Info ================================ #
            # tf.print("[DynGNN] layer", layer_id, "new_state shape =", tf.shape(new_state))
            # ================================ Debug Info ================================ #

        all_node_embedding = x
        edge_embedding = tf.gather(all_node_embedding, edge_node_indices)  # [E, hidden_dim]

        return {
            "new_state_list": new_state_list,
            "all_node_embedding": all_node_embedding,
            "edge_embedding": edge_embedding,
        }

    
# Helper: Batched Call func
def run_dyn_gnn_batched(
    dyn_gnn_module, 
    snapshot, 
    prev_state, 
    training=False
):
    node_feature = tf.cast(snapshot["node_feature"], tf.float32)        # [B,N,dn]
    edge_index = tf.cast(snapshot["edge_index"], tf.int32)              # [B,2,M]
    edge_feature = tf.cast(snapshot["edge_feature"], tf.float32)        # [B,M,de]
    edge_node_indices = tf.cast(snapshot["edge_node_indices"], tf.int32)# [B,E]

    B = tf.shape(node_feature)[0]
    N = tf.shape(node_feature)[1]
    M = tf.shape(edge_index)[2]
    E = tf.shape(edge_node_indices)[1]
    L = tf.shape(prev_state)[1]

    # 1) flatten nodes / edges / mask
    flat_node_feature = tf.reshape(node_feature, [B * N, -1])   # [B*N, dn]
    flat_edge_feature = tf.reshape(edge_feature, [B * M, -1])   # [B*M, de]
    # DynENV-EDIT
    edge_mask = tf.cast(snapshot["edge_mask"], tf.float32)      # [B, M]
    flat_edge_mask = tf.reshape(edge_mask, [-1])                # [B*M]

    # 2) build node offset
    node_offset = tf.reshape(tf.range(B, dtype=tf.int32) * N, [B, 1])   # [B,1]

    # 3) offset edge_index
    src = edge_index[:, 0, :] + node_offset   # [B,M]
    dst = edge_index[:, 1, :] + node_offset   # [B,M]
    flat_edge_index = tf.stack([
        tf.reshape(src, [-1]),
        tf.reshape(dst, [-1]),
    ], axis=0)   # [2, B*M]

    # 4) offset edge_node_indices
    flat_edge_node_indices = tf.reshape(edge_node_indices + node_offset, [-1])  # [B*E]

    # 5) flatten prev_state per layer
    prev_state_list = []
    for l in range(dyn_gnn_module.num_layers):
        prev_l = prev_state[:, l, :, :]             # [B,N,d]
        prev_l = tf.reshape(prev_l, [B * N, -1])    # [B*N,d]
        prev_state_list.append(prev_l)

    # 6) directly call internal layers on big graph
    x = dyn_gnn_module._input_proj(flat_node_feature)
    new_state_list = []
    for l in range(dyn_gnn_module.num_layers):
        x = dyn_gnn_module._layers[l](
            current_x=x,
            edge_index=flat_edge_index,
            edge_feature=flat_edge_feature,
            prev_state=prev_state_list[l],
            edge_mask=flat_edge_mask,       # DynENV-EDIT
        )
        new_state_list.append(tf.reshape(x, [B, N, dyn_gnn_module.hidden_dim]))

    flat_all_node_embedding = x                           # [B*N,d]
    all_node_embedding = tf.reshape(x, [B, N, dyn_gnn_module.hidden_dim])
    flat_edge_embedding = tf.gather(flat_all_node_embedding, flat_edge_node_indices)
    edge_embedding = tf.reshape(flat_edge_embedding, [B, E, dyn_gnn_module.hidden_dim])

    new_state_batched = tf.stack(new_state_list, axis=1)  # [B,L,N,d]

    # DEBUG
    # tf.print("[BATCHED GNN TEST] B,N,M =", B, N, M)
    # tf.print("[BATCHED GNN TEST] flat_edge_mask shape =", tf.shape(flat_edge_mask))

    return {
        "new_state_list": new_state_batched,
        "all_node_embedding": all_node_embedding,
        "edge_embedding": edge_embedding,
    }
