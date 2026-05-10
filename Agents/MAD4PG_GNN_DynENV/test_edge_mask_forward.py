import copy
import numpy as np
import tensorflow as tf

from Utilities.FileOperator import load_obj
from Agents.MAD4PG_GNN_DynENV.dynamic_gnn import DynGNNModule


def main():
    # 1) 加载你已经保存好的 DynENV 环境对象
    env_path = "/home/darli0/DeepReinforcementLearningmain/Game-Theoretic-Deep-Reinforcement-Learning-main/saved_env/2026-04-21-17-17-59/convex_environment_3cc1d1ff5f0144748437e1f4a7f9f957.pkl"
    env = load_obj(env_path)

    env.reset()
    snapshot = env.get_current_snapshot()

    print("===== MASK TEST: SNAPSHOT INFO =====")
    print("node_feature shape =", snapshot["node_feature"].shape)
    print("edge_index shape   =", snapshot["edge_index"].shape)
    print("edge_feature shape =", snapshot["edge_feature"].shape)
    print("edge_mask shape    =", snapshot["edge_mask"].shape)
    print("num_edges_raw      =", snapshot["num_edges_raw"])
    print("edge_mask sum      =", np.sum(snapshot["edge_mask"]))

    # 2) 初始化 DynGNN
    node_feat_dim = snapshot["node_feature"].shape[-1]
    edge_feat_dim = snapshot["edge_feature"].shape[-1]

    dyn_gnn = DynGNNModule(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=32,
        num_layers=2,
        updater_type="gru",
        name="dyn_gnn_mask_test",
    )

    # 3) 第一次 forward：使用真实 edge_mask
    out_masked = dyn_gnn(
        snapshot=snapshot,
        prev_state_list=None,
        training=False,
    )

    # 4) 第二次 forward：把 edge_mask 改成全 1
    snapshot_all_ones = dict(snapshot)
    snapshot_all_ones["edge_mask"] = np.ones_like(snapshot["edge_mask"], dtype=np.float32)

    out_all_ones = dyn_gnn(
        snapshot=snapshot_all_ones,
        prev_state_list=None,
        training=False,
    )

    # 5) 对比输出差异
    node_a = out_masked["all_node_embedding"].numpy()
    node_b = out_all_ones["all_node_embedding"].numpy()

    edge_a = out_masked["edge_embedding"].numpy()
    edge_b = out_all_ones["edge_embedding"].numpy()

    node_diff = np.abs(node_a - node_b)
    edge_diff = np.abs(edge_a - edge_b)

    print("\n===== MASK TEST: DIFF =====")
    print("node embedding max abs diff  =", node_diff.max())
    print("node embedding mean abs diff =", node_diff.mean())
    print("edge embedding max abs diff  =", edge_diff.max())
    print("edge embedding mean abs diff =", edge_diff.mean())


if __name__ == "__main__":
    main()