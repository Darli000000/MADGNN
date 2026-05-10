import numpy as np
import tensorflow as tf

from Agents.MAD4PG_GNN.dynamic_gnn import DynGNNModule
from Utilities.FileOperator import load_obj

# 假设你已经按你当前项目方式创建好了 env
# 例如：
# from Environment.make_environment import make_environment
# env = make_environment(...)
# 这里用你项目里现有创建环境的方式即可
# 创建环境
# EDIT：使用已创建的环境文件 - Darli0 ver2026.3.6
environment_file_name = "/home/darli0/DeepReinforcementLearningmain/Game-Theoretic-Deep-Reinforcement-Learning-main/saved_env/2026-03-06-10-17-32/convex_environment_df285cc3dcef492bbe1d3cf2520b144b.pkl"
env = load_obj(environment_file_name)

# 1) reset 环境
timestep = env.reset()

# 2) 拿当前 snapshot
snapshot = env.get_current_snapshot()

# 3) 如果 edge_node_indices 还是 dict，可以先看一下
print("time_index:", snapshot["time_index"])
print("node_feature shape:", snapshot["node_feature"].shape)
print("edge_index shape:", snapshot["edge_index"].shape)
print("edge_feature shape:", snapshot["edge_feature"].shape)
print("edge_node_indices:", snapshot["edge_node_indices"])

# 4) 创建 DynGNN
dyn_gnn = DynGNNModule(
    node_feat_dim=12,   # 你现在 snapshot node_feature 是 12 维
    edge_feat_dim=4,    # 你现在 snapshot edge_feature 是 4 维
    hidden_dim=32,
    num_layers=2,
    updater_type="gru",   # 或 "moving_average" / "mlp"
)

# 5) 第一次前向（prev_state_list = None）
out = dyn_gnn(snapshot=snapshot, prev_state_list=None)

print("\n===== First Forward =====")
print("time_index first:", snapshot["time_index"])
print("all_node_embedding shape:", out["all_node_embedding"].shape)
print("edge_embedding shape:", out["edge_embedding"].shape)
print("num new_state_list:", len(out["new_state_list"]))
for i, s in enumerate(out["new_state_list"]):
    print(f"layer {i} state shape:", s.shape)

# 期望：
# all_node_embedding shape == (36, 32)
# edge_embedding shape == (9, 32)
# len(new_state_list) == 2
# each state shape == (36, 32)

# 6) 做一步环境推进（这里只是为了测试两次前向）
# 先随便给个合法 action。
# 项目中：action_size = vehicle_number_within_edges * edge_number
# 也就是 3 * 9 = 27
dummy_action = np.zeros(
    (env._config.edge_number, env._config.action_size),
    dtype=np.float64
)

for e in range(env._config.edge_number):
    for local_vehicle_slot in range(env._config.vehicle_number_within_edges):
        dummy_action[e, local_vehicle_slot * env._config.edge_number + e] = 1.0

# next_timestep = env.step(dummy_action)
# EDIT
step_out = env.step(dummy_action)
next_timestep = step_out[0]

snapshot_next = env.get_current_snapshot()

# 7) 第二次前向：把第一次的 new_state_list 作为 prev_state_list
out2 = dyn_gnn(snapshot=snapshot_next, prev_state_list=out["new_state_list"])

print("\n===== Second Forward =====")
print("time_index second:", snapshot_next["time_index"])
print("all_node_embedding shape:", out2["all_node_embedding"].shape)
print("edge_embedding shape:", out2["edge_embedding"].shape)
print("num new_state_list:", len(out2["new_state_list"]))
for i, s in enumerate(out2["new_state_list"]):
    print(f"layer {i} state shape:", s.shape)

# 8) 可选：简单比较两次 edge_embedding 是否不同
diff = tf.reduce_mean(tf.abs(out["edge_embedding"] - out2["edge_embedding"]))
print("mean abs diff between two edge embeddings:", diff.numpy())

