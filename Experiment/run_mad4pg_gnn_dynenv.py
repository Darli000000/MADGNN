import launchpad as lp
from Environment.environment_dynamic import make_environment_spec    # EDIT: 原本使用的是environment_old，现在改回来，有bug再说
from Agents.MAD4PG_GNN_DynENV.networks import make_default_networks
from Agents.MAD4PG_GNN_DynENV.agent_distributed import DistributedD4PG
from Utilities.FileOperator import load_obj


def main(_):
    
    # debug
    print("Running mad4pg_gnn_dynenv!")
    # different scenario
    # scneario 1
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_1/convex_environment_b6447224a61e446183f13dd40a04b17b.pkl"
    # scneario 2
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_2/convex_environment_f1c365156c98462b9ae0b920d0063533.pkl"
    # scenario 3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_3/convex_environment_0ff6ea4dcd184438aeb3389520f60aa9.pkl"
    # scenario 4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/scenarios/scenario_4/convex_environment_1bc5da3127734abc9d015bccf84bc1c0.pkl"
    
    # different bandwidth 
    # bandwidth 10 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth10/convex_environment_0c3404cb7b094635b93478b7ed8414d4.pkl"
    # bandwidth 15 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth15/convex_environment_2aa46263ed1543a9b1724f2ae1e15517.pkl"
    # bandwidth 25 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth25/convex_environment_73e7ad98699f41ac9e940690c9bbf274.pkl"
    # bandwidth 30 MHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/bandwidth/bandwidth30/convex_environment_99c0184f2a2f44ffa51e8570a3c56e44.pkl"
    
    # different power 
    # power 100 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/100mW/convex_environment_a7be501bbb0449e78ba3d18a915190f0.pkl"
    # power 550 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/550mW/convex_environment_d242a140f3a54b03af77a22a3e4698fe.pkl"
    # power 1450 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/1450mW/convex_environment_13da0cdb1f0f40849099c080b17e60bf.pkl"
    # power 1900 mW
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/power/1900mW/convex_environment_4f77862d11a1479898416ea261c93b66.pkl"
    
    # different compuation resources 
    # CPU 1-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/1GHz/convex_environment_b80ebdb0027045288b59f66247950cb0.pkl"
    # CPU 2-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/2GHz/convex_environment_62b140db2ca04cac84eab306ba2323d2.pkl"
    # CPU 4-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/4GHz/convex_environment_aa0f303f501d427296ef9c93a2261868.pkl"
    # CPU 5-10GHz
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/computation/5GHz/convex_environment_68eaeca4ef604e68b4753ad37530e431.pkl"
    
    # different task number
    # 0.1
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_1/convex_environment_323579c648bf4169abcefc1c8036f79c.pkl"
    # 0.3
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_3/convex_environment_590cd268b35a4e79b5b5216ee06e9ef3.pkl"
    # 0.4
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_4/convex_environment_00a13ba8916649c08c02f374dff640df.pkl"
    # 0.6
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_6/convex_environment_05727ef5311540ca84b4c596a73987cd.pkl"
    # 0.7
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_7/convex_environment_c2ea75aa7cce404e9d9f8af15d49369f.pkl"
    # 0.9
    # environment_file_name = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/task_number/0_9/convex_environment_02893eba453a40638713178e264ec23e.pkl"
    
    # EDIT：Darli0 ver2026.4.20
    # /home/darli0/DeepReinforcementLearningmain/Game-Theoretic-Deep-Reinforcement-Learning-main/saved_env/2026-04-20-15-07-26/convex_environment_ab73e900a4e64412bd4824f5b9a4e621.pkl
    # Remember Using A Dynamic Environment!!!
    environment_file_name = "/root/Game-Theoretic-Deep-Reinforcement-Learning-main/saved_env/2026-04-28-13-06-18/convex_environment_770ccb6a62d94c608a4759d5d2ef6849.pkl"

    environment = load_obj(environment_file_name)
    
    spec = make_environment_spec(environment)  

    print("============ ENV SPEC DEBUG ============")
    print("edge_actions.shape =", spec.edge_actions.shape)
    print("edge_observations.shape =", spec.edge_observations.shape)
    print("critic_actions.shape =", spec.critic_actions.shape)
    print("rewards.shape =", spec.rewards.shape)
    print("discounts.shape =", spec.discounts.shape)

    print("edge_actions.minimum.shape =", spec.edge_actions.minimum.shape)
    print("edge_actions.maximum.shape =", spec.edge_actions.maximum.shape)

    print("environment._config.max_vehicle_slots_per_edge =", environment._config.max_vehicle_slots_per_edge)
    print("environment._config.action_size =", environment._config.action_size)
    print("environment._config.observation_size =", environment._config.observation_size)
    print("environment._config.critic_network_action_size =", environment._config.critic_network_action_size)
    print("========================================")  
    
    networks = make_default_networks(
        agent_number=9,
        action_spec=spec.edge_actions,
    )

    agent_action_size = spec.edge_actions.shape[0]    # DynENV-EDIT

    agent = DistributedD4PG(
        agent_number=9,
        agent_action_size=agent_action_size, # DynENV-EDIT
        environment_file=environment_file_name,
        networks=networks,
        num_actors=10, # EDIT:originally 10, 1 for test
        environment_spec=spec,
        batch_size=256, # EDIT:256->64
        prefetch_size=4,
        min_replay_size=1000,
        max_replay_size=1000000,
        samples_per_insert=8.0,
        n_step=1,
        sigma=0.1, # EDIT: 0.3 is too big,
        discount=0.996,
        target_update_period=50,# EDIT 100->5
        variable_update_period=300, # EDIT 1000->10
        max_actor_steps=20000*300, # EDIT：太长，原本是300*25000，300的意思是size of time slot == 300
        log_every=5.0,
        gnn_hidden_dim=24,    # EDIT : gnn相关的参数可以控制gnn模块的基本构造
        gnn_num_layers=1,
        gnn_updater_type='gru',
        gnn_is_dynmic=True # EDIT：控制是否是动态gnn，即是否有prev state
    )

    program = agent.build()
    
    lp.launch(program, launch_type="local_mt", serialize_py_nodes=False)
        