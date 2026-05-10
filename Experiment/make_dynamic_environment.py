import sys
sys.path.append(r"/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/")

from typing import Optional, List, Tuple
import numpy as np
from Environment.environment_dynamic import init_distance_matrix_and_radio_coverage_matrix, define_size_of_spaces
from Environment.environment_dynamic import vehicularNetworkEnv as ConvexResourceAllocationEnv
from Environment.environment_random_action import vehicularNetworkEnv as RandomResourceAllocationEnv
from Environment.environment_local_processing import vehicularNetworkEnv as LocalOffloadingEnv
from Environment.environment_offloaded_other_edge_nodes import vehicularNetworkEnv as EdgeOffloadEnv
from Environment.environment_old import vehicularNetworkEnv as OldEnv
from Environment.environment_global_actions import vehicularNetworkEnv as GlobalActionEnv
from Environment.environmentDynConfig import vehicularNetworkEnvConfig
from Environment.dataStruct import vehicleList, timeSlots, taskList, edgeList
from Utilities.FileOperator import save_obj, init_file_name

def get_default_environment(
        flatten_space: Optional[bool] = False,
        occuiped: Optional[bool] = False,
        for_mad5pg: Optional[bool] = True,
    ):
    
    environment_config = vehicularNetworkEnvConfig(
        task_request_rate=0.7,
    )
    environment_config.vehicle_seeds += [i for i in range(environment_config.vehicle_number)]
    
    time_slots= timeSlots(
        start=environment_config.time_slot_start,
        end=environment_config.time_slot_end,
        slot_length=environment_config.time_slot_length,
    )
    
    task_list = taskList(
        tasks_number=environment_config.task_number,
        minimum_data_size=environment_config.task_minimum_data_size,
        maximum_data_size=environment_config.task_maximum_data_size,
        minimum_computation_cycles=environment_config.task_minimum_computation_cycles,
        maximum_computation_cycles=environment_config.task_maximum_computation_cycles,
        minimum_delay_thresholds=environment_config.task_minimum_delay_thresholds,
        maximum_delay_thresholds=environment_config.task_maximum_delay_thresholds,
        seed=environment_config.task_seed,
    )
    
    vehicle_list = vehicleList(
        edge_number=environment_config.edge_number,
        communication_range=environment_config.communication_range,
        vehicle_number=environment_config.vehicle_number,
        time_slots=time_slots,
        trajectories_file_name=environment_config.trajectories_file_name,
        slot_number=environment_config.time_slot_number,
        task_number=environment_config.task_number,
        task_request_rate=environment_config.task_request_rate,
        seeds=environment_config.vehicle_seeds,
    )
    
    # =============== DEBUG =============== #
    vehicles = vehicle_list.get_vehicle_list()
    print("====== dynamic vehicle list [DEBUG] ======")
    print("loaded vehicle count =", len(vehicles))

    for i, v in enumerate(vehicles[:5]):
        locs = v.get_vehicle_trajectory().get_locations()
        print(f"vehicle {i}: traj_len={len(locs)} first=({locs[0].get_x()}, {locs[0].get_y()}) "
            f"last=({locs[-1].get_x()}, {locs[-1].get_y()})")

    # x,y test
    for i, v in enumerate(vehicles[:10]):
        locs = v.get_vehicle_trajectory().get_locations()
        xs = [p.get_x() for p in locs]
        ys = [p.get_y() for p in locs]
        print(f"vehicle {i}: x_range=({min(xs):.2f}, {max(xs):.2f}), y_range=({min(ys):.2f}, {max(ys):.2f})")
    
    # crossing area test
    edge_xs = [500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500]
    edge_ys = [2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500]
    comm_range = 500.0

    for i, v in enumerate(vehicles[:10]):
        touched_edges = set()
        locs = v.get_vehicle_trajectory().get_locations()
        for p in locs:
            for e, (ex, ey) in enumerate(zip(edge_xs, edge_ys)):
                d = ((p.get_x() - ex) ** 2 + (p.get_y() - ey) ** 2) ** 0.5
                if d <= comm_range:
                    touched_edges.add(e)
        print(f"vehicle {i}: touched_edges={sorted(list(touched_edges))}")
    # =============== DEBUG =============== #
    
    edge_list = edgeList(
        edge_number=environment_config.edge_number,
        power=environment_config.edge_power,
        bandwidth=environment_config.edge_bandwidth,
        minimum_computing_cycles=environment_config.edge_minimum_computing_cycles,
        maximum_computing_cycles=environment_config.edge_maximum_computing_cycles,
        communication_range=environment_config.communication_range,
        edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
        edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
        seed=environment_config.edge_seed,
    )
    
    distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(env_config=environment_config, vehicle_list=vehicle_list, edge_list=edge_list)
    
    # ============ 非 Dynamic Environment 逻辑 ============ #
    # environment_config.max_vehicle_slots_per_edge = int(environment_config.vehicle_number / environment_config.edge_number)
    # environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
    #         environment_config.critic_network_action_size = define_size_of_spaces(vehicle_number_within_edges=environment_config.max_vehicle_slots_per_edge, edge_number=environment_config.edge_number)   
    # ============ 非 Dynamic Environment 逻辑 ============ #

    # ============== Dynamic Environment 逻辑 ============== #
    # DynENV-EDIT : TODO 待修改，vehicle_number_within_edges逻辑改成max_vehicle_slots_per_edge
    environment_config.action_size, environment_config.observation_size, environment_config.reward_size, \
            environment_config.critic_network_action_size = define_size_of_spaces(vehicle_number_within_edges=environment_config.max_vehicle_slots_per_edge, edge_number=environment_config.edge_number)
    # ============== Dynamic Environment 逻辑 ============== #

    print("====== dynamic basic env info [DEBUG] ======")
    print("[DEBUG]environment_config.action_size: ", environment_config.action_size)
    print("[DEBUG]environment_config.observation_size: ", environment_config.observation_size)
    print("[DEBUG]environment_config.reward_size: ", environment_config.reward_size)
    print("[DEBUG]environment_config.critic_network_action_size: ", environment_config.critic_network_action_size)
    print("[DEBUG]environment_config.max_vehicle_slots_per_edge: ", environment_config.max_vehicle_slots_per_edge)

    counts = []
    for e in range(environment_config.edge_number):
        for t in range(environment_config.time_slot_number):
            counts.append(len(vehicle_observed_index_within_edges[e][t]))

    counts = np.array(counts)
    print("====== max vehicle num within edge count [DEBUG] ======")
    print("[DEBUG] observed count max =", counts.max())
    print("[DEBUG] observed count mean =", counts.mean())
    print("[DEBUG] observed count p90 =", np.percentile(counts, 90))
    print("[DEBUG] observed count p95 =", np.percentile(counts, 95))
    
    convexEnvironment = ConvexResourceAllocationEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
        flatten_space = flatten_space,
        occuiped = occuiped,
        for_mad5pg = for_mad5pg, 
    )

    print("\n====== convex environment [DEBUG] ======")
    convexEnvironment.reset()
    for _ in range(100):
        if _ <= 2:
            snap = convexEnvironment.get_current_snapshot()
            convexEnvironment.debug_print_snapshot(snap, max_edges=10)
        
        if _ >=50 and _ <= 51:
            snap = convexEnvironment.get_current_snapshot()
            convexEnvironment.debug_print_snapshot(snap, max_edges=10)
            
        if _ >=98:
            snap = convexEnvironment.get_current_snapshot()
            convexEnvironment.debug_print_snapshot(snap, max_edges=10)
        
        convexEnvironment.step(np.zeros(convexEnvironment.action_spec().shape))
    
    # randomEnvironment = RandomResourceAllocationEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # localEnvironment = LocalOffloadingEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # edgeEnvironment = EdgeOffloadEnv(
    #     envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    # oldEnvironment = OldEnv(
    #             envConfig = environment_config,
    #     time_slots = time_slots,
    #     task_list = task_list,
    #     vehicle_list = vehicle_list,
    #     edge_list = edge_list,
    #     distance_matrix = distance_matrix, 
    #     channel_condition_matrix = channel_condition_matrix, 
    #     vehicle_index_within_edges = vehicle_index_within_edges,
    #     vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
    #     flatten_space = flatten_space,
    #     occuiped = occuiped,
    #     for_mad5pg = for_mad5pg, 
    # )
    
    globalActionEnv = GlobalActionEnv(
        envConfig = environment_config,
        time_slots = time_slots,
        task_list = task_list,
        vehicle_list = vehicle_list,
        edge_list = edge_list,
        distance_matrix = distance_matrix, 
        channel_condition_matrix = channel_condition_matrix, 
        vehicle_index_within_edges = vehicle_index_within_edges,
        vehicle_observed_index_within_edges = vehicle_observed_index_within_edges,
        flatten_space = flatten_space,
        occuiped = occuiped,
        for_mad5pg = for_mad5pg, 
    )
    
    file_name = init_file_name()
    # save_obj(randomEnvironment, file_name["random_environment_name"])
    save_obj(convexEnvironment, file_name["convex_environment_name"])
    # save_obj(localEnvironment, file_name["local_environment_name"])
    # save_obj(edgeEnvironment, file_name["edge_environment_name"])
    # save_obj(oldEnvironment, file_name["old_environment_name"])
    save_obj(globalActionEnv, file_name["global_environment_name"])

if __name__ == "__main__":
    # for d4pg
    # get_default_environment(flatten_space=True)
    # get_default_environment(flatten_space=False)
    # for mad4pg
    get_default_environment(for_mad5pg=True)