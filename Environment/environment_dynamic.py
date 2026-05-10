
"""Vehicular Network Environments."""
import time
import dm_env
from dm_env import specs
from acme.types import NestedSpec
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
import Environment.environmentDynConfig as env_config
from Environment.dataStruct import timeSlots, taskList, edgeList, vehicleList
from Environment.utilities import compute_channel_gain, generate_complex_normal_distribution, compute_transmission_rate, compute_SINR, cover_mW_to_W
np.set_printoptions(threshold=np.inf)
from Log.logger import myapp

# EDIT
from typing import Dict, Any, List, Tuple

class vehicularNetworkEnv(dm_env.Environment):
    """Vehicular Network Environment built on the dm_env framework."""
    
    def __init__(
        self, 
        envConfig: Optional[env_config.vehicularNetworkEnvConfig] = None,
        time_slots: Optional[timeSlots] = None,
        task_list: Optional[taskList] = None,
        vehicle_list: Optional[vehicleList] = None,
        edge_list: Optional[edgeList] = None,
        distance_matrix: Optional[np.ndarray] = None, 
        channel_condition_matrix: Optional[np.ndarray] = None, 
        vehicle_index_within_edges: Optional[List[List[List[int]]]] = None,
        vehicle_observed_index_within_edges: Optional[List[List[List[int]]]] = None,
        flatten_space: Optional[bool] = False,
        occuiped: Optional[bool] = False,
        for_mad5pg: Optional[bool] = True,
    ) -> None:
        """Initialize the environment."""
        if envConfig is None:
            # self._config = env_config.vehicularNetworkEnvConfig()
            # self._config.vehicle_seeds += [i for i in range(self._config.vehicle_number)]
            # self._config.vehicle_number_within_edges = int(self._config.vehicle_number / self._config.edge_number)
            # self._config.action_size, self._config.observation_size, self._config.reward_size, \
            #     self._config.critic_network_action_size = define_size_of_spaces(self._config.vehicle_number_within_edges, self._config.edge_number, self._config.task_assigned_number)
            
            # DynENV-EDIT :envConfig不能为None
            print("envConfig cannot be None!")

        else:
            self._config = envConfig
        
        if distance_matrix is None:
            self._distance_matrix, self._channel_condition_matrix, self._vehicle_index_within_edges, self._vehicle_observed_index_within_edges = init_distance_matrix_and_radio_coverage_matrix(
                env_config=self._config,
                vehicle_list=vehicle_list,
                edge_list=edge_list,
            )
        else:
            self._distance_matrix = distance_matrix
            self._channel_condition_matrix = channel_condition_matrix
            self._vehicle_index_within_edges = vehicle_index_within_edges
            self._vehicle_observed_index_within_edges = vehicle_observed_index_within_edges
        
        if time_slots is None:
            self._time_slots: timeSlots = timeSlots(
                start=self._config.time_slot_start,
                end=self._config.time_slot_end,
                slot_length=self._config.time_slot_length,
            )
        else:
            self._time_slots = time_slots
        if task_list is None:
            self._task_list: taskList = taskList(
                tasks_number=self._config.task_number,
                minimum_data_size=self._config.task_minimum_data_size,
                maximum_data_size=self._config.task_maximum_data_size,
                minimum_computation_cycles=self._config.task_minimum_computation_cycles,
                maximum_computation_cycles=self._config.task_maximum_computation_cycles,
                minimum_delay_thresholds=self._config.task_minimum_delay_thresholds,
                maximum_delay_thresholds=self._config.task_maximum_delay_thresholds,
                seed=self._config.task_seed,
            )
        else:
            self._task_list = task_list
        if vehicle_list is None:
            self._vehicle_list: vehicleList = vehicleList(
                vehicle_number=self._config.vehicle_number,
                time_slots=self._time_slots,
                trajectories_file_name=self._config.trajectories_file_name,
                slot_number=self._config.time_slot_number,
                task_number=self._config.task_number,
                task_request_rate=self._config.task_request_rate,
                seeds=self._config.vehicle_seeds,
            )
        else:
            self._vehicle_list = vehicle_list
        if edge_list is None:
            self._edge_list: edgeList = edgeList(
                edge_number=self._config.edge_number,
                power=self._config.edge_power,
                bandwidth=self._config.edge_bandwidth,
                minimum_computing_cycles=self._config.edge_minimum_computing_cycles,
                maximum_computing_cycles=self._config.edge_maximum_computing_cycles,
                communication_range=self._config.communication_range,
                edge_xs=[500, 1500, 2500, 500, 1500, 2500, 500, 1500, 2500],
                edge_ys=[2500, 2500, 2500, 1500, 1500, 1500, 500, 500, 500],
                seed=self._config.edge_seed,
            )
        else:
            self._edge_list = edge_list
                    
        self._reward: np.ndarray = np.zeros(self._config.reward_size)
        
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        # DynENV-EDIT
        self._snapshot_max_edges = None
        self._snapshot_max_edges = self._compute_snapshot_max_edges()

        self._reset_next_step: bool = True
        self._flatten_space: bool = flatten_space
        self._occuiped: bool = occuiped
        self._for_mad5pg: bool = for_mad5pg

        
    def reset(self) -> dm_env.TimeStep:
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        Returns the first `TimeStep` of a new episode.
        """
        self._time_slots.reset()
        self._occupied_power = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        self._occupied_computing_resources = np.zeros(shape=(self._config.edge_number, self._config.time_slot_number))
        
        self._reset_next_step = False
        
        return dm_env.restart(observation=self._observation())

    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """        
        if self._reset_next_step:
            return self.reset()

        # 核心步骤计算凸优化reward    
        self._reward, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number = self.compute_reward_with_convex_optimization(action)
        # print("compute_reward time taken: ", time.time() - time_start)
        
        time_start = time.time()
        observation = self._observation()
        # print("observation time taken: ", time.time() - time_start)
        # check for termination
        if self._time_slots.is_end():
            self._reset_next_step = True
            return dm_env.termination(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
        self._time_slots.add_time()
        return dm_env.transition(observation=observation, reward=self._reward), cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, \
            average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
    
    def get_transmission_power_with_convex_optimization(self):
        
        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        
        for edge_index in range(self._config.edge_number):
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            tasks_number_within_edge = len(vehicle_index_within_edge)
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                if self._occuiped:
                    if edge_power - edge_occupied_power <= 0:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * (edge_power - edge_occupied_power)
                else:
                    vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 /tasks_number_within_edge * edge_power
                        
        
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            edge_inter_interference = np.zeros((self._config.edge_number))
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        inter_interference = np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                        edge_inter_interference[other_edge_index] += inter_interference
            
            if vehicle_index_within_edge != []:
                for vehicle_index in vehicle_index_within_edge:
                    channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_edge:
                        if other_vehicle_index != vehicle_index:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            if other_channel_condition < channel_condition:
                                vehicle_intar_edge_inference[vehicle_index, -1] += np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                            
        """Compute the SINR"""
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    vehicle_SINR[vehicle_index, -1] = compute_SINR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                        transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][-1],
                        inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][-1],)
        
        """Updata the transmission power"""
        
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))

        for edge_index in range(self._config.edge_number):
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            tasks_number_within_edge = len(vehicle_index_within_edge)
            the_edge = self._edge_list.get_edge_by_index(edge_index)
            
            edge_power = the_edge.get_power()
            edge_occupied_power = self._occupied_power[edge_index][self._time_slots.now()]
            
            divided_sum = 0
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                divided_sum = divided_sum + ( vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]))
            
            for i in range(int(tasks_number_within_edge)):
                vehicle_index = vehicle_index_within_edge[i]
                if self._occuiped:
                    if edge_power - edge_occupied_power <= 0:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_transmission_power[vehicle_index][edge_index] = vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]) / divided_sum * (edge_power - edge_occupied_power)
                else:
                    vehicle_edge_transmission_power[vehicle_index][edge_index] = vehicle_SINR[vehicle_index, -1] / (1 + vehicle_SINR[vehicle_index, -1]) / divided_sum * edge_power

            # for i in range(int(tasks_number_within_edge)):
            #     vehicle_index = vehicle_index_within_edge[i]
            #     if self._occuiped:
            #         if edge_power - edge_occupied_power <= 0:
            #             vehicle_edge_transmission_power[vehicle_index][edge_index] = 0
            #         else:
            #             vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * (edge_power - edge_occupied_power)
            #     else:
            #         vehicle_edge_transmission_power[vehicle_index][edge_index] = 1 / tasks_number_within_edge * edge_power

        # print("vehicle_edge_transmission_power: ", vehicle_edge_transmission_power)
        return vehicle_edge_transmission_power
    
    # ============================ Penalty Computing Helper ============================ #
    # DynENV-EDIT
    def _will_leave_edge_coverage(
        self,
        vehicle_index: int,
        edge_index: int,
        now_t: int,
        horizon: int = 3,
    ) -> bool:
        """
        判断车辆在未来 horizon 个时隙内，是否会离开当前 edge 的通信范围。
        """
        edge = self._edge_list.get_edge_by_index(edge_index)
        comm_range = edge.get_communication_range()

        max_t = min(now_t + horizon, self._config.time_slot_end)

        for future_t in range(now_t + 1, max_t + 1):
            dist = self._distance_matrix[vehicle_index][edge_index][future_t]
            if dist > comm_range:
                return True

        return False

    # DynENV-EDIT
    def _boundary_risk_penalty(
        self,
        vehicle_index: int,
        edge_index: int,
        now_t: int,
        safe_margin: float = 50.0,
        penalty_max: float = 1.0,
    ) -> float:
        """
        根据车辆当前距离覆盖边界的裕量，给出连续边界风险惩罚。
        
        safe_margin:
            距离边界多少米以内，开始触发惩罚。
        penalty_max:
            最大边界惩罚（单位：秒），最终会直接加到 service time 上。
        """
        edge = self._edge_list.get_edge_by_index(edge_index)
        comm_range = edge.get_communication_range()

        dist = self._distance_matrix[vehicle_index][edge_index][now_t]
        margin = comm_range - dist

        if margin >= safe_margin:
            return 0.0

        if margin <= 0.0:
            return float(penalty_max)

        ratio = (safe_margin - margin) / safe_margin
        return float(penalty_max * ratio)
    
    # DynENV-EDIT
    def _handover_penalty(
        self,
        vehicle_index: int,
        edge_index: int,
        now_t: int,
        horizon: int = 3,
        penalty_value: float = 1.0,
    ) -> float:
        """
        若车辆未来短窗口内会离开当前 edge 覆盖，则返回固定切换惩罚。
        """
        if self._will_leave_edge_coverage(
            vehicle_index=vehicle_index,
            edge_index=edge_index,
            now_t=now_t,
            horizon=horizon,
        ):
            return float(penalty_value)
        return 0.0
    
    def _reference_adjacent_wired_time(self, task_index: int) -> float:
        sample_data = self._task_list.get_task_by_index(task_index).get_data_size()
        return float(
            sample_data / self._config.wired_transmission_rate
            * self._config.wired_transmission_discount
            * 1000.0
        )
    # ============================ Penalty Computing Helper ============================ #

    def _ignored_task_failure_time(self, task_index: int) -> float:
        """
        对于有任务但未进入 selected slots 的车辆，
        直接返回一个必然超过 deadline 的 service time。
        """
        return float(self._task_list.get_task_by_index(task_index).get_delay_threshold() + 1.0)
        
    def compute_reward_with_convex_optimization(
        self,
        action: np.ndarray,
    ):
        actions = np.array(action)
        
        punished_time = 30

        # 如果用的是扁平动作向量（flatten_space=True），先按 [E, A_e] 还原
        # 目前不需要关，因为目前设置的self._flatten_space为false
        if self._flatten_space:
            actions = np.reshape(np.array(actions), newshape=(self._config.edge_number, self._config.action_size))

        # 预分配：针对每辆车×每个边 / “-1列=选定/合成通道” 的统计面板
        vehicle_SINR = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_execution_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_wired_transmission_time = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        vehicle_intar_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        vehicle_inter_edge_inference = np.zeros((self._config.vehicle_number, self._config.edge_number + 1))
        
        # 记录矩阵（谁给谁算、谁用多少功率/CPU）
        vehicle_edge_transmission_power = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_task_assignment = np.zeros((self._config.vehicle_number, self._config.edge_number))
        vehicle_edge_computation_resources = np.zeros((self._config.vehicle_number, self._config.edge_number))    
                
        cumulative_reward = 0    
        successful_serviced_number = 0
        task_required_number = 0
        
        task_offloaded_number = 0
        
        # 子问题1：传输功率分配
        vehicle_edge_transmission_power = self.get_transmission_power_with_convex_optimization()
        
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]

            # ==================================== DynENV-EDIT : slot selector logic ==================================== #
            # vehicle_observed_index_within_edge = self._vehicle_observed_index_within_edges[edge_index][self._time_slots.now()]
            # vehicle_number_within_edge = len(vehicle_observed_index_within_edge)
            selected_vehicle_indices = self._select_vehicle_slots_for_edge(edge_index, self._time_slots.now())
            vehicle_number_within_edge = len(selected_vehicle_indices)

            # ==================================== DynENV-EDIT : slot selector logic ==================================== #
            
            the_edge = self._edge_list.get_edge_by_index(edge_index)

            task_assignment = np.array(actions[edge_index, :])
            # DynENV-EDIT : 变量名已修改
            # task_assignment = np.reshape(task_assignment, newshape=(self._config.vehicle_number_within_edges, self._config.edge_number))
            task_assignment = np.reshape(
                task_assignment,
                newshape=(self._config.max_vehicle_slots_per_edge, self._config.edge_number)
            )
            # DynENV-EDIT : 变量名已修改

            # ==================================== DynENV-EDIT : DEBUG ==================================== #
            # if self._time_slots.now() < 2 and edge_index < 2 or self._time_slots.now() > 50 and self._time_slots.now() < 53 and edge_index < 2:
            #     print(f"[ACT DEBUG] t={self._time_slots.now()} edge={edge_index}")
            #     print("selected vehicles =", selected_vehicle_indices)
            #     print("task_assignment shape =", task_assignment.shape)
            # ==================================== DynENV-EDIT : DEBUG ==================================== #

            for i in range(int(vehicle_number_within_edge)):
                processing_edge_index = int(task_assignment[i, :].argmax())
                
                # ==================================== DynENV-EDIT : slot selector logic ==================================== #
                #vehicle_index = vehicle_observed_index_within_edge[i]      
                vehicle_index = selected_vehicle_indices[i]

                # DEBUG
                # if self._time_slots.now() < 3 and edge_index < 2:
                #     print(f"slot {i} -> vehicle {vehicle_index}, assigned_to_edge {processing_edge_index}")
                # ==================================== DynENV-EDIT : slot selector logic ==================================== #

                if self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now()) != -1:
                    
                    vehicle_edge_task_assignment[vehicle_index][processing_edge_index] = 1
                
                    if processing_edge_index != edge_index:
                        task_offloaded_number += 1
                        task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                        data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                        wired_transmission_time = data_size / self._config.wired_transmission_rate * self._config.wired_transmission_discount * \
                                the_edge.get_edge_location().get_distance(self._edge_list.get_edge_by_index(processing_edge_index).get_edge_location())
                        for e in range(self._config.edge_number + 1):
                            vehicle_wired_transmission_time[vehicle_index, e] = wired_transmission_time
        
        # 计算子问题2 计算资源分配 Computation Resource Allocation
        for edge_index in range(self._config.edge_number):

            edge_computing_speed = self._edge_list.get_edge_by_index(edge_index).get_computing_speed()
            edge_occupied_computing_speed = self._occupied_computing_resources[edge_index][self._time_slots.now()]
            
            task_sum = int(np.sum(vehicle_edge_task_assignment[:, edge_index]))
            
            task_vehicle_index = np.where(vehicle_edge_task_assignment[:, edge_index] == 1)[0]
            
            task_computation_resource_allocation = np.zeros((task_sum, ))
            for i in range(task_sum):
                vehicle_index = task_vehicle_index[i]
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                task_computation_resource_allocation[i] = np.sqrt(edge_computing_speed * data_size * computation_cycles)
            new_task_computation_resource_allocation = (task_computation_resource_allocation) / np.sum((task_computation_resource_allocation))

            for i in range(task_sum):
                vehicle_index = task_vehicle_index[i]
                if self._occuiped:
                    if edge_computing_speed - edge_occupied_computing_speed <= 0:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = 0
                    else:
                        vehicle_edge_computation_resources[vehicle_index][edge_index] = new_task_computation_resource_allocation[i] * (edge_computing_speed - edge_occupied_computing_speed)
                else:
                    vehicle_edge_computation_resources[vehicle_index][edge_index] = new_task_computation_resource_allocation[i] * edge_computing_speed
                task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                computation_cycles = self._task_list.get_task_by_index(task_index).get_computation_cycles()
                if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                    if float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]) < punished_time:
                        vehicle_execution_time[vehicle_index, -1] = float(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index])
                    else:
                        vehicle_execution_time[vehicle_index, -1] = punished_time
                else:
                    vehicle_execution_time[vehicle_index, -1] = punished_time
                    
                for e in range(self._config.edge_number):  # e is the edge node which do nothing
                    if e == edge_index:
                        vehicle_execution_time[vehicle_index, e] = punished_time
                    else:
                        vehicle_execution_time[vehicle_index, e] = vehicle_execution_time[vehicle_index, -1]
                        
                if self._occuiped:
                    if vehicle_edge_computation_resources[vehicle_index][edge_index] != 0:
                        occupied_time = int(np.floor(data_size * computation_cycles / vehicle_edge_computation_resources[vehicle_index][edge_index]))
                        if self._occuiped and occupied_time > 0:
                            start_time = int(self._time_slots.now() + 1)
                            end_time = int(self._time_slots.now() + occupied_time + 1)
                            if end_time < self._config.time_slot_number:
                                for i in range(start_time, end_time):
                                    self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
                            else:
                                for i in range(start_time, int(self._config.time_slot_number)):
                                    self._occupied_computing_resources[edge_index][i] += vehicle_edge_computation_resources[vehicle_index][edge_index]
            
        """Compute the inference"""
        for edge_index in range(self._config.edge_number):
            
            vehicle_index_within_edge = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            
            edge_inter_interference = np.zeros((self._config.edge_number))
            
            for other_edge_index in range(self._config.edge_number):
                if other_edge_index != edge_index:
                    vehicle_index_within_other_edge = self._vehicle_index_within_edges[other_edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_other_edge:
                        other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                        inter_interference = np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][other_edge_index])
                        edge_inter_interference[other_edge_index] += inter_interference
            
            if vehicle_index_within_edge != []:
                for vehicle_index in vehicle_index_within_edge:
                    for e in range(self._config.edge_number):
                        vehicle_inter_edge_inference[vehicle_index, -1] += edge_inter_interference[e]
                        for other_edge_index in range(self._config.edge_number):
                            if e == other_edge_index:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += 0
                            else:
                                vehicle_inter_edge_inference[vehicle_index, other_edge_index] += edge_inter_interference[e]
                    channel_condition = self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()]
                    for other_vehicle_index in vehicle_index_within_edge:
                        if other_vehicle_index != vehicle_index:
                            other_channel_condition = self._channel_condition_matrix[other_vehicle_index][edge_index][self._time_slots.now()]
                            if other_channel_condition < channel_condition:
                                vehicle_intar_edge_inference[vehicle_index, -1] += np.power(np.absolute(other_channel_condition), 2) * cover_mW_to_W(vehicle_edge_transmission_power[other_vehicle_index][edge_index])
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                            vehicle_intar_edge_inference[vehicle_index, e] = 0
                        else:
                            vehicle_intar_edge_inference[vehicle_index, e] = vehicle_intar_edge_inference[vehicle_index, -1]
                            
        """Compute the SINR and transimission time"""
        for edge_index in range(self._config.edge_number):
            if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
                for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index).get_data_size()
                    
                    for e in range(self._config.edge_number):
                        if e == edge_index:
                            vehicle_transmission_time[vehicle_index, e] = punished_time
                        else:
                            vehicle_SINR[vehicle_index, e] = compute_SINR(
                                white_gaussian_noise=self._config.white_gaussian_noise, 
                                channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                                transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                                intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][e],
                                inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][e],)
                            transmission_rate = compute_transmission_rate(
                                SINR=vehicle_SINR[vehicle_index, e], 
                                bandwidth=self._config.edge_bandwidth)
                            if transmission_rate != 0:
                                if float(data_size / transmission_rate) < punished_time:
                                    vehicle_transmission_time[vehicle_index, e] = float(data_size / transmission_rate)
                                else:
                                    vehicle_transmission_time[vehicle_index, e] = punished_time
                            else:
                                vehicle_transmission_time[vehicle_index, e] = punished_time
                    
                    vehicle_SINR[vehicle_index, -1] = compute_SINR(
                        white_gaussian_noise=self._config.white_gaussian_noise, 
                        channel_condition=self._channel_condition_matrix[vehicle_index][edge_index][self._time_slots.now()],
                        transmission_power=vehicle_edge_transmission_power[vehicle_index][edge_index],
                        intra_edge_interference=vehicle_intar_edge_inference[vehicle_index][-1],
                        inter_edge_interference=vehicle_inter_edge_inference[vehicle_index][-1],)
                    
                    transmission_rate = compute_transmission_rate(
                        SINR=vehicle_SINR[vehicle_index, -1], 
                        bandwidth=self._config.edge_bandwidth)
                    
                    if transmission_rate != 0:
                        if float(data_size / transmission_rate) < punished_time:
                            vehicle_transmission_time[vehicle_index, -1] = float(data_size / transmission_rate)
                        else:
                            vehicle_transmission_time[vehicle_index, -1] = punished_time
                    else:
                        vehicle_transmission_time[vehicle_index, -1] = punished_time
                    
                    # myapp.debug(f"data_size: {data_size}")
                    # myapp.debug(f"transmission_rate: {transmission_rate}")
                    # myapp.debug(f"transmission_time: {data_size / transmission_rate}")
                    # print("data_size: ", data_size)
                    # print("transmission_rate: ", transmission_rate)
                    # print("transmission_time: ", data_size / transmission_rate)
                    
                    if self._occuiped:
                        if transmission_rate != 0:
                            occupied_time = int(np.floor(data_size / transmission_rate))
                            if self._occuiped and occupied_time > 0:
                                start_time = int(self._time_slots.now() + 1)
                                end_time = int(self._time_slots.now() + occupied_time + 1)
                                if end_time < self._config.time_slot_number:
                                    for i in range(start_time, end_time):
                                        self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]
                                else:
                                    for i in range(start_time, int(self._config.time_slot_number)):
                                        self._occupied_power[edge_index][i] += vehicle_edge_transmission_power[vehicle_index][edge_index]        

        # successful_serviced = np.zeros(self._config.edge_number + 1)    
        
        # service_time = np.zeros(self._config.edge_number + 1)        
        # rewards = np.zeros(self._config.edge_number + 1)
        # edge_task_requested_number = np.zeros(self._config.edge_number)
    
        # average_transmision_time = 0
        # average_wired_transmission_time = 0        
        # average_execution_time = 0
        
        # for edge_index in range(self._config.edge_number):
        #     if self._vehicle_index_within_edges[edge_index][self._time_slots.now()] != []:
        #         for vehicle_index in self._vehicle_index_within_edges[edge_index][self._time_slots.now()]:
        #             task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())
        #             task_required_number += 1

        #             # DynENV-EDIT : Add penalty
        #             # TODO：有待改进，先使用当前task的有线传输时延来计算两个penalty
        #             reference_wired_time = self._reference_adjacent_wired_time(task_index)

        #             handover_penalty = self._handover_penalty(
        #                 vehicle_index=vehicle_index,
        #                 edge_index=edge_index,
        #                 now_t=self._time_slots.now(),
        #                 horizon=3,
        #                 penalty_value=reference_wired_time,
        #             )

        #             boundary_penalty = self._boundary_risk_penalty(
        #                 vehicle_index=vehicle_index,
        #                 edge_index=edge_index,
        #                 now_t=self._time_slots.now(),
        #                 safe_margin=100.0,
        #                 penalty_max=0.5 * reference_wired_time,
        #             )

        #             task_service_time = (
        #                 vehicle_transmission_time[vehicle_index, -1]
        #                 + vehicle_wired_transmission_time[vehicle_index, -1]
        #                 + vehicle_execution_time[vehicle_index, -1]
        #                 + handover_penalty
        #                 + boundary_penalty
        #             )

        #             average_transmision_time += vehicle_transmission_time[vehicle_index, -1]
        #             average_wired_transmission_time += vehicle_wired_transmission_time[vehicle_index, -1]
        #             average_execution_time += vehicle_execution_time[vehicle_index, -1]
        #             service_time[-1] += task_service_time

        #             if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
        #                 successful_serviced_number += 1
        #                 successful_serviced[-1] += 1

        #             for e in range(self._config.edge_number):
        #                 if e != edge_index:
        #                     edge_task_requested_number[e] += 1

        #                     cf_task_service_time = (
        #                         vehicle_transmission_time[vehicle_index, e]
        #                         + vehicle_wired_transmission_time[vehicle_index, e]
        #                         + vehicle_execution_time[vehicle_index, e]
        #                         + handover_penalty
        #                         + boundary_penalty
        #                     )

        #                     service_time[e] += cf_task_service_time
        #                     if cf_task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
        #                         successful_serviced[e] += 1   
        successful_serviced = np.zeros(self._config.edge_number + 1)

        service_time = np.zeros(self._config.edge_number + 1)
        rewards = np.zeros(self._config.edge_number + 1)
        edge_task_requested_number = np.zeros(self._config.edge_number)

        average_transmision_time = 0.0
        average_wired_transmission_time = 0.0
        average_execution_time = 0.0

        for edge_index in range(self._config.edge_number):
            current_vehicle_indices = self._vehicle_index_within_edges[edge_index][self._time_slots.now()]
            selected_vehicle_indices = self._select_vehicle_slots_for_edge(edge_index, self._time_slots.now())
            selected_vehicle_set = set(selected_vehicle_indices)

            if current_vehicle_indices != []:
                for vehicle_index in current_vehicle_indices:
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index).get_requested_task_by_slot_index(self._time_slots.now())

                    # 没任务：不计入当前时隙需求
                    if task_index == -1:
                        continue

                    task_required_number += 1
                    is_selected = (vehicle_index in selected_vehicle_set)

                    # 可以选择固定的penalty或者与wired_time同步的动态penalty
                    reference_wired_time = 1.0
                    # reference_wired_time = self._reference_adjacent_wired_time(task_index)

                    handover_penalty = self._handover_penalty(
                        vehicle_index=vehicle_index,
                        edge_index=edge_index,
                        now_t=self._time_slots.now(),
                        horizon=3,
                        penalty_value=reference_wired_time,
                    )

                    boundary_penalty = self._boundary_risk_penalty(
                        vehicle_index=vehicle_index,
                        edge_index=edge_index,
                        now_t=self._time_slots.now(),
                        safe_margin=50.0,
                        penalty_max=0.5 * reference_wired_time,
                    )

                    # -----------------------------
                    # Global reward branch
                    # -----------------------------
                    if not is_selected:
                        task_service_time = self._ignored_task_failure_time(task_index)
                    else:
                        task_service_time = (
                            vehicle_transmission_time[vehicle_index, -1]
                            + vehicle_wired_transmission_time[vehicle_index, -1]
                            + vehicle_execution_time[vehicle_index, -1]
                            + handover_penalty
                            + boundary_penalty
                        )

                        average_transmision_time += vehicle_transmission_time[vehicle_index, -1]
                        average_wired_transmission_time += vehicle_wired_transmission_time[vehicle_index, -1]
                        average_execution_time += vehicle_execution_time[vehicle_index, -1]

                    service_time[-1] += task_service_time

                    if task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                        successful_serviced_number += 1
                        successful_serviced[-1] += 1

                    # -----------------------------
                    # Edge-level counterfactual branch
                    # -----------------------------
                    for e in range(self._config.edge_number):
                        if e != edge_index:
                            edge_task_requested_number[e] += 1

                            if not is_selected:
                                cf_task_service_time = self._ignored_task_failure_time(task_index)
                            else:
                                cf_task_service_time = (
                                    vehicle_transmission_time[vehicle_index, e]
                                    + vehicle_wired_transmission_time[vehicle_index, e]
                                    + vehicle_execution_time[vehicle_index, e]
                                    + handover_penalty
                                    + boundary_penalty
                                )

                            service_time[e] += cf_task_service_time
                            if cf_task_service_time <= self._task_list.get_task_by_index(task_index).get_delay_threshold():
                                successful_serviced[e] += 1

        rewards[-1] = successful_serviced[-1] / task_required_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = successful_serviced[edge_index] / task_required_number
        for edge_index in range(self._config.edge_number):
            rewards[edge_index] = rewards[-1] - rewards[edge_index]
    
        
        cumulative_reward = successful_serviced[-1] / task_required_number
        
        average_vehicle_SINR = np.sum(vehicle_SINR[:, -1])
        
        average_vehicle_intar_interference = np.sum(vehicle_intar_edge_inference[:, -1])
        average_vehicle_inter_interference = np.sum(vehicle_inter_edge_inference[:, -1])
        average_vehicle_interference = average_vehicle_intar_interference + average_vehicle_inter_interference

        # penalty time
        average_penalty_time = 0.0
        # 伪代码：average_penalty_time += handover_penalty + boundary_penalty       
        # TODO：考虑用不用加penalty time 
        average_service_time = average_transmision_time + average_wired_transmission_time + average_execution_time + average_penalty_time
        
        # print("successful_serviced_number: ", successful_serviced_number)
        # print("task_required_number: ", task_required_number)
        
        return rewards, cumulative_reward, average_vehicle_SINR, average_vehicle_intar_interference, average_vehicle_inter_interference, average_vehicle_interference, average_transmision_time, average_wired_transmission_time, average_execution_time, average_service_time, successful_serviced_number, task_offloaded_number, task_required_number
    

    """Define the action spaces of edge in critic network."""
    def critic_network_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        critic_network_action_shape = (self._config.critic_network_action_size, )
        return specs.BoundedArray(
            shape=(critic_network_action_shape),
            dtype=float,
            minimum=np.zeros(critic_network_action_shape),
            maximum=np.ones(critic_network_action_shape),
            name='critic_actions',
        )

    """Define the gloabl observation spaces."""
    def observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, ) 
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
            observation_shape = (self._config.edge_number, observation_size)
            if self._flatten_space:
                observation_shape = (self._config.edge_number * observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='observations'
        )
    
    def edge_observation_spec(self) -> specs.BoundedArray:
        """Define and return the observation space."""
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        observation_shape = (observation_size, )
        return specs.BoundedArray(
            shape=observation_shape,
            dtype=float,
            minimum=np.zeros(observation_shape),
            maximum=np.ones(observation_shape),
            name='edge_observations'
        )
    
    """Define the gloabl action spaces."""
    def action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self._config.edge_number, self._config.action_size)
        if self._flatten_space:
            action_shape = (self._config.edge_number * self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )

    def edge_action_spec(self) -> specs.BoundedArray:
        """Define and return the action space."""
        action_shape = (self._config.action_size, )
        return specs.BoundedArray(
            shape=action_shape,
            dtype=float,
            minimum=np.zeros(action_shape),
            maximum=np.ones(action_shape),
            name='actions'
        )
    
    def reward_spec(self):
        """Define and return the reward space."""
        reward_shape = (self._config.reward_size, )
        return specs.Array(
            shape=reward_shape, 
            dtype=float, 
            name='rewards'
        )
    
    def _observation(self) -> np.ndarray:
        """Return the observation of the environment."""
        """
        observation_shape = (self._config.edge_number, self._config.observation_size)
        The observation space is the location, task size, and computation cycles of each vehicle, then the aviliable transmission power, and computation resoucers
        """
        if self._occuiped:
            observation_size = self._config.observation_size
            if not self._for_mad5pg:
                observation_size -= 2
        else:
            observation_size = self._config.observation_size - 2 * self._config.edge_number
            if not self._for_mad5pg:
                observation_size -= 2
        # print("self._config.observation_size: ", self._config.observation_size)
        # print("self._for_mad5pg: ", self._for_mad5pg)
        # print("observation_size: ", observation_size)
        observation = np.zeros(shape=(self._config.edge_number, observation_size))

        # EDIT: observation info
        # ================================================================================================= #
        # print("\n[DEBUG _observation] time_slot:", self._time_slots.now())
        # print("[DEBUG _observation] observation_size:", observation_size)
        # print("[DEBUG _observation] edge_number:", self._config.edge_number)
        # print("[DEBUG _observation] max_vehicle_slots_per_edge:", self._config.max_vehicle_slots_per_edge)
        # ================================================================================================= #
        
        for j in range(self._config.edge_number):
            # ==================================== DynENV-EDIT : slot selector logic ==================================== #
            # vehicle_observed_index_within_edges = self._vehicle_observed_index_within_edges[j][self._time_slots.now()]
            # vehicle_index_within_edges = self._vehicle_index_within_edges[j][self._time_slots.now()]
            # vehicle_number_in_edge = len(vehicle_observed_index_within_edges)
            raw_vehicle_observed_index_within_edges = self._vehicle_observed_index_within_edges[j][self._time_slots.now()]
            selected_vehicle_indices = self._select_vehicle_slots_for_edge(j, self._time_slots.now())
            vehicle_number_in_edge = len(selected_vehicle_indices)

            # Test DEBUG
            # if self._time_slots.now() < 2 and j < 2 or self._time_slots.now() > 50 and self._time_slots.now() < 53 and j < 2:
            #     print(f"[OBS DEBUG] t={self._time_slots.now()} edge={j}")
            #     print("raw observed =", raw_vehicle_observed_index_within_edges)
            #     print("selected =", selected_vehicle_indices)
            #     print("selected_count =", vehicle_number_in_edge)
            #     print("max_slots =", self._config.max_vehicle_slots_per_edge)

            # ==================================== DynENV-EDIT : slot selector logic ==================================== #
            index = 0
            # if vehicle_number_in_edge != 3:
            #     print("vehicle_number_in_edge: ", len(vehicle_index_within_edges))

            # EDIT: observation info
            # ================================================================================================= #
            # print(f"\n[DEBUG _observation] ===== edge {j} =====")
            # print("vehicle_observed_index_within_edges:", vehicle_observed_index_within_edges)
            # print("vehicle_index_within_edges:", vehicle_index_within_edges)
            # print("vehicle_number_in_edge:", vehicle_number_in_edge)
            # ================================================================================================= #

            for i in range(vehicle_number_in_edge):
                try:
                    # DynENV-EDIT : slot selector logic
                    # vehicle_index = vehicle_observed_index_within_edges[i]
                    vehicle_index = selected_vehicle_indices[i]

                    distance = self._distance_matrix[vehicle_index][j][self._time_slots.now()]
                    task_index = self._vehicle_list.get_vehicle_by_index(vehicle_index=vehicle_index).get_requested_task_by_slot_index(slot_index=self._time_slots.now())
                    data_size = self._task_list.get_task_by_index(task_index=task_index).get_data_size()
                    computing_cycles = self._task_list.get_task_by_index(task_index=task_index).get_computation_cycles()
                    delay_threshold = self._task_list.get_task_by_index(task_index=task_index).get_delay_threshold()
                    observation[j][index] = float(vehicle_index / self._config.vehicle_number)
                    index += 1
                    observation[j][index] = float(distance / self._config.communication_range)
                    index += 1
                    if task_index == -1:
                        observation[j][index] = 0
                        index += 1
                        observation[j][index] = 0
                        index += 1
                        # observation[j][index] = 0
                        # index += 1
                        observation[j][index] = 0
                        index += 1
                    else:
                        observation[j][index] = 1
                        index += 1
                        observation[j][index] = float((data_size - self._config.task_minimum_data_size) / (self._config.task_maximum_data_size - self._config.task_minimum_data_size))
                        index += 1
                        # observation[j][index] = float((computing_cycles  - self._config.task_minimum_computation_cycles) / (self._config.task_maximum_computation_cycles - self._config.task_minimum_computation_cycles))
                        # index += 1
                        observation[j][index] = float((delay_threshold - self._config.task_minimum_delay_thresholds) / (self._config.task_maximum_delay_thresholds - self._config.task_minimum_delay_thresholds))
                        index += 1

                    # EDIT: observation info
                    # ================================================================================================= #
                    # print(f"[DEBUG _observation] edge {j}, local vehicle slot {i}, vehicle_index={vehicle_index}")
                    # print("distance:", distance)
                    # print("task_index:", task_index)
                    # if task_index != -1:
                    #     print("data_size:", data_size)
                    #     print("computing_cycles:", computing_cycles)
                    #     print("delay_threshold:", delay_threshold)
                    # ================================================================================================= #

                except IndexError:
                    pass
            # print("index 1: ", index)
            # DynENV-EDIT : 已修改，需要优化老的vehicle_number_within_edges逻辑，改为新的max_vehicle_slots_per_edge
            index = self._config.max_vehicle_slots_per_edge * 5
            for i in range(self._config.edge_number):
                edge_compuation_speed = self._edge_list.get_edge_by_index(edge_index=i).get_computing_speed()
                observation[j][index] = float((edge_compuation_speed - self._config.edge_minimum_computing_cycles)/ (self._config.edge_maximum_computing_cycles - self._config.edge_minimum_computing_cycles))
                index += 1
            # print("index 2: ", index)
            if self._occuiped and not self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
            if self._for_mad5pg and not self._occuiped:
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
            # print("index 3: ", index)
            if self._occuiped and self._for_mad5pg:
                for i in range(self._config.edge_number):
                    observation[j][index] = self._occupied_power[i][self._time_slots.now()] / ( 1.01 * self._config.edge_power)
                    index += 1
                    observation[j][index] = self._occupied_computing_resources[i][self._time_slots.now()] / ( 1.01 * self._config.edge_maximum_computing_cycles)
                    index += 1
                observation[j][-2] = self._time_slots.now() / (self._config.time_slot_number - 1)
                observation[j][-1] = j / (self._config.edge_number - 1)
        
        # debug
        #print("flatten_space:", self._flatten_space)

        if self._flatten_space:
            if self._occuiped:
                observation_size = self._config.observation_size
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
            else:
                observation_size = self._config.observation_size - 2 * self._config.edge_number
                if not self._for_mad5pg:
                    observation_size -= 2
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
                else:
                    observation = np.reshape(observation, newshape=(self._config.edge_number * observation_size, ))
        
        # EDIT: observation info
        # ================================================================================================= #
        # print("\n[DEBUG _observation] final observation shape:", observation.shape)
        # print(observation)
        # ================================================================================================= #
        
        return observation

    # ==================================== DynENV-EDIT : helper func ==================================== #
    def _select_vehicle_slots_for_edge(self, edge_index: int, now_t: int) -> List[int]:
        raw_vehicle_indices = self._vehicle_observed_index_within_edges[edge_index][now_t]
        max_slots = self._config.max_vehicle_slots_per_edge

        if len(raw_vehicle_indices) <= max_slots:
            return list(raw_vehicle_indices)

        return list(raw_vehicle_indices[:max_slots])
    # ==================================== DynENV-EDIT : helper func ==================================== #

    
    # ==================================== Edit: Snapshot Builder ==================================== #
    def _get_current_time_index(self) -> int:
        return int(self._time_slots.now())

    def _get_graph_connection_range(self) -> float:
        # 图连接半径略大于真实通信半径，用于提前感知边界切换，可以修改具体数值
        return float(self._config.communication_range * 1.2) # DynENV-EDIT : 复杂的环境适当改大一点

    def _get_edge_grid_adjacency(self) -> List[Tuple[int, int]]:
        """
        固定 3x3 edge 网格邻接。
        第一版返回无向边对，每条边只出现一次。
        
        包含：
        - 四邻接：上下左右（通过 right/down 生成）
        - 对角邻接：左下、右下
        
        对于 3x3 网格：
        - 四邻接无向边共 12 条
        - 对角无向边共 8 条
        - 总计 20 条无向边
        """
        edge_num = self._config.edge_number
        width = int(np.sqrt(edge_num))
        assert width * width == edge_num, "First version assumes edge_number is a perfect square."

        pairs = []
        for r in range(width):
            for c in range(width):
                idx = r * width + c

                # right
                if c + 1 < width:
                    right_idx = r * width + (c + 1)
                    pairs.append((idx, right_idx))

                # down
                if r + 1 < width:
                    down_idx = (r + 1) * width + c
                    pairs.append((idx, down_idx))

                # down-right diagonal
                if r + 1 < width and c + 1 < width:
                    dr_idx = (r + 1) * width + (c + 1)
                    pairs.append((idx, dr_idx))

                # down-left diagonal
                if r + 1 < width and c - 1 >= 0:
                    dl_idx = (r + 1) * width + (c - 1)
                    pairs.append((idx, dl_idx))

        return pairs

    def _get_active_vehicle_indices(self, now_t: int) -> List[int]:
        """
        当前时刻至少落在一个 graph_connection_range 内的车辆，视为 active vehicles。
        """
        active = []
        graph_range = self._get_graph_connection_range()

        for vehicle in self._vehicle_list.get_vehicle_list():
            v_idx = vehicle.get_vehicle_index()
            v_loc = vehicle.get_vehicle_location(now_t)
            if v_loc is None:
                continue

            is_active = False
            for edge in self._edge_list.get_edge_list():
                e_loc = edge.get_edge_location()
                dist = v_loc.get_distance(e_loc)
                if dist <= graph_range:
                    is_active = True
                    break

            if is_active:
                active.append(v_idx)

        return active


    def _build_edge_node_feature(self, edge, now_t: int) -> np.ndarray:
        """
        边缘结点特征构建
        """
        e_idx = edge.get_edge_index()
        e_loc = edge.get_edge_location()

        x = e_loc.get_x()
        y = e_loc.get_y()
        computing_speed = edge.get_computing_speed()
        comm_range = edge.get_communication_range()

        feat = np.array([
            0.0, 1.0,  # node type one-hot: edge
            _normalize(e_idx, 0, self._config.edge_number - 1),
            _normalize(x, 0.0, self._config.map_length),
            _normalize(y, 0.0, self._config.map_width),

            0.0,  # has_task (not used for edge)
            0.0,  # data_size_norm
            0.0,  # computation_cycles_norm
            0.0,  # delay_threshold_norm

            _normalize(
                computing_speed,
                self._config.edge_minimum_computing_cycles,
                self._config.edge_maximum_computing_cycles
            ),
            _normalize(
                comm_range,
                0.0,
                self._config.communication_range * np.sqrt(2.0)
            ),
            _normalize(
                now_t,
                self._config.time_slot_start,
                self._config.time_slot_end
            ),
        ], dtype=np.float32)

        return feat

    def _build_vehicle_node_feature(self, vehicle, now_t: int) -> np.ndarray:
        v_idx = vehicle.get_vehicle_index()
        v_loc = vehicle.get_vehicle_location(now_t)

        x = 0.0 if v_loc is None else v_loc.get_x()
        y = 0.0 if v_loc is None else v_loc.get_y()

        requested_task_idx = vehicle.get_requested_task_by_slot_index(now_t)

        if requested_task_idx != -1:
            has_task = 1.0
            task = self._task_list.get_task_by_index(requested_task_idx)

            data_size_norm = _normalize(
                task.get_data_size(),
                self._config.task_minimum_data_size,
                self._config.task_maximum_data_size
            )
            computation_cycles_norm = _normalize(
                task.get_computation_cycles(),
                self._config.task_minimum_computation_cycles,
                self._config.task_maximum_computation_cycles
            )
            delay_threshold_norm = _normalize(
                task.get_delay_threshold(),
                self._config.task_minimum_delay_thresholds,
                self._config.task_maximum_delay_thresholds
            )
        else: # NOTICE：requested_task_idx == -1，直接设置相应属性为0，避免出现负数bug。注意：没有任务可以直接设置为0吗？需要商榷
            has_task = 0.0
            data_size_norm = 0.0
            computation_cycles_norm = 0.0
            delay_threshold_norm = 0.0

        feat = np.array([
            1.0, 0.0,  # node type one-hot: vehicle
            _normalize(v_idx, 0, self._config.vehicle_number - 1),
            _normalize(x, 0.0, self._config.map_length),
            _normalize(y, 0.0, self._config.map_width),

            has_task,
            data_size_norm,
            computation_cycles_norm,
            delay_threshold_norm,

            0.0,  # edge computing speed (not used for vehicle)
            0.0,  # comm_range (not used for vehicle)
            _normalize(
                now_t,
                self._config.time_slot_start,
                self._config.time_slot_end
            ),
        ], dtype=np.float32)

        return feat

    # DynENV-EDIT
    def _count_snapshot_edges_raw(self, now_t: int) -> int:
        graph_range = self._get_graph_connection_range()
        edge_list = self._edge_list.get_edge_list()
        all_vehicle_indices = list(range(self._config.vehicle_number))

        # edge-edge fixed adjacency, double directed
        ee_pairs = self._get_edge_grid_adjacency()
        edge_count = len(ee_pairs) * 2

        # vehicle-edge dynamic edges, double directed
        for v_idx in all_vehicle_indices:
            vehicle = self._vehicle_list.get_vehicle_by_index(v_idx)
            v_loc = vehicle.get_vehicle_location(now_t)
            if v_loc is None:
                continue

            for edge in edge_list:
                dist = v_loc.get_distance(edge.get_edge_location())
                if dist <= graph_range:
                    edge_count += 2

        return edge_count
    
    # DynENV-EDIT
    def _compute_snapshot_max_edges(self) -> int:
        max_edges = 0
        edge_counts = []

        for t in range(self._config.time_slot_start, self._config.time_slot_end + 1):
            e_t = self._count_snapshot_edges_raw(t)
            edge_counts.append(e_t)
            max_edges = max(max_edges, e_t)

        print("[SNAPSHOT DEBUG] max_edges =", max_edges)
        print("[SNAPSHOT DEBUG] mean_edges =", np.mean(edge_counts))
        print("[SNAPSHOT DEBUG] p90_edges =", np.percentile(edge_counts, 90))
        print("[SNAPSHOT DEBUG] p95_edges =", np.percentile(edge_counts, 95))
        return int(max_edges)
    
    # DynENV-EDIT
    def _pad_snapshot_edges(self, edge_index_raw, edge_feature_raw, edge_type_raw):
        E_t = edge_feature_raw.shape[0]
        E_max = self._snapshot_max_edges

        edge_index = np.zeros((2, E_max), dtype=np.int32)
        edge_feature = np.zeros((E_max, 4), dtype=np.float32)
        edge_type = np.zeros((E_max,), dtype=np.int32)
        edge_mask = np.zeros((E_max,), dtype=np.float32)

        if E_t > 0:
            edge_index[:, :E_t] = edge_index_raw.astype(np.int32)
            edge_feature[:E_t] = edge_feature_raw.astype(np.float32)
            edge_type[:E_t] = edge_type_raw.astype(np.int32)
            edge_mask[:E_t] = 1.0

        return edge_index, edge_feature, edge_type, edge_mask
    
    # DynENV-EDIT 
    def _build_snapshot_edges_raw(
        self,
        now_t: int,
        edge_node_indices: Dict[int, int],
        vehicle_node_indices: Dict[int, int],
    ):
        """
        构造当前时刻的“真实边”（不做 padding）。

        Returns
        -------
        edge_index_raw : np.ndarray
            shape [2, E_t], dtype int32/int64
        edge_feature_raw : np.ndarray
            shape [E_t, 4], dtype float32
        edge_type_raw : np.ndarray
            shape [E_t], dtype int32

        edge_type:
            0 = vehicle-edge
            1 = edge-edge
        """
        graph_range = self._get_graph_connection_range()
        edge_list = self._edge_list.get_edge_list()
        all_vehicle_indices = list(range(self._config.vehicle_number))

        edge_index_src = []
        edge_index_dst = []
        edge_features = []
        edge_types = []

        # =========================================================
        # (A) edge-edge fixed grid adjacency
        # =========================================================
        ee_pairs = self._get_edge_grid_adjacency()
        max_map_dist = np.sqrt(self._config.map_length ** 2 + self._config.map_width ** 2)

        for e1, e2 in ee_pairs:
            edge1 = self._edge_list.get_edge_by_index(e1)
            edge2 = self._edge_list.get_edge_by_index(e2)

            loc1 = edge1.get_edge_location()
            loc2 = edge2.get_edge_location()
            dist = loc1.get_distance(loc2)

            feat = np.array([
                _normalize(dist, 0.0, max_map_dist),
                _normalize(
                    self._config.wired_transmission_rate,
                    0.0,
                    self._config.wired_transmission_rate
                ),
                1.0,  # wired flag
                0.0,  # within coverage flag (not used for edge-edge)
            ], dtype=np.float32)

            n1 = edge_node_indices[e1]
            n2 = edge_node_indices[e2]

            # 双向边
            edge_index_src.extend([n1, n2])
            edge_index_dst.extend([n2, n1])
            edge_features.extend([feat, feat.copy()])
            edge_types.extend([1, 1])

        # =========================================================
        # (B) vehicle-edge dynamic edges
        # 说明：
        # - 节点集固定：所有 vehicle 都在图里
        # - 边集动态：只有当前时刻 dist <= graph_range 的 vehicle-edge 边进入 raw graph
        # =========================================================
        for v_idx in all_vehicle_indices:
            vehicle = self._vehicle_list.get_vehicle_by_index(v_idx)
            v_loc = vehicle.get_vehicle_location(now_t)
            if v_loc is None:
                continue

            v_node = vehicle_node_indices[v_idx]

            for edge in edge_list:
                e_idx = edge.get_edge_index()
                e_loc = edge.get_edge_location()

                dist = v_loc.get_distance(e_loc)
                if dist <= graph_range:
                    within_coverage = 1.0 if dist <= edge.get_communication_range() else 0.0

                    channel_gain = 0.0
                    if hasattr(self, "_channel_condition_matrix"):
                        try:
                            channel_gain = float(self._channel_condition_matrix[v_idx][e_idx][now_t])
                        except Exception:
                            channel_gain = 0.0

                    feat = np.array([
                        _normalize(dist, 0.0, graph_range),
                        channel_gain,
                        0.0,              # wired flag
                        within_coverage,  # 是否落在真实通信覆盖范围内
                    ], dtype=np.float32)

                    e_node = edge_node_indices[e_idx]

                    # 双向边
                    edge_index_src.extend([v_node, e_node])
                    edge_index_dst.extend([e_node, v_node])
                    edge_features.extend([feat, feat.copy()])
                    edge_types.extend([0, 0])

        # =========================================================
        # pack raw edges
        # =========================================================
        if len(edge_features) == 0:
            edge_index_raw = np.zeros((2, 0), dtype=np.int32)
            edge_feature_raw = np.zeros((0, 4), dtype=np.float32)
            edge_type_raw = np.zeros((0,), dtype=np.int32)
        else:
            edge_index_raw = np.array([edge_index_src, edge_index_dst], dtype=np.int32)
            edge_feature_raw = np.stack(edge_features, axis=0).astype(np.float32)
            edge_type_raw = np.array(edge_types, dtype=np.int32)

        return edge_index_raw, edge_feature_raw, edge_type_raw
    

    def _build_snapshot_v1(self) -> Dict[str, Any]:
        """
        全局 snapshot builder（同质图版本）。
        返回一个 numpy dict，后面可直接转给 replay / GNN。
        """
        now_t = self._get_current_time_index()
        graph_range = self._get_graph_connection_range()

        edge_list = self._edge_list.get_edge_list()
        vehicle_list = self._vehicle_list.get_vehicle_list()

        # ---------- nodes ---------- #
        edge_indices = [edge.get_edge_index() for edge in edge_list]
        # DynENV-EDIT : 启用active_vehicle_indices
        # active_vehicle_indices = self._get_active_vehicle_indices(now_t)
        all_vehicle_indices = list(range(self._config.vehicle_number))

        # 建立全局 node id -> snapshot node index 映射
        # 用 ("edge", e_idx) / ("vehicle", v_idx) 标识
        node_keys = []
        node_features = []
        node_types = []

        edge_node_indices = {}
        vehicle_node_indices = {}

        # 先放 edge nodes
        for edge in edge_list:
            e_idx = edge.get_edge_index()
            node_idx = len(node_keys)
            node_keys.append(("edge", e_idx))
            edge_node_indices[e_idx] = node_idx
            node_features.append(self._build_edge_node_feature(edge, now_t))
            node_types.append(1)  # 1 = edge

        # 再放 active vehicle nodes
        # DynENV-EDIT : 弃用active_vehicle_indices        
        # for v_idx in active_vehicle_indices:
        for v_idx in all_vehicle_indices:
            vehicle = self._vehicle_list.get_vehicle_by_index(v_idx)
            node_idx = len(node_keys)
            node_keys.append(("vehicle", v_idx))
            vehicle_node_indices[v_idx] = node_idx
            node_features.append(self._build_vehicle_node_feature(vehicle, now_t))
            node_types.append(0)  # 0 = vehicle

        node_feature = np.stack(node_features, axis=0).astype(np.float32)
        node_type = np.array(node_types, dtype=np.int32)

        # DynENV-EDIT
        # ---------- edges ---------- #
        edge_index_raw, edge_feature_raw, edge_type_raw = self._build_snapshot_edges_raw(
            now_t=now_t,
            edge_node_indices=edge_node_indices,
            vehicle_node_indices=vehicle_node_indices,
        )
        edge_index, edge_feature, edge_type, edge_mask = self._pad_snapshot_edges(
            edge_index_raw, edge_feature_raw, edge_type_raw
        )

        snapshot = {
            "time_index": int(now_t),
            "node_feature": node_feature,             # [N, d_node]
            "node_type": node_type,                   # [N]
            "edge_index": edge_index,                 # [2, E]
            "edge_feature": edge_feature,             # [E, d_edge]
            "edge_type": edge_type,                   # [E]
            "edge_mask": edge_mask,                   # DynENV-EDIT : select raw edges
            "edge_node_indices": edge_node_indices,   # dict: edge_id -> node_idx
            "vehicle_node_indices": vehicle_node_indices,  # dict: vehicle_id -> node_idx
            "num_nodes": int(node_feature.shape[0]),
            "num_edges": int(edge_index.shape[1]) if edge_index.size > 0 else 0,      # = E_max
            "num_edges_raw": int(edge_feature_raw.shape[0]),    # DynENV-EDIT
        }
        return snapshot
    
    def get_current_snapshot(self):
        return self._build_snapshot_v1()

    def get_current_local_and_global_state(self):
        return {
            "observation": self._observation(),
            "snapshot": self._build_snapshot_v1(),
        }


    # debug function
    def debug_print_snapshot(self, snapshot: Dict[str, Any], max_edges: int = 20):
        print("\n========== SNAPSHOT DEBUG ==========")
        print("time_index:", snapshot["time_index"])
        print("node_feature shape:", snapshot["node_feature"].shape)
        print("edge_index shape:", snapshot["edge_index"].shape)
        print("edge_feature shape:", snapshot["edge_feature"].shape)
        print("num edge nodes:", len(snapshot["edge_node_indices"]))
        print("num vehicle nodes:", len(snapshot["vehicle_node_indices"]))

        print("edge_node_indices:", snapshot["edge_node_indices"])
        print("vehicle_node_indices:", snapshot["vehicle_node_indices"])

        print("edge_mask shape:", snapshot["edge_mask"].shape)
        print("num_edges_raw:", snapshot["num_edges_raw"])
        print("edge_mask sum:", np.sum(snapshot["edge_mask"]))

        print("\nFirst few node features:")
        for i in range(min(15, snapshot["node_feature"].shape[0])): # 可以改min()中的第一个数“15”，来控制log多少个node features
            print(f"node {i}, type={snapshot['node_type'][i]}, feat={snapshot['node_feature'][i]}")

        print("\nFirst few edges:")
        num_edges = snapshot["edge_index"].shape[1]
        for k in range(min(max_edges, num_edges)):
            src = snapshot["edge_index"][0, k]
            dst = snapshot["edge_index"][1, k]
            etype = snapshot["edge_type"][k]
            efeat = snapshot["edge_feature"][k]
            print(f"edge {k}: {src} -> {dst}, type={etype}, feat={efeat}")
        print("====================================\n")

    # ==================================== Edit: Snapshot Builder ==================================== #


Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    edge_observations: NestedSpec
    critic_actions: NestedSpec
    actions: NestedSpec
    edge_actions: NestedSpec
    rewards: NestedSpec
    discounts: NestedSpec


def make_environment_spec(environment: vehicularNetworkEnv) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        edge_observations=environment.edge_observation_spec(),
        critic_actions=environment.critic_network_action_spec(),
        actions=environment.action_spec(),
        edge_actions=environment.edge_action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec())
    

def define_size_of_spaces(
    vehicle_number_within_edges: int,  # NOTICE : 这里由于是形式参数，不需要再将vehicle_number_within_edges改为新的max_vehicle_slots_per_edge了
    edge_number: int,
    task_assigned_number: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """The action space is task assignment"""
    # action_size for mad4pg
    action_size = vehicle_number_within_edges * edge_number
    
    # action_size = vehicle_number_within_edges + vehicle_number_within_edges * edge_number
    
    """The observation space is the location, task size, computing cycles of each vehicle, then the aviliable transmission power, and computation resoucers"""
    observation_size = vehicle_number_within_edges * 5 + edge_number * 3 + 2
    
    """The reward space is the reward of each edge node and the gloabl reward
    reward[-1] is the global reward.
    reward[0:edge_number] are the edge rewards.
    """
    reward_size = edge_number + 1
    
    """Defined the shape of the action space in critic network"""
    critici_network_action_size = edge_number * action_size
    
    return action_size, observation_size, reward_size, critici_network_action_size

    
def init_distance_matrix_and_radio_coverage_matrix(
    env_config: env_config,
    vehicle_list: vehicleList,
    edge_list: edgeList,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]]]:
    """Initialize the distance matrix and radio coverage."""
    matrix_shpae = (env_config.vehicle_number, env_config.edge_number, env_config.time_slot_number)
    distance_matrix = np.zeros(matrix_shpae)
    channel_condition_matrix = [[[[] for _ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)] for _ in range(env_config.vehicle_number)]
    """Get the radio coverage information of each edge node."""
    vehicle_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    vehicle_observed_index_within_edges = [[[] for __ in range(env_config.time_slot_number)] for _ in range(env_config.edge_number)]
    for i in range(env_config.vehicle_number):
        for j in range(env_config.edge_number):
            for k in range(env_config.time_slot_number):
                distance = vehicle_list.get_vehicle_by_index(i).get_distance_between_edge(k, edge_list.get_edge_by_index(j).get_edge_location())
                distance_matrix[i][j][k] = distance
                channel_condition_matrix[i][j][k] = compute_channel_gain(
                    rayleigh_distributed_small_scale_fading=generate_complex_normal_distribution(),
                    distance=distance,
                    path_loss_exponent=env_config.path_loss_exponent,
                )
                if distance_matrix[i][j][k] <= env_config.communication_range:
                    requested_task_index = vehicle_list.get_vehicle_by_index(i).get_requested_task_by_slot_index(k)
                    vehicle_observed_index_within_edges[j][k].append(i)
                    if requested_task_index != -1:
                        vehicle_index_within_edges[j][k].append(i)
    return distance_matrix, channel_condition_matrix, vehicle_index_within_edges, vehicle_observed_index_within_edges
    

# ======================== EDIT ======================== #
def _normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return float((value - vmin) / (vmax - vmin))


def _safe_float(x):
    if x is None:
        return 0.0
    return float(x)

# ======================== EDIT ======================== #