import traci
import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class TrafficLightPhase(Enum):
    NORTH_SOUTH_GREEN = 0
    NORTH_SOUTH_YELLOW = 1
    EAST_WEST_GREEN = 4
    EAST_WEST_YELLOW = 3


@dataclass
class TrafficMetrics:
    waiting_time: float
    throughput: int
    average_speed: float
    queue_length: Dict[str, int]
    emissions: float
    travel_time: float


class SumoTrafficEnv:
    def __init__(
        self,
        sumo_config_file: str = "sumo_config/intersection.sumocfg",
        use_gui: bool = True,
        step_length: float = 1.0,
        seed: int = None,
    ):
        """
        SUMO Traffic Environment for RL training

        Args:
            sumo_config_file: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            step_length: Simulation step length in seconds
            seed: Random seed for reproducibility
        """
        self.sumo_config_file = sumo_config_file
        self.use_gui = use_gui
        self.step_length = step_length
        self.seed = seed if seed is not None else random.randint(0, 100000)

        self.tl_id = "intersection1"
        self.current_phase = TrafficLightPhase.NORTH_SOUTH_GREEN
        self.phase_duration = 0
        self.min_phase_duration = 15
        self.yellow_duration = 3

        self.max_queue_length = 20
        self.max_waiting_time = 300
        self.observation_radius = 100

        self.action_space_size = 3

        self.episode_stats = {
            "total_waiting_time": 0,
            "vehicles_completed": 0,
            "total_travel_time": 0,
            "total_emissions": 0,
            "timestep": 0,
        }

        self.active_vehicles = set()
        self.completed_vehicles = set()
        self.vehicle_start_times = {}

        # Track previous metrics for delta-based rewards
        self.prev_total_queue = 0
        self.prev_avg_waiting = 0.0
        self.prev_avg_speed = 0.0

        self.connection_label = f"intersection_{self.seed}"
        self.sumo_cmd = None
        self.connected = False

        self._setup_sumo_command()

    def _setup_sumo_command(self):
        """Setup SUMO command based on configuration"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        self.sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_config_file,
            "--step-length",
            str(self.step_length),
            "--seed",
            str(self.seed),
            "--no-warnings",
            "true",
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            "300",
            "--time-to-teleport",
            "-1",  # Disable teleporting
        ]

        if not self.use_gui:
            self.sumo_cmd.extend(["--quit-on-end", "true"])

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        if self.connected:
            self.close()

        traci.start(self.sumo_cmd, label=self.connection_label)
        self.connected = True

        self.current_phase = TrafficLightPhase.NORTH_SOUTH_GREEN
        self.phase_duration = 0
        self.episode_stats = {
            "total_waiting_time": 0,
            "vehicles_completed": 0,
            "total_travel_time": 0,
            "total_emissions": 0,
            "timestep": 0,
        }

        self.active_vehicles = set()
        self.completed_vehicles = set()
        self.vehicle_start_times = {}

        # Reset previous metrics for delta-based rewards
        self.prev_total_queue = 0
        self.prev_avg_waiting = 0.0
        self.prev_avg_speed = 0.0

        traci.trafficlight.setPhase(self.tl_id, self.current_phase.value)

        for _ in range(5):
            traci.simulationStep()

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.episode_stats["timestep"] += 1

        # Apply action (traffic light control)
        action_reward = self._apply_action(action)

        # Advance simulation
        traci.simulationStep()
        self.phase_duration += self.step_length

        # Cache vehicle IDs once per step for optimization
        self._cached_vehicle_ids = traci.vehicle.getIDList()

        # Update vehicle tracking
        self._update_vehicle_tracking()

        # Calculate metrics and reward
        metrics = self._calculate_metrics()
        reward = self._calculate_reward(metrics, action_reward)

        # Check if episode is done
        done = self._is_episode_done()

        # Prepare info
        info = {
            "metrics": metrics,
            "phase": self.current_phase.name,
            "phase_duration": self.phase_duration,
            "timestep": self.episode_stats["timestep"],
            **self.episode_stats,
        }

        # Clear caches
        self._cached_vehicle_ids = None
        self._cached_avg_speed = None

        return self.get_state(), reward, done, info

    def _apply_action(self, action: int) -> float:
        """Apply traffic light action and return immediate reward"""
        reward = 0

        if action == 0:
            return reward

        if self.phase_duration < self.min_phase_duration:
            return -10

        target_phase = None
        if action == 1:
            if self.current_phase != TrafficLightPhase.NORTH_SOUTH_GREEN:
                target_phase = TrafficLightPhase.NORTH_SOUTH_GREEN
        elif action == 2:
            if self.current_phase != TrafficLightPhase.EAST_WEST_GREEN:
                target_phase = TrafficLightPhase.EAST_WEST_GREEN

        if target_phase is not None:
            if self.current_phase == TrafficLightPhase.NORTH_SOUTH_GREEN:
                self._set_phase(TrafficLightPhase.NORTH_SOUTH_YELLOW)
                yellow_steps = max(1, int(self.yellow_duration / self.step_length))
                for _ in range(yellow_steps):
                    traci.simulationStep()
                    self.phase_duration += self.step_length
                self._set_phase(target_phase)
            elif self.current_phase == TrafficLightPhase.EAST_WEST_GREEN:
                self._set_phase(TrafficLightPhase.EAST_WEST_YELLOW)
                yellow_steps = max(1, int(self.yellow_duration / self.step_length))
                for _ in range(yellow_steps):
                    traci.simulationStep()
                    self.phase_duration += self.step_length
                self._set_phase(target_phase)
            else:
                self._set_phase(target_phase)

        return reward

    def _set_phase(self, phase: TrafficLightPhase):
        """Set traffic light phase"""
        traci.trafficlight.setPhase(self.tl_id, phase.value)
        self.current_phase = phase
        self.phase_duration = 0

    def _update_vehicle_tracking(self):
        """Update vehicle tracking for statistics"""
        # Use cached vehicle IDs if available
        if hasattr(self, '_cached_vehicle_ids') and self._cached_vehicle_ids is not None:
            current_vehicles = set(self._cached_vehicle_ids)
        else:
            current_vehicles = set(traci.vehicle.getIDList())
        current_time = traci.simulation.getTime()

        new_vehicles = current_vehicles - self.active_vehicles
        for veh_id in new_vehicles:
            self.vehicle_start_times[veh_id] = current_time

        completed = self.active_vehicles - current_vehicles
        for veh_id in completed:
            if veh_id not in self.completed_vehicles:
                self.completed_vehicles.add(veh_id)
                self.episode_stats["vehicles_completed"] += 1

                if veh_id in self.vehicle_start_times:
                    travel_time = current_time - self.vehicle_start_times[veh_id]
                    self.episode_stats["total_travel_time"] += travel_time
                    del self.vehicle_start_times[veh_id]

        self.active_vehicles = current_vehicles

        if len(self.vehicle_start_times) > len(current_vehicles) * 2:
            cutoff_time = current_time - 600
            old_vehicles = [
                veh_id
                for veh_id, start_time in self.vehicle_start_times.items()
                if start_time < cutoff_time and veh_id not in current_vehicles
            ]
            for veh_id in old_vehicles:
                del self.vehicle_start_times[veh_id]

    def _calculate_metrics(self) -> TrafficMetrics:
        """Calculate current traffic metrics (optimized)"""
        # Use cached vehicle IDs if available
        if hasattr(self, '_cached_vehicle_ids') and self._cached_vehicle_ids is not None:
            vehicle_ids = self._cached_vehicle_ids
        else:
            vehicle_ids = traci.vehicle.getIDList()

        # Optimize: Calculate waiting time, speed, and emissions in single loop
        total_waiting = 0.0
        total_speed = 0.0
        total_emissions = 0.0

        if vehicle_ids:
            for veh_id in vehicle_ids:
                try:
                    total_waiting += traci.vehicle.getWaitingTime(veh_id)
                    total_speed += traci.vehicle.getSpeed(veh_id)
                    total_emissions += traci.vehicle.getCO2Emission(veh_id)
                except Exception:
                    # Vehicle may have left during iteration
                    continue

            avg_speed = total_speed / len(vehicle_ids)
        else:
            avg_speed = 0.0

        # Cache average speed for reuse in get_state()
        self._cached_avg_speed = avg_speed

        self.episode_stats["total_waiting_time"] += total_waiting
        self.episode_stats["total_emissions"] += total_emissions

        # Calculate queue lengths per direction
        queue_lengths = self._get_queue_lengths()

        # Calculate throughput (vehicles completed this step)
        current_completed = len(self.completed_vehicles)
        prev_completed = self.episode_stats.get("prev_completed", 0)
        throughput = max(0, current_completed - prev_completed)
        self.episode_stats["prev_completed"] = current_completed

        # Average travel time
        avg_travel_time = self.episode_stats["total_travel_time"] / max(
            1, self.episode_stats["vehicles_completed"]
        )

        return TrafficMetrics(
            waiting_time=total_waiting,
            throughput=throughput,
            average_speed=avg_speed,
            queue_length=queue_lengths,
            emissions=total_emissions,
            travel_time=avg_travel_time,
        )

    def _get_queue_lengths(self) -> Dict[str, int]:
        """Get queue lengths for each approach"""
        queue_lengths = {"north": 0, "south": 0, "east": 0, "west": 0}

        edges = ["north_in", "south_in", "east_in", "west_in"]
        directions = ["north", "south", "east", "west"]

        for edge, direction in zip(edges, directions):
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge)

            queue_count = 0
            for veh_id in vehicles_on_edge:
                speed = traci.vehicle.getSpeed(veh_id)
                waiting_time = traci.vehicle.getWaitingTime(veh_id)

                if speed < 5.0 or waiting_time > 5.0:
                    queue_count += 1

            queue_lengths[direction] = queue_count

        return queue_lengths

    def _calculate_reward(self, metrics: TrafficMetrics, action_reward: float) -> float:
        """Calculate reward based on traffic metrics (delta-based to avoid congestion bias)"""
        reward = action_reward
        initial_reward = reward

        step_multiplier = self.step_length

        # POSITIVE: Throughput (vehicles completing their journey)
        reward += metrics.throughput * 10

        # Calculate current metrics for comparison
        total_queue = sum(metrics.queue_length.values())
        normalized_waiting = metrics.waiting_time / step_multiplier
        avg_speed = metrics.average_speed

        # DELTA-BASED REWARDS: Reward improvements, penalize worsening
        # This prevents unfair penalties when system naturally congests

        # Queue length delta (reward reduction in queues)
        queue_delta = self.prev_total_queue - total_queue
        reward += queue_delta * 2.0

        # Waiting time delta (reward reduction in waiting)
        waiting_delta = self.prev_avg_waiting - normalized_waiting
        reward += waiting_delta * 0.05

        # Speed delta (reward speed increases)
        speed_delta = avg_speed - self.prev_avg_speed
        reward += speed_delta * 5.0

        # Update previous metrics for next step
        self.prev_total_queue = total_queue
        self.prev_avg_waiting = normalized_waiting
        self.prev_avg_speed = avg_speed

        # ABSOLUTE PENALTIES: Still penalize severe conditions
        # But with reduced weights to allow delta rewards to dominate
        if total_queue > 20:  # Severe congestion
            reward -= (total_queue - 20) * 0.1

        if normalized_waiting > 50:  # Excessive waiting
            reward -= (normalized_waiting - 50) * 0.01

        # EMISSIONS: Small penalty
        normalized_emissions = metrics.emissions / step_multiplier
        reward -= normalized_emissions * 0.00005  # Reduced weight

        # BALANCE: Encourage balanced queue lengths across directions
        queue_values = list(metrics.queue_length.values())
        if queue_values and max(queue_values) > 0:
            balance_factor = min(queue_values) / max(queue_values)
            reward += balance_factor * 5.0

        # DEBUG: Print reward breakdown for first few steps and every 100 steps
        show_debug = False
        if not hasattr(self, "step_counter"):
            self.step_counter = 1
            show_debug = True
        elif self.step_counter < 10:
            show_debug = True
        elif self.step_counter % 100 == 0:
            show_debug = True

        if show_debug:
            print(
                f"DEBUG: step={self.step_counter}, "
                f"throughput={metrics.throughput * 10:.1f}, "
                f"queue_Δ={queue_delta * 2.0:.1f}, "
                f"wait_Δ={waiting_delta * 0.05:.1f}, "
                f"speed_Δ={speed_delta * 5.0:.1f}, "
                f"queues={total_queue}, "
                f"speed={avg_speed:.1f}m/s, "
                f"final={reward:.1f}"
            )

        self.step_counter += 1

        return reward

    def _is_episode_done(self) -> bool:
        """Check if episode should end"""
        is_fast_mode = "fast" in self.sumo_config_file.lower()
        simulation_duration = 1800 if is_fast_mode else 3600
        max_timesteps = int(simulation_duration / self.step_length)

        if self.episode_stats["timestep"] >= max_timesteps:
            return True

        return False

    def get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Queue lengths for each direction
        queue_lengths = self._get_queue_lengths()
        north_queue = min(queue_lengths["north"], self.max_queue_length)
        south_queue = min(queue_lengths["south"], self.max_queue_length)
        east_queue = min(queue_lengths["east"], self.max_queue_length)
        west_queue = min(queue_lengths["west"], self.max_queue_length)

        # Average waiting times per direction
        waiting_times = self._get_average_waiting_times()
        north_wait = min(waiting_times["north"], self.max_waiting_time)
        south_wait = min(waiting_times["south"], self.max_waiting_time)
        east_wait = min(waiting_times["east"], self.max_waiting_time)
        west_wait = min(waiting_times["west"], self.max_waiting_time)

        # Current phase and duration
        current_phase = self.current_phase.value
        phase_duration = min(self.phase_duration, 120)  # Cap at 2 minutes

        # Additional features
        total_vehicles = len(traci.vehicle.getIDList())
        avg_speed = self._get_average_speed()

        state = np.array(
            [
                north_queue,
                south_queue,
                east_queue,
                west_queue,
                north_wait,
                south_wait,
                east_wait,
                west_wait,
                current_phase,
                phase_duration,
                total_vehicles,
                avg_speed,
            ],
            dtype=np.float32,
        )

        return state

    def _get_average_waiting_times(self) -> Dict[str, float]:
        """Get average waiting times for each direction"""
        waiting_times = {"north": 0, "south": 0, "east": 0, "west": 0}

        edges = ["north_in", "south_in", "east_in", "west_in"]
        directions = ["north", "south", "east", "west"]

        for edge, direction in zip(edges, directions):
            vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge)
            if vehicles_on_edge:
                avg_wait = np.mean(
                    [
                        traci.vehicle.getWaitingTime(veh_id)
                        for veh_id in vehicles_on_edge
                    ]
                )
                waiting_times[direction] = avg_wait

        return waiting_times

    def _get_average_speed(self) -> float:
        """Get average speed of all vehicles (optimized)"""
        # Use cached vehicle IDs if available
        if hasattr(self, '_cached_vehicle_ids') and self._cached_vehicle_ids is not None:
            vehicle_ids = self._cached_vehicle_ids
        else:
            vehicle_ids = traci.vehicle.getIDList()

        if not vehicle_ids:
            return 0.0

        # Use cached speed if we already calculated it in _calculate_metrics
        if hasattr(self, '_cached_avg_speed') and self._cached_avg_speed is not None:
            return self._cached_avg_speed

        total_speed = 0.0
        for veh_id in vehicle_ids:
            try:
                total_speed += traci.vehicle.getSpeed(veh_id)
            except Exception:
                continue

        return total_speed / len(vehicle_ids) if vehicle_ids else 0.0

    def get_state_space_size(self) -> int:
        """Return the size of the state space"""
        return 12

    def get_action_space_size(self) -> int:
        """Return the size of the action space"""
        return self.action_space_size

    def close(self):
        """Close the SUMO simulation"""
        if self.connected:
            try:
                traci.close()
                self.connected = False
            except Exception:
                pass

    def render(self, mode="human"):
        """Render the environment (SUMO GUI handles this)"""
        if not self.use_gui:
            print(
                f"Step: {self.episode_stats['timestep']}, "
                f"Phase: {self.current_phase.name}, "
                f"Vehicles: {len(traci.vehicle.getIDList())}"
            )

    def get_traffic_light_state(self) -> str:
        """Get current traffic light state"""
        return traci.trafficlight.getRedYellowGreenState(self.tl_id)

    def get_detailed_metrics(self) -> Dict:
        """Get detailed traffic metrics for analysis"""
        vehicle_ids = traci.vehicle.getIDList()

        metrics = {
            "current_vehicles": len(vehicle_ids),
            "completed_vehicles": self.episode_stats["vehicles_completed"],
            "total_waiting_time": self.episode_stats["total_waiting_time"],
            "average_travel_time": (
                self.episode_stats["total_travel_time"]
                / max(1, self.episode_stats["vehicles_completed"])
            ),
            "total_emissions": self.episode_stats["total_emissions"],
            "queue_lengths": self._get_queue_lengths(),
            "waiting_times": self._get_average_waiting_times(),
            "average_speed": self._get_average_speed(),
            "current_phase": self.current_phase.name,
            "phase_duration": self.phase_duration,
            "timestep": self.episode_stats["timestep"],
        }

        return metrics

    def get_detailed_traffic_info(self) -> Dict:
        """Get detailed traffic info by direction for agent decision making"""
        direction_lanes = {
            "north": ["north_in_0", "north_in_1"],
            "south": ["south_in_0", "south_in_1"],
            "east": ["east_in_0", "east_in_1"],
            "west": ["west_in_0", "west_in_1"],
        }

        traffic_info = {}

        for direction, lanes in direction_lanes.items():
            direction_data = {
                "total_vehicles": 0,
                "stopped_vehicles": 0,
                "avg_waiting_time": 0.0,
                "max_waiting_time": 0.0,
                "avg_speed": 0.0,
            }

            all_vehicles = []
            for lane_id in lanes:
                try:
                    vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    all_vehicles.extend(vehicles)
                except Exception:
                    continue

            if all_vehicles:
                total_wait = 0.0
                total_speed = 0.0
                stopped_count = 0
                max_wait = 0.0

                for veh_id in all_vehicles:
                    try:
                        wait_time = traci.vehicle.getWaitingTime(veh_id)
                        speed = traci.vehicle.getSpeed(veh_id)

                        if isinstance(wait_time, (int, float)):
                            total_wait += wait_time
                            max_wait = max(max_wait, wait_time)

                        if isinstance(speed, (int, float)):
                            total_speed += speed
                            if speed < 1.0:
                                stopped_count += 1
                    except Exception:
                        continue

                direction_data["total_vehicles"] = len(all_vehicles)
                direction_data["stopped_vehicles"] = stopped_count
                direction_data["avg_waiting_time"] = total_wait / len(all_vehicles)
                direction_data["max_waiting_time"] = max_wait
                direction_data["avg_speed"] = total_speed / len(all_vehicles)

            traffic_info[direction] = direction_data

        return traffic_info

    def get_current_phase_name(self) -> str:
        """Get current traffic light phase name"""
        return self.current_phase.name

    def get_vehicles_in_lanes(self, lanes: List[str]) -> List[str]:
        """Get all vehicle IDs in specified lanes"""
        all_vehicles = []
        for lane_id in lanes:
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                all_vehicles.extend(vehicles)
            except Exception:
                continue
        return all_vehicles
