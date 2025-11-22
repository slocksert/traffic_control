import numpy as np
import random
import json
import os
from typing import Dict, Tuple
from collections import defaultdict
from src.utils.gpu_accelerator import get_accelerator


class SumoQLearningAgent:
    def __init__(
        self,
        state_space_size: int = 12,
        action_space_size: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Q-Learning agent specifically adapted for SUMO Traffic Environment

        Args:
            state_space_size: Size of the SUMO state space (12 features)
            action_space_size: Size of the SUMO action space (3 actions)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        self.traffic_flow_memory = []
        self.phase_efficiency = defaultdict(list)
        self.congestion_threshold = 5

        self.action_usage_count = defaultdict(int)
        self.min_action_exploration = 100

        self.gpu_accelerator = get_accelerator()
        print(f"Q-Learning agent using: {self.gpu_accelerator.device}")

        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_next_states = []
        self.batch_size = 32

    def get_action(self, state, training=False):
        """Base method for getting action from Q-table"""
        state_key = self._state_to_key(state)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = (
            reward
            + self.discount_factor * self.q_table[next_state_key][best_next_action]
        )
        td_error = td_target - self.q_table[state_key][action]

        self.q_table[state_key][action] += self.learning_rate * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_all_q_values(self, state):
        """Get all Q-values for a given state"""
        state_key = self._state_to_key(state)
        return self.q_table[state_key].copy()

    def save_model(self, filepath):
        """Save Q-table to file"""
        model_data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": float(self.epsilon),
            "learning_rate": float(self.learning_rate),
            "discount_factor": float(self.discount_factor),
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2, default=self._json_serializer)

    def load_model(self, filepath):
        """Load Q-table from file"""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                model_data = json.load(f)

            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
            for k, v in model_data["q_table"].items():
                self.q_table[k] = np.array(v)

            self.epsilon = model_data.get("epsilon", self.epsilon)
            return True
        return False

    def print_stats(self):
        """Print basic agent statistics"""
        print(f"Q-table size: {len(self.q_table)}")
        print(f"Epsilon: {self.epsilon:.4f}")

    def _state_to_key(self, state):
        """Convert state array to hashable key"""
        if isinstance(state, (list, tuple)):
            return tuple(state)
        elif isinstance(state, np.ndarray):
            return tuple(state.tolist())
        else:
            return str(state)

    def choose_action(self, state, environment=None):
        """
        Choose action with improved epsilon-greedy to prioritize west congestion
        """
        if environment:
            traffic_info = environment.get_detailed_traffic_info()
            west_problem = self._detect_west_congestion(traffic_info)

            if west_problem:
                return 2

        return self.get_action(state, training=True)

    def _detect_west_congestion(self, traffic_info):
        """Detect specific west congestion"""
        west_data = traffic_info.get("west", {})

        west_total = west_data.get("total_vehicles", 0)
        west_stopped = west_data.get("stopped_vehicles", 0)
        west_max_wait = west_data.get("max_waiting_time", 0)
        if west_total > 0:
            stopped_ratio = west_stopped / west_total
            avg_wait = west_data.get("avg_waiting_time", 0)

            if (
                stopped_ratio >= 0.5
                or west_max_wait > 30
                or (west_stopped >= 2 and avg_wait > 15)
            ):
                return True

        return False

    def update_with_sumo_metrics(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        sumo_metrics: Dict,
    ) -> None:
        """
        GPU-accelerated Q-table update with SUMO-specific information

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            sumo_metrics: Additional SUMO metrics for learning
        """
        # Accumulate experiences for batch processing
        self.batch_states.append(state)
        self.batch_actions.append(action)
        self.batch_rewards.append(reward)
        self.batch_next_states.append(next_state)

        # Process batch when full or force update
        if len(self.batch_states) >= self.batch_size:
            self._process_batch_update()
        else:
            # Fall back to individual update for immediate learning
            self.update(state, action, reward, next_state)

        # Track phase efficiency for analysis
        current_phase = int(state[8])
        if current_phase in [0, 2]:  # Green phases only
            throughput = sumo_metrics.get("throughput", 0)
            self.phase_efficiency[current_phase].append(throughput)

        # Store traffic flow patterns
        self.traffic_flow_memory.append(
            {
                "timestep": sumo_metrics.get("timestep", 0),
                "queues": state[0:4].tolist(),
                "waiting_times": state[4:8].tolist(),
                "total_vehicles": state[10],
                "avg_speed": state[11],
                "action": action,
                "reward": reward,
            }
        )

        # Keep memory manageable
        if len(self.traffic_flow_memory) > 1000:
            self.traffic_flow_memory = self.traffic_flow_memory[-500:]

    def _process_batch_update(self) -> None:
        """
        Process accumulated batch using GPU acceleration
        """
        if not self.batch_states:
            return

        try:
            # Batch processing disabled: GPU acceleration expects
            # numeric indices but we use string keys.
            raise Exception("Batch processing disabled - using individual updates")

            # GPU-accelerated Q-learning update
            self.q_table = self.gpu_accelerator.accelerated_qlearning_update(
                self.q_table,
                self.batch_states,
                self.batch_actions,
                self.batch_rewards,
                self.batch_next_states,
                self.learning_rate,
                self.discount_factor,
            )

            # Update learning statistics (learning_steps is inherited from parent)
            if hasattr(self, "learning_steps"):
                self.learning_steps += len(self.batch_states)

        except Exception:
            # Fall back to individual updates (silently)
            for i in range(len(self.batch_states)):
                self.update(
                    self.batch_states[i],
                    self.batch_actions[i],
                    self.batch_rewards[i],
                    self.batch_next_states[i],
                )

        # Clear batch
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_next_states = []

    def force_batch_update(self) -> None:
        """Force processing of current batch (useful at episode end)"""
        if self.batch_states:
            self._process_batch_update()

    def get_traffic_insights(self) -> Dict:
        """
        Analyze traffic patterns learned by the agent

        Returns:
            Dictionary with traffic insights
        """
        if not self.traffic_flow_memory:
            return {"message": "No traffic data available"}

        recent_data = self.traffic_flow_memory[-100:]

        avg_queues = np.mean([data["queues"] for data in recent_data], axis=0)

        avg_waits = np.mean([data["waiting_times"] for data in recent_data], axis=0)

        traffic_levels = [data["total_vehicles"] for data in recent_data]
        peak_traffic = max(traffic_levels) if traffic_levels else 0

        phase_stats = {}
        for phase, throughputs in self.phase_efficiency.items():
            if throughputs:
                phase_stats[f"phase_{phase}_avg_throughput"] = np.mean(throughputs)
                phase_stats[f"phase_{phase}_max_throughput"] = max(throughputs)

        insights = {
            "average_queue_lengths": {
                "north": float(avg_queues[0]),
                "south": float(avg_queues[1]),
                "east": float(avg_queues[2]),
                "west": float(avg_queues[3]),
            },
            "average_waiting_times": {
                "north": float(avg_waits[0]),
                "south": float(avg_waits[1]),
                "east": float(avg_waits[2]),
                "west": float(avg_waits[3]),
            },
            "peak_traffic_vehicles": peak_traffic,
            "phase_efficiency": phase_stats,
            "total_observations": len(self.traffic_flow_memory),
            "current_epsilon": self.epsilon,
            "action_usage_distribution": dict(self.action_usage_count),
        }

        return insights

    def recommend_action(self, state: np.ndarray) -> Tuple[int, str]:
        """
        Recommend action with explanation based on current traffic state

        Args:
            state: Current SUMO state

        Returns:
            Tuple of (recommended_action, explanation)
        """
        north_queue, south_queue, east_queue, west_queue = state[0:4]
        north_wait, south_wait, east_wait, west_wait = state[4:8]
        current_phase, phase_duration = state[8:10]

        ns_pressure = north_queue + south_queue + (north_wait + south_wait) / 10
        ew_pressure = east_queue + west_queue + (east_wait + west_wait) / 10

        west_problem = west_queue > 2 or west_wait > 30
        east_problem = east_queue > 2 or east_wait > 30

        q_values = self.get_all_q_values(state)
        best_action = int(np.argmax(q_values))

        if west_problem or east_problem:
            if current_phase in [0, 2] and (west_problem or east_problem):
                best_action = 2
            elif (
                current_phase in [4, 6]
                and phase_duration < 20
                and (west_problem or east_problem)
            ):
                best_action = 0

        if phase_duration < 15:
            best_action = 0

        explanations = {
            0: "Maintain current phase",
            1: "Switch to North-South green",
            2: "Switch to East-West green",
        }

        if best_action == 0:
            if west_problem or east_problem:
                reason = (
                    f"Extending current phase to resolve congestion "
                    f"(duration: {phase_duration:.1f}s)"
                )
            else:
                reason = (
                    f"Current phase is working well (duration: {phase_duration:.1f}s)"
                )
        elif best_action == 1:
            reason = (
                f"North-South has higher pressure "
                f"({ns_pressure:.1f} vs {ew_pressure:.1f})"
            )
        else:
            if west_problem:
                reason = (
                    f"WEST CONGESTION! W_queue={west_queue}, "
                    f"W_wait={west_wait:.1f}s"
                )
            elif east_problem:
                reason = (
                    f"EAST CONGESTION! E_queue={east_queue}, "
                    f"E_wait={east_wait:.1f}s"
                )
            else:
                reason = (
                    f"East-West has higher pressure "
                    f"({ew_pressure:.1f} vs {ns_pressure:.1f})"
                )

        if (
            max(north_queue, south_queue, east_queue, west_queue)
            > self.congestion_threshold
        ):
            reason += " - CONGESTION DETECTED!"

        explanation = f"{explanations[best_action]} - {reason}"

        return best_action, explanation

    def save_sumo_model(self, filepath: str) -> None:
        """
        Save model with SUMO-specific data

        Args:
            filepath: Path to save the model
        """
        try:
            self.save_model(filepath)
        except Exception as e:
            print(f"Error saving base model: {e}")

        sumo_data_path = filepath.replace(".json", "_sumo_data.json")

        try:
            sumo_specific_data = {
                "traffic_flow_memory": self.traffic_flow_memory[-100:],
                "phase_efficiency": {
                    str(k): v for k, v in self.phase_efficiency.items()
                },
                "action_usage_count": dict(self.action_usage_count),
                "traffic_insights": self.get_traffic_insights(),
            }

            with open(sumo_data_path, "w") as f:
                json.dump(
                    sumo_specific_data,
                    f,
                    indent=2,
                    default=self._json_serializer,
                )
            print(f"SUMO-specific data saved to {sumo_data_path}")
        except Exception as e:
            print(f"Error saving SUMO-specific data: {e}")
            try:
                minimal_data = {
                    "action_usage_count": dict(self.action_usage_count),
                    "epsilon": float(self.epsilon),
                }
                with open(sumo_data_path, "w") as f:
                    json.dump(
                        minimal_data,
                        f,
                        indent=2,
                        default=self._json_serializer,
                    )
                print(f"Minimal SUMO data saved to {sumo_data_path}")
            except Exception as e2:
                print(f"Could not save any SUMO data: {e2}")

    def load_sumo_model(self, filepath: str) -> None:
        """
        Load model with SUMO-specific data

        Args:
            filepath: Path to load the model from
        """
        self.load_model(filepath)

        sumo_data_path = filepath.replace(".json", "_sumo_data.json")

        if os.path.exists(sumo_data_path):
            with open(sumo_data_path, "r") as f:
                sumo_data = json.load(f)

            self.traffic_flow_memory = sumo_data.get("traffic_flow_memory", [])

            phase_eff = sumo_data.get("phase_efficiency", {})
            self.phase_efficiency = defaultdict(list)
            for k, v in phase_eff.items():
                self.phase_efficiency[int(k)] = v

            action_usage = sumo_data.get("action_usage_count", {})
            self.action_usage_count = defaultdict(int)
            for k, v in action_usage.items():
                self.action_usage_count[int(k)] = v

            print(f"SUMO-specific data loaded from {sumo_data_path}")
        else:
            print(f"No SUMO-specific data found at {sumo_data_path}")

    def print_sumo_stats(self) -> None:
        """Print SUMO-specific training statistics"""
        try:
            self.print_stats()  # Base stats
        except Exception as e:
            print(f"Error printing base stats: {e}")
            # Print basic info manually
            print("=== Q-Learning Agent Statistics ===")
            print(f"Total Steps: {getattr(self, 'total_steps', 0)}")
            print(f"Current Epsilon: {self.epsilon:.4f}")
            print(f"Q-table Size: {len(self.q_table)}")
            print("=" * 35)

        print("\n=== SUMO Traffic Insights ===")
        insights = self.get_traffic_insights()

        print("Average Queue Lengths:")
        for direction, length in insights["average_queue_lengths"].items():
            print(f"  {direction.capitalize()}: {length:.1f} vehicles")

        print("Average Waiting Times:")
        for direction, time in insights["average_waiting_times"].items():
            print(f"  {direction.capitalize()}: {time:.1f} seconds")

        print(f"Peak Traffic: {insights['peak_traffic_vehicles']} vehicles")

        # Action usage distribution analysis
        if insights["action_usage_distribution"]:
            print("Action Usage Distribution:")
            total_actions = sum(insights["action_usage_distribution"].values())
            for action, count in insights["action_usage_distribution"].items():
                percentage = (count / total_actions * 100) if total_actions > 0 else 0
                action_name = {
                    0: "Maintain",
                    1: "NS-Green",
                    2: "EW-Green",
                }.get(action, f"Action-{action}")
                print(f"  {action_name}: {count} times ({percentage:.1f}%)")

        if insights["phase_efficiency"]:
            print("Phase Efficiency:")
            for phase, throughput in insights["phase_efficiency"].items():
                print(f"  {phase}: {throughput:.1f} vehicles/step")

        # GPU acceleration stats
        gpu_info = self.gpu_accelerator.get_device_info()
        print("\nGPU Acceleration:")
        print(f"  Device: {gpu_info['device']}")
        print(f"  GPU Operations: {gpu_info['gpu_operations']}")
        print(f"  CPU Operations: {gpu_info['cpu_operations']}")

        print("=" * 30)

    def cleanup(self) -> None:
        """Cleanup resources including GPU"""
        self.force_batch_update()

        pass

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and other objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, "item"):  # Para escalares numpy
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
