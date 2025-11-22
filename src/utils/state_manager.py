import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import deque


@dataclass
class TrafficMetrics:
    """Container for traffic performance metrics"""

    average_wait_time: float = 0.0
    throughput: float = 0.0  # vehicles per minute
    pedestrian_wait_time: float = 0.0
    total_vehicles_passed: int = 0
    total_pedestrians_crossed: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0


class StateManager:
    """
    Manages state representation and normalization for traffic control environment
    """

    def __init__(self, max_queue_size: int = 10, max_wait_time: int = 100):
        """
        Initialize state manager

        Args:
            max_queue_size: Maximum queue size for discretization
            max_wait_time: Maximum wait time for discretization
        """
        self.max_queue_size = max_queue_size
        self.max_wait_time = max_wait_time

        # State space dimensions
        self.state_dims = {
            "ns_count": max_queue_size + 1,  # 0 to max_queue_size
            "ew_count": max_queue_size + 1,  # 0 to max_queue_size
            "ns_wait": max_wait_time + 1,  # 0 to max_wait_time
            "ew_wait": max_wait_time + 1,  # 0 to max_wait_time
            "ped_waiting": 2,  # 0 or 1
            "light_state": 4,  # 0, 1, 2, 3
            "time_since_change": max_wait_time + 1,  # 0 to max_wait_time
        }

        # Metrics tracking
        self.episode_metrics_history: List[TrafficMetrics] = []
        self.current_metrics = TrafficMetrics()
        # Rolling window for statistics
        self.metrics_window = deque(maxlen=100)

    def normalize_state(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Normalize and discretize raw state values

        Args:
            raw_state: Raw state array
                [ns_count, ew_count, ns_wait, ew_wait,
                 ped_waiting, light_state, time_since_change]

        Returns:
            Normalized and discretized state
        """
        (
            ns_count,
            ew_count,
            ns_wait,
            ew_wait,
            ped_waiting,
            light_state,
            time_since_change,
        ) = raw_state

        # Discretize continuous values
        ns_count = min(int(ns_count), self.max_queue_size)
        ew_count = min(int(ew_count), self.max_queue_size)
        ns_wait = min(int(ns_wait), self.max_wait_time)
        ew_wait = min(int(ew_wait), self.max_wait_time)
        ped_waiting = int(ped_waiting > 0)  # Binary
        light_state = int(light_state)
        time_since_change = min(int(time_since_change), self.max_wait_time)

        return np.array(
            [
                ns_count,
                ew_count,
                ns_wait,
                ew_wait,
                ped_waiting,
                light_state,
                time_since_change,
            ]
        )

    def get_state_space_size(self) -> int:
        """
        Calculate total state space size

        Returns:
            Total number of possible states
        """
        size = 1
        for dim in self.state_dims.values():
            size *= dim
        return size

    def state_to_index(self, state: np.ndarray) -> int:
        """
        Convert state array to unique index

        Args:
            state: Normalized state array

        Returns:
            Unique state index
        """
        (
            ns_count,
            ew_count,
            ns_wait,
            ew_wait,
            ped_waiting,
            light_state,
            time_since_change,
        ) = state

        index = 0
        multiplier = 1

        # Convert to index using base conversion
        values = [
            time_since_change,
            light_state,
            ped_waiting,
            ew_wait,
            ns_wait,
            ew_count,
            ns_count,
        ]
        dims = [
            self.state_dims["time_since_change"],
            self.state_dims["light_state"],
            self.state_dims["ped_waiting"],
            self.state_dims["ew_wait"],
            self.state_dims["ns_wait"],
            self.state_dims["ew_count"],
            self.state_dims["ns_count"],
        ]

        for value, dim in zip(values, dims):
            index += value * multiplier
            multiplier *= dim

        return index

    def index_to_state(self, index: int) -> np.ndarray:
        """
        Convert state index back to state array

        Args:
            index: State index

        Returns:
            State array
        """
        state = np.zeros(7, dtype=int)

        dims = [
            self.state_dims["time_since_change"],
            self.state_dims["light_state"],
            self.state_dims["ped_waiting"],
            self.state_dims["ew_wait"],
            self.state_dims["ns_wait"],
            self.state_dims["ew_count"],
            self.state_dims["ns_count"],
        ]

        temp_index = index
        for i, dim in enumerate(dims):
            state[6 - i] = temp_index % dim
            temp_index //= dim

        return state

    def update_metrics(
        self, env_info: Dict[str, Any], reward: float, episode_length: int
    ) -> None:
        """
        Update current episode metrics

        Args:
            env_info: Environment information dictionary
            reward: Episode reward
            episode_length: Episode length
        """
        self.current_metrics.total_vehicles_passed = env_info.get("cars_passed", 0)
        self.current_metrics.total_pedestrians_crossed = env_info.get(
            "pedestrians_crossed", 0
        )
        self.current_metrics.episode_reward = reward
        self.current_metrics.episode_length = episode_length

        # Calculate derived metrics
        if episode_length > 0:
            self.current_metrics.throughput = (
                self.current_metrics.total_vehicles_passed / episode_length
            ) * 60  # per minute

        if (
            env_info.get("total_wait_time", 0) > 0
            and self.current_metrics.total_vehicles_passed > 0
        ):
            self.current_metrics.average_wait_time = (
                env_info["total_wait_time"] / self.current_metrics.total_vehicles_passed
            )

        # Add to metrics window for rolling statistics
        self.metrics_window.append(self.current_metrics)

    def finalize_episode(self) -> TrafficMetrics:
        """
        Finalize current episode metrics and add to history

        Returns:
            Final episode metrics
        """
        final_metrics = TrafficMetrics(
            average_wait_time=self.current_metrics.average_wait_time,
            throughput=self.current_metrics.throughput,
            pedestrian_wait_time=self.current_metrics.pedestrian_wait_time,
            total_vehicles_passed=self.current_metrics.total_vehicles_passed,
            total_pedestrians_crossed=self.current_metrics.total_pedestrians_crossed,
            episode_reward=self.current_metrics.episode_reward,
            episode_length=self.current_metrics.episode_length,
        )

        self.episode_metrics_history.append(final_metrics)
        self.current_metrics = TrafficMetrics()  # Reset for next episode

        return final_metrics

    def get_average_metrics(self, last_n_episodes: int = 100) -> Dict[str, float]:
        """
        Get average metrics over last N episodes

        Args:
            last_n_episodes: Number of recent episodes to average

        Returns:
            Dictionary of average metrics
        """
        if not self.episode_metrics_history:
            return {
                "avg_wait_time": 0.0,
                "avg_throughput": 0.0,
                "avg_pedestrian_wait": 0.0,
                "avg_vehicles_passed": 0.0,
                "avg_pedestrians_crossed": 0.0,
                "avg_reward": 0.0,
                "avg_episode_length": 0.0,
            }

        recent_episodes = self.episode_metrics_history[-last_n_episodes:]

        return {
            "avg_wait_time": np.mean([m.average_wait_time for m in recent_episodes]),
            "avg_throughput": np.mean([m.throughput for m in recent_episodes]),
            "avg_pedestrian_wait": np.mean(
                [m.pedestrian_wait_time for m in recent_episodes]
            ),
            "avg_vehicles_passed": np.mean(
                [m.total_vehicles_passed for m in recent_episodes]
            ),
            "avg_pedestrians_crossed": np.mean(
                [m.total_pedestrians_crossed for m in recent_episodes]
            ),
            "avg_reward": np.mean([m.episode_reward for m in recent_episodes]),
            "avg_episode_length": np.mean([m.episode_length for m in recent_episodes]),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary

        Returns:
            Dictionary with performance statistics
        """
        if not self.episode_metrics_history:
            return {"total_episodes": 0}

        all_rewards = [m.episode_reward for m in self.episode_metrics_history]
        all_wait_times = [m.average_wait_time for m in self.episode_metrics_history]
        all_throughputs = [m.throughput for m in self.episode_metrics_history]

        # Recent performance (last 100 episodes)
        recent_metrics = self.get_average_metrics(100)

        # Overall statistics
        summary = {
            "total_episodes": len(self.episode_metrics_history),
            "overall_stats": {
                "mean_reward": np.mean(all_rewards),
                "std_reward": np.std(all_rewards),
                "max_reward": np.max(all_rewards),
                "min_reward": np.min(all_rewards),
                "mean_wait_time": np.mean(all_wait_times),
                "mean_throughput": np.mean(all_throughputs),
            },
            "recent_stats": recent_metrics,
            "improvement": {
                "reward_trend": self._calculate_trend(
                    [m.episode_reward for m in self.episode_metrics_history[-50:]]
                ),
                "wait_time_trend": self._calculate_trend(
                    [m.average_wait_time for m in self.episode_metrics_history[-50:]]
                ),
                "throughput_trend": self._calculate_trend(
                    [m.throughput for m in self.episode_metrics_history[-50:]]
                ),
            },
        }

        return summary

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction for a list of values

        Args:
            values: List of values

        Returns:
            Trend direction ('improving', 'declining', 'stable')
        """
        if len(values) < 10:
            return "insufficient_data"

        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if abs(slope) < 0.01:  # Threshold for stability
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "declining"

    def reset_metrics(self) -> None:
        """Reset all metrics and history"""
        self.episode_metrics_history = []
        self.current_metrics = TrafficMetrics()
        self.metrics_window.clear()

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics for external analysis

        Returns:
            Dictionary with all metrics data
        """
        return {
            "episode_history": [
                {
                    "average_wait_time": m.average_wait_time,
                    "throughput": m.throughput,
                    "pedestrian_wait_time": m.pedestrian_wait_time,
                    "total_vehicles_passed": m.total_vehicles_passed,
                    "total_pedestrians_crossed": m.total_pedestrians_crossed,
                    "episode_reward": m.episode_reward,
                    "episode_length": m.episode_length,
                }
                for m in self.episode_metrics_history
            ],
            "summary": self.get_performance_summary(),
            "state_space_info": {
                "dimensions": self.state_dims,
                "total_states": self.get_state_space_size(),
            },
        }

    def compare_agents(
        self, other_manager: "StateManager", last_n_episodes: int = 100
    ) -> Dict[str, Any]:
        """
        Compare performance with another state manager

        Args:
            other_manager: Another StateManager instance
            last_n_episodes: Number of recent episodes to compare

        Returns:
            Comparison results
        """
        self_metrics = self.get_average_metrics(last_n_episodes)
        other_metrics = other_manager.get_average_metrics(last_n_episodes)

        comparison = {}
        for key in self_metrics:
            if key in other_metrics:
                self_val = self_metrics[key]
                other_val = other_metrics[key]

                if other_val != 0:
                    improvement = ((self_val - other_val) / other_val) * 100
                else:
                    improvement = 0

                comparison[key] = {
                    "self": self_val,
                    "other": other_val,
                    "improvement_percent": improvement,
                    "better": (
                        improvement > 0 if "wait" not in key else improvement < 0
                    ),
                }

        return comparison

    def get_state_representation_info(self) -> str:
        """
        Get human-readable information about state representation

        Returns:
            String description of state representation
        """
        info = "State Representation:\n"
        info += f"- NS Vehicle Count: 0-{self.max_queue_size}\n"
        info += f"- EW Vehicle Count: 0-{self.max_queue_size}\n"
        info += f"- NS Wait Time: 0-{self.max_wait_time}\n"
        info += f"- EW Wait Time: 0-{self.max_wait_time}\n"
        info += "- Pedestrian Waiting: 0 (no) or 1 (yes)\n"
        info += "- Light State: 0 (NS), 1 (EW), 2 (Ped), 3 (All Red)\n"
        info += f"- Time Since Change: 0-{self.max_wait_time}\n"
        info += f"Total State Space Size: {self.get_state_space_size():,} states"

        return info
