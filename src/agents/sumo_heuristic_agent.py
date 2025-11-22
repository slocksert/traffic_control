import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TrafficState:
    north_queue: float
    south_queue: float
    east_queue: float
    west_queue: float
    north_wait: float
    south_wait: float
    east_wait: float
    west_wait: float
    current_phase: int
    phase_duration: float
    total_vehicles: float
    avg_speed: float


class SumoHeuristicAgent:
    def __init__(
        self,
        min_phase_duration: float = 5.0,
        max_phase_duration: float = 45.0,
        congestion_threshold: int = 4,
        wait_time_threshold: float = 15.0,
        switch_threshold: float = 1.2,
    ):
        """
        Heuristic agent for SUMO traffic control using intelligent rules

        Args:
            min_phase_duration: Minimum phase duration in seconds
            max_phase_duration: Maximum phase duration in seconds
            congestion_threshold: Queue length to consider congested
            wait_time_threshold: Waiting time to prioritize switching
            switch_threshold: Pressure ratio to trigger phase switch
        """
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.congestion_threshold = congestion_threshold
        self.wait_time_threshold = wait_time_threshold
        self.switch_threshold = switch_threshold

        self.decisions_made = 0
        self.phase_switches = 0
        self.emergency_switches = 0
        self.congestion_events = 0

        self.decision_history = []

    def get_action(self, state: np.ndarray) -> int:
        """
        Get action based on heuristic rules

        Args:
            state: SUMO state vector

        Returns:
            Action (0=maintain, 1=switch_to_ns, 2=switch_to_ew)
        """
        traffic_state = TrafficState(
            north_queue=state[0],
            south_queue=state[1],
            east_queue=state[2],
            west_queue=state[3],
            north_wait=state[4],
            south_wait=state[5],
            east_wait=state[6],
            west_wait=state[7],
            current_phase=int(state[8]),
            phase_duration=state[9],
            total_vehicles=state[10],
            avg_speed=state[11],
        )

        action, reason = self._decide_action(traffic_state)

        self.decisions_made += 1
        if action != 0:
            self.phase_switches += 1

        self.decision_history.append(
            {
                "state": traffic_state,
                "action": action,
                "reason": reason,
                "timestep": self.decisions_made,
            }
        )

        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

        return action

    def _decide_action(self, state: TrafficState) -> Tuple[int, str]:
        """
        Core decision logic with explanations

        Args:
            state: Parsed traffic state

        Returns:
            Tuple of (action, reason)
        """
        ns_pressure = self._calculate_pressure(
            state.north_queue,
            state.south_queue,
            state.north_wait,
            state.south_wait,
        )
        ew_pressure = self._calculate_pressure(
            state.east_queue,
            state.west_queue,
            state.east_wait,
            state.west_wait,
        )

        emergency_action = self._check_emergency_conditions(
            state, ns_pressure, ew_pressure
        )
        if emergency_action is not None:
            self.emergency_switches += 1
            return emergency_action

        if state.phase_duration < self.min_phase_duration:
            dur_msg = (
                f"Minimum phase duration not met "
                f"({state.phase_duration:.1f}s < "
                f"{self.min_phase_duration}s)"
            )
            return (0, dur_msg)

        if state.phase_duration > self.max_phase_duration:
            if state.current_phase in [0, 1]:
                msg = (
                    f"Maximum phase duration exceeded, "
                    f"switching to EW ({state.phase_duration:.1f}s)"
                )
                return (2, msg)
            else:
                msg = (
                    f"Maximum phase duration exceeded, "
                    f"switching to NS ({state.phase_duration:.1f}s)"
                )
                return (1, msg)

        current_direction = "NS" if state.current_phase in [0, 1] else "EW"

        if state.current_phase in [0, 1]:
            if ew_pressure > ns_pressure * self.switch_threshold:
                msg = (
                    f"EW pressure significantly higher "
                    f"({ew_pressure:.1f} vs {ns_pressure:.1f})"
                )
                return (2, msg)
        else:
            if ns_pressure > ew_pressure * self.switch_threshold:
                msg = (
                    f"NS pressure significantly higher "
                    f"({ns_pressure:.1f} vs {ew_pressure:.1f})"
                )
                return (1, msg)

        if state.avg_speed < 5.0 and state.total_vehicles > 20:
            self.congestion_events += 1

            if ns_pressure > ew_pressure and state.current_phase not in [0, 1]:
                msg = (
                    f"Congestion detected, clearing NS traffic "
                    f"(speed: {state.avg_speed:.1f})"
                )
                return (1, msg)
            elif ew_pressure > ns_pressure and state.current_phase not in [
                2,
                3,
            ]:
                msg = (
                    f"Congestion detected, clearing EW traffic "
                    f"(speed: {state.avg_speed:.1f})"
                )
                return (2, msg)

        msg = (
            f"Maintaining {current_direction} phase "
            f"(pressure: NS={ns_pressure:.1f}, EW={ew_pressure:.1f})"
        )
        return (0, msg)

    def _calculate_pressure(
        self, queue1: float, queue2: float, wait1: float, wait2: float
    ) -> float:
        """
        Calculate pressure for a direction considering both queues and waiting times

        Args:
            queue1, queue2: Queue lengths for the direction
            wait1, wait2: Waiting times for the direction

        Returns:
            Combined pressure value
        """
        queue_pressure = queue1 + queue2
        wait_pressure = (wait1 + wait2) / 10

        return queue_pressure + wait_pressure * 2

    def _check_emergency_conditions(
        self, state: TrafficState, ns_pressure: float, ew_pressure: float
    ) -> Tuple[int, str]:
        """
        Check for emergency conditions that require immediate action

        Args:
            state: Traffic state
            ns_pressure, ew_pressure: Calculated pressures

        Returns:
            Emergency action and reason, or None
        """
        max_queue = max(
            state.north_queue,
            state.south_queue,
            state.east_queue,
            state.west_queue,
        )
        if max_queue > self.congestion_threshold * 2:
            is_ns = state.north_queue == max_queue or state.south_queue == max_queue
            if is_ns:
                if state.current_phase not in [0, 1]:
                    msg = f"EMERGENCY: Extreme NS congestion " f"({max_queue} vehicles)"
                    return 1, msg
            else:
                if state.current_phase not in [2, 3]:
                    msg = f"EMERGENCY: Extreme EW congestion " f"({max_queue} vehicles)"
                    return 2, msg

        max_wait = max(
            state.north_wait,
            state.south_wait,
            state.east_wait,
            state.west_wait,
        )
        if max_wait > self.wait_time_threshold * 2:
            is_ns = state.north_wait == max_wait or state.south_wait == max_wait
            if is_ns:
                if state.current_phase not in [0, 1]:
                    msg = f"EMERGENCY: Excessive NS waiting ({max_wait:.1f}s)"
                    return 1, msg
            else:
                if state.current_phase not in [2, 3]:
                    msg = f"EMERGENCY: Excessive EW waiting ({max_wait:.1f}s)"
                    return 2, msg

        if ns_pressure > ew_pressure * 2.0 and state.current_phase not in [
            0,
            1,
        ]:
            msg = (
                f"EMERGENCY: Extreme NS pressure imbalance "
                f"({ns_pressure:.1f} vs {ew_pressure:.1f})"
            )
            return (1, msg)
        elif ew_pressure > ns_pressure * 2.0 and state.current_phase not in [
            2,
            3,
        ]:
            msg = (
                f"EMERGENCY: Extreme EW pressure imbalance "
                f"({ew_pressure:.1f} vs {ns_pressure:.1f})"
            )
            return (2, msg)

        return None

    def get_decision_explanation(self, state: np.ndarray) -> str:
        """
        Get detailed explanation for the recommended action

        Args:
            state: SUMO state vector

        Returns:
            Detailed explanation string
        """
        traffic_state = TrafficState(
            north_queue=state[0],
            south_queue=state[1],
            east_queue=state[2],
            west_queue=state[3],
            north_wait=state[4],
            south_wait=state[5],
            east_wait=state[6],
            west_wait=state[7],
            current_phase=int(state[8]),
            phase_duration=state[9],
            total_vehicles=state[10],
            avg_speed=state[11],
        )

        action, reason = self._decide_action(traffic_state)

        ns_pressure = self._calculate_pressure(
            traffic_state.north_queue,
            traffic_state.south_queue,
            traffic_state.north_wait,
            traffic_state.south_wait,
        )
        ew_pressure = self._calculate_pressure(
            traffic_state.east_queue,
            traffic_state.west_queue,
            traffic_state.east_wait,
            traffic_state.west_wait,
        )

        phase_str = "NS" if traffic_state.current_phase in [0, 1] else "EW"
        explanation = f"""
HEURISTIC AGENT DECISION:
Action: {['Maintain', 'Switch to NS', 'Switch to EW'][action]}
Reason: {reason}

Current Situation:
- Phase: {phase_str} (duration: {traffic_state.phase_duration:.1f}s)
- Total vehicles: {traffic_state.total_vehicles:.0f}
- Average speed: {traffic_state.avg_speed:.1f} m/s

Traffic Pressure:
- North-South: {ns_pressure:.1f} (queues: \
N={traffic_state.north_queue:.0f}, S={traffic_state.south_queue:.0f})
- East-West: {ew_pressure:.1f} (queues: \
E={traffic_state.east_queue:.0f}, W={traffic_state.west_queue:.0f})

Waiting Times:
- North: {traffic_state.north_wait:.1f}s, \
South: {traffic_state.south_wait:.1f}s
- East: {traffic_state.east_wait:.1f}s, \
West: {traffic_state.west_wait:.1f}s
        """.strip()

        return explanation

    def get_statistics(self) -> Dict:
        """
        Get agent performance statistics

        Returns:
            Dictionary with statistics
        """
        if self.decisions_made == 0:
            return {"message": "No decisions made yet"}

        switch_rate = self.phase_switches / self.decisions_made
        emergency_rate = self.emergency_switches / self.decisions_made
        congestion_rate = self.congestion_events / self.decisions_made

        stats = {
            "total_decisions": self.decisions_made,
            "phase_switches": self.phase_switches,
            "emergency_switches": self.emergency_switches,
            "congestion_events": self.congestion_events,
            "switch_rate": switch_rate,
            "emergency_rate": emergency_rate,
            "congestion_rate": congestion_rate,
        }

        if self.decision_history:
            recent_decisions = self.decision_history[-100:]
            action_counts = {0: 0, 1: 0, 2: 0}
            for decision in recent_decisions:
                action_counts[decision["action"]] += 1

            stats["recent_action_distribution"] = {
                "maintain": action_counts[0],
                "switch_to_ns": action_counts[1],
                "switch_to_ew": action_counts[2],
            }

        return stats

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("=== HEURISTIC AGENT STATISTICS ===")
        print(f"Total Decisions: {stats['total_decisions']}")
        sw_rate = stats["switch_rate"]
        print(f"Phase Switches: {stats['phase_switches']} ({sw_rate:.1%})")
        em_rate = stats["emergency_rate"]
        print(f"Emergency Switches: {stats['emergency_switches']} " f"({em_rate:.1%})")
        cong_rate = stats["congestion_rate"]
        print(f"Congestion Events: {stats['congestion_events']} " f"({cong_rate:.1%})")

        if "recent_action_distribution" in stats:
            print("\nRecent Action Distribution:")
            dist = stats["recent_action_distribution"]
            total = sum(dist.values())
            for action, count in dist.items():
                percentage = count / total * 100 if total > 0 else 0
                print(f"  {action}: {count} ({percentage:.1f}%)")

        print("=" * 35)

    def reset_statistics(self):
        """Reset all statistics"""
        self.decisions_made = 0
        self.phase_switches = 0
        self.emergency_switches = 0
        self.congestion_events = 0
        self.decision_history = []
