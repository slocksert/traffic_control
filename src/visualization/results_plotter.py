import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import json

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class ResultsPlotter:
    """
    Comprehensive visualization tool for traffic control simulation results
    """

    def __init__(self, output_dir: str = "plots", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize results plotter

        Args:
            output_dir: Directory to save plots
            figsize: Default figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)

    def plot_training_progress(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        window_size: int = 100,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training progress with rewards and episode lengths

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            window_size: Window size for rolling average
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        episodes = range(1, len(episode_rewards) + 1)

        ax1.plot(
            episodes,
            episode_rewards,
            alpha=0.3,
            color=self.colors[0],
            label="Episode Reward",
        )

        if len(episode_rewards) >= window_size:
            rolling_rewards = (
                pd.Series(episode_rewards).rolling(window=window_size).mean()
            )
            ax1.plot(
                episodes,
                rolling_rewards,
                color=self.colors[1],
                linewidth=2,
                label=f"Rolling Average ({window_size} episodes)",
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Training Progress - Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            episodes,
            episode_lengths,
            alpha=0.3,
            color=self.colors[2],
            label="Episode Length",
        )

        if len(episode_lengths) >= window_size:
            rolling_lengths = (
                pd.Series(episode_lengths).rolling(window=window_size).mean()
            )
            ax2.plot(
                episodes,
                rolling_lengths,
                color=self.colors[3],
                linewidth=2,
                label=f"Rolling Average ({window_size} episodes)",
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Training Progress - Episode Lengths")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training progress plot saved to {save_path}")
        else:
            plt.show()

    def plot_agent_comparison(
        self,
        agents_data: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = ["rewards", "wait_times", "throughput"],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare multiple agents across different metrics

        Args:
            agents_data: Dictionary with agent names as keys and metrics as values
            metrics: List of metrics to compare
            save_path: Optional path to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        agent_names = list(agents_data.keys())

        for i, metric in enumerate(metrics):
            ax = axes[i]

            data_for_plot = []
            labels = []

            for agent_name in agent_names:
                if metric in agents_data[agent_name]:
                    data_for_plot.append(agents_data[agent_name][metric])
                    labels.append(agent_name)

            if data_for_plot:
                box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)

                for patch, color in zip(box_plot["boxes"], self.colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Agent comparison plot saved to {save_path}")
        else:
            plt.show()

    def plot_convergence_analysis(
        self,
        q_values_history: List[Dict[str, float]],
        epsilon_history: List[float],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot Q-learning convergence analysis

        Args:
            q_values_history: History of Q-values for key states
            epsilon_history: History of epsilon values
            save_path: Optional path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        if q_values_history:
            episodes = range(len(q_values_history))

            all_keys = set()
            for q_dict in q_values_history:
                all_keys.update(q_dict.keys())

            for i, key in enumerate(list(all_keys)[:5]):
                values = [q_dict.get(key, 0) for q_dict in q_values_history]
                ax1.plot(
                    episodes,
                    values,
                    label=f"State-Action: {key}",
                    color=self.colors[i % len(self.colors)],
                )

            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Q-Value")
            ax1.set_title("Q-Values Convergence")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if epsilon_history:
            episodes = range(len(epsilon_history))
            ax2.plot(episodes, epsilon_history, color=self.colors[0], linewidth=2)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_title("Exploration Rate (Epsilon) Decay")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Convergence analysis plot saved to {save_path}")
        else:
            plt.show()

    def plot_traffic_metrics(
        self,
        metrics_data: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot comprehensive traffic metrics

        Args:
            metrics_data: Dictionary with metric names and values
            save_path: Optional path to save plot
        """
        n_metrics = len(metrics_data)
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i < len(axes):
                ax = axes[i]
                episodes = range(1, len(values) + 1)

                ax.plot(
                    episodes,
                    values,
                    alpha=0.5,
                    color=self.colors[i % len(self.colors)],
                )

                if len(values) >= 50:
                    rolling_avg = pd.Series(values).rolling(window=50).mean()
                    ax.plot(
                        episodes,
                        rolling_avg,
                        color=self.colors[i % len(self.colors)],
                        linewidth=2,
                        label="50-episode average",
                    )
                    ax.legend()

                ax.set_xlabel("Episode")
                ax.set_ylabel(metric_name.replace("_", " ").title())
                ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
                ax.grid(True, alpha=0.3)

        for i in range(len(metrics_data), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Traffic metrics plot saved to {save_path}")
        else:
            plt.show()

    def plot_state_space_exploration(
        self,
        state_visit_counts: Dict[str, int],
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot state space exploration statistics

        Args:
            state_visit_counts: Dictionary mapping state strings to visit counts
            top_n: Number of top states to show
            save_path: Optional path to save plot
        """
        if not state_visit_counts:
            print("No state visit data available")
            return

        sorted_states = sorted(
            state_visit_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_states = sorted_states[:top_n]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        states, counts = zip(*top_states)
        state_labels = [f"State {i+1}" for i in range(len(states))]

        bars = ax1.bar(state_labels, counts, color=self.colors[0], alpha=0.7)
        ax1.set_xlabel("State")
        ax1.set_ylabel("Visit Count")
        ax1.set_title(f"Top {top_n} Most Visited States")
        ax1.tick_params(axis="x", rotation=45)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count}",
                ha="center",
                va="bottom",
            )

        all_counts = list(state_visit_counts.values())
        ax2.hist(
            all_counts,
            bins=30,
            color=self.colors[1],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Visit Count")
        ax2.set_ylabel("Number of States")
        ax2.set_title("Distribution of State Visit Counts")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"State space exploration plot saved to {save_path}")
        else:
            plt.show()

    def plot_action_distribution(
        self,
        action_history: List[int],
        action_names: List[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot distribution of actions taken

        Args:
            action_history: List of actions taken
            action_names: Optional names for actions
            save_path: Optional path to save plot
        """
        if not action_history:
            print("No action history available")
            return

        if action_names is None:
            action_names = [
                "Maintain",
                "NS Green",
                "EW Green",
                "Pedestrian Green",
            ]

        unique_actions, counts = np.unique(action_history, return_counts=True)
        action_labels = [
            (action_names[action] if action < len(action_names) else f"Action {action}")
            for action in unique_actions
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        bars = ax1.bar(
            action_labels,
            counts,
            color=self.colors[: len(unique_actions)],
            alpha=0.7,
        )
        ax1.set_xlabel("Action")
        ax1.set_ylabel("Count")
        ax1.set_title("Action Distribution")
        ax1.tick_params(axis="x", rotation=45)

        total_actions = len(action_history)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_actions) * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
            )

        ax2.pie(
            counts,
            labels=action_labels,
            autopct="%1.1f%%",
            colors=self.colors[: len(unique_actions)],
        )
        ax2.set_title("Action Distribution (Percentage)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Action distribution plot saved to {save_path}")
        else:
            plt.show()

    def create_comprehensive_report(
        self,
        experiment_data: Dict[str, Any],
        report_name: str = "traffic_control_report",
    ) -> str:
        """
        Create a comprehensive visual report

        Args:
            experiment_data: Complete experiment data
            report_name: Name for the report

        Returns:
            Path to report directory
        """
        report_dir = (
            self.output_dir
            / f"{report_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        report_dir.mkdir(exist_ok=True)

        print(f"Creating comprehensive report in {report_dir}")

        for agent_name, agent_data in experiment_data.get("agents", {}).items():
            if "episode_rewards" in agent_data and "episode_lengths" in agent_data:
                self.plot_training_progress(
                    agent_data["episode_rewards"],
                    agent_data["episode_lengths"],
                    save_path=report_dir / f"{agent_name}_training_progress.png",
                )

        if len(experiment_data.get("agents", {})) > 1:
            agents_comparison_data = {}
            for agent_name, agent_data in experiment_data["agents"].items():
                agents_comparison_data[agent_name] = {
                    "rewards": agent_data.get("episode_rewards", []),
                    "wait_times": agent_data.get("wait_times", []),
                    "throughput": agent_data.get("throughput", []),
                }

            self.plot_agent_comparison(
                agents_comparison_data,
                save_path=report_dir / "agent_comparison.png",
            )

        for agent_name, agent_data in experiment_data.get("agents", {}).items():
            metrics_data = {
                "average_wait_time": agent_data.get("wait_times", []),
                "throughput": agent_data.get("throughput", []),
                "vehicles_passed": agent_data.get("vehicles_passed", []),
                "pedestrians_crossed": agent_data.get("pedestrians_crossed", []),
            }

            metrics_data = {k: v for k, v in metrics_data.items() if v}

            if metrics_data:
                self.plot_traffic_metrics(
                    metrics_data,
                    save_path=report_dir / f"{agent_name}_traffic_metrics.png",
                )

        for agent_name, agent_data in experiment_data.get("agents", {}).items():
            if "q_learning" in agent_name.lower():
                if "state_visits" in agent_data:
                    self.plot_state_space_exploration(
                        agent_data["state_visits"],
                        save_path=report_dir / f"{agent_name}_state_exploration.png",
                    )

                if "action_history" in agent_data:
                    self.plot_action_distribution(
                        agent_data["action_history"],
                        save_path=report_dir / f"{agent_name}_action_distribution.png",
                    )

        summary_path = report_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("Traffic Control Simulation Report\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n")
            f.write(f"Report Directory: {report_dir}\n\n")

            f.write("Experiment Configuration:\n")
            config = experiment_data.get("config", {})
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\nAgents Tested:\n")
            for agent_name in experiment_data.get("agents", {}):
                f.write(f"  - {agent_name}\n")

            f.write("\nGenerated Plots:\n")
            for plot_file in report_dir.glob("*.png"):
                f.write(f"  - {plot_file.name}\n")

        print(f"Comprehensive report created in {report_dir}")
        return str(report_dir)

    def save_plot_data(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save plot data for later use

        Args:
            data: Data to save
            filename: Filename for data
        """
        filepath = self.output_dir / f"{filename}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Plot data saved to {filepath}")

    def set_style(self, style: str = "seaborn-v0_8") -> None:
        """
        Set matplotlib style

        Args:
            style: Style name
        """
        plt.style.use(style)
        print(f"Plot style set to: {style}")
