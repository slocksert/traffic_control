"""
SUMO Traffic Control Main Application

This application provides a complete traffic control simulation using SUMO
with AI agents for intelligent traffic light management.

Usage:
    python main_sumo.py --mode [train|test|demo] --agent [qlearning|heuristic]
"""

from src.environments.sumo_traffic_env import SumoTrafficEnv
from src.agents.sumo_q_learning_agent import SumoQLearningAgent
from src.agents.sumo_heuristic_agent import SumoHeuristicAgent
from src.utils.data_persistence import TestResultsDatabase
from src.analysis.generate_test_report import generate_test_report
from src.analysis.generate_training_report import generate_training_report

import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


class SumoTrafficController:
    def __init__(self, config: Dict):
        """
        Initialize the SUMO Traffic Controller

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.env: Optional[SumoTrafficEnv] = None
        self.agent: Optional[Union[SumoQLearningAgent, SumoHeuristicAgent]] = None
        self.results = {
            "episodes": [],
            "rewards": [],
            "metrics": [],
            "avg_waiting_time": [],
            "throughput": [],
            "avg_speed": [],
        }

    def setup_environment(self):
        """Setup SUMO environment"""
        print("Setting up SUMO environment...")

        self.env = SumoTrafficEnv(
            sumo_config_file=self.config["sumo_config"],
            use_gui=self.config["use_gui"],
            step_length=self.config["step_length"],
            seed=self.config["seed"],
        )

        print(f"Environment created with config: {self.config['sumo_config']}")
        print(f"State space size: {self.env.get_state_space_size()}")
        print(f"Action space size: {self.env.get_action_space_size()}")

    def setup_agent(self):
        """Setup AI agent"""
        print(f"Setting up {self.config['agent_type']} agent...")

        if self.env is None:
            raise RuntimeError("Environment must be setup before agent")

        if self.config["agent_type"] == "qlearning":
            self.agent = SumoQLearningAgent(
                state_space_size=self.env.get_state_space_size(),
                action_space_size=self.env.get_action_space_size(),
                learning_rate=self.config["learning_rate"],
                discount_factor=self.config["discount_factor"],
                epsilon=self.config["epsilon"],
                epsilon_decay=self.config["epsilon_decay"],
            )

            if self.config.get("load_model"):
                try:
                    self.agent.load_sumo_model(self.config["load_model"])
                    print(f"Loaded model from {self.config['load_model']}")
                except Exception as e:
                    print(f"Could not load model: {e}")

        elif self.config["agent_type"] == "heuristic":
            self.agent = SumoHeuristicAgent(
                min_phase_duration=self.config.get("min_phase_duration", 10),
                max_phase_duration=self.config.get("max_phase_duration", 60),
                congestion_threshold=self.config.get("congestion_threshold", 8),
            )

        else:
            raise ValueError(f"Unknown type: {self.config['agent_type']}")

        print(f"Agent setup complete: {type(self.agent).__name__}")

    def run_episode(self, episode_num: int, training: bool = True) -> Dict:
        """
        Run a single episode

        Args:
            episode_num: Episode number
            training: Whether this is a training episode

        Returns:
            Episode statistics
        """
        if self.env is None or self.agent is None:
            raise RuntimeError(
                "Environment and agent must be setup before " "running episodes"
            )

        state = self.env.reset()
        total_reward = 0
        step_count = 0
        episode_metrics = []

        print(f"\n{'='*50}")
        mode_str = "(Training)" if training else "(Testing)"
        print(f"Episode {episode_num + 1} {mode_str}")
        print(f"{'='*50}")

        try:
            while True:
                if isinstance(self.agent, SumoQLearningAgent):
                    action = self.agent.choose_action(state, environment=self.env)
                else:
                    action = self.agent.get_action(state)

                next_state, reward, done, info = self.env.step(action)

                if training and isinstance(self.agent, SumoQLearningAgent):
                    self.agent.update_with_sumo_metrics(
                        state, action, reward, next_state, info
                    )

                total_reward += reward
                step_count += 1
                episode_metrics.append(info["metrics"])

                if step_count % 100 == 0:
                    # Get current vehicle count from state (index 10 is total_vehicles)
                    active_vehicles = int(state[10])
                    print(
                        f"Step {step_count}: Reward={reward:.2f}, "
                        f"Active={active_vehicles}, "
                        f"Completed={info['vehicles_completed']}, "
                        f"Phase={info['phase']}"
                    )

                state = next_state

                if done:
                    break

                if not training and step_count > 1000:
                    break

        except KeyboardInterrupt:
            print("\nEpisode interrupted by user")
        except Exception as e:
            print(f"Error during episode: {e}")

        if episode_metrics:
            avg_waiting_time = np.mean([m.waiting_time for m in episode_metrics])
            total_throughput = sum([m.throughput for m in episode_metrics])
            avg_speed = np.mean([m.average_speed for m in episode_metrics])
        else:
            avg_waiting_time = 0
            total_throughput = 0
            avg_speed = 0

        episode_stats = {
            "episode": episode_num,
            "total_reward": total_reward,
            "steps": step_count,
            "avg_waiting_time": avg_waiting_time,
            "total_throughput": total_throughput,
            "avg_speed": avg_speed,
            "vehicles_completed": info.get("vehicles_completed", 0),
        }

        print(f"\nEpisode {episode_num + 1} Complete:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Avg Waiting Time: {avg_waiting_time:.2f}s")
        print(f"  Total Throughput: {total_throughput}")
        print(f"  Avg Speed: {avg_speed:.2f} m/s")
        print(f"  Vehicles Completed: {episode_stats['vehicles_completed']}")

        return episode_stats

    def train(self):
        """Train the agent and generate comprehensive learning analysis"""
        if not isinstance(self.agent, SumoQLearningAgent):
            print("Training is only available for Q-Learning agent")
            return

        num_eps = self.config["num_episodes"]
        print(f"\nStarting training for {num_eps} episodes...")

        # Initialize database
        db = TestResultsDatabase()

        # Create training run entry
        train_id = db.create_training_run(
            agent_type=self.config["agent_type"], config=self.config
        )

        print(f"Training run ID: {train_id}")

        for episode in range(self.config["num_episodes"]):
            episode_stats = self.run_episode(episode, training=True)

            self.results["episodes"].append(episode_stats["episode"])
            self.results["rewards"].append(episode_stats["total_reward"])
            self.results["avg_waiting_time"].append(episode_stats["avg_waiting_time"])
            self.results["throughput"].append(episode_stats["total_throughput"])
            self.results["avg_speed"].append(episode_stats["avg_speed"])

            # Save to database (batch mode - no auto commit)
            db.add_training_episode(
                train_id, episode_stats, epsilon=self.agent.epsilon, auto_commit=False
            )

            # Commit batch every 10 episodes for safety
            if (episode + 1) % 10 == 0:
                db.commit_batch(train_id)

            if (episode + 1) % self.config.get("save_interval", 50) == 0:
                ep_num = episode + 1
                model_path = f"models/sumo_qlearning_episode_{ep_num}.json"
                os.makedirs("models", exist_ok=True)
                self.agent.save_sumo_model(model_path)
                print(f"Model saved to {model_path}")

            if (episode + 1) % 10 == 0:
                recent_rewards = self.results["rewards"][-10:]
                avg_reward = np.mean(recent_rewards)
                ep_range = f"{episode-8}-{episode+1}"
                print(f"\nTraining Progress (Episodes {ep_range}):")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Current Epsilon: {self.agent.epsilon:.4f}")
                self.agent.print_sumo_stats()

        final_model_path = "models/sumo_qlearning_final.json"
        os.makedirs("models", exist_ok=True)
        self.agent.save_sumo_model(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")

        # Final batch commit and update epsilon
        db.commit_batch(train_id)
        db.update_final_epsilon(train_id, self.agent.epsilon)

        self.save_training_history()
        self.plot_training_results()

        # Generate comprehensive training report
        print(f"\n{'='*50}")
        print("GENERATING COMPREHENSIVE TRAINING REPORT")
        print(f"{'='*50}")

        try:
            # Get data from database
            df = db.get_training_results(train_id)
            train_info = db.get_training_info(train_id)

            # Generate complete report with learning curves and analysis
            generate_training_report(
                df, train_info, output_dir="training_results"
            )

            print("\n✓ Relatório de treinamento gerado com sucesso!")
            print("✓ Localização: {report_dir}")
            print("\nArquivos principais:")
            print(
                "  • relatorio_treinamento.txt - Relatório com análise de aprendizado"
            )
            print("  • curvas_aprendizado.png - Curvas de aprendizado (6 gráficos)")
            print("  • progresso_aprendizado.csv - Tabela de progresso")
            print("  • dados_treinamento.json - Dados completos em JSON")

        except Exception as e:
            print(f"\n⚠ Erro ao gerar relatório de treinamento: {e}")
            import traceback

            traceback.print_exc()

        finally:
            db.close()

    def test(self):
        """Test the agent and generate comprehensive analysis report"""
        test_eps = self.config["test_episodes"]
        print(f"\nStarting testing for {test_eps} episodes...")

        # Initialize database
        db = TestResultsDatabase()

        # Create test run entry
        run_id = db.create_test_run(
            agent_type=self.config["agent_type"],
            model_path=self.config.get("load_model"),
            config=self.config,
        )

        print(f"Test run ID: {run_id}")

        test_results = []

        for episode in range(self.config["test_episodes"]):
            episode_stats = self.run_episode(episode, training=False)
            test_results.append(episode_stats)

            # Save to database (batch mode - no auto commit)
            db.add_episode_result(run_id, episode_stats, auto_commit=False)

        # Final commit for all test episodes
        db.commit_test_batch(run_id)

        # Calculate summary statistics
        avg_reward = np.mean([r["total_reward"] for r in test_results])
        avg_waiting_time = np.mean([r["avg_waiting_time"] for r in test_results])
        avg_throughput = np.mean([r["total_throughput"] for r in test_results])
        avg_speed = np.mean([r["avg_speed"] for r in test_results])

        print(f"\n{'='*50}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Episodes: {len(test_results)}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Waiting Time: {avg_waiting_time:.2f}s")
        print(f"Average Throughput: {avg_throughput:.1f}")
        print(f"Average Speed: {avg_speed:.2f} m/s")

        if isinstance(self.agent, SumoQLearningAgent):
            self.agent.print_sumo_stats()
        elif hasattr(self.agent, "print_statistics"):
            self.agent.print_statistics()

        # Generate comprehensive test report
        print(f"\n{'='*50}")
        print("GENERATING COMPREHENSIVE TEST REPORT")
        print(f"{'='*50}")

        try:
            # Get data from database
            df = db.get_run_results(run_id)
            run_info = db.get_run_info(run_id)

            # Generate complete report with graphs, tables, and analysis
            generate_test_report(df, run_info, output_dir="test_results")

            print("\n✓ Relatório completo gerado com sucesso!")
            print("✓ Localização: {report_dir}")
            print("\nArquivos principais:")
            print(
                "  • relatorio_completo.txt - Relatório em texto com análise crítica"
            )
            print("  • grafico_desempenho_geral.png - Gráficos de desempenho")
            print("  • grafico_metricas_erro.png - Análise de erros")
            print("  • tabela_metricas.csv - Tabela de métricas")
            print("  • dados_completos.json - Dados completos em JSON")

        except Exception as e:
            print(f"\n⚠ Erro ao gerar relatório: {e}")
            import traceback

            traceback.print_exc()

        finally:
            db.close()

    def demo(self):
        """Run interactive demo"""
        print("\n" + "=" * 50)
        print("INTERACTIVE DEMO MODE")
        print("=" * 50)
        print("Press Ctrl+C to exit")

        if self.env is None or self.agent is None:
            raise RuntimeError(
                "Environment and agent must be setup before " "running demo"
            )

        if self.config["mode"] == "demo":
            max_demo_episodes = self.config.get("num_episodes", 1)
        else:
            max_demo_episodes = 1
        if self.config["mode"] == "demo" and self.config.get("num_episodes") == 100:
            max_demo_episodes = 1
        current_episode = 0

        demo_results = {
            "episodes": [],
            "rewards": [],
            "metrics": [],
            "decisions": [],
            "summary": {},
        }

        state = self.env.reset()
        step = 0

        print(f"Running demo for {max_demo_episodes} episode(s)...")

        try:
            while current_episode < max_demo_episodes:
                episode_reward = 0
                episode_decisions = []
                episode_metrics = []
                episode_step = 0

                print(f"\n{'='*30}")
                print(f"EPISODE {current_episode + 1}/{max_demo_episodes}")
                print(f"{'='*30}")

                while True:
                    if isinstance(self.agent, SumoQLearningAgent):
                        action, explanation = self.agent.recommend_action(state)
                        ep_step = f"{current_episode + 1}, Step {step + 1}"
                        print(f"\nEpisode {ep_step}")
                        print(f"Q-Learning Recommendation: {explanation}")
                    elif hasattr(self.agent, "get_decision_explanation"):
                        explanation = self.agent.get_decision_explanation(state)
                        action = self.agent.get_action(state)
                        ep_step = f"{current_episode + 1}, Step {step + 1}"
                        print(f"\nEpisode {ep_step}")
                        print(f"{explanation}")
                    else:
                        action = self.agent.get_action(state)
                        explanation = f"Action: {action}"
                        ep_step = f"{current_episode + 1}, Step {step + 1}"
                        print(f"\nEpisode {ep_step}")
                        print(f"Action: {action}")

                    next_state, reward, done, info = self.env.step(action)

                    print(f"Reward: {reward:.2f}")
                    traffic = info["metrics"].throughput
                    print(f"Current Traffic: {traffic} vehicles")

                    episode_reward += reward
                    episode_decisions.append(
                        {
                            "step": episode_step,
                            "state": (
                                state.tolist()
                                if hasattr(state, "tolist")
                                else list(state)
                            ),
                            "action": int(action),
                            "explanation": explanation,
                            "reward": float(reward),
                            "next_state": (
                                next_state.tolist()
                                if hasattr(next_state, "tolist")
                                else list(next_state)
                            ),
                        }
                    )
                    metrics = info["metrics"]
                    episode_metrics.append(
                        {
                            "step": episode_step,
                            "waiting_time": float(metrics.waiting_time),
                            "throughput": int(metrics.throughput),
                            "average_speed": float(metrics.average_speed),
                            "queue_lengths": dict(metrics.queue_length),
                            "emissions": float(metrics.emissions),
                            "phase": info.get("phase", "unknown"),
                            "phase_duration": float(info.get("phase_duration", 0)),
                        }
                    )

                    state = next_state
                    step += 1
                    episode_step += 1

                    time.sleep(self.config.get("demo_delay", 2))

                    if done:
                        break

                wait_times = [m["waiting_time"] for m in episode_metrics]
                avg_wait = float(np.mean(wait_times))
                total_tput = sum([m["throughput"] for m in episode_metrics])
                speeds = [m["average_speed"] for m in episode_metrics]
                avg_spd = float(np.mean(speeds))

                episode_summary = {
                    "episode": current_episode,
                    "total_reward": float(episode_reward),
                    "steps": episode_step,
                    "avg_waiting_time": avg_wait,
                    "total_throughput": total_tput,
                    "avg_speed": avg_spd,
                    "vehicles_completed": info.get("vehicles_completed", 0),
                }

                demo_results["episodes"].append(episode_summary)
                demo_results["rewards"].append(float(episode_reward))
                demo_results["metrics"].extend(episode_metrics)
                demo_results["decisions"].extend(episode_decisions)

                current_episode += 1
                print(f"\nEpisode {current_episode} completed!")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Steps: {episode_step}")
                print(f"  Avg Waiting Time: {avg_wait:.2f}s")
                print(f"  Total Throughput: {total_tput}")
                print(f"  Avg Speed: {avg_spd:.2f} m/s")

                if current_episode < max_demo_episodes:
                    print(f"Starting episode {current_episode + 1}...")
                    state = self.env.reset()
                    step = 0
                else:
                    break

            if demo_results["episodes"]:
                eps = demo_results["episodes"]
                ep_steps = [ep["steps"] for ep in eps]
                ep_waits = [ep["avg_waiting_time"] for ep in eps]
                ep_tputs = [ep["total_throughput"] for ep in eps]
                ep_speeds = [ep["avg_speed"] for ep in eps]

                demo_results["summary"] = {
                    "total_episodes": len(eps),
                    "avg_reward": float(np.mean(demo_results["rewards"])),
                    "total_reward": float(sum(demo_results["rewards"])),
                    "avg_episode_length": float(np.mean(ep_steps)),
                    "avg_waiting_time": float(np.mean(ep_waits)),
                    "total_throughput": sum(ep_tputs),
                    "avg_speed": float(np.mean(ep_speeds)),
                    "agent_type": self.config["agent_type"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                self._save_demo_results(demo_results)

            print("Demo completed!")

        except KeyboardInterrupt:
            print("\nDemo ended by user")
            if demo_results["episodes"]:
                demo_results["summary"] = {
                    "total_episodes": len(demo_results["episodes"]),
                    "avg_reward": (
                        float(np.mean(demo_results["rewards"]))
                        if demo_results["rewards"]
                        else 0
                    ),
                    "interrupted": True,
                    "agent_type": self.config["agent_type"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save_demo_results(demo_results)

    def _save_demo_results(self, demo_results):
        """Save demo results to files"""
        import json
        import os

        os.makedirs("data/demo_results", exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        agent_type = self.config["agent_type"]

        detailed_filename = (
            f"data/demo_results/" f"demo_{agent_type}_{timestamp}_detailed.json"
        )
        with open(detailed_filename, "w") as f:
            json.dump(demo_results, f, indent=2, default=str)

        summary_filename = (
            f"data/demo_results/" f"demo_{agent_type}_{timestamp}_summary.json"
        )
        summary_report = {
            "config": {
                "agent_type": self.config["agent_type"],
                "episodes": len(demo_results["episodes"]),
                "sumo_config": self.config["sumo_config"],
                "step_length": self.config["step_length"],
                "demo_delay": self.config["demo_delay"],
            },
            "performance": demo_results["summary"],
            "episode_summaries": demo_results["episodes"],
        }

        with open(summary_filename, "w") as f:
            json.dump(summary_report, f, indent=2, default=str)

        text_report = self._generate_text_report(demo_results)
        text_filename = (
            f"data/demo_results/" f"demo_{agent_type}_{timestamp}_report.txt"
        )
        with open(text_filename, "w") as f:
            f.write(text_report)

        print("\nDemo results saved:")
        print(f"  Detailed data: {detailed_filename}")
        print(f"  Summary: {summary_filename}")
        print(f"  Text report: {text_filename}")

    def _generate_text_report(self, demo_results):
        """Generate human-readable text report"""
        summary = demo_results["summary"]
        episodes = demo_results["episodes"]

        report = f"""
SUMO Traffic Control Demo Report
===============================
Generated: {summary.get('timestamp', 'Unknown')}
Agent Type: {summary.get('agent_type', 'Unknown')}

OVERALL PERFORMANCE:
- Total Episodes: {summary.get('total_episodes', 0)}
- Average Reward per Episode: {summary.get('avg_reward', 0):.2f}
- Total Reward: {summary.get('total_reward', 0):.2f}
- Average Episode Length: {summary.get('avg_episode_length', 0):.1f} steps
- Average Waiting Time: {summary.get('avg_waiting_time', 0):.2f} seconds
- Total Throughput: {summary.get('total_throughput', 0)} vehicles
- Average Speed: {summary.get('avg_speed', 0):.2f} m/s

EPISODE BREAKDOWN:
"""

        for i, ep in enumerate(episodes):
            report += f"""
Episode {i+1}:
  Reward: {ep.get('total_reward', 0):.2f}
  Steps: {ep.get('steps', 0)}
  Avg Waiting Time: {ep.get('avg_waiting_time', 0):.2f}s
  Throughput: {ep.get('total_throughput', 0)} vehicles
  Avg Speed: {ep.get('avg_speed', 0):.2f} m/s
  Vehicles Completed: {ep.get('vehicles_completed', 0)}
"""

        if demo_results["decisions"]:
            actions = [d["action"] for d in demo_results["decisions"]]
            action_counts = {
                0: actions.count(0),
                1: actions.count(1),
                2: actions.count(2),
            }
            total_decisions = len(actions)

            report += f"""
DECISION ANALYSIS:
- Total Decisions: {total_decisions}
- Maintain Phase (Action 0): {action_counts[0]} \
({action_counts[0]/total_decisions*100:.1f}%)
- Switch to NS Green (Action 1): {action_counts[1]} \
({action_counts[1]/total_decisions*100:.1f}%)
- Switch to EW Green (Action 2): {action_counts[2]} \
({action_counts[2]/total_decisions*100:.1f}%)

PERFORMANCE INSIGHTS:
"""

            if summary.get("avg_reward", 0) > 0:
                report += "Positive average reward - " "Agent is performing well\n"
            else:
                report += "Negative average reward - " "Agent may need more training\n"

            if summary.get("avg_waiting_time", 0) < 60:
                report += "Low average waiting time - Good traffic flow\n"
            else:
                report += "High average waiting time - " "Traffic congestion issues\n"

            if summary.get("avg_speed", 0) > 8:
                report += "Good average speed - " "Efficient traffic movement\n"
            else:
                report += "Low average speed - Traffic moving slowly\n"

        return report

    def save_training_history(self):
        """Save training history to JSON for analysis"""
        import json

        os.makedirs("models", exist_ok=True)
        history_path = "models/training_history.json"

        with open(history_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Training history saved to {history_path}")

    def plot_training_results(self):
        """Plot training results"""
        if not self.results["episodes"]:
            print("No training results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("SUMO Traffic Control Training Results")

        axes[0, 0].plot(self.results["episodes"], self.results["rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")

        axes[0, 1].plot(self.results["episodes"], self.results["avg_waiting_time"])
        axes[0, 1].set_title("Average Waiting Time")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Waiting Time (s)")

        axes[1, 0].plot(self.results["episodes"], self.results["throughput"])
        axes[1, 0].set_title("Throughput")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Vehicles Processed")

        axes[1, 1].plot(self.results["episodes"], self.results["avg_speed"])
        axes[1, 1].set_title("Average Speed")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Speed (m/s)")

        plt.tight_layout()
        plt.savefig("sumo_training_results.png", dpi=300, bbox_inches="tight")
        print("Training results plotted and saved as " "'sumo_training_results.png'")

        if self.config.get("show_plots", False):
            plt.show()

    def cleanup(self):
        """Cleanup resources"""
        if self.env:
            self.env.close()

        if self.agent and hasattr(self.agent, "cleanup"):
            self.agent.cleanup()

        print("Cleanup complete")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SUMO Traffic Control")

    parser.add_argument(
        "--mode",
        choices=["train", "test", "demo"],
        default="demo",
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--agent",
        choices=["qlearning", "heuristic"],
        default="heuristic",
        help="Agent type (default: heuristic)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes (default: 10)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without SUMO GUI (faster training)",
    )
    parser.add_argument("--load-model", help="Path to pre-trained model")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast training mode (30min episodes, 2s steps)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Enable GPU acceleration (ROCm/CUDA)"
    )

    return parser.parse_args()


def main():
    """Main function"""
    Args = parse_arguments()

    if Args.gpu:
        os.environ.setdefault("ROCM_PATH", "/opt/rocm")
        os.environ.setdefault("HIP_PATH", "/opt/rocm")
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
        os.environ.setdefault("GPU_DEVICE_ORDINAL", "0")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
        os.environ.setdefault("HSA_ENABLE_SDMA", "0")
        os.environ.setdefault("AMD_DIRECT_DISPATCH", "1")

    config = {
        "mode": Args.mode,
        "agent_type": Args.agent,
        "num_episodes": Args.episodes,
        "test_episodes": Args.episodes,  # Added for test mode
        "sumo_config": (
            "sumo_config/intersection_fast.sumocfg"
            if Args.fast
            else "sumo_config/intersection.sumocfg"
        ),
        "use_gui": not Args.no_gui,
        "step_length": 2.0 if Args.fast else 1.0,
        "seed": 42,
        "demo_delay": 0.5,
        "load_model": Args.load_model,
        "learning_rate": 0.3,
        "discount_factor": 0.8,
        "epsilon": 1.0 if Args.mode == "train" else 0.0,
        "epsilon_decay": 0.9999,
        "min_phase_duration": 5,
        "max_phase_duration": 45,
        "congestion_threshold": 4,
        "save_interval": 50,
        "show_plots": False,
        "use_gpu": Args.gpu,
    }

    print("SUMO Traffic Control System")
    print("=" * 50)
    print(f"Mode: {config['mode']}")
    print(f"Agent: {config['agent_type']}")
    print(f"Episodes: {config['num_episodes']}")
    print(f"GUI: {'Enabled' if config['use_gui'] else 'Disabled'}")
    print(f"GPU: {'Enabled' if Args.gpu else 'Disabled (CPU only)'}")
    if Args.fast:
        print("Training: FAST (30min episodes, 2s steps)")
    print("=" * 50)

    if not os.path.exists(config["sumo_config"]):
        sumo_cfg = config["sumo_config"]
        print(f"ERROR: SUMO config file not found: {sumo_cfg}")
        print("Please generate the network files first. " "See setup instructions.")
        return 1

    controller = SumoTrafficController(config)

    try:
        controller.setup_environment()
        controller.setup_agent()

        if config["mode"] == "train":
            controller.train()
        elif config["mode"] == "test":
            controller.test()
        elif config["mode"] == "demo":
            controller.demo()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        controller.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
