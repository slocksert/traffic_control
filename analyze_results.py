#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class TrafficResultsAnalyzer:
    def __init__(self, results_dir: str = "data/demo_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)

        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def load_results(self, pattern: str = "*_summary.json") -> Dict[str, Any]:
        files = list(self.results_dir.glob(pattern))
        results = {}

        for file in files:
            agent_type = "qlearning" if "qlearning" in file.name else "heuristic"

            with open(file, 'r') as f:
                data = json.load(f)

            if agent_type not in results:
                results[agent_type] = []
            results[agent_type].append(data)

        return results

    def aggregate_metrics(self, results: Dict[str, Any]) -> pd.DataFrame:
        rows = []

        for agent_type, sessions in results.items():
            for session in sessions:
                episodes = session.get('episode_summaries', [])

                if episodes:
                    for episode in episodes:
                        row = {
                            'agent': agent_type.upper(),
                            'avg_reward': episode.get('total_reward', 0),
                            'avg_waiting_time': episode.get('avg_waiting_time', 0),
                            'total_throughput': episode.get('total_throughput', 0),
                            'avg_speed': episode.get('avg_speed', 0),
                            'episode': episode.get('episode', 0),
                        }
                        rows.append(row)
                else:
                    perf = session.get('performance', {})
                    row = {
                        'agent': agent_type.upper(),
                        'avg_reward': perf.get('avg_reward', 0),
                        'avg_waiting_time': perf.get('avg_waiting_time', 0),
                        'total_throughput': perf.get('total_throughput', 0),
                        'avg_speed': perf.get('avg_speed', 0),
                        'episode': 0,
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def create_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.groupby('agent').agg({
            'avg_reward': ['mean', 'std'],
            'avg_waiting_time': ['mean', 'std'],
            'total_throughput': ['mean', 'std'],
            'avg_speed': ['mean', 'std'],
        }).round(2)

        return summary

    def plot_comparative_metrics(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('avg_waiting_time', 'Tempo de Espera Médio (s)', axes[0, 0]),
            ('total_throughput', 'Throughput Total (veículos)', axes[0, 1]),
            ('avg_speed', 'Velocidade Média (m/s)', axes[1, 0]),
            ('avg_reward', 'Recompensa Média', axes[1, 1])
        ]

        for metric, title, ax in metrics:
            sns.barplot(data=df, x='agent', y=metric, ax=ax, errorbar='sd')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Agente', fontsize=10)
            ax.set_ylabel(title.split('(')[0].strip(), fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comparative_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Gráfico comparativo salvo em: {self.plots_dir / 'comparative_metrics.png'}")
        plt.close()

    def plot_boxplot_comparison(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = [
            ('avg_waiting_time', 'Tempo de Espera (s)', axes[0, 0]),
            ('total_throughput', 'Throughput (veículos)', axes[0, 1]),
            ('avg_speed', 'Velocidade (m/s)', axes[1, 0]),
            ('avg_reward', 'Recompensa', axes[1, 1])
        ]

        for metric, title, ax in metrics:
            sns.boxplot(data=df, x='agent', y=metric, ax=ax)
            ax.set_title(f'Distribuição: {title}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Agente', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'boxplot_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Box plot salvo em: {self.plots_dir / 'boxplot_comparison.png'}")
        plt.close()

    def plot_training_evolution(self, training_file: str = "models/training_history.json"):
        if not Path(training_file).exists():
            print(f"Arquivo de histórico de treino não encontrado: {training_file}")
            return

        with open(training_file, 'r') as f:
            history = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        episodes = history.get('episodes', [])
        rewards = history.get('rewards', [])
        waiting_times = history.get('avg_waiting_time', [])
        throughput = history.get('throughput', [])
        avg_speed = history.get('avg_speed', [])

        if episodes and rewards:
            axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue')

            if len(rewards) >= 10:
                rolling = pd.Series(rewards).rolling(window=10).mean()
                axes[0, 0].plot(episodes, rolling, color='red', linewidth=2, label='Média móvel (10 ep.)')

            axes[0, 0].set_title('Evolução da Recompensa', fontweight='bold')
            axes[0, 0].set_xlabel('Episódio')
            axes[0, 0].set_ylabel('Recompensa Total')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

        if episodes and waiting_times:
            axes[0, 1].plot(episodes, waiting_times, alpha=0.3, color='orange')

            if len(waiting_times) >= 10:
                rolling = pd.Series(waiting_times).rolling(window=10).mean()
                axes[0, 1].plot(episodes, rolling, color='red', linewidth=2, label='Média móvel (10 ep.)')

            axes[0, 1].set_title('Tempo de Espera Médio', fontweight='bold')
            axes[0, 1].set_xlabel('Episódio')
            axes[0, 1].set_ylabel('Tempo de Espera (s)')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

        if episodes and throughput:
            axes[1, 0].plot(episodes, throughput, alpha=0.3, color='green')

            if len(throughput) >= 10:
                rolling = pd.Series(throughput).rolling(window=10).mean()
                axes[1, 0].plot(episodes, rolling, color='red', linewidth=2, label='Média móvel (10 ep.)')

            axes[1, 0].set_title('Throughput Total', fontweight='bold')
            axes[1, 0].set_xlabel('Episódio')
            axes[1, 0].set_ylabel('Veículos Processados')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        if episodes and avg_speed:
            axes[1, 1].plot(episodes, avg_speed, alpha=0.3, color='purple')

            if len(avg_speed) >= 10:
                rolling = pd.Series(avg_speed).rolling(window=10).mean()
                axes[1, 1].plot(episodes, rolling, color='red', linewidth=2, label='Média móvel (10 ep.)')

            axes[1, 1].set_title('Velocidade Média', fontweight='bold')
            axes[1, 1].set_xlabel('Episódio')
            axes[1, 1].set_ylabel('Velocidade (m/s)')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_evolution.png', dpi=300, bbox_inches='tight')
        print(f"Gráfico de evolução do treino salvo em: {self.plots_dir / 'training_evolution.png'}")
        plt.close()

    def export_comparison_table(self, df: pd.DataFrame, filename: str = "comparison_table.csv"):
        summary = self.create_comparison_table(df)

        output_path = self.plots_dir / filename
        summary.to_csv(output_path)
        print(f"\nTabela de comparação salva em: {output_path}")

        print("\n=== TABELA DE COMPARAÇÃO ===")
        print(summary)
        print("=" * 50)

        return summary

    def calculate_improvement(self, df: pd.DataFrame) -> Dict[str, float]:
        if 'QLEARNING' not in df['agent'].values or 'HEURISTIC' not in df['agent'].values:
            print("Dados insuficientes para calcular melhoria")
            return {}

        qlearning = df[df['agent'] == 'QLEARNING']
        heuristic = df[df['agent'] == 'HEURISTIC']

        improvements = {}

        for metric in ['avg_reward', 'total_throughput', 'avg_speed']:
            q_mean = qlearning[metric].mean()
            h_mean = heuristic[metric].mean()

            if h_mean != 0:
                improvement = ((q_mean - h_mean) / abs(h_mean)) * 100
                improvements[metric] = improvement

        for metric in ['avg_waiting_time']:
            q_mean = qlearning[metric].mean()
            h_mean = heuristic[metric].mean()

            if h_mean != 0:
                improvement = ((h_mean - q_mean) / abs(h_mean)) * 100
                improvements[metric] = improvement

        print("\n=== MELHORIA PERCENTUAL (Q-Learning vs Heurístico) ===")
        for metric, value in improvements.items():
            print(f"{metric}: {value:+.2f}%")
        print("=" * 50)

        return improvements

    def plot_action_distribution(self, results: Dict[str, Any]):
        """Plota distribuição de ações dos agentes"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (agent_type, sessions) in enumerate(results.items()):
            if idx >= 2:
                break

            action_counts = {0: 0, 1: 0, 2: 0}

            for session in sessions:
                decisions = session.get('decisions', [])
                for decision in decisions:
                    action = decision.get('action', 0)
                    action_counts[action] = action_counts.get(action, 0) + 1

            if sum(action_counts.values()) == 0:
                continue

            labels = ['Manter Fase', 'Verde N-S', 'Verde L-O']
            sizes = [action_counts[0], action_counts[1], action_counts[2]]
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            ax = axes[idx]
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            ax.set_title(f'Distribuição de Ações - {agent_type.upper()}',
                        fontsize=12, fontweight='bold')

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Distribuição de ações salva em: {self.plots_dir / 'action_distribution.png'}")
        plt.close()

    def generate_full_report(self):
        print("=" * 60)
        print("GERANDO RELATÓRIO COMPLETO DE ANÁLISE")
        print("=" * 60)

        results = self.load_results()

        if not results:
            print("Nenhum resultado encontrado em:", self.results_dir)
            print("Execute primeiro: python main.py --mode test --agent qlearning/heuristic")
            return

        print(f"\nResultados carregados para {len(results)} agente(s)")

        df = self.aggregate_metrics(results)

        if df.empty:
            print("Nenhuma métrica encontrada nos resultados")
            return

        print(f"\nTotal de {len(df)} sessões de teste")

        self.export_comparison_table(df)

        self.plot_comparative_metrics(df)

        self.plot_boxplot_comparison(df)

        self.plot_action_distribution(results)

        self.plot_training_evolution()

        self.calculate_improvement(df)

        print("\n" + "=" * 60)
        print("RELATÓRIO COMPLETO GERADO COM SUCESSO!")
        print(f"Gráficos salvos em: {self.plots_dir}")
        print("=" * 60)


def main():
    analyzer = TrafficResultsAnalyzer()
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()
