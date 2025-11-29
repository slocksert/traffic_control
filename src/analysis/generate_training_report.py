"""
Automatic Training Report Generator

Generates comprehensive reports with learning curves, convergence analysis,
and performance metrics for training runs
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime


# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def generate_training_report(
    df: pd.DataFrame, train_info: Dict, output_dir: str = "training_results"
):
    """
    Generate complete training report with learning curves and analysis

    Args:
        df: DataFrame with training episode results (includes epsilon column)
        train_info: Dictionary with training run information
        output_dir: Directory to save results

    Returns:
        Path to generated report directory
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = train_info.get("agent_type", "unknown")
    report_dir = Path(output_dir) / f"training_{agent_type}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GERANDO RELATÓRIO COMPLETO DE TREINAMENTO")
    print("=" * 80)

    # 1. Calculate metrics
    print("\n1. Calculando métricas de aprendizado...")
    metrics = _calculate_training_metrics(df)

    # 2. Generate tables
    print("2. Gerando tabelas de progresso...")
    _generate_training_tables(df, metrics, report_dir)

    # 3. Generate graphs
    print("3. Gerando gráficos de aprendizado...")
    _generate_training_graphs(df, metrics, report_dir)

    # 4. Generate learning analysis
    print("4. Gerando análise de convergência...")
    analysis_text = _generate_learning_analysis(df, metrics, train_info)

    # 5. Create complete text report
    print("5. Criando relatório completo...")
    _create_training_report(df, metrics, analysis_text, train_info, report_dir)

    # 6. Save data as JSON
    print("6. Salvando dados em JSON...")
    _save_training_json(df, metrics, train_info, report_dir)

    print(f"\n{'=' * 80}")
    print("RELATÓRIO DE TREINAMENTO GERADO COM SUCESSO!")
    print(f"{'=' * 80}")
    print(f"\nDiretório: {report_dir}")
    print("\nArquivos gerados:")
    for file in sorted(report_dir.glob("*")):
        if file.is_file():
            print(f"  - {file.name}")

    return report_dir


def _calculate_training_metrics(df: pd.DataFrame) -> Dict:
    """Calculate training-specific metrics"""

    # Split into early and late training
    mid_point = len(df) // 2
    early_df = df.iloc[:mid_point]
    late_df = df.iloc[mid_point:]

    metrics = {
        "total_episodes": len(df),
        # Overall metrics
        "reward": {
            "mean": float(df["total_reward"].mean()),
            "std": float(df["total_reward"].std()),
            "min": float(df["total_reward"].min()),
            "max": float(df["total_reward"].max()),
            "final_avg": float(df["total_reward"].tail(10).mean()),  # Last 10 episodes
        },
        # Learning progress
        "learning": {
            "early_reward_avg": float(early_df["total_reward"].mean()),
            "late_reward_avg": float(late_df["total_reward"].mean()),
            "improvement": float(
                late_df["total_reward"].mean() - early_df["total_reward"].mean()
            ),
            "improvement_pct": (
                float(
                    (late_df["total_reward"].mean() - early_df["total_reward"].mean())
                    / abs(early_df["total_reward"].mean())
                    * 100
                )
                if early_df["total_reward"].mean() != 0
                else 0
            ),
        },
        # Epsilon decay
        "epsilon": {
            "initial": float(df["epsilon"].iloc[0]) if "epsilon" in df.columns else 1.0,
            "final": float(df["epsilon"].iloc[-1]) if "epsilon" in df.columns else 0.0,
        },
        # Traffic metrics
        "waiting_time": {
            "mean": float(df["avg_waiting_time"].mean()),
            "std": float(df["avg_waiting_time"].std()),
            "final_avg": float(df["avg_waiting_time"].tail(10).mean()),
        },
        "throughput": {
            "mean": float(df["total_throughput"].mean()),
            "total": int(df["total_throughput"].sum()),
            "final_avg": float(df["total_throughput"].tail(10).mean()),
        },
        "speed": {
            "mean": float(df["avg_speed"].mean()),
            "final_avg": float(df["avg_speed"].tail(10).mean()),
        },
    }

    # Convergence analysis (check last 20% stability)
    last_20_pct = df.tail(len(df) // 5)
    if len(last_20_pct) > 0:
        reward_cv = (
            last_20_pct["total_reward"].std() / abs(last_20_pct["total_reward"].mean())
            if last_20_pct["total_reward"].mean() != 0
            else 0
        )
        metrics["convergence"] = {
            "last_20pct_reward_mean": float(last_20_pct["total_reward"].mean()),
            "last_20pct_reward_std": float(last_20_pct["total_reward"].std()),
            "coefficient_of_variation": float(reward_cv),
            "is_converged": reward_cv < 0.2,  # CV < 0.2 indicates convergence
        }

    return metrics


def _generate_training_tables(df: pd.DataFrame, metrics: Dict, output_dir: Path):
    """Generate training progress tables"""
    # Summary table
    table_data = []

    table_data.append(
        {
            "Fase": "Início (primeiros 50%)",
            "Recompensa Média": f"{metrics['learning']['early_reward_avg']:.2f}",
            "Tempo Espera (s)": f"{df.iloc[:len(df)//2]['avg_waiting_time'].mean():.2f}",
            "Throughput": f"{df.iloc[:len(df)//2]['total_throughput'].mean():.1f}",
        }
    )

    table_data.append(
        {
            "Fase": "Final (últimos 50%)",
            "Recompensa Média": f"{metrics['learning']['late_reward_avg']:.2f}",
            "Tempo Espera (s)": (
                f"{df.iloc[len(df)//2:]['avg_waiting_time'].mean():.2f}"
            ),
            "Throughput": f"{df.iloc[len(df)//2:]['total_throughput'].mean():.1f}",
        }
    )

    table_data.append(
        {
            "Fase": "Melhoria",
            "Recompensa Média": f"{metrics['learning']['improvement']:+.2f} ({metrics['learning']['improvement_pct']:+.1f}%)",
            "Tempo Espera (s)": "-",
            "Throughput": "-",
        }
    )

    table_df = pd.DataFrame(table_data)

    csv_path = output_dir / "progresso_aprendizado.csv"
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   Tabela de progresso salva: {csv_path.name}")

    # Detailed episodes
    episodes_path = output_dir / "episodios_treinamento.csv"
    df.to_csv(episodes_path, index=False, encoding="utf-8-sig")
    print(f"   Dados detalhados salvos: {episodes_path.name}")


def _generate_training_graphs(df: pd.DataFrame, metrics: Dict, output_dir: Path):
    """Generate training-specific graphs"""
    colors = sns.color_palette("husl", 8)

    # Main training graph (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Curvas de Aprendizado - Progresso do Treinamento",
        fontsize=16,
        fontweight="bold",
    )

    episodes = df["episode_number"].values + 1

    # 1. Learning curve with moving average
    ax = axes[0, 0]
    ax.plot(
        episodes,
        df["total_reward"],
        alpha=0.3,
        color=colors[0],
        label="Recompensa por episódio",
    )

    # Calculate moving averages
    window_sizes = [10, 50]
    for i, window in enumerate(window_sizes):
        if len(df) >= window:
            ma = df["total_reward"].rolling(window=window, min_periods=1).mean()
            ax.plot(
                episodes,
                ma,
                linewidth=2,
                color=colors[i + 1],
                label=f"Média Móvel ({window} eps)",
            )

    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa Total")
    ax.set_title("Curva de Aprendizado")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Epsilon decay
    if "epsilon" in df.columns:
        ax = axes[0, 1]
        ax.plot(episodes, df["epsilon"], color=colors[3], linewidth=2)
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Epsilon (Taxa de Exploração)")
        ax.set_title("Decaimento do Epsilon")
        ax.grid(True, alpha=0.3)

    # 3. Waiting time improvement
    ax = axes[1, 0]
    ax.plot(episodes, df["avg_waiting_time"], alpha=0.4, color=colors[4])
    if len(df) >= 50:
        ma = df["avg_waiting_time"].rolling(window=50, min_periods=1).mean()
        ax.plot(
            episodes, ma, linewidth=2, color=colors[5], label="Média Móvel (50 eps)"
        )
        ax.legend()
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Tempo Médio de Espera (s)")
    ax.set_title("Evolução do Tempo de Espera")
    ax.grid(True, alpha=0.3)

    # 4. Throughput progress
    ax = axes[1, 1]
    ax.plot(episodes, df["total_throughput"], alpha=0.4, color=colors[6])
    if len(df) >= 50:
        ma = df["total_throughput"].rolling(window=50, min_periods=1).mean()
        ax.plot(
            episodes, ma, linewidth=2, color=colors[7], label="Média Móvel (50 eps)"
        )
        ax.legend()
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Throughput (veículos)")
    ax.set_title("Evolução do Throughput")
    ax.grid(True, alpha=0.3)

    # 5. Reward distribution: early vs late
    ax = axes[2, 0]
    mid = len(df) // 2
    early_rewards = df["total_reward"].iloc[:mid]
    late_rewards = df["total_reward"].iloc[mid:]

    bp = ax.boxplot(
        [early_rewards, late_rewards],
        labels=["Primeiros 50%", "Últimos 50%"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][1].set_facecolor(colors[1])

    # Add mean markers
    means = [early_rewards.mean(), late_rewards.mean()]
    ax.plot([1, 2], means, "D", color="red", markersize=8, label="Média", zorder=3)

    ax.set_ylabel("Recompensa Total")
    ax.set_title("Comparação: Início vs Final do Treinamento")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Convergence analysis
    ax = axes[2, 1]
    window = 20
    if len(df) >= window:
        rolling_std = df["total_reward"].rolling(window=window).std()
        rolling_mean = df["total_reward"].rolling(window=window).mean()
        cv = rolling_std / rolling_mean.abs()

        ax.plot(episodes[window - 1 :], cv.dropna(), color=colors[2], linewidth=2)
        ax.axhline(
            0.2, color="red", linestyle="--", label="Limiar de Convergência (0.2)"
        )
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Coeficiente de Variação")
        ax.set_title(f"Análise de Convergência (janela de {window} eps)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    graph_path = output_dir / "curvas_aprendizado.png"
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    print(f"   Gráfico gerado: {graph_path.name}")
    plt.close()


def _generate_learning_analysis(
    df: pd.DataFrame, metrics: Dict, train_info: Dict
) -> str:
    """Generate learning analysis text"""
    analysis = []

    analysis.append("=" * 80)
    analysis.append("ANÁLISE DO PROCESSO DE APRENDIZADO")
    analysis.append("=" * 80)
    analysis.append("")

    # 1. Learning progress
    analysis.append("1. PROGRESSO DO APRENDIZADO")
    analysis.append("-" * 40)

    improvement = metrics["learning"]["improvement"]
    improvement_pct = metrics["learning"]["improvement_pct"]

    if improvement > 0:
        analysis.append(
            f"✓ Melhoria significativa: +{improvement:.2f}"
            f" recompensa ({improvement_pct:+.1f}%)"
        )
        analysis.append(
            f"  Recompensa inicial (primeiros 50%): "
            f"{metrics['learning']['early_reward_avg']:.2f}"
        )
        analysis.append(
            f"  Recompensa final (últimos 50%): "
            f"{metrics['learning']['late_reward_avg']:.2f}"
        )
    else:
        analysis.append(
            f"⚠ Aprendizado limitado: "
            f"{improvement:.2f} recompensa ({improvement_pct:+.1f}%)"
        )
        analysis.append("  Considere ajustar hiperparâmetros ou aumentar episódios")

    analysis.append("")

    # 2. Convergence
    is_converged = False  # Default value
    if "convergence" in metrics:
        analysis.append("2. CONVERGÊNCIA")
        analysis.append("-" * 40)

        cv = metrics["convergence"]["coefficient_of_variation"]
        is_converged = metrics["convergence"]["is_converged"]

        if is_converged:
            analysis.append(f"✓ Modelo convergiu (CV={cv:.3f} < 0.2)")
            analysis.append("  Performance estabilizada nos episódios finais")
        else:
            analysis.append(f"⚠ Modelo ainda não convergiu completamente (CV={cv:.3f})")
            analysis.append("  Considere mais episódios de treinamento")

        last_20_mean = metrics["convergence"]["last_20pct_reward_mean"]
        last_20_std = metrics["convergence"]["last_20pct_reward_std"]
        analysis.append(
            f"  Últimos 20% episódios: {last_20_mean:.2f} ± {last_20_std:.2f}"
        )

    analysis.append("")

    # 3. Exploration vs Exploitation
    analysis.append("3. EXPLORAÇÃO vs EXPLOITAÇÃO")
    analysis.append("-" * 40)

    eps_initial = metrics["epsilon"]["initial"]
    eps_final = metrics["epsilon"]["final"]
    analysis.append(f"  Epsilon inicial: {eps_initial:.4f}")
    analysis.append(f"  Epsilon final: {eps_final:.4f}")
    analysis.append(f"  Decay: {((eps_initial - eps_final) / eps_initial * 100):.1f}%")

    if eps_final < 0.1:
        analysis.append("✓ Boa transição para exploitação")
    elif eps_final < 0.3:
        analysis.append("~ Ainda em fase de exploração moderada")
    else:
        analysis.append("⚠ Alta taxa de exploração - considere mais episódios")

    analysis.append("")

    # 4. Traffic performance
    analysis.append("4. DESEMPENHO NO CONTROLE DE TRÁFEGO")
    analysis.append("-" * 40)

    wait_final = metrics["waiting_time"]["final_avg"]
    throughput_final = metrics["throughput"]["final_avg"]

    analysis.append(f"  Tempo de espera (últimos 10 eps): {wait_final:.2f}s")
    analysis.append(f"  Throughput (últimos 10 eps): {throughput_final:.1f} veículos")

    if wait_final < 60:
        analysis.append("✓ Bom controle de tempo de espera")
    else:
        analysis.append("⚠ Tempo de espera elevado - revisar estratégia")

    analysis.append("")

    # 5. Recommendations
    analysis.append("5. RECOMENDAÇÕES")
    analysis.append("-" * 40)

    num_episodes = metrics["total_episodes"]

    if is_converged and improvement > 0:
        analysis.append("✓ Treinamento bem-sucedido!")
        analysis.append("  • Modelo pronto para testes finais")
        analysis.append("  • Considere salvar este modelo como baseline")
    else:
        if num_episodes < 100:
            analysis.append("• Aumentar número de episódios (recomendado: >= 100)")
        if not is_converged:
            analysis.append("• Modelo precisa de mais treinamento para convergir")
        if improvement < 0:
            analysis.append("• Revisar função de recompensa e hiperparâmetros")
            analysis.append("• Verificar se o ambiente está configurado corretamente")

    analysis.append("")
    analysis.append("=" * 80)

    return "\n".join(analysis)


def _create_training_report(
    df: pd.DataFrame,
    metrics: Dict,
    analysis_text: str,
    train_info: Dict,
    output_dir: Path,
):
    """Create complete training text report"""
    report_path = output_dir / "relatorio_treinamento.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SISTEMA DE CONTROLE DE TRÁFEGO - RELATÓRIO DE TREINAMENTO\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tipo de Agente: {train_info.get('agent_type', 'Desconhecido')}\n")
        f.write(f"Total de Episódios: {metrics['total_episodes']}\n")
        f.write("\n")

        # Configuration
        if train_info.get("config"):
            f.write("CONFIGURAÇÃO:\n")
            f.write("-" * 80 + "\n")
            for key, value in train_info["config"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Summary
        f.write("RESUMO DO APRENDIZADO:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Recompensa inicial (média):\n"
            f"              {metrics['learning']['early_reward_avg']:.2f}\n"
        )
        f.write(
            f"Recompensa final (média):\n"
            f"              {metrics['learning']['late_reward_avg']:.2f}\n"
        )
        f.write(
            f"Melhoria: {metrics['learning']['improvement']:+.2f}\n"
            f"              ({metrics['learning']['improvement_pct']:+.1f}%)\n"
        )
        f.write(
            f"Epsilon: {metrics['epsilon']['initial']:.4f} → \n"
            f"              {metrics['epsilon']['final']:.4f}\n"
        )

        if "convergence" in metrics:
            f.write(
                f"Convergência:\n"
                f"        {'Sim' if metrics['convergence']['is_converged'] else 'Não'} "
            )
            f.write(f"(CV={metrics['convergence']['coefficient_of_variation']:.3f})\n")

        f.write("\n\n")

        # Analysis
        f.write(analysis_text)
        f.write("\n\n")

        # Files
        f.write("ARQUIVOS GERADOS:\n")
        f.write("-" * 80 + "\n")
        for file in sorted(output_dir.glob("*")):
            if file.is_file() and file != report_path:
                f.write(f"  - {file.name}\n")

    print(f"   Relatório completo salvo: {report_path.name}")


def _save_training_json(
    df: pd.DataFrame, metrics: Dict, train_info: Dict, output_dir: Path
):
    """Save training data as JSON"""
    import json

    # Custom JSON encoder for numpy and bool types
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return super().default(obj)

    data = {
        "metadata": train_info,
        "metrics": metrics,
        "episodes": df.to_dict("records"),
    }

    json_path = output_dir / "dados_treinamento.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

    print(f"   Dados JSON salvos: {json_path.name}")
