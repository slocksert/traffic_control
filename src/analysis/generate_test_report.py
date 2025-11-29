"""
Automatic Test Report Generator

Generates comprehensive reports with tables, graphs, and critical analysis
after test runs complete
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime
import scipy.stats as stats

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def generate_test_report(
    df: pd.DataFrame, run_info: Dict, output_dir: str = "test_results"
):
    """
    Generate complete test report with graphs, tables, and analysis

    Args:
        df: DataFrame with episode results
        run_info: Dictionary with test run information
        output_dir: Directory to save results

    Returns:
        Path to generated report directory
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = run_info.get("agent_type", "unknown")
    report_dir = Path(output_dir) / f"report_{agent_type}_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GERANDO RELATÓRIO COMPLETO DE RESULTADOS DE TESTE")
    print("=" * 80)

    # 1. Calculate metrics
    print("\n1. Calculando métricas estatísticas...")
    metrics = _calculate_metrics(df)

    # 2. Generate tables
    print("2. Gerando tabelas comparativas...")
    _generate_tables(df, metrics, report_dir)

    # 3. Generate graphs
    print("3. Gerando gráficos...")
    _generate_graphs(df, metrics, report_dir)

    # 4. Generate critical analysis
    print("4. Gerando análise crítica...")
    analysis_text = _generate_critical_analysis(df, metrics, run_info)

    # 5. Create complete text report
    print("5. Criando relatório completo...")
    _create_complete_report(df, metrics, analysis_text, run_info, report_dir)

    # 6. Save data as JSON
    print("6. Salvando dados em JSON...")
    _save_json_data(df, metrics, run_info, report_dir)

    print(f"\n{'=' * 80}")
    print("RELATÓRIO GERADO COM SUCESSO!")
    print(f"{'=' * 80}")
    print(f"\nDiretório: {report_dir}")
    print("\nArquivos gerados:")
    for file in sorted(report_dir.glob("*")):
        if file.is_file():
            print(f"  - {file.name}")

    return report_dir


def _calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistical metrics"""
    metrics = {
        "total_episodes": len(df),
        # Reward metrics
        "reward": {
            "mean": float(df["total_reward"].mean()),
            "std": float(df["total_reward"].std()),
            "min": float(df["total_reward"].min()),
            "max": float(df["total_reward"].max()),
            "median": float(df["total_reward"].median()),
            "q1": float(df["total_reward"].quantile(0.25)),
            "q3": float(df["total_reward"].quantile(0.75)),
        },
        # Waiting time metrics
        "waiting_time": {
            "mean": float(df["avg_waiting_time"].mean()),
            "std": float(df["avg_waiting_time"].std()),
            "min": float(df["avg_waiting_time"].min()),
            "max": float(df["avg_waiting_time"].max()),
            "median": float(df["avg_waiting_time"].median()),
        },
        # Throughput metrics
        "throughput": {
            "mean": float(df["total_throughput"].mean()),
            "std": float(df["total_throughput"].std()),
            "min": int(df["total_throughput"].min()),
            "max": int(df["total_throughput"].max()),
            "total": int(df["total_throughput"].sum()),
        },
        # Speed metrics
        "speed": {
            "mean": float(df["avg_speed"].mean()),
            "std": float(df["avg_speed"].std()),
            "min": float(df["avg_speed"].min()),
            "max": float(df["avg_speed"].max()),
        },
        # Episode length
        "steps": {
            "mean": float(df["steps"].mean()),
            "std": float(df["steps"].std()),
            "min": int(df["steps"].min()),
            "max": int(df["steps"].max()),
        },
        # Vehicles completed
        "vehicles": {
            "mean": float(df["vehicles_completed"].mean()),
            "total": int(df["vehicles_completed"].sum()),
            "std": float(df["vehicles_completed"].std()),
            "min": int(df["vehicles_completed"].min()),
            "max": int(df["vehicles_completed"].max()),
        },
    }

    # Coefficient of variation
    for metric_name in ["reward", "waiting_time", "throughput", "speed"]:
        mean_val = metrics[metric_name]["mean"]
        std_val = metrics[metric_name]["std"]
        if mean_val != 0:
            metrics[metric_name]["cv"] = float(abs(std_val / mean_val))
        else:
            metrics[metric_name]["cv"] = 0.0

    # Error metrics (RMSE)
    mean_reward = metrics["reward"]["mean"]
    residuals = df["total_reward"] - mean_reward
    metrics["error"] = {
        "rmse_reward": float(np.sqrt(np.mean(residuals**2))),
        "mae_reward": float(np.mean(np.abs(residuals))),
    }

    return metrics


def _generate_tables(df: pd.DataFrame, metrics: Dict, output_dir: Path):
    """Generate comparison tables"""
    # Metrics summary table
    table_data = []

    metric_configs = [
        ("Recompensa Total", "reward"),
        ("Tempo de Espera (s)", "waiting_time"),
        ("Throughput (veículos)", "throughput"),
        ("Velocidade Média (m/s)", "speed"),
        ("Tamanho do Episódio (steps)", "steps"),
        ("Veículos Completados", "vehicles"),
    ]

    for label, key in metric_configs:
        if key in metrics:
            m = metrics[key]
            row = {
                "Métrica": label,
                "Média": f"{m['mean']:.2f}",
                "Desvio Padrão": f"{m['std']:.2f}",
                "Mínimo": f"{m.get('min', 0):.2f}",
                "Máximo": f"{m.get('max', 0):.2f}",
                "Mediana": f"{m.get('median', m['mean']):.2f}",
            }
            if "cv" in m:
                row["CV"] = f"{m['cv']:.3f}"
            table_data.append(row)

    table_df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = output_dir / "tabela_metricas.csv"
    table_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   Tabela CSV salva: {csv_path.name}")

    # Save detailed episode data
    episodes_path = output_dir / "episodios_detalhados.csv"
    df.to_csv(episodes_path, index=False, encoding="utf-8-sig")
    print(f"   Dados detalhados salvos: {episodes_path.name}")


def _generate_graphs(df: pd.DataFrame, metrics: Dict, output_dir: Path):
    """Generate all required graphs"""
    colors = sns.color_palette("husl", 8)

    # Graph 1: Performance Overview (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Visão Geral de Desempenho - Resultados do Teste",
        fontsize=16,
        fontweight="bold",
    )

    episodes = df["episode_number"].values + 1

    # 1.1 Rewards over episodes
    ax = axes[0, 0]
    ax.plot(
        episodes,
        df["total_reward"],
        "o-",
        color=colors[0],
        alpha=0.6,
        label="Recompensa",
    )
    ax.axhline(
        df["total_reward"].mean(),
        color="red",
        linestyle="--",
        label=f'Média: {df["total_reward"].mean():.2f}',
    )
    ax.fill_between(
        episodes,
        df["total_reward"].mean() - df["total_reward"].std(),
        df["total_reward"].mean() + df["total_reward"].std(),
        alpha=0.2,
        color="red",
    )
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa Total")
    ax.set_title("Recompensas por Episódio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.2 Waiting time distribution
    ax = axes[0, 1]
    ax.hist(
        df["avg_waiting_time"], bins=15, color=colors[1], alpha=0.7, edgecolor="black"
    )
    ax.axvline(
        df["avg_waiting_time"].mean(),
        color="red",
        linestyle="--",
        label=f'Média: {df["avg_waiting_time"].mean():.2f}s',
    )
    ax.axvline(
        df["avg_waiting_time"].median(),
        color="green",
        linestyle=":",
        label=f'Mediana: {df["avg_waiting_time"].median():.2f}s',
    )
    ax.set_xlabel("Tempo Médio de Espera (s)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição do Tempo de Espera")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.3 Throughput over episodes
    ax = axes[0, 2]
    ax.plot(episodes, df["total_throughput"], "o-", color=colors[2], alpha=0.6)
    ax.axhline(
        df["total_throughput"].mean(),
        color="red",
        linestyle="--",
        label=f'Média: {df["total_throughput"].mean():.1f}',
    )
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Throughput Total (veículos)")
    ax.set_title("Throughput por Episódio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.4 Average speed over episodes
    ax = axes[1, 0]
    ax.plot(episodes, df["avg_speed"], "s-", color=colors[3], alpha=0.6)
    ax.axhline(
        df["avg_speed"].mean(),
        color="red",
        linestyle="--",
        label=f'Média: {df["avg_speed"].mean():.2f} m/s',
    )
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Velocidade Média (m/s)")
    ax.set_title("Velocidade Média por Episódio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.5 Box plots comparison (normalized)
    ax = axes[1, 1]
    box_data = [df["total_reward"], df["avg_waiting_time"], df["avg_speed"]]
    box_labels = ["Recompensa", "Tempo Espera (s)", "Velocidade (m/s)"]

    normalized_data = []
    for data in box_data:
        if data.std() != 0:
            normalized_data.append((data - data.mean()) / data.std())
        else:
            normalized_data.append(data - data.mean())

    bp = ax.boxplot(normalized_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Valor Normalizado (z-score)")
    ax.set_title("Distribuição das Métricas (Normalizado)")
    ax.grid(True, alpha=0.3)

    # 1.6 Episode length vs reward scatter
    ax = axes[1, 2]
    scatter = ax.scatter(
        df["steps"],
        df["total_reward"],
        c=df["avg_waiting_time"],
        cmap="RdYlGn_r",
        s=100,
        alpha=0.6,
        edgecolors="black",
    )
    ax.set_xlabel("Tamanho do Episódio (steps)")
    ax.set_ylabel("Recompensa Total")
    ax.set_title("Tamanho do Episódio vs Recompensa")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Tempo Espera (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    graph_path = output_dir / "grafico_desempenho_geral.png"
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    print(f"   Gráfico gerado: {graph_path.name}")
    plt.close()

    # Graph 2: Error Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Análise de Métricas de Erro", fontsize=16, fontweight="bold")

    # 2.1 Residuals plot
    ax = axes[0]
    mean_reward = df["total_reward"].mean()
    residuals = df["total_reward"] - mean_reward

    ax.scatter(episodes, residuals, alpha=0.6, color=colors[0])
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Média")
    ax.axhline(residuals.std(), color="orange", linestyle=":", label="+1 Desvio Padrão")
    ax.axhline(
        -residuals.std(), color="orange", linestyle=":", label="-1 Desvio Padrão"
    )
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Resíduo (Real - Média)")
    ax.set_title(f"Resíduos da Recompensa\nRMSE: {np.sqrt(np.mean(residuals**2)):.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.2 Error distribution
    ax = axes[1]
    ax.hist(
        residuals, bins=15, color=colors[1], alpha=0.7, edgecolor="black", density=True
    )

    # Fit normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        "r-",
        linewidth=2,
        label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})",
    )
    ax.set_xlabel("Valor do Resíduo")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição dos Resíduos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    error_path = output_dir / "grafico_metricas_erro.png"
    plt.savefig(error_path, dpi=300, bbox_inches="tight")
    print(f"   Gráfico gerado: {error_path.name}")
    plt.close()


def _generate_critical_analysis(df: pd.DataFrame, metrics: Dict, run_info: Dict) -> str:
    """Generate critical analysis text"""
    analysis = []

    analysis.append("=" * 80)
    analysis.append("ANÁLISE CRÍTICA DOS RESULTADOS DE TESTE")
    analysis.append("=" * 80)
    analysis.append("")

    # 1. Overall performance
    analysis.append("1. DESEMPENHO GERAL")
    analysis.append("-" * 40)

    reward_mean = metrics["reward"]["mean"]
    reward_std = metrics["reward"]["std"]
    reward_cv = metrics["reward"]["cv"]

    if reward_mean > 0:
        analysis.append(
            f"✓ Recompensa média positiva ({reward_mean:.2f}):\n"
            f"              Agente demonstra aprendizado efetivo"
        )
    else:
        analysis.append(
            f"⚠ Recompensa média negativa ({reward_mean:.2f}):\n"
            f"              Agente pode precisar de mais treinamento"
        )

    if reward_cv < 0.3:
        analysis.append(
            f"✓ Baixa variabilidade (CV={reward_cv:.3f}):\n"
            f"              Desempenho consistente entre episódios"
        )
    elif reward_cv < 0.6:
        analysis.append(
            f"~ Variabilidade moderada (CV={reward_cv:.3f}):\n"
            f"              Alguma inconsistência nas decisões"
        )
    else:
        analysis.append(
            f"⚠ Alta variabilidade (CV={reward_cv:.3f}): Desempenho instável"
        )

    analysis.append("")

    # 2. Traffic efficiency
    analysis.append("2. EFICIÊNCIA DO FLUXO DE TRÁFEGO")
    analysis.append("-" * 40)

    wait_mean = metrics["waiting_time"]["mean"]
    if wait_mean < 30:
        analysis.append(
            f"✓ Excelente tempo de espera ({wait_mean:.2f}s média):\n"
            f"              Congestionamento mínimo"
        )
    elif wait_mean < 60:
        analysis.append(
            f"~ Tempo de espera aceitável ({wait_mean:.2f}s média):\n"
            f"              Fluxo moderado"
        )
    else:
        analysis.append(
            f"⚠ Tempo de espera elevado ({wait_mean:.2f}s média):\n"
            f"              Problemas significativos de congestionamento"
        )

    speed_mean = metrics["speed"]["mean"]
    if speed_mean > 8:
        analysis.append(
            f"✓ Boa velocidade média ({speed_mean:.2f} m/s):\n"
            f"              Movimento eficiente do tráfego"
        )
    elif speed_mean > 5:
        analysis.append(
            f"~ Velocidade moderada ({speed_mean:.2f} m/s):\n"
            f"              Espaço para melhoria"
        )
    else:
        analysis.append(
            f"⚠ Velocidade baixa ({speed_mean:.2f} m/s):\n"
            f"              Tráfego lento"
        )

    throughput_total = metrics["throughput"]["total"]
    analysis.append(f"  Total de veículos processados: {throughput_total}")
    analysis.append(f"  Média por episódio: {metrics['throughput']['mean']:.1f}")

    analysis.append("")

    # 3. Statistical reliability
    analysis.append("3. CONFIABILIDADE ESTATÍSTICA")
    analysis.append("-" * 40)

    n_episodes = metrics["total_episodes"]
    analysis.append(f"  Número de episódios de teste: {n_episodes}")

    if n_episodes >= 30:
        analysis.append("✓ Tamanho amostral suficiente para significância estatística")
    elif n_episodes >= 10:
        analysis.append(
            "~ Tamanho amostral moderado, considere mais episódios para robustez"
        )
    else:
        analysis.append(
            "⚠ Resultados podem não ser estatisticamente significativos"
        )

    # Confidence interval (95%)
    ci_margin = 1.96 * (reward_std / np.sqrt(n_episodes))
    ci_lower = reward_mean - ci_margin
    ci_upper = reward_mean + ci_margin
    analysis.append(f"  IC 95% para recompensa média: [{ci_lower:.2f}, {ci_upper:.2f}]")

    # Error metrics
    rmse = metrics["error"]["rmse_reward"]
    mae = metrics["error"]["mae_reward"]
    analysis.append(f"  RMSE (recompensa): {rmse:.2f}")
    analysis.append(f"  MAE (recompensa): {mae:.2f}")

    analysis.append("")

    # 4. Recommendations
    analysis.append("4. RECOMENDAÇÕES")
    analysis.append("-" * 40)

    if reward_mean < 0 or reward_cv > 0.5:
        analysis.append(
            "•Considerar episódios adicionais de treinamento para melhorar estabilidade"
        )
        analysis.append(
            "• Revisar hiperparâmetros (taxa de aprendizado, decay do epsilon)"
        )

    if wait_mean > 60:
        analysis.append(
            "• Otimizar função de recompensa para penalizar mais o tempo de espera"
        )
        analysis.append("• Investigar estratégias de temporização de fases")

    if speed_mean < 5:
        analysis.append("• Considerar técnicas de otimização de fluxo de tráfego")
        analysis.append("• Analisar padrões de congestionamento em pistas específicas")

    if reward_cv < 0.3 and reward_mean > 0:
        analysis.append(
            "• Agente mostra desempenho estável - pronto para testes de implantação"
        )
        analysis.append("• Considerar testes em cenários variados de tráfego")

    analysis.append("")
    analysis.append("=" * 80)

    return "\n".join(analysis)


def _create_complete_report(
    df: pd.DataFrame,
    metrics: Dict,
    analysis_text: str,
    run_info: Dict,
    output_dir: Path,
):
    """Create complete text report"""
    report_path = output_dir / "relatorio_completo.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SISTEMA DE CONTROLE DE TRÁFEGO - RELATÓRIO DE RESULTADOS DE TESTE\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tipo de Agente: {run_info.get('agent_type', 'Desconhecido')}\n")
        if run_info.get("model_path"):
            f.write(f"Modelo: {run_info['model_path']}\n")
        f.write("\n")

        # Configuration
        if run_info.get("config"):
            f.write("CONFIGURAÇÃO:\n")
            f.write("-" * 80 + "\n")
            for key, value in run_info["config"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Summary metrics table
        f.write("RESUMO DAS MÉTRICAS:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Métrica':<30} {'Média':<12} {'Desvio':<12} {'Min':<10} {'Max':<10}\n"
        )
        f.write("-" * 80 + "\n")

        metric_lines = [
            ("Recompensa Total", metrics["reward"]),
            ("Tempo de Espera (s)", metrics["waiting_time"]),
            ("Throughput (veículos)", metrics["throughput"]),
            ("Velocidade Média (m/s)", metrics["speed"]),
            ("Veículos Completados", metrics["vehicles"]),
        ]

        for label, m in metric_lines:
            f.write(
                f"{label:<30} {m['mean']:>11.2f} {m['std']:>11.2f}"
                f" {m['min']:>9.2f} {m['max']:>9.2f}\n"
            )

        f.write("\n\n")

        # Critical analysis
        f.write(analysis_text)
        f.write("\n\n")

        # Files generated
        f.write("ARQUIVOS GERADOS:\n")
        f.write("-" * 80 + "\n")
        for file in sorted(output_dir.glob("*")):
            if file.is_file() and file != report_path:
                f.write(f"  - {file.name}\n")

    print(f"   Relatório completo salvo: {report_path.name}")


def _save_json_data(df: pd.DataFrame, metrics: Dict, run_info: Dict, output_dir: Path):
    """Save complete data as JSON"""
    import json

    # Custom JSON encoder for numpy types
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
        "metadata": run_info,
        "metrics": metrics,
        "episodes": df.to_dict("records"),
    }

    json_path = output_dir / "dados_completos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)

    print(f"   Dados JSON salvos: {json_path.name}")
