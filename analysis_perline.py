"""
Figuras individuais por linha (pasta per_line/) do experimento v4.

Para cada config (linha x sentido, circulares agrupando os onibus):
- sensitivity_{linha}_{sentido}.png: 2x2 com boxplots de aderencia por
  cada parametro + taxa de conclusao por fator de aumento
- heatmap_{linha}_{sentido}.png: 2 heatmaps (aderencia mediana e taxa de
  conclusao) no grid raio minimo x fator de aumento
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from analysis_common import load_results, ordered_configs

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

PER_LINE_DIR = cfg.ANALYSIS_DIR / "per_line"


# ============================================================
# SENSIBILIDADE POR LINHA (2x2)
# ============================================================
def make_per_line_sensitivity(out_dir: Path = PER_LINE_DIR):
    df = load_results()
    out_dir.mkdir(parents=True, exist_ok=True)

    for linha, sentido, _ in ordered_configs(df):
        g = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        if g.empty:
            continue

        tag = "(todos os onibus)" if sentido == "circular" else ""
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6),
                                 constrained_layout=True)

        sns.boxplot(
            data=g, x="angular_samples", y="coverage_pct",
            hue="angular_samples", legend=False, palette="Blues",
            linewidth=0.7, fliersize=1.2, ax=axes[0, 0],
        )
        axes[0, 0].set_title("(a) Amostras angulares", fontsize=8)
        axes[0, 0].set_xlabel("amostras")
        axes[0, 0].set_ylabel("aderencia (%)")

        sns.boxplot(
            data=g, x="meters", y="coverage_pct",
            hue="meters", legend=False, palette="Greens",
            linewidth=0.7, fliersize=1.2, ax=axes[0, 1],
        )
        axes[0, 1].set_title("(b) Raio minimo", fontsize=8)
        axes[0, 1].set_xlabel("raio min (m)")
        axes[0, 1].set_ylabel("")

        sns.boxplot(
            data=g, x="increase_meters", y="coverage_pct",
            hue="increase_meters", legend=False, palette="Oranges",
            linewidth=0.7, fliersize=1.2, ax=axes[1, 0],
        )
        axes[1, 0].set_title("(c) Fator de aumento", fontsize=8)
        axes[1, 0].set_xlabel("fator de aumento")
        axes[1, 0].set_ylabel("aderencia (%)")

        rate = (
            g.groupby("increase_meters")["completed"]
            .mean().mul(100).reset_index()
        )
        sns.barplot(
            data=rate, x="increase_meters", y="completed",
            hue="increase_meters", legend=False, palette="Oranges",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("(d) Taxa de conclusao", fontsize=8)
        axes[1, 1].set_xlabel("fator de aumento")
        axes[1, 1].set_ylabel("conclusoes (%)")
        axes[1, 1].set_ylim(0, 115)
        for container in axes[1, 1].containers:
            axes[1, 1].bar_label(container, fmt="%.0f", fontsize=6.5)

        for ax in axes.flat:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.35)
            ax.grid(False, axis="x")
            ax.tick_params(labelsize=7)
            if ax.get_ylabel().startswith("aderencia"):
                ax.set_ylim(0, 105)

        fig.suptitle(
            f"Linha {linha} ({sentido}) {tag} — "
            f"sensibilidade dos parametros ({len(g)} execucoes)",
            fontsize=9.5,
        )
        fig.savefig(out_dir / f"sensitivity_{linha}_{sentido}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[sensitivity {linha}/{sentido}] ok")


# ============================================================
# HEATMAPS POR LINHA (1x2)
# ============================================================
def make_per_line_heatmaps(out_dir: Path = PER_LINE_DIR):
    df = load_results()
    out_dir.mkdir(parents=True, exist_ok=True)

    for linha, sentido, _ in ordered_configs(df):
        g = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        if g.empty:
            continue

        cov = g.pivot_table(
            index="meters", columns="increase_meters",
            values="coverage_pct", aggfunc="median",
        )
        comp = (
            g.groupby(["meters", "increase_meters"])["completed"]
            .mean().mul(100).reset_index()
            .pivot(index="meters", columns="increase_meters",
                   values="completed")
        )

        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8),
                                 constrained_layout=True)

        sns.heatmap(
            cov, ax=axes[0], cmap="RdYlGn", vmin=0, vmax=100,
            annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "%", "shrink": 0.85},
            annot_kws={"size": 7},
        )
        axes[0].set_title("(a) Aderencia mediana (%)", fontsize=8.5)

        sns.heatmap(
            comp, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=100,
            annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "%", "shrink": 0.85},
            annot_kws={"size": 7},
        )
        axes[1].set_title("(b) Taxa de conclusao (%)", fontsize=8.5)

        for ax in axes:
            ax.set_xlabel("fator de aumento", fontsize=8)
            ax.set_ylabel("raio min (m)", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Linha {linha} ({sentido}) — "
            f"raio minimo x fator de aumento",
            fontsize=9.5,
        )
        fig.savefig(out_dir / f"heatmap_{linha}_{sentido}.png",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[heatmap {linha}/{sentido}] ok")
