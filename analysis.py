"""
Analise do experimento v4.

Figuras globais:
- fig_01: sensibilidade dos parametros
- fig_02: heatmaps de aderencia mediana por tipo de linha
- fig_03: matriz de taxa de conclusao (config x fator)
- fig_06: aderencia vs MAE do shape
- fig_07: comprimento reconstruido vs shape
- fig_08: escalabilidade de tempo
- fig_10: motivos de parada por config (barras empilhadas)
- fig_11: p95 vs media das distancias (cauda)
- fig_12: aderencia vs volume de dados de entrada
- fig_13: compressao de pontos (rota vs entrada)

Figuras por linha (per_line/) e em mapa vem dos modulos auxiliares.
Tabela: best_configs_v4.csv

Uso:
    cd experimento_v4
    python analysis.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from analysis_common import (
    load_results, ordered_configs, ORDEM_CONFIGS, LINE_COLORS,
)
from analysis_maps import (
    ANALYSIS_DIR,
    make_fig_04_sensitivity_grid,
    make_all_sensitivity_grids,
    make_fig_05_success_gallery,
    make_success_images,
    make_best_route_images,
    make_fig_before_after,
    make_fig_14_direction_separation,
    shape_length_km,
)
from analysis_perline import (
    make_per_line_sensitivity,
    make_per_line_heatmaps,
)
from core.shapes import load_shape_models

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

IEEE_DOUBLE = 7.16
IEEE_SINGLE = 3.5


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


# ============================================================
# FIG 01 — SENSIBILIDADE DOS PARAMETROS
# ============================================================
def make_fig_01(df, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(IEEE_DOUBLE, 2.4),
                             constrained_layout=True)

    sns.boxplot(
        data=df, x="angular_samples", y="coverage_pct",
        hue="angular_samples", legend=False, palette="Blues",
        linewidth=0.7, fliersize=1.2, ax=axes[0],
    )
    axes[0].set_xlabel("amostras angulares")
    axes[0].set_ylabel("aderencia aos dados (%)")
    axes[0].set_title("(a) Amostras angulares", fontsize=8)

    sns.boxplot(
        data=df, x="meters", y="coverage_pct",
        hue="meters", legend=False, palette="Greens",
        linewidth=0.7, fliersize=1.2, ax=axes[1],
    )
    axes[1].set_xlabel("raio minimo (m)")
    axes[1].set_ylabel("")
    axes[1].set_title("(b) Raio minimo", fontsize=8)

    sns.boxplot(
        data=df, x="increase_meters", y="coverage_pct",
        hue="increase_meters", legend=False, palette="Oranges",
        linewidth=0.7, fliersize=1.2, ax=axes[2],
    )
    axes[2].set_xlabel("fator de aumento do raio")
    axes[2].set_ylabel("")
    axes[2].set_title("(c) Fator de aumento", fontsize=8)

    rate = (
        df.groupby("increase_meters")["completed"]
        .mean().mul(100).reset_index()
    )
    sns.barplot(
        data=rate, x="increase_meters", y="completed",
        hue="increase_meters", legend=False, palette="Oranges",
        ax=axes[3],
    )
    axes[3].set_xlabel("fator de aumento do raio")
    axes[3].set_ylabel("conclusoes (%)")
    axes[3].set_title("(d) Taxa de conclusao", fontsize=8)
    axes[3].set_ylim(0, 118)
    for container in axes[3].containers:
        axes[3].bar_label(container, fmt="%.0f", fontsize=6.5, padding=2)

    for ax in axes:
        _clean(ax)
        ax.grid(False, axis="x")
        if ax.get_ylabel().startswith("aderencia"):
            ax.set_ylim(0, 105)

    fig.savefig(out_dir / "fig_01_param_sensitivity.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_01] ok")


# ============================================================
# FIG 02 — HEATMAPS POR TIPO DE LINHA
# ============================================================
def make_fig_02(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE, 2.7),
                             constrained_layout=True)

    for ax, (tipo, titulo) in zip(
        axes,
        [("ida/volta", "(a) Linhas ida/volta"),
         ("circular", "(b) Linhas circulares")],
    ):
        sub = df[df["tipo"] == tipo]
        table = sub.pivot_table(
            index="meters", columns="increase_meters",
            values="coverage_pct", aggfunc="median",
        )
        sns.heatmap(
            table, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
            annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
            cbar=(tipo == "circular"),
            cbar_kws={"label": "aderencia mediana (%)", "shrink": 0.9},
            annot_kws={"size": 7},
        )
        ax.set_xlabel("fator de aumento", fontsize=8)
        ax.set_ylabel("raio minimo (m)", fontsize=8)
        ax.set_title(titulo, fontsize=8.5)
        ax.tick_params(labelsize=7)

    fig.suptitle("Aderencia mediana aos dados GPS (%)", fontsize=9.5)
    fig.savefig(out_dir / "fig_02_coverage_heatmaps.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_02] ok")


# ============================================================
# FIG 03 — MATRIZ DE CONCLUSAO
# ============================================================
def make_fig_03(df, out_dir):
    rows = []
    for cfg_name, g in df.groupby("config"):
        for inc, gg in g.groupby("increase_meters"):
            rows.append({
                "config": cfg_name,
                "inc": inc,
                "rate": 100.0 * gg["completed"].mean(),
            })
    mat = pd.DataFrame(rows).pivot(
        index="config", columns="inc", values="rate"
    ).reindex([o for o in ORDEM_CONFIGS if o in df["config"].unique()])

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.4, 6.4),
                           constrained_layout=True)
    sns.heatmap(
        mat, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
        annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
        cbar_kws={"label": "conclusoes (%)", "shrink": 0.5},
        annot_kws={"size": 7},
    )
    ax.set_xlabel("fator de aumento do raio", fontsize=8)
    ax.set_ylabel("")
    ax.set_title("Taxa de conclusao do trajeto por config", fontsize=8.5)
    ax.tick_params(labelsize=7)

    fig.savefig(out_dir / "fig_03_completion_matrix.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_03] ok")


# ============================================================
# FIG 06 — ADERENCIA vs MAE DO SHAPE
# ============================================================
def make_fig_06(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.0),
                           constrained_layout=True)

    for linha, g in df.groupby("linha_s"):
        ax.scatter(
            g["mae_shape_m"], g["coverage_pct"],
            s=14, alpha=0.65, color=LINE_COLORS.get(linha, "#666666"),
            label=f"Linha {linha}", edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("MAE vs shape oficial (m, escala log)")
    ax.set_ylabel("aderencia aos dados GPS (%)")
    ax.set_title(
        "Aderencia aos dados vs divergencia do shape\n"
        "canto superior direito = shape defeituoso",
        fontsize=8.5,
    )
    _clean(ax)

    ncols = 4
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center", ncol=1,
        fontsize=6.0, frameon=False, title="Linha", title_fontsize=7,
    )

    fig.savefig(out_dir / "fig_06_adherence_vs_shape_mae.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_06] ok")


# ============================================================
# FIG 07 — COMPRIMENTO RECONSTRUIDO vs SHAPE
# ============================================================
def make_fig_07(df, shape_models, out_dir):
    df = df.copy()
    df["shape_km"] = df.apply(
        lambda r: shape_length_km(shape_models, r["linha_s"], r["shape_id"]),
        axis=1,
    )
    sub = df.dropna(subset=["shape_km"])

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.4),
                           constrained_layout=True)

    lim = (0, sub["shape_km"].max() * 1.9)
    ax.plot(lim, lim, linestyle="--", color="#888", lw=0.9,
            label="reconstrucao = shape")

    for cfg_name, g in sub.groupby("config"):
        cor = LINE_COLORS.get(cfg_name.split()[0], "#666")
        done = g[g["completed"]]
        part = g[~g["completed"]]
        if not done.empty:
            ax.scatter(
                done["shape_km"], done["route_length_km"],
                s=16, alpha=0.8, color=cor, label=cfg_name,
                edgecolors="none",
            )
        if not part.empty:
            ax.scatter(
                part["shape_km"], part["route_length_km"],
                s=18, alpha=0.45, color=cor, marker="x",
            )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("comprimento do shape oficial (km)")
    ax.set_ylabel("comprimento reconstruido (km)")
    ax.set_title(
        "Comprimento da rota: cheios = completos, x = incompletos\n"
        "acima da diagonal = ziguezague/excursoes",
        fontsize=8.5,
    )
    _clean(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center",
        fontsize=5.2, frameon=False, title="Config", title_fontsize=7,
        ncol=2,
    )

    fig.savefig(out_dir / "fig_07_route_length_vs_shape.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_07] ok")


# ============================================================
# FIG 08 — ESCALABILIDADE DE TEMPO
# ============================================================
def make_fig_08(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE, 2.7),
                             constrained_layout=True)

    pal = {90: "#9ECAE1", 180: "#4292C6", 360: "#084594"}
    for smp, g in df.groupby("angular_samples"):
        axes[0].scatter(
            g["n_input_points"] / 1000.0, g["runtime_s"],
            s=10, alpha=0.55, color=pal[smp], label=f"{smp}",
            edgecolors="none",
        )
    axes[0].set_xlabel("pontos GPS de entrada (milhares)")
    axes[0].set_ylabel("tempo de execucao (s)")
    axes[0].set_title("(a) Tempo vs volume de dados", fontsize=8)

    sns.boxplot(
        data=df, x="angular_samples", y="runtime_s",
        hue="angular_samples", legend=False, palette=pal,
        linewidth=0.7, fliersize=1.2, ax=axes[1],
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("amostras angulares")
    axes[1].set_ylabel("tempo de execucao (s, log)")
    axes[1].set_title("(b) Custo por resolucao angular", fontsize=8)

    for ax in axes:
        _clean(ax)
        ax.grid(False, axis="x")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncol=3,
        fontsize=7, frameon=False, title="amostras angulares",
        title_fontsize=7,
    )

    fig.savefig(out_dir / "fig_08_runtime_scaling.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_08] ok")


# ============================================================
# FIG 10 — MOTIVOS DE PARADA POR CONFIG
# ============================================================
REASON_MAP = {
    "chegada_destino": "chegada ao destino",
    "loop_fechado": "loop fechado",
    "passos_sem_dados": "sem dados (raio max)",
    "limite_comprimento": "limite de comprimento",
    "raio_max_excedido": "raio max excedido",
    "sem_pontos_ativos": "sem pontos ativos",
    "max_iterations": "max iteracoes",
}

REASON_COLORS = {
    "chegada ao destino": "#2CA02C",
    "loop fechado": "#90D892",
    "sem dados (raio max)": "#D62728",
    "limite de comprimento": "#FF7F0E",
    "raio max excedido": "#8C564B",
    "sem pontos ativos": "#7F7F7F",
    "max iteracoes": "#BBBBBB",
}


def make_fig_10(df, out_dir):
    d = df.copy()
    d["motivo"] = d["stop_reason"].map(REASON_MAP).fillna("outros")

    counts = (
        d.groupby(["config", "motivo"]).size()
        .reset_index(name="n")
        .pivot(index="config", columns="motivo", values="n")
        .fillna(0)
    )
    counts = counts.reindex(
        [o for o in ORDEM_CONFIGS if o in counts.index]
    )

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 2.4, 6.4),
                           constrained_layout=True)

    left = np.zeros(len(counts))
    cats = [c for c in counts.columns if c in REASON_COLORS]
    cats += [c for c in counts.columns if c not in REASON_COLORS]
    y = np.arange(len(counts))
    for cat in cats:
        vals = counts[cat].to_numpy()
        ax.barh(
            y, vals, left=left, color=REASON_COLORS.get(cat, "#999"),
            label=cat, height=0.62, edgecolor="white", linewidth=0.4,
        )
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("numero de execucoes")
    ax.set_title("Motivos de parada por config "
                 "(36 execucoes por config, 72 nas circulares)",
                 fontsize=8.5)

    _clean(ax)
    ax.grid(False, axis="x")
    ax.grid(True, axis="x", alpha=0.35)

    fig.legend(
        loc="outside lower center", ncol=3, fontsize=6.5, frameon=False,
    )

    fig.savefig(out_dir / "fig_10_stop_reasons.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_10] ok")


# ============================================================
# FIG 11 — P95 vs MEDIA DAS DISTANCIAS
# ============================================================
def make_fig_11(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.1),
                           constrained_layout=True)

    for linha, g in df.groupby("linha_s"):
        done = g[g["completed"]]
        part = g[~g["completed"]]
        cor = LINE_COLORS.get(linha, "#666")
        if not done.empty:
            ax.scatter(
                done["mean_gps_dist_m"], done["p95_gps_dist_m"],
                s=14, alpha=0.75, color=cor, label=f"{linha}",
                edgecolors="none",
            )
        if not part.empty:
            ax.scatter(
                part["mean_gps_dist_m"], part["p95_gps_dist_m"],
                s=16, alpha=0.4, color=cor, marker="x",
            )

    xs = np.linspace(0.5, df["mean_gps_dist_m"].max() * 1.5, 10)
    ax.plot(xs, xs, linestyle="--", color="#888", lw=0.9,
            label="p95 = media")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distancia media GPS -> rota (m, log)")
    ax.set_ylabel("p95 da distancia GPS -> rota (m, log)")
    ax.set_title(
        "Cauda das distancias: circulos = completos, x = incompletos\n"
        "p95 >> media = fugas localizadas da rota",
        fontsize=8.5,
    )
    _clean(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center",
        fontsize=6.0, frameon=False, title="Linha", title_fontsize=7,
    )

    fig.savefig(out_dir / "fig_11_p95_vs_mean.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_11] ok")


# ============================================================
# FIG 12 — ADERENCIA vs VOLUME DE DADOS
# ============================================================
def make_fig_12(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.1),
                           constrained_layout=True)

    cores = {"ida/volta": "#1F77B4", "circular": "#9467BD"}
    for tipo, g in df.groupby("tipo"):
        done = g[g["completed"]]
        part = g[~g["completed"]]
        if not done.empty:
            ax.scatter(
                done["n_input_points"], done["coverage_pct"],
                s=15, alpha=0.75, color=cores[tipo], label=f"{tipo} (ok)",
                edgecolors="none",
            )
        if not part.empty:
            ax.scatter(
                part["n_input_points"], part["coverage_pct"],
                s=17, alpha=0.45, color=cores[tipo], marker="x",
                label=f"{tipo} (parcial)",
            )

    ax.set_xscale("log")
    ax.set_xlabel("pontos GPS de entrada (escala log)")
    ax.set_ylabel("aderencia aos dados (%)")
    ax.set_title(
        "Aderencia vs volume de dados de entrada\n"
        "volume alto nao basta: a separacao de sentido e decisiva",
        fontsize=8.5,
    )
    _clean(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center",
        fontsize=6.5, frameon=False,
    )

    fig.savefig(out_dir / "fig_12_adherence_vs_volume.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_12] ok")


# ============================================================
# FIG 13 — COMPRESSAO DE PONTOS
# ============================================================
def make_fig_13(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.1),
                           constrained_layout=True)

    lim = (df["n_input_points"].min() * 0.7,
           df["n_input_points"].max() * 2.0)
    ax.plot(lim, lim, linestyle="--", color="#888", lw=0.9,
            label="rota = entrada")

    for linha, g in df.groupby("linha_s"):
        ax.scatter(
            g["n_input_points"], g["n_route_points"],
            s=13, alpha=0.6, color=LINE_COLORS.get(linha, "#666"),
            label=linha, edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pontos GPS de entrada (log)")
    ax.set_ylabel("pontos da rota gerada (log)")
    ax.set_title(
        "Compressao: a rota usa poucos pontos para\n"
        "representar dezenas de milhares de pings",
        fontsize=8.5,
    )
    _clean(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside right center",
        fontsize=6.0, frameon=False, title="Linha", title_fontsize=7,
    )

    fig.savefig(out_dir / "fig_13_data_compression.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_13] ok")


# ============================================================
# TABELA — MELHORES CONFIGS
# ============================================================
def save_best_configs(df, out_dir):
    rows = []
    for linha, sentido, nb in ordered_configs(df, with_buses=True):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        if nb is not None:
            sub = sub[sub["n_buses"] == nb]
        if sub.empty:
            continue
        done = sub[sub["completed"]]
        pool = done if not done.empty else sub
        r = pool.loc[pool["coverage_pct"].idxmax()]
        rows.append({
            "linha": linha,
            "sentido": sentido,
            "n_buses": int(r["n_buses"]),
            "completed": bool(r["completed"]),
            "coverage_pct": round(r["coverage_pct"], 2),
            "mean_gps_dist_m": round(r["mean_gps_dist_m"], 2),
            "p95_gps_dist_m": round(r["p95_gps_dist_m"], 2),
            "mae_shape_m": round(r["mae_shape_m"], 2),
            "route_length_km": round(r["route_length_km"], 2),
            "angular_samples": int(r["angular_samples"]),
            "meters": r["meters"],
            "increase_meters": r["increase_meters"],
            "runtime_s": round(r["runtime_s"], 2),
            "run_id": r["run_id"],
        })
    out = pd.DataFrame(rows).sort_values(["linha", "sentido", "n_buses"])
    out.to_csv(out_dir / "best_configs_v4.csv", index=False)
    print(f"[tabela] best_configs_v4.csv ({len(out)} configs)")


# ============================================================
# MAIN
# ============================================================
def main():
    out_dir = Path(ANALYSIS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results()
    print(f"Execucoes: {len(df):,} | completas: {int(df['completed'].sum())}")

    print("Carregando shapes...")
    shape_models = load_shape_models(cfg.SHAPE_XZ_PATH)

    print("Figuras estatisticas globais...")
    make_fig_01(df, out_dir)
    make_fig_02(df, out_dir)
    make_fig_03(df, out_dir)
    make_fig_06(df, out_dir)
    make_fig_07(df, shape_models, out_dir)
    make_fig_08(df, out_dir)
    make_fig_10(df, out_dir)
    make_fig_11(df, out_dir)
    make_fig_12(df, out_dir)
    make_fig_13(df, out_dir)

    print("Figuras individuais por linha...")
    make_per_line_sensitivity()
    make_per_line_heatmaps()

    print("Figuras em mapa (carrega dados GPS, pode demorar)...")
    make_fig_04_sensitivity_grid(df, shape_models, out_dir)
    make_all_sensitivity_grids(df, shape_models)
    make_fig_05_success_gallery(df, shape_models, out_dir)
    make_success_images(df, shape_models, out_dir)
    make_best_route_images(df, shape_models, out_dir)
    make_fig_before_after(df, shape_models, out_dir)
    make_fig_14_direction_separation(df, shape_models, out_dir)

    save_best_configs(df, out_dir)
    print(f"\nFinalizado. Analises em: {out_dir}")


if __name__ == "__main__":
    main()
