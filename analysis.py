"""
Análise global do experimento v8.

Figuras (outputs/analysis/):
- fig_01: sensibilidade dos parâmetros (bandas 5/10/15 m)
- fig_02: heatmaps raio × fator por tipo de linha
- fig_03: matriz de conclusão (config × fator)
- fig_04: distribuição das bandas de aderência (5/10/15 m)
- fig_05: efeito do número de viagens (global)
- fig_06: escalabilidade de tempo
- fig_07: aderência vs MAE do shape
- fig_08: comprimento reconstruído vs shape
- fig_09: motivos de parada
- fig_10: p95 vs média das distâncias
- fig_11: compressão de pontos

Por linha: per_line/ (sensitivity + trip_effect) e mapas.
Tabela: best_configs_v10.csv

Uso:
    cd experimento_v8
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
    load_results, ordered_configs, ORDEM_LINHAS, LINE_COLORS,
    BANDS, BAND_COL, BAND_COLORS, best_row,
)
from analysis_perline import make_all_per_line, PERLINE_DIR
from analysis_maps import make_line_maps, make_gallery
from core.shapes import load_shape_models

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

IEEE_DOUBLE = 7.16
IEEE_SINGLE = 3.5
REF_BAND = "coverage_15m"


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


# ============================================================
# FIG 01 — SENSIBILIDADE DOS PARAMETROS
# ============================================================
def make_fig_01(df, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(IEEE_DOUBLE, 2.5),
                             constrained_layout=True)

    specs = [
        ("angular_samples", "amostras angulares", "Blues",
         "(a) Amostras angulares", "amostras angulares", "aderência \u2264 15 m (%)"),
        ("meters", "raio mínimo (m)", "Greens",
         "(b) Raio mínimo", "raio mínimo (m)", ""),
        ("increase_meters", "fator de escala", "Oranges",
         "(c) Fator de escala", "fator de escala", ""),
        ("n_trips", "nº de viagens", "Purples",
         "(d) Nº de viagens", "nº de viagens", ""),
    ]
    for ax, (col, _lbl, pal, title, xlab, ylab) in zip(axes, specs):
        sns.boxplot(
            data=df, x=col, y=REF_BAND, hue=col, legend=False, palette=pal,
            linewidth=0.7, fliersize=1.2, ax=ax,
        )
        ax.set_xlabel(xlab, fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 105)
        _clean(ax)
        ax.grid(False, axis="x")

    fig.savefig(out_dir / "fig_01_param_sensitivity.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_01] ok")


# ============================================================
# FIG 02 — HEATMAPS POR TIPO DE LINHA
# ============================================================
def make_fig_02(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE, 2.8),
                             constrained_layout=True)
    for ax, (tipo, titulo) in zip(
        axes,
        [("ida/volta", "(a) Linhas ida/volta"),
         ("circular", "(b) Linhas circulares")],
    ):
        sub = df[df["tipo"] == tipo]
        table = sub.pivot_table(index="meters", columns="increase_meters",
                                values=REF_BAND, aggfunc="median")
        sns.heatmap(
            table, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
            annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
            cbar=(tipo == "circular"),
            cbar_kws={"label": "aderência mediana \u2264 15 m (%)",
                      "shrink": 0.9},
            annot_kws={"size": 7},
        )
        ax.set_xlabel("fator de escala do raio", fontsize=8)
        ax.set_ylabel("raio mínimo (m)", fontsize=8)
        ax.set_title(titulo, fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle("Aderência mediana aos dados GPS (\u2264 15 m)", fontsize=9.5)
    fig.savefig(out_dir / "fig_02_coverage_heatmaps.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_02] ok")


# ============================================================
# FIG 03 — MATRIZ DE CONCLUSAO
# ============================================================
def make_fig_03(df, out_dir):
    rows = []
    for cfg_name, g in df.groupby("config"):
        for inc, gg in g.groupby("increase_meters"):
            rows.append({"config": cfg_name, "inc": inc,
                         "rate": 100.0 * gg["completed"].mean()})
    mat = pd.DataFrame(rows).pivot(index="config", columns="inc",
                                   values="rate")
    ordem = [c for c in (df["config"].drop_duplicates().tolist())
             if c in mat.index]
    mat = mat.reindex(ordem)

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.4, 8.0),
                           constrained_layout=True)
    sns.heatmap(
        mat, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
        annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
        cbar_kws={"label": "conclusões (%)", "shrink": 0.4},
        annot_kws={"size": 5.5},
    )
    ax.set_xlabel("fator de escala do raio", fontsize=8)
    ax.set_ylabel("")
    ax.set_title("Taxa de conclusão por config", fontsize=8.5)
    ax.tick_params(labelsize=5.5)
    fig.savefig(out_dir / "fig_03_completion_matrix.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_03] ok")


# ============================================================
# FIG 04 — DISTRIBUICAO DAS BANDAS
# ============================================================
def make_fig_04(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE, 2.7),
                             constrained_layout=True)

    long = df.melt(value_vars=[BAND_COL[b] for b in BANDS],
                   var_name="banda", value_name="cobertura")
    long["banda"] = long["banda"].map(
        {BAND_COL[b]: f"\u2264 {b} m" for b in BANDS}
    )
    sns.violinplot(data=long, x="banda", y="cobertura", ax=axes[0],
                   hue="banda", legend=False, palette="crest", cut=0,
                   linewidth=0.6)
    axes[0].set_xlabel("banda de aderência")
    axes[0].set_ylabel("cobertura (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Distribuição por banda", fontsize=8)

    med = df.groupby("tipo")[[BAND_COL[b] for b in BANDS]].median()
    med.columns = [f"\u2264 {b} m" for b in BANDS]
    med.plot(kind="bar", ax=axes[1], width=0.75,
             color=[BAND_COLORS[b] for b in BANDS], legend=True)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("cobertura mediana (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("(b) Mediana por tipo de linha", fontsize=8)
    axes[1].tick_params(axis="x", rotation=0, labelsize=7)
    axes[1].legend(fontsize=6.5, frameon=False, title=None)

    for ax in axes:
        _clean(ax)
    axes[1].grid(False, axis="x")
    fig.savefig(out_dir / "fig_04_adherence_bands.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_04] ok")


# ============================================================
# FIG 05 — EFEITO DO NUMERO DE VIAGENS (GLOBAL)
# ============================================================
def make_fig_05(df, out_dir):
    trips = sorted(df["n_trips"].unique())
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.0),
                           constrained_layout=True)

    for b in BANDS:
        col = BAND_COL[b]
        med = df.groupby("n_trips")[col].median().reindex(trips)
        mx = df.groupby("n_trips")[col].max().reindex(trips)
        ax.plot(trips, med, "-o", color=BAND_COLORS[b], lw=1.5, ms=4,
                label=f"mediana \u2264 {b} m")
        ax.plot(trips, mx, "--s", color=BAND_COLORS[b], lw=1.0, ms=3,
                alpha=0.6, label=f"máx. \u2264 {b} m")

    global_med = df.groupby("n_trips")["coverage_pct"].mean().reindex(trips)
    ax.plot(trips, global_med, "-^", color="#333333", lw=1.2, ms=4,
            alpha=0.7, label="média (15 m)")

    ax.set_xscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips])
    ax.set_xlabel("número de viagens (volume de dados)")
    ax.set_ylabel("cobertura GPS (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Efeito do volume de dados na aderência", fontsize=8.5)
    _clean(ax)
    ax.legend(fontsize=6.0, frameon=False, ncol=2)
    fig.savefig(out_dir / "fig_05_trip_effect.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_05] ok")


# ============================================================
# FIG 06 — ESCALABILIDADE DE TEMPO
# ============================================================
def make_fig_06(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE, 2.7),
                             constrained_layout=True)
    pal = {90: "#9ECAE1", 180: "#4292C6", 360: "#084594"}
    for smp, g in df.groupby("angular_samples"):
        axes[0].scatter(g["n_input_points"] / 1000.0, g["runtime_s"],
                        s=8, alpha=0.5, color=pal[smp], label=f"{smp}",
                        edgecolors="none")
    axes[0].set_xlabel("pontos GPS de entrada (milhares)")
    axes[0].set_ylabel("tempo de execução (s)")
    axes[0].set_title("(a) Tempo vs volume de dados", fontsize=8)

    sns.boxplot(data=df, x="angular_samples", y="runtime_s",
                hue="angular_samples", legend=False, palette=pal,
                linewidth=0.7, fliersize=1.2, ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("amostras angulares")
    axes[1].set_ylabel("tempo de execução (s, log)")
    axes[1].set_title("(b) Custo por resolução angular", fontsize=8)

    for ax in axes:
        _clean(ax)
        ax.grid(False, axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3,
               fontsize=7, frameon=False, title="amostras angulares",
               title_fontsize=7)
    fig.savefig(out_dir / "fig_06_runtime_scaling.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_06] ok")


# ============================================================
# FIG 07 — ADERENCIA vs MAE DO SHAPE
# ============================================================
def make_fig_07(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.0),
                           constrained_layout=True)
    for linha, g in df.groupby("linha_s"):
        ax.scatter(g["mae_shape_m"], g[REF_BAND], s=12, alpha=0.6,
                   color=LINE_COLORS.get(linha, "#666"), label=f"Linha {linha}",
                   edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("MAE vs shape oficial (m, log)")
    ax.set_ylabel("aderência aos dados GPS (%)")
    ax.set_title("Aderência aos dados vs divergência do shape\n"
                 "canto superior direito = shape defeituoso", fontsize=8.5)
    _clean(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", ncol=1,
               fontsize=6.0, frameon=False, title="Linha", title_fontsize=7)
    fig.savefig(out_dir / "fig_07_adherence_vs_shape_mae.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_07] ok")


# ============================================================
# FIG 08 — COMPRIMENTO vs SHAPE
# ============================================================
def make_fig_08(df, shape_models, out_dir):
    from analysis_maps import get_config_map

    rows = []
    for linha, sentido in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sid = get_config_map().get((linha, sentido), (None,))[0]
        if sid is None:
            continue
        model = shape_models.get((linha, str(sid).strip()))
        if model is None:
            continue
        rows.append((linha, sentido, float(model["length_km"])))
    skm = pd.DataFrame(rows, columns=["linha_s", "sentido", "shape_km"])
    sub = df.merge(skm, on=["linha_s", "sentido"], how="inner")

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.4),
                           constrained_layout=True)
    lim = (0, max(sub["shape_km"].max(), sub["route_length_km"].max()) * 1.15)
    ax.plot(lim, lim, "--", color="#888", lw=0.9, label="reconstrução = shape")
    for cfg_name, g in sub.groupby("config"):
        cor = LINE_COLORS.get(cfg_name.split()[0], "#666")
        done = g[g["completed"]]
        part = g[~g["completed"]]
        if not done.empty:
            ax.scatter(done["shape_km"], done["route_length_km"], s=12,
                       alpha=0.75, color=cor, edgecolors="none")
        if not part.empty:
            ax.scatter(part["shape_km"], part["route_length_km"], s=14,
                       alpha=0.4, color=cor, marker="x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("comprimento do shape oficial (km)")
    ax.set_ylabel("comprimento reconstruído (km)")
    ax.set_title("Comprimento: cheios = completos, x = incompletos\n"
                 "acima da diagonal = ziguezague", fontsize=8.5)
    _clean(ax)
    fig.savefig(out_dir / "fig_08_route_length_vs_shape.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_08] ok")


def _shape_km(lons, lats):
    from core.geometry import meters_per_deg_lon
    m_lon = meters_per_deg_lon(float(np.nanmean(lats)))
    xs = lons * m_lon
    ys = lats * cfg.M_PER_DEG_LAT
    return float(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2).sum() / 1000.0)


# ============================================================
# FIG 09 — MOTIVOS DE PARADA
# ============================================================
REASON_MAP = {
    "chegada_destino": "chegada ao destino",
    "loop_fechado": "loop fechado",
    "passos_sem_dados": "sem dados (raio máx)",
    "limite_comprimento": "limite de comprimento",
    "raio_max_excedido": "raio máx excedido",
    "sem_pontos_ativos": "sem pontos ativos",
    "max_iterations": "max iterações",
    "erro": "erro",
}
REASON_COLORS = {
    "chegada ao destino": "#2CA02C",
    "loop fechado": "#90D892",
    "sem dados (raio máx)": "#D62728",
    "limite de comprimento": "#FF7F0E",
    "raio máx excedido": "#8C564B",
    "sem pontos ativos": "#7F7F7F",
    "max iterações": "#BBBBBB",
    "erro": "#000000",
}


def make_fig_09(df, out_dir):
    d = df.copy()
    d["motivo"] = d["stop_reason"].map(REASON_MAP).fillna("outros")
    counts = (d.groupby(["config", "motivo"]).size().reset_index(name="n")
              .pivot(index="config", columns="motivo", values="n").fillna(0))
    ordem = [c for c in df["config"].drop_duplicates().tolist()
             if c in counts.index]
    counts = counts.reindex(ordem)

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 2.4, 8.0),
                           constrained_layout=True)
    left = np.zeros(len(counts))
    cats = [c for c in counts.columns if c in REASON_COLORS]
    cats += [c for c in counts.columns if c not in REASON_COLORS]
    y = np.arange(len(counts))
    for cat in cats:
        vals = counts[cat].to_numpy()
        ax.barh(y, vals, left=left, color=REASON_COLORS.get(cat, "#999"),
                label=cat, height=0.62, edgecolor="white", linewidth=0.4)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("número de execuções")
    ax.set_title("Motivos de parada por config "
                 "(150 execuções por config)", fontsize=8.5)
    _clean(ax)
    ax.grid(False, axis="x")
    fig.legend(loc="outside lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.savefig(out_dir / "fig_09_stop_reasons.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_09] ok")


# ============================================================
# FIG 10 — P95 vs MEDIA
# ============================================================
def make_fig_10(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.1),
                           constrained_layout=True)
    for linha, g in df.groupby("linha_s"):
        done = g[g["completed"]]
        part = g[~g["completed"]]
        cor = LINE_COLORS.get(linha, "#666")
        if not done.empty:
            ax.scatter(done["mean_gps_dist_m"], done["p95_gps_dist_m"], s=12,
                       alpha=0.7, color=cor, label=f"{linha}",
                       edgecolors="none")
        if not part.empty:
            ax.scatter(part["mean_gps_dist_m"], part["p95_gps_dist_m"], s=14,
                       alpha=0.4, color=cor, marker="x")
    xs = np.linspace(0.5, df["mean_gps_dist_m"].max() * 1.5, 10)
    ax.plot(xs, xs, "--", color="#888", lw=0.9, label="p95 = média")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distância média GPS \u2192 rota (m, log)")
    ax.set_ylabel("p95 da distância GPS \u2192 rota (m, log)")
    ax.set_title("Cauda das distâncias: círculos = completos, x = incompletos\n"
                 "p95 >> média = fugas localizadas", fontsize=8.5)
    _clean(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=6.0,
               frameon=False, title="Linha", title_fontsize=7)
    fig.savefig(out_dir / "fig_10_p95_vs_mean.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_10] ok")


# ============================================================
# FIG 11 — COMPRESSAO
# ============================================================
def make_fig_11(df, out_dir):
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE + 1.8, 3.1),
                           constrained_layout=True)
    lim = (df["n_input_points"].min() * 0.7,
           df["n_input_points"].max() * 2.0)
    ax.plot(lim, lim, "--", color="#888", lw=0.9, label="rota = entrada")
    for linha, g in df.groupby("linha_s"):
        ax.scatter(g["n_input_points"], g["n_route_points"], s=11, alpha=0.55,
                   color=LINE_COLORS.get(linha, "#666"), label=linha,
                   edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pontos GPS de entrada (log)")
    ax.set_ylabel("pontos da rota gerada (log)")
    ax.set_title("Compressão: a rota usa poucos pontos para\n"
                 "representar dezenas de milhares de amostras de GPS", fontsize=8.5)
    _clean(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", fontsize=6.0,
               frameon=False, title="Linha", title_fontsize=7)
    fig.savefig(out_dir / "fig_11_data_compression.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[fig_11] ok")


# ============================================================
# TABELA — MELHORES CONFIGS
# ============================================================
def save_best_configs(df, out_dir):
    rows = []
    for (linha, sentido, nt) in ordered_configs(df):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)
                 & (df["n_trips"] == nt)]
        if sub.empty:
            continue
        r = best_row(sub)
        rows.append({
            "linha": linha, "sentido": sentido, "n_trips": int(nt),
            "completed": bool(r["completed"]),
            "coverage_5m": round(r["coverage_5m"], 2),
            "coverage_10m": round(r["coverage_10m"], 2),
            "coverage_15m": round(r["coverage_15m"], 2),
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
    out = pd.DataFrame(rows).sort_values(["linha", "sentido", "n_trips"])
    out.to_csv(out_dir / "best_configs_v10.csv", index=False)
    print(f"[tabela] best_configs_v10.csv ({len(out)} configs)")
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    out_dir = Path(cfg.ANALYSIS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results()
    print(f"Execuções: {len(df):,} | completas: {int(df['completed'].sum())}")

    print("Carregando shapes...")
    shape_models = load_shape_models(cfg.SHAPE_XZ_PATH)

    print("Figuras globais...")
    make_fig_01(df, out_dir)
    make_fig_02(df, out_dir)
    make_fig_03(df, out_dir)
    make_fig_04(df, out_dir)
    make_fig_05(df, out_dir)
    make_fig_06(df, out_dir)
    make_fig_07(df, out_dir)
    make_fig_08(df, shape_models, out_dir)
    make_fig_09(df, out_dir)
    make_fig_10(df, out_dir)
    make_fig_11(df, out_dir)

    print("Figuras por linha...")
    make_all_per_line(df)

    print("Mapas por linha (carrega dados GPS)...")
    make_line_maps(df)
    make_gallery(df)

    save_best_configs(df, out_dir)
    print(f"\nFinalizado. Análises em: {out_dir}")


if __name__ == "__main__":
    main()
