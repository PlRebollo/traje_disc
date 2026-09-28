"""
Figuras individuais por linha — experimento v8.

Para cada par (linha, sentido) gera:

  sensitivity_{linha}_{sentido}.png
      3 painéis (bandas 5 m, 10 m, 15 m) com heatmaps da aderência
      mediana em função do raio mínimo (linhas) e do fator de escala
      (colunas), no melhor número de viagens.

  trip_effect_{linha}_{sentido}.png
      Aderência (5/10/15 m) em função do número de viagens
      (1, 2, 5, 10, 20, 50, 100, 200, 500): mediana, faixa interquartil e
      máximo.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from analysis_common import BANDS, BAND_COL, BAND_COLORS, best_row

warnings.filterwarnings("ignore")

ANALYSIS_DIR = Path(cfg.ANALYSIS_DIR)
PERLINE_DIR = ANALYSIS_DIR / "per_line"


def _cmap_band(b):
    return {5: "Reds", 10: "Oranges", 15: "Greens"}[b]


def sensitivity_figure(sub: pd.DataFrame, linha: str, sentido: str,
                       out_dir: Path):
    """Heatmaps raio × fator por banda de aderência."""
    if sub.empty:
        return
    # melhor nº de viagens (pela banda de referência)
    nt_best = int(best_row(sub)["n_trips"])
    s = sub[sub["n_trips"] == nt_best]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                             constrained_layout=True)
    for ax, b in zip(axes, BANDS):
        col = BAND_COL[b]
        table = s.pivot_table(index="meters", columns="increase_meters",
                              values=col, aggfunc="median")
        sns.heatmap(
            table, ax=ax, cmap=_cmap_band(b), vmin=0, vmax=100,
            annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
            cbar_kws={"label": f"aderência \u2264 {b} m (%)", "shrink": 0.9},
            annot_kws={"size": 6.5},
        )
        ax.set_xlabel("fator de escala do raio", fontsize=7.5)
        ax.set_ylabel("raio mínimo (m)" if b == BANDS[0] else "", fontsize=7.5)
        ax.set_title(f"({chr(97 + BANDS.index(b))}) banda {b} m", fontsize=8)
        ax.tick_params(labelsize=6.5)

    tipo = "circular" if sentido == "circular" else "ida/volta"
    fig.suptitle(
        f"Linha {linha} — sentido {sentido} ({tipo}) | "
        f"melhor nº de viagens = {nt_best}",
        fontsize=9,
    )
    tag = f"{linha}_{sentido}"
    fig.savefig(out_dir / f"sensitivity_{tag}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def trip_effect_figure(sub: pd.DataFrame, linha: str, sentido: str,
                       out_dir: Path):
    """Aderência vs nº de viagens, por banda."""
    if sub.empty:
        return
    trips = sorted(sub["n_trips"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             constrained_layout=True)

    # (a) mediana + IQR por banda
    ax = axes[0]
    for b in BANDS:
        col = BAND_COL[b]
        med = sub.groupby("n_trips")[col].median().reindex(trips)
        q1 = sub.groupby("n_trips")[col].quantile(0.25).reindex(trips)
        q3 = sub.groupby("n_trips")[col].quantile(0.75).reindex(trips)
        ax.plot(trips, med, "-o", color=BAND_COLORS[b], lw=1.4, ms=4,
                label=f"\u2264 {b} m")
        ax.fill_between(trips, q1, q3, color=BAND_COLORS[b], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips])
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("aderência mediana (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Mediana e faixa interquartil", fontsize=8)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=6.5, frameon=False)

    # (b) melhor execução por banda
    ax = axes[1]
    for b in BANDS:
        col = BAND_COL[b]
        best = sub.groupby("n_trips")[col].max().reindex(trips)
        ax.plot(trips, best, "-s", color=BAND_COLORS[b], lw=1.4, ms=4,
                label=f"\u2264 {b} m")
    ax.set_xscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips])
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("melhor aderência (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Melhor combinação de parâmetros", fontsize=8)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=6.5, frameon=False)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(labelsize=7)

    fig.suptitle(f"Linha {linha} — sentido {sentido}: efeito do nº de viagens",
                 fontsize=9)
    tag = f"{linha}_{sentido}"
    fig.savefig(out_dir / f"trip_effect_{tag}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def make_all_per_line(df: pd.DataFrame, out_dir: Path = PERLINE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        sensitivity_figure(sub, linha, sentido, out_dir)
        trip_effect_figure(sub, linha, sentido, out_dir)
        n += 1
        print(f"  [per_line] {linha}/{sentido} ok")
    print(f"[per_line] {n} configs -> {out_dir}")
    return n


if __name__ == "__main__":
    from analysis_common import load_results
    dfr = load_results()
    make_all_per_line(dfr)
