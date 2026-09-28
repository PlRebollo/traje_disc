"""
Análise por FAIXAS DE DADOS — experimento v9.

Separa o desempenho em regimes de volume de dados para destacar (a) os
casos que funcionam, (b) os limitados por falta de dados e (c) os
limitados pela própria linha (shape/etiquetagem/geometria).

Figuras (outputs/analysis/regimes/):
- rg_01: desempenho por faixa de dados (o agregado esconde os bons regimes)
- rg_02: matriz linha/sentido × faixa de dados
- rg_03: limitado por dados vs limitado pela linha
- rg_04: alinhamento GPS↔shape explica o teto das linhas
- rg_05: anatomia das bandas de aderência nos casos que não fecham
- rg_06: exemplos visuais (funciona / custa dados / problema de shape)
- rg_07: custo em dados e teto por linha

Tabela: regimes_summary.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

import config as cfg
from analysis_common import load_results

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

ANALYSIS_DIR = Path(cfg.ANALYSIS_DIR)
OUT = ANALYSIS_DIR / "regimes"
DIAG_CSV = OUT / "diag_gps_vs_shape.csv"
DV_CSV = ANALYSIS_DIR / "data_volume" / "data_volume_summary.csv"
PER_LINE = ANALYSIS_DIR / "per_line"

REF = "coverage_15m"
COL_IV = "#1F77B4"
COL_CIRC = "#2CA02C"
CLASS_COLORS = {"funciona": "#2CA02C", "parcial": "#FF7F0E",
                "limitado pela linha": "#D62728", "poucos dados": "#999999"}

BANDS_T = [(1, 2, "1–2"), (5, 10, "5–10"), (20, 50, "20–50"),
           (100, 200, "100–200"), (500, 500, "500")]


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def _band(nt):
    for lo, hi, lab in BANDS_T:
        if lo <= nt <= hi:
            return lab
    return "?"


def classify(v_rich):
    if v_rich >= 95:
        return "funciona"
    if v_rich >= 90:
        return "parcial"
    return "limitado pela linha"


def build_summary(df):
    df = df.copy()
    df["cfg"] = df["linha_s"] + "/" + df["sentido"]
    df["band"] = df["n_trips"].map(_band)
    rows = []
    for cfgname, g in df.groupby("cfg"):
        v500 = g.loc[g["n_trips"] == 500, REF].median()
        vlow = g.loc[g["n_trips"] <= 2, REF].median()
        rows.append(dict(
            cfg=cfgname, sentido=g["sentido"].iloc[0],
            tipo="circular" if g["sentido"].iloc[0] == "circular"
            else "ida/volta",
            v_low=vlow, v_rich=v500, ganho=v500 - vlow,
            cl=classify(v500)))
    sm = pd.DataFrame(rows)
    if DIAG_CSV.exists():
        dg = pd.read_csv(DIAG_CSV)
        sm = sm.merge(dg[["cfg", "gps_med", "frac_far", "n_veh"]],
                      on="cfg", how="left")
    if DV_CSV.exists():
        dv = pd.read_csv(DV_CSV).rename(columns={"label": "cfg",
                                                 "n_trips": "n90"})
        sm = sm.merge(dv[["cfg", "n90", "cov"]].rename(
            columns={"cov": "teto"}), on="cfg", how="left")
    sm = sm.sort_values("v_rich", ascending=False)
    sm.to_csv(OUT / "regimes_summary.csv", index=False)
    return sm


# ============================================================
# RG_01 — DESEMPENHO POR FAIXA
# ============================================================
def rg_01(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)
    bands = [b[2] for b in BANDS_T]

    ax = axes[0]
    for tipo, cor in [("ida/volta", COL_IV), ("circular", COL_CIRC)]:
        sub = df[df["tipo"] == tipo]
        med = [sub.loc[sub["band"] == b, REF].median() for b in bands]
        q1 = [sub.loc[sub["band"] == b, REF].quantile(.25) for b in bands]
        q3 = [sub.loc[sub["band"] == b, REF].quantile(.75) for b in bands]
        x = np.arange(len(bands))
        ax.plot(x, med, "-o", color=cor, lw=1.7, ms=4, label=tipo)
        ax.fill_between(x, q1, q3, color=cor, alpha=0.13)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands)
    ax.set_xlabel("número de viagens (faixa)")
    ax.set_ylabel("aderência mediana \u2264 15 m (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Aderência por faixa de dados", fontsize=8)
    _clean(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    ax = axes[1]
    n = [len(df[(df["band"] == b) & (df["tipo"] == "ida/volta")])
         for b in bands]
    comp_iv = [100 * df[(df["band"] == b) & (df["tipo"] == "ida/volta")]
               ["completed"].mean() for b in bands]
    comp_ci = [100 * df[(df["band"] == b) & (df["tipo"] == "circular")]
               ["completed"].mean() for b in bands]
    x = np.arange(len(bands))
    ax.bar(x - 0.2, comp_iv, 0.4, color=COL_IV, label="ida/volta")
    ax.bar(x + 0.2, comp_ci, 0.4, color=COL_CIRC, label="circular")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_xlabel("número de viagens (faixa)")
    ax.set_ylabel("conclusão (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Conclusão por faixa", fontsize=8)
    _clean(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    fig.suptitle("O agregado mistura regimes: com \u2265 100 viagens o modelo "
                 "funciona bem", fontsize=9)
    fig.savefig(out / "rg_01_regimes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rg_01] ok")


# ============================================================
# RG_02 — MATRIZ LINHA × FAIXA
# ============================================================
def rg_02(df, sm, out):
    bands = [b[2] for b in BANDS_T]
    piv = df.pivot_table(index="cfg", columns="band", values=REF,
                         aggfunc="median")
    piv = piv.reindex(columns=bands)
    piv = piv.reindex(sm["cfg"])  # ordena pelo desempenho rico
    fig, ax = plt.subplots(figsize=(7.16, 7.4), constrained_layout=True)
    sns.heatmap(piv, ax=ax, cmap="RdYlGn", vmin=0, vmax=100, annot=True,
                fmt=".0f", linewidths=0.4, linecolor="white",
                cbar_kws={"label": "aderência mediana \u2264 15 m (%)",
                          "shrink": 0.5}, annot_kws={"size": 6.2})
    ax.set_xlabel("faixa de número de viagens", fontsize=8)
    ax.set_ylabel("")
    ax.set_yticklabels(piv.index, fontsize=6.5)
    ax.set_title("Aderência por linha/sentido e faixa de dados\n"
                 "(linhas de baixo nunca ficam verdes = problema da linha)",
                 fontsize=9)
    fig.savefig(out / "rg_02_band_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rg_02] ok")


# ============================================================
# RG_03 — LIMITADO POR DADOS vs PELA LINHA
# ============================================================
def rg_03(sm, out):
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for cl, cor in CLASS_COLORS.items():
        s = sm[sm["cl"] == cl]
        if s.empty:
            continue
        ax.scatter(s["v_low"], s["v_rich"], s=42, color=cor, label=cl,
                   edgecolors="white", linewidths=0.6, zorder=3)
    lim = [-3, 103]
    ax.plot(lim, lim, "--", color="#888", lw=0.9)
    ax.axhline(90, ls=":", color="#888", lw=0.9)
    for _, r in sm.iterrows():
        ax.annotate(r["cfg"], (r["v_low"], r["v_rich"]), fontsize=5.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("aderência no regime pobre (1–2 viagens) (%)")
    ax.set_ylabel("aderência no regime rico (500 viagens) (%)")
    ax.set_title("Muito acima da diagonal = salvos pelos dados;\n"
                 "abaixo de 90 no topo = teto da própria linha", fontsize=8.5)
    _clean(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(out / "rg_03_data_vs_line.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rg_03] ok")


# ============================================================
# RG_04 — ALINHAMENTO GPS↔SHAPE
# ============================================================
def rg_04(sm, out):
    s = sm.dropna(subset=["frac_far", "v_rich"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for cl, cor in CLASS_COLORS.items():
        sub = s[s["cl"] == cl]
        if sub.empty:
            continue
        ax.scatter(sub["frac_far"], sub["v_rich"], s=sub["n_veh"] * 1.1 + 12,
                   color=cor, alpha=0.75, edgecolors="white", linewidths=0.6,
                   label=cl, zorder=3)
    for _, r in s.iterrows():
        if r["frac_far"] > 3 or r["v_rich"] < 95:
            ax.annotate(r["cfg"], (r["frac_far"], r["v_rich"]), fontsize=5.8,
                        xytext=(3, -6), textcoords="offset points")
    ax.axhline(90, ls=":", color="#888", lw=0.9)
    ax.set_xlabel("pings a > 100 m do shape oficial (no regime rico, %)")
    ax.set_ylabel("aderência mediana \u2264 15 m (500 viagens) (%)")
    z = np.polyfit(s["frac_far"], s["v_rich"], 1)
    xs = np.linspace(0, s["frac_far"].max() * 1.05, 20)
    ax.plot(xs, np.polyval(z, xs), "--", color="#666", lw=1.0,
            label="tendência")
    ax.set_title("Quando parte dos pings não segue o shape oficial,\n"
                 "a aderência fica presa abaixo de 100%", fontsize=8.5)
    _clean(ax)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    fig.savefig(out / "rg_04_gps_shape_alignment.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[rg_04] ok")


# ============================================================
# RG_05 — ANATOMIA DAS BANDAS
# ============================================================
def rg_05(df, sm, out):
    ordem = sm["cfg"].tolist()
    sub = df[df["n_trips"] == 500]
    fig, ax = plt.subplots(figsize=(7.16, 4.6), constrained_layout=True)
    y = np.arange(len(ordem))
    c5 = [sub[sub["cfg"] == c]["coverage_5m"].median() for c in ordem]
    c10 = [sub[sub["cfg"] == c]["coverage_10m"].median() for c in ordem]
    c15 = [sub[sub["cfg"] == c]["coverage_15m"].median() for c in ordem]
    ax.barh(y, c5, color="#2CA02C", label="\u2264 5 m")
    ax.barh(y, np.array(c10) - np.array(c5), left=c5, color="#FF7F0E",
            label="5–10 m")
    ax.barh(y, np.array(c15) - np.array(c10), left=c10, color="#D62728",
            label="10–15 m")
    ax.set_yticks(y)
    ax.set_yticklabels(ordem, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("aderência por banda no regime rico (500 viagens) (%)")
    ax.set_title("Quanto da nuvem cada rota cobre (verde = dentro de 5 m)",
                 fontsize=8.5)
    _clean(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(out / "rg_05_bands_anatomy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rg_05] ok")


# ============================================================
# RG_06 — EXEMPLOS VISUAIS
# ============================================================
def rg_06(out):
    exemplos = [
        ("658_volta", "Funciona (dados baratos): 96,8% com 10 viagens"),
        ("603_ida", "Funciona já com 1 viagem: 94,3%"),
        ("505_volta", "Precisa de dados: 4% (1 viagem) → 97,4% (500)"),
        ("607_ida", "Limitada pelo shape: 19% dos pings fora do shape"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 7.0),
                             constrained_layout=True)
    for ax, (tag, title) in zip(axes.ravel(), exemplos):
        p = PER_LINE / f"map_{tag}.png"
        if p.exists():
            ax.imshow(plt.imread(p))
        ax.axis("off")
        ax.set_title(title, fontsize=7.5)
    fig.suptitle("Exemplos: o que funciona, o que custa dados e o que é "
                 "problema de shape", fontsize=9)
    fig.savefig(out / "rg_06_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[rg_06] ok")


# ============================================================
# RG_07 — CUSTO EM DADOS E TETO
# ============================================================
def rg_07(sm, out):
    s = sm.dropna(subset=["n90"]).sort_values("n90")
    fig, ax = plt.subplots(figsize=(7.16, 4.6), constrained_layout=True)
    y = np.arange(len(s))
    ax.barh(y, s["n90"], color=[CLASS_COLORS[c] for c in s["cl"]],
            edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(s["cfg"], fontsize=6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_xlabel("viagens necessárias para atingir 90%")
    ax.set_title("Custo em dados por linha (cor = teto alcançado no regime "
                 "rico)", fontsize=8.5)
    _clean(ax)
    ax.grid(False, axis="y")
    ax.legend(handles=[plt.Line2D([], [], color=c, marker="s", ls="",
                                  label=k)
                       for k, c in CLASS_COLORS.items() if k != "poucos dados"],
              fontsize=6.5, frameon=False, loc="lower right")
    fig.savefig(out / "rg_07_data_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[rg_07] ok")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_results()
    df["cfg"] = df["linha_s"] + "/" + df["sentido"]
    df["band"] = df["n_trips"].map(_band)
    df["tipo"] = np.where(df["sentido"] == "circular", "circular", "ida/volta")
    sm = build_summary(df)
    print(f"configs: {len(sm)} | funciona={sum(sm.cl=='funciona')} "
          f"parcial={sum(sm.cl=='parcial')} limitado={sum(sm.cl=='limitado pela linha')}")

    rg_01(df, OUT)
    rg_02(df, sm, OUT)
    rg_03(sm, OUT)
    rg_04(sm, OUT)
    rg_05(df, sm, OUT)
    rg_06(OUT)
    rg_07(sm, OUT)
    print(f"\nPronto. Tabela: {OUT / 'regimes_summary.csv'}")


if __name__ == "__main__":
    main()
