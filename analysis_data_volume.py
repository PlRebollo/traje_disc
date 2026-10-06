"""
Análise de VOLUME DE DADOS — experimento v8.

Pergunta central: o desempenho do modelo depende da QUANTIDADE de dados GPS
(nº de viagens / amostras de GPS de entrada)? Estas figuras isolam esse efeito,
separando-o do modelo (que é idêntico ao do v7) e dos hiperparâmetros.

Figuras (outputs/analysis/data_volume/):
- dv_01: curva mestra — aderência (5/10/15 m) e conclusão vs volume de dados
- dv_02: circular vs ida/volta — onde a escassez de dados mais dói
- dv_03: robustez — dispersão das soluções encolhe conforme há mais dados
- dv_04: modo de falha — por que a rota não completa sem dados
- dv_05: qualidade geométrica — distância GPS→rota e MAE do shape
- dv_06: saturação linha a linha (small multiples)
- dv_07: mapa de calor dados × fator de escala (dados > tuning)
- dv_08: v7 vs v8 casados por volume de amostras de GPS (mesmo modelo, mesmo plateau)
- dv_09: custo em dados — viagens necessárias para atingir 90%

Tabela: data_volume_summary.csv

Uso:
    cd experimento_v8
    python analysis_data_volume.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from analysis_common import load_results, BANDS, BAND_COL, BAND_COLORS

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

ANALYSIS_DIR = Path(cfg.ANALYSIS_DIR)
OUT_DIR = ANALYSIS_DIR / "data_volume"
V7_SUMMARY = cfg.BASE / "experimento_v7" / "outputs" / "experiment_summary.csv"

REF = "coverage_15m"
COL_CIRC = "#2CA02C"
COL_IV = "#1F77B4"
COL_FAIL = "#D62728"

REASON_MAP = {
    "chegada_destino": "chegou ao destino",
    "loop_fechado": "loop fechado",
    "passos_sem_dados": "sem dados (raio máx)",
    "limite_comprimento": "limite de comprimento",
    "raio_max_excedido": "raio máx excedido",
    "sem_pontos_ativos": "sem pontos ativos",
    "max_iterations": "max iterações",
    "erro": "erro",
}
REASON_COLORS = {
    "chegou ao destino": "#2CA02C",
    "loop fechado": "#90D892",
    "sem dados (raio máx)": "#D62728",
    "limite de comprimento": "#FF7F0E",
    "raio máx excedido": "#8C564B",
    "sem pontos ativos": "#7F7F7F",
    "max iterações": "#BBBBBB",
    "erro": "#000000",
}


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=7)


def _vol_curve(df, y, x="n_input_points", nbins=16):
    """Curva mediana+IQR de y em função de x (bins log-espaçados)."""
    d = df[[x, y]].dropna()
    d = d[d[x] > 0]
    edges = np.logspace(np.log10(d[x].min()), np.log10(d[x].max()), nbins)
    b = pd.cut(d[x], edges)
    g = d.groupby(b, observed=True)[y].agg(
        med="median", q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75), n="size")
    g["xc"] = [iv.mid for iv in g.index]
    return g.reset_index(drop=True)


# ============================================================
# DV_01 — CURVA MESTRA
# ============================================================
def dv_01(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)

    ax = axes[0]
    for b in BANDS:
        c = _vol_curve(df, BAND_COL[b])
        ax.plot(c["xc"], c["med"], "-", color=BAND_COLORS[b], lw=1.6,
                label=f"mediana \u2264 {b} m")
        if b == 15:
            ax.fill_between(c["xc"], c["q1"], c["q3"], color=BAND_COLORS[b],
                            alpha=0.15)
    ax.axvspan(df["n_input_points"].min() * 0.8, 4000, color=COL_FAIL,
               alpha=0.07)
    ax.text(500, 6, "escassez de dados", color=COL_FAIL, fontsize=7,
            rotation=0, ha="left")
    ax.axhline(90, ls=":", color="#888", lw=0.9)
    ax.axhline(80, ls=":", color="#BBB", lw=0.8)
    c15 = _vol_curve(df, REF)
    reach = c15[c15["med"] >= 90]
    if len(reach):
        xr = reach["xc"].iloc[0]
        ax.axvline(xr, ls="--", color="#444", lw=0.9)
        ax.annotate(f"90% a partir de\n~{xr:,.0f} amostras de GPS".replace(",", "."),
                    xy=(xr, 90), xytext=(xr * 1.3, 42), fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"))
    ax.set_xscale("log")
    ax.set_xlabel("amostras de GPS de entrada")
    ax.set_ylabel("cobertura GPS (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Aderência ao dado cresce com o volume", fontsize=8)
    _clean(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    ax = axes[1]
    g = df.groupby(pd.cut(df["n_input_points"],
                          np.logspace(np.log10(df["n_input_points"].min()),
                                      np.log10(df["n_input_points"].max()), 14)),
                   observed=True).agg(comp=("completed", "mean"),
                                      n=("completed", "size"))
    xc = [iv.mid for iv in g.index]
    ax.plot(xc, 100 * g["comp"], "-o", color="#444", lw=1.5, ms=3.5)
    ax.set_xscale("log")
    ax.set_xlabel("amostras de GPS de entrada")
    ax.set_ylabel("execuções que completaram (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Conclusão do trajeto vs volume", fontsize=8)
    _clean(ax)

    fig.suptitle("Efeito do volume de dados GPS no desempenho do modelo",
                 fontsize=9.5)
    fig.savefig(out / "dv_01_master_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_01] ok")


# ============================================================
# DV_02 — CIRCULAR vs IDA/VOLTA
# ============================================================
def dv_02(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)
    trips = sorted(df["n_trips"].unique())

    for ax, (col, name) in zip(
            axes, [(REF, "(a) Aderência \u2264 15 m"),
                   ("completed", "(b) Conclusão do trajeto")]):
        for tipo, cor in [("circular", COL_CIRC), ("ida/volta", COL_IV)]:
            sub = df[df["tipo"] == tipo]
            med = sub.groupby("n_trips")[col].median().reindex(trips)
            if col == "completed":
                med = 100 * sub.groupby("n_trips")[col].mean().reindex(trips)
            ax.plot(trips, med, "-o", color=cor, lw=1.6, ms=3.5,
                    label=tipo)
            if col == REF:
                q1 = sub.groupby("n_trips")[col].quantile(.25).reindex(trips)
                q3 = sub.groupby("n_trips")[col].quantile(.75).reindex(trips)
                ax.fill_between(trips, q1, q3, color=cor, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks(trips)
        ax.set_xticklabels([str(t) for t in trips], fontsize=6.5)
        ax.set_xlabel("número de viagens")
        ax.set_ylim(0, 105)
        ax.set_title(name, fontsize=8)
        _clean(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower right")

    axes[0].set_ylabel("cobertura GPS (%)")
    axes[1].set_ylabel("% das execuções")
    fig.suptitle("Linhas circulares amostram o anel inteiro; ida/volta "
                 "precisam de muito mais dados", fontsize=9)
    fig.savefig(out / "dv_02_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_02] ok")


# ============================================================
# DV_03 — ROBUSTEZ
# ============================================================
def dv_03(df, out):
    trips = sorted(df["n_trips"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)

    ax = axes[0]
    dfx = df.assign(nt=df["n_trips"].astype(str))
    sns.boxplot(data=dfx, x="nt", y=REF, hue="nt", legend=False,
                palette="crest", linewidth=0.6, fliersize=1.0, ax=ax,
                order=[str(t) for t in trips])
    best = df.groupby("n_trips")[REF].max().reindex(trips)
    ax.plot(range(len(trips)), best, "-s", color="#D62728", lw=1.2, ms=3.5,
            label="melhor execução")
    ax.set_xticklabels([str(t) for t in trips], fontsize=6.5)
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("cobertura \u2264 15 m (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Soluções: caixa = metade central", fontsize=8)
    _clean(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    ax = axes[1]
    iqr = (df.groupby("n_trips")[REF].quantile(.75)
           - df.groupby("n_trips")[REF].quantile(.25)).reindex(trips)
    fail = df.assign(bad=df[REF] < 50).groupby("n_trips")["bad"].mean()
    fail = (100 * fail).reindex(trips)
    ax.plot(trips, iqr, "-o", color="#6A51A3", lw=1.5, ms=3.5,
            label="largura interquartil (pp)")
    ax.plot(trips, fail, "-^", color=COL_FAIL, lw=1.5, ms=3.5,
            label="execuções com \u2264 15 m abaixo de 50%")
    ax.set_xscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips], fontsize=6.5)
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("dispersão (pp) / taxa (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(b) Variabilidade e falhas caem com os dados", fontsize=8)
    _clean(ax)
    ax.legend(fontsize=6.5, frameon=False)

    fig.suptitle("Com poucos dados, o resultado depende do acaso dos "
                 "hiperparâmetros; com muitos, torna-se robusto", fontsize=9)
    fig.savefig(out / "dv_03_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_03] ok")


# ============================================================
# DV_04 — MODO DE FALHA
# ============================================================
def dv_04(df, out):
    trips = sorted(df["n_trips"].unique())
    d = df.copy()
    d["motivo"] = d["stop_reason"].map(REASON_MAP).fillna("outros")
    tab = (d.groupby(["n_trips", "motivo"]).size().unstack(fill_value=0))
    tab = tab.div(tab.sum(axis=1), axis=0) * 100
    cats = [c for c in REASON_COLORS if c in tab.columns]
    cats += [c for c in tab.columns if c not in cats]
    tab = tab.reindex(trips)

    fig, ax = plt.subplots(figsize=(7.16, 3.1), constrained_layout=True)
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab))
    for cat in cats:
        v = tab[cat].to_numpy()
        ax.bar(x, v, bottom=bottom, color=REASON_COLORS.get(cat, "#999"),
               label=cat, width=0.78, edgecolor="white", linewidth=0.5)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in trips])
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("proporção das execuções (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Por que a rota para — muda com o volume de dados",
                 fontsize=8.5)
    _clean(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=6.2, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out / "dv_04_failure_mode.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_04] ok")


# ============================================================
# DV_05 — QUALIDADE GEOMETRICA
# ============================================================
def dv_05(df, out):
    trips = sorted(df["n_trips"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)

    ax = axes[0]
    for col, lbl, cor in [("mean_gps_dist_m", "distância média GPS\u2192rota",
                           "#6A51A3"),
                          ("p95_gps_dist_m", "p95 GPS\u2192rota", "#D62728")]:
        med = df.groupby("n_trips")[col].median().reindex(trips)
        q1 = df.groupby("n_trips")[col].quantile(.25).reindex(trips)
        q3 = df.groupby("n_trips")[col].quantile(.75).reindex(trips)
        ax.plot(trips, med, "-o", color=cor, lw=1.5, ms=3.5, label=lbl)
        ax.fill_between(trips, q1, q3, color=cor, alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips], fontsize=6.5)
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("distância (m, log)")
    ax.set_title("(a) Quão longe a rota passa dos amostras de GPS", fontsize=8)
    _clean(ax)
    ax.legend(fontsize=6.2, frameon=False)

    ax = axes[1]
    med = df.groupby("n_trips")["mae_shape_m"].median().reindex(trips)
    q1 = df.groupby("n_trips")["mae_shape_m"].quantile(.25).reindex(trips)
    q3 = df.groupby("n_trips")["mae_shape_m"].quantile(.75).reindex(trips)
    ax.plot(trips, med, "-o", color="#1F77B4", lw=1.5, ms=3.5,
            label="MAE rota \u2194 shape oficial")
    ax.fill_between(trips, q1, q3, color="#1F77B4", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(trips)
    ax.set_xticklabels([str(t) for t in trips], fontsize=6.5)
    ax.set_xlabel("número de viagens")
    ax.set_ylabel("MAE (m)")
    ax.set_title("(b) Aproximação ao trajeto oficial", fontsize=8)
    _clean(ax)
    ax.legend(fontsize=6.2, frameon=False)

    fig.suptitle("Com poucos dados a rota se afasta dos amostras de GPS e do trajeto "
                 "oficial", fontsize=9)
    fig.savefig(out / "dv_05_geometry_quality.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[dv_05] ok")


# ============================================================
# DV_06 — SATURACAO POR LINHA
# ============================================================
def dv_06(df, out):
    trips = sorted(df["n_trips"].unique())
    linhas = sorted(df["linha_s"].unique())
    fig, axes = plt.subplots(4, 4, figsize=(7.16, 6.2), sharex=True,
                             sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for i, (ax, linha) in enumerate(zip(axes, linhas)):
        sub = df[df["linha_s"] == linha]
        for sentido, cor in [("circular", COL_CIRC), ("ida", COL_IV),
                             ("volta", "#9467BD")]:
            s = sub[sub["sentido"] == sentido]
            if s.empty:
                continue
            med = s.groupby("n_trips")[REF].median().reindex(trips)
            ax.plot(trips, med, "-o", color=cor, lw=1.3, ms=2.8,
                    label=sentido)
        ax.axhline(90, ls=":", color="#888", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"Linha {linha}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        _clean(ax)
    for ax in axes[len(linhas):]:
        ax.axis("off")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("cov. \u2264 15 m (%)", fontsize=6.5)
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    for ax in axes[:len(linhas)]:
        ax.set_xlabel("viagens", fontsize=6.5)
    fig.suptitle("Saturação da aderência por linha — todas convergem com "
                 "dados suficientes", fontsize=9.5)
    fig.savefig(out / "dv_06_saturation_per_line.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[dv_06] ok")


# ============================================================
# DV_07 — HEATMAP DADOS x FATOR
# ============================================================
def dv_07(df, out):
    incs = sorted(df["increase_meters"].unique())
    trips = sorted(df["n_trips"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6),
                             constrained_layout=True)
    for ax, tipo in zip(axes, ["ida/volta", "circular"]):
        sub = df[df["tipo"] == tipo]
        mat = sub.pivot_table(index="n_trips", columns="increase_meters",
                              values=REF, aggfunc="median").reindex(trips)
        sns.heatmap(mat, ax=ax, cmap="RdYlGn", vmin=0, vmax=100,
                    annot=True, fmt=".0f", linewidths=0.4, linecolor="white",
                    cbar=(tipo == "circular"),
                    cbar_kws={"label": "cov. \u2264 15 m (%)", "shrink": 0.85},
                    annot_kws={"size": 5.4})
        ax.set_xlabel("fator de escala do raio", fontsize=7.5)
        ax.set_ylabel("nº de viagens" if tipo == "ida/volta" else "",
                      fontsize=7.5)
        ax.set_title(f"({'a' if tipo == 'ida/volta' else 'b'}) {tipo}",
                     fontsize=8)
        ax.tick_params(labelsize=6.2)
    fig.suptitle("Sem dados, nenhum ajuste de fator salva; com dados, "
                 "quase todo fator funciona", fontsize=9)
    fig.savefig(out / "dv_07_data_vs_factor.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[dv_07] ok")


# ============================================================
# DV_08 — v9 (anterior) vs v10 (refinado), por volume de amostras
# ============================================================
V9_SUMMARY = cfg.BASE / "experimento_v9_sem_limite" / "outputs" / \
    "experiment_summary.csv"


def dv_08(df, out):
    if not V9_SUMMARY.exists():
        print("[dv_08] resumo do v9 ausente — pulando")
        return
    v9 = pd.read_csv(V9_SUMMARY)
    v10 = df.copy()
    v9["tipo"] = np.where(v9["sentido"] == "circular", "circular", "ida/volta")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9),
                             constrained_layout=True)
    for ax, (key, cor, name) in zip(
            axes, [(v9, "#FF7F0E", "modelo anterior"),
                   (v10, "#1F77B4", "modelo refinado")]):
        c = _vol_curve(key, REF, nbins=12)
        ax.plot(c["xc"], c["med"], "-o", color=cor, lw=1.7, ms=3.5,
                label=f"{name}: mediana")
        ax.fill_between(c["xc"], c["q1"], c["q3"], color=cor, alpha=0.14)
        ax.axhline(90, ls=":", color="#888", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("amostras de GPS de entrada")
        ax.set_ylim(0, 105)
        ax.set_title(name, fontsize=8)
        _clean(ax)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].set_ylabel("cobertura \u2264 15 m (%)")
    fig.suptitle("Cobertura em função do volume de amostras: modelo anterior "
                 "vs refinado", fontsize=9)
    fig.savefig(out / "dv_08_v9_vs_v10.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_08] ok")


# ============================================================
# DV_09 — CUSTO EM DADOS PARA 90%
# ============================================================
def dv_09(df, out):
    trips = sorted(df["n_trips"].unique())
    rows = []
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        med = sub.groupby("n_trips")[REF].median().reindex(trips)
        ok = med[med >= 90]
        pts = sub.groupby("n_trips")["n_input_points"].median()
        if len(ok):
            nt = int(ok.index.min())
            rows.append(dict(label=f"{linha}/{sentido}", n_trips=nt,
                             points=float(pts.get(nt, np.nan)),
                             cov=float(ok.max())))
        else:
            rows.append(dict(label=f"{linha}/{sentido}", n_trips=np.nan,
                             points=np.nan, cov=float(med.max())))
    t = pd.DataFrame(rows).sort_values(["n_trips", "cov"])
    t.to_csv(out / "data_volume_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.6),
                             constrained_layout=True)

    ax = axes[0]
    d = t.dropna(subset=["n_trips"]).sort_values("n_trips")
    y = np.arange(len(d))
    ax.barh(y, d["n_trips"], color="#2CA02C", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("viagens necessárias para atingir 90%")
    ax.set_title(f"(a) Atingem 90% ({len(d)}/{len(t)})", fontsize=8)
    _clean(ax)
    ax.grid(False, axis="x")

    ax = axes[1]
    d2 = t[t["n_trips"].isna()]
    y = np.arange(len(d2))
    ax.barh(y, d2["cov"], color=COL_FAIL, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(d2["label"], fontsize=6.5)
    ax.axvline(90, ls=":", color="#888", lw=0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("melhor mediana alcançada (%)")
    ax.set_title(f"(b) Não atingem 90% ({len(d2)}) — teto da linha",
                 fontsize=8)
    _clean(ax)
    ax.grid(False, axis="x")

    fig.suptitle("Quanto de dado cada linha precisa — e qual o teto de quem "
                 "não chega lá", fontsize=9)
    fig.savefig(out / "dv_09_data_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[dv_09] ok")
    return t


# ============================================================
# MAIN
# ============================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Execuções: {len(df):,} | linhas: {df['linha_s'].nunique()}")

    dv_01(df, OUT_DIR)
    dv_02(df, OUT_DIR)
    dv_03(df, OUT_DIR)
    dv_04(df, OUT_DIR)
    dv_05(df, OUT_DIR)
    dv_06(df, OUT_DIR)
    dv_07(df, OUT_DIR)
    dv_08(df, OUT_DIR)
    t = dv_09(df, OUT_DIR)

    print("\n=== Síntese: quanto de dado muda o jogo ===")
    lo = df[df["n_trips"] <= 2]
    hi = df[df["n_trips"] >= 100]
    print(f"  v1-v2 : cov15 mediana {lo[REF].median():.1f}% | "
          f"conclusão {100 * lo['completed'].mean():.1f}%")
    print(f"  v100+ : cov15 mediana {hi[REF].median():.1f}% | "
          f"conclusão {100 * hi['completed'].mean():.1f}%")
    print(f"  Tabela: {OUT_DIR / 'data_volume_summary.csv'}")


if __name__ == "__main__":
    main()
