"""
Análise dos limiares de corte por terminal (data-driven).

Mostra, a partir dos próprios dados, qual é a zona de terminal:
  - distribuição da distância à amostra de GPS até o terminal mais próximo;
  - velocidade mediana em função dessa distância;
  - fração de amostras dentro de cada raio.

A ideia é justificar limiares de "perto" (zona de manobra/parada) e
"longe" (quando o veículo já saiu de fato) sem escolhê-los ao acaso.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from core.trips import load_gps_data  # noqa: E402
from core.trips import build_configs  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402


def main():
    sm = load_shape_models(cfg.SHAPE_XZ_PATH)
    configs = build_configs(sm, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    tmap = {}
    for (linha, _s, _sid, st, en) in configs:
        if linha not in cfg.LINHAS_IDA_VOLTA:
            continue
        tmap.setdefault(linha, [])
        tmap[linha].append(tuple(st))
        if en is not None:
            tmap[linha].append(tuple(en))

    df = load_gps_data(cfg.PATH_POSITIONS, linhas=cfg.LINHAS,
                       split_terminals=tmap, max_gap_s=900,
                       max_speed_kmh=None, min_dt_s=1)

    rows_t = []
    for k_, pts_ in tmap.items():
        for p in pts_:
            rows_t.append({"linha": str(k_), "t_lon": float(p[0]),
                           "t_lat": float(p[1])})
    term = pl.DataFrame(rows_t)
    idx = df.with_row_index("_rid")
    j = idx.join(term, on="linha", how="inner")
    j = j.with_columns(
        (
            ((pl.col("lon") - pl.col("t_lon")) * cfg.M_PER_DEG_LAT
             * (pl.col("lat") * np.pi / 180).cos()) ** 2
            + ((pl.col("lat") - pl.col("t_lat")) * cfg.M_PER_DEG_LAT) ** 2
        ).sqrt().alias("d_t")
    )
    dmin = j.group_by("_rid").agg(pl.col("d_t").min().alias("d_term"))
    sub = (idx.join(dmin, on="_rid", how="inner")
           .select(["linha", "d_term", "speed_kmh"]).to_pandas())
    sub = sub[np.isfinite(sub["d_term"])]
    sub = sub[sub["speed_kmh"].notna() & (sub["speed_kmh"] < 120)]
    print(f"amostras de GPS ida/volta: {len(sub):,}")

    edges = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]
    sub["faixa"] = pd.cut(sub["d_term"], edges)
    g = sub.groupby("faixa", observed=True).agg(
        n=("d_term", "size"),
        speed=("speed_kmh", "median"),
    )
    g["centro"] = [iv.mid for iv in g.index]
    g["pct"] = 100 * g["n"] / g["n"].sum()
    g.to_csv(Path(cfg.OUT_ROOT) / "limiares_terminal.csv")
    print(g.round(2).to_string())

    # a que distância a velocidade se aproxima do cruzeiro?
    cruise = sub.loc[sub["d_term"] > 1000, "speed_kmh"].median()
    print(f"velocidade de cruzeiro (d>1000m): {cruise:.1f} km/h")
    for frac in (0.5, 0.75, 0.9):
        alvo = frac * cruise
        cand = g[g["speed"] >= alvo]
        dmin_ = cand["centro"].min() if len(cand) else np.nan
        print(f"  velocidade atinge {int(frac*100)}% do cruzeiro (~{alvo:.0f} "
              f"km/h) a partir de ~{dmin_:.0f} m do terminal")

    frac_within = {t: 100 * (sub["d_term"] <= t).mean()
                   for t in (100, 150, 200, 300, 500, 750, 1000)}
    print("fração de amostras dentro de X m:", {k: round(v, 2)
                                                 for k, v in frac_within.items()})

    OUT = Path(cfg.OUT_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6),
                             constrained_layout=True)

    ax = axes[0]
    ax.semilogy(g["centro"], g["pct"], "-o", color="#1f6fb2", ms=3.5)
    ax.set_xlabel("distância ao terminal (m)")
    ax.set_ylabel("fração das amostras (%, log)")
    ax.set_title("(a) Onde ficam as amostras", fontsize=8)

    ax = axes[1]
    ax.plot(g["centro"], g["speed"], "-o", color="#d62728", ms=3.5)
    ax.axhline(cruise, ls=":", color="#888", lw=0.9)
    ax.axvline(150, ls="--", color="#2ca02c", lw=1.0)
    ax.axvline(500, ls="--", color="#2ca02c", lw=1.0)
    ax.set_xlabel("distância ao terminal (m)")
    ax.set_ylabel("velocidade mediana (km/h)")
    ax.set_title("(b) Velocidade x distância\n(zona de manobra)", fontsize=8)

    ax = axes[2]
    xs = sorted(frac_within)
    ax.plot(xs, [frac_within[x] for x in xs], "-o", color="#6a51a3", ms=3.5)
    for t in (150, 500):
        ax.axvline(t, ls="--", color="#2ca02c", lw=1.0)
    ax.set_xlabel("raio X (m)")
    ax.set_ylabel("amostras com d_term \u2264 X (%)")
    ax.set_title("(c) Cobertura acumulada", fontsize=8)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, alpha=0.3)
        a.tick_params(labelsize=7)
    fig.suptitle("Definição dos limiares de corte por terminal a partir dos "
                 "dados", fontsize=9.5)
    fig.savefig(OUT / "limiares_terminal.png", dpi=200, bbox_inches="tight")
    print("figura -> outputs/limiares_terminal.png")


if __name__ == "__main__":
    main()
