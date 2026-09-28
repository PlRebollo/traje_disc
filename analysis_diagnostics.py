"""
Diagnóstico de desempenho — experimento v8.

Investiga os casos de baixa aderência separando:
  - erro de dados (shape/terminal/etiquetagem incorretos);
  - limitação do modelo (caminhada trava, zigzagueia, etc.).

Gera:
  - diag_{linha}_{sentido}.png : mapa com nuvem GPS, shape oficial, rota
    reconstruída (colorida pelo avanço) e ponto de parada;
  - fig_perf_diagnostico.png  : painel comparativo (cobertura, MAE vs shape,
    razão de comprimento, cobertura do trajeto);
  - fig_runtime_boxplots.png  : boxplots de tempo por variável do modelo.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import contextily as ctx
import pyproj

import config as cfg
from analysis_common import load_results, best_row, ORDEM_LINHAS, LINE_COLORS
from core.shapes import load_shape_models, line_shapes
from core.trips import (load_gps_data, build_trip_summary, select_trips,
                        validate_trips, fetch_trip_points, build_configs)
from core.geometry import distance_matrix_to_polyline
from core.metrics import route_as_polyline_model

warnings.filterwarnings("ignore")

ANALYSIS_DIR = Path(cfg.ANALYSIS_DIR)
DIAG_DIR = ANALYSIS_DIR / "diagnostics"

FWD = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

_CACHE = {}


def _load():
    if "sm" not in _CACHE:
        _CACHE["sm"] = load_shape_models(cfg.SHAPE_XZ_PATH)
        configs = build_configs(_CACHE["sm"], cfg.LINHAS_CIRCULARES,
                                cfg.LINHAS_IDA_VOLTA)
        _CACHE["cmap"] = {(l, s): (sid, st, en)
                          for (l, s, sid, st, en) in configs}
        tmap = None
        if getattr(cfg, "SEGMENT_AT_TERMINALS", False):
            tmap = {}
            for (l, _s, _sid, st, en) in configs:
                if l not in cfg.LINHAS_IDA_VOLTA:
                    continue
                tmap.setdefault(l, [])
                tmap[l].append(tuple(st))
                if en is not None:
                    tmap[l].append(tuple(en))
        _CACHE["dfpl"] = load_gps_data(cfg.PATH_POSITIONS,
                                       linhas=sorted(set(cfg.LINHAS)),
                                       split_terminals=tmap)
        _CACHE["ts"] = build_trip_summary(_CACHE["dfpl"])
    return _CACHE


def _basemap_zoom(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ext = max(xmax - xmin, ymax - ymin, 1.0)
    z = int(round(np.log2(156543.03392 / (ext / 1080.0))))
    return int(np.clip(z, 8, 18))


def _basemap(ax):
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldStreetMap,
                        attribution=False, zoom=_basemap_zoom(ax))
    except Exception:
        ax.set_facecolor("#F4F4F4")


def _gps_for(linha, sentido, nt):
    c = _load()
    sid, start, end = c["cmap"][(linha, sentido)]
    sel = select_trips(c["ts"], linha, start, end, max_vehicles=None,
                       start_radius_m=cfg.START_RADIUS_M)
    if sel.empty:
        return pd.DataFrame(columns=["lon", "lat"]), start, end, sid
    sel = validate_trips(c["dfpl"], sel, c["sm"], linha, sid,
                         max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
    if not sel.empty and nt is not None:
        sel = sel.sort_values("n_points", ascending=False).reset_index(
            drop=True).iloc[:nt]
    if sel.empty:
        return pd.DataFrame(columns=["lon", "lat"]), start, end, sid
    gps = fetch_trip_points(c["dfpl"], sel, start=start, end=end,
                            per_point_filter=getattr(cfg, "PER_POINT_FILTER",
                                                    True) and (end is not None),
                            max_points=cfg.MAX_INPUT_POINTS
                            ).dropna(subset=["lon", "lat"]).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)
    return gps[["lon", "lat"]], start, end, sid


def diag_map(df, linha, sentido, out_dir=DIAG_DIR):
    """Mapa detalhado: GPS, shape, rota (por avanço) e ponto de parada."""
    c = _load()
    sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
    best = sub[sub["completed"]] if sub["completed"].any() else sub
    r = best.loc[best["coverage_15m"].idxmax()]
    nt = int(r["n_trips"])
    gps, start, end, sid = _gps_for(linha, sentido, nt)
    shape = c["sm"][(linha, str(sid).strip())]

    route_path = cfg.ROUTES_DIR / f"{linha}_{sentido}_v{nt}_melhor.csv"
    if not route_path.exists():
        return
    route = pd.read_csv(route_path)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    if not gps.empty:
        gx, gy = FWD.transform(gps["lon"].to_numpy(), gps["lat"].to_numpy())
        ax.scatter(gx, gy, s=1.0, c="#C0C0C0", alpha=0.35, linewidths=0,
                   rasterized=True, label="pings GPS")
    sx, sy = FWD.transform(np.asarray(shape["lons"]), np.asarray(shape["lats"]))
    ax.plot(sx, sy, "--", color="#D62728", lw=1.3, alpha=0.9,
            label="shape oficial")

    rx, ry = FWD.transform(route["lon"].to_numpy(), route["lat"].to_numpy())
    pts = np.linspace(0, 1, len(rx))
    ax.scatter(rx, ry, c=pts, cmap="viridis", s=9, zorder=5,
               edgecolors="none", label="rota (roxo=início, amarelo=fim)")
    ax.plot(*FWD.transform(*start), "o", color="#2CA02C", ms=9, mec="white",
            mew=1.2, zorder=7, label="partida A")
    if end is not None:
        ax.plot(*FWD.transform(*end), "o", color="#FF7F0E", ms=9, mec="white",
                mew=1.2, zorder=7, label="chegada B")
    # ponto de parada
    ax.plot(rx[-1], ry[-1], "X", color="#D62728", ms=11, mec="white", mew=1.2,
            zorder=8, label="parada")

    ratio = r["route_length_km"] / shape["length_km"]
    ax.set_title(
        f"Linha {linha} — {sentido}  (n={nt} viagens)\n"
        f"cov: 5m={r['coverage_5m']:.0f}% 10m={r['coverage_10m']:.0f}% "
        f"15m={r['coverage_15m']:.0f}% | MAE shape={r['mae_shape_m']:.0f} m | "
        f"rota/shape={ratio:.2f} | {r['stop_reason']}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    _basemap(ax)
    ax.legend(fontsize=6, loc="lower right", frameon=True, ncol=2)
    fig.savefig(out_dir / f"diag_{linha}_{sentido}.png", dpi=190,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  [diag] {linha}/{sentido} ratio={ratio:.2f} mae={r['mae_shape_m']:.0f}m")


def make_all_diag(df, out_dir=DIAG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        diag_map(df, linha, sentido, out_dir)
    print(f"[diag] mapas -> {out_dir}")


# ============================================================
# PAINEL DE DIAGNOSTICO
# ============================================================
def perf_panel(df, out_dir=ANALYSIS_DIR):
    c = _load()
    rows = []
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        r = best_row(sub)
        sid = r["shape_id"]
        m = c["sm"].get((linha, str(sid).strip()))
        skm = m["length_km"] if m else np.nan
        rows.append(dict(config=f"{linha}/{sentido}", cov15=r["coverage_15m"],
                         mae=r["mae_shape_m"], ratio=r["route_length_km"] / skm,
                         completed=int(r["completed"])))
    d = pd.DataFrame(rows).sort_values("cov15")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 6.0), constrained_layout=True)
    cores = ["#2CA02C" if r_ >= 80 else "#FF7F0E" if r_ >= 50 else "#D62728"
             for r_ in d["cov15"]]
    y = np.arange(len(d))

    axes[0].barh(y, d["cov15"], color=cores, edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["config"], fontsize=6.5)
    axes[0].axvline(80, ls="--", c="#888", lw=0.9)
    axes[0].set_xlabel("aderência \u2264 15 m (%)")
    axes[0].set_xlim(0, 105)
    axes[0].set_title("(a) Aderência aos dados", fontsize=8.5)

    axes[1].barh(y, d["mae"], color=cores, edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].axvline(50, ls="--", c="#888", lw=0.9)
    axes[1].set_xlabel("MAE rota vs shape (m)")
    axes[1].set_title("(b) Divergência do shape oficial", fontsize=8.5)

    axes[2].barh(y, d["ratio"], color=cores, edgecolor="white")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].axvline(1.0, ls="--", c="#2CA02C", lw=1.0)
    axes[2].axvline(1.25, ls=":", c="#D62728", lw=1.0)
    axes[2].set_xlabel("comprimento rota / shape")
    axes[2].set_title("(c) Excesso de comprimento", fontsize=8.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(labelsize=6.5)
        ax.invert_yaxis()

    fig.suptitle("Diagnóstico de desempenho por config — verde \u2265 80%, "
                 "laranja 50-80%, vermelho < 50%", fontsize=9)
    fig.savefig(out_dir / "fig_perf_diagnostico.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[perf] fig_perf_diagnostico.png")


# ============================================================
# BOXPLOTS DE TEMPO POR VARIAVEL
# ============================================================
def runtime_boxplots(df, out_dir=ANALYSIS_DIR):
    variaveis = [
        ("angular_samples", "amostras angulares (resolução)", [90, 180, 360]),
        ("meters", "raio mínimo (m)", None),
        ("increase_meters", "fator de escala do raio", None),
        ("n_trips", "nº de viagens", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9),
                             constrained_layout=True)
    pal = ["Blues", "Greens", "Oranges", "Purples"]
    for ax, (col, xlab, order), p in zip(axes, variaveis, pal):
        sns.boxplot(data=df, x=col, y="runtime_s", hue=col, legend=False,
                    palette=p, linewidth=0.7, fliersize=1.0, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel(xlab, fontsize=7.5)
        ax.set_ylabel("tempo de execução (s, log)" if col == "angular_samples"
                      else "", fontsize=7.5)
        ax.set_title(f"({chr(97 + variaveis.index((col, xlab, order)))}) "
                     f"{xlab.split(' (')[0]}", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.suptitle("Tempo de execução por variável do modelo", fontsize=9)
    fig.savefig(out_dir / "fig_runtime_boxplots.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[runtime] fig_runtime_boxplots.png")


def runtime_by_line(df, out_dir=ANALYSIS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4),
                             constrained_layout=True)
    ordem = [l for l in ORDEM_LINHAS if l in set(df["linha_s"])]
    sns.boxplot(data=df, x="linha_s", y="runtime_s", order=ordem,
                hue="linha_s", legend=False, palette="Set2",
                linewidth=0.7, fliersize=1.0, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("linha")
    axes[0].set_ylabel("tempo (s, log)")
    axes[0].set_title("(a) Tempo por linha", fontsize=8.5)
    axes[0].tick_params(axis="x", labelsize=6.5, rotation=0)

    sns.scatterplot(data=df, x="n_input_points", y="runtime_s",
                    hue="angular_samples", palette={90: "#9ECAE1", 180: "#4292C6",
                                                    360: "#084594"},
                    s=8, alpha=0.45, ax=axes[1])
    axes[1].set_xlabel("pontos GPS de entrada")
    axes[1].set_ylabel("tempo (s)")
    axes[1].set_title("(b) Tempo vs volume de dados", fontsize=8.5)
    axes[1].legend(fontsize=6.5, frameon=False, title="N",
                   title_fontsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.savefig(out_dir / "fig_runtime_by_line.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[runtime] fig_runtime_by_line.png")


def quantitative_diagnosis(df, out_dir=ANALYSIS_DIR):
    """
    Separa erro de DADO de limitação do MODELO, por config:

      gps_med      : mediana da distância dos pings ao shape oficial
                     (baixo = o shape corresponde ao dado observado)
      gps_p90      : cauda dessa distância
      frac_far     : % de pings a >100 m do shape (indício de 2º corredor
                     ou trip mal etiquetada)
      start_gap    : distância do início das trips ao terminal A da shape
                     (alto = ponto inicial mal definido)
      shape_cov30  : % do trajeto oficial coberto pela rota (<=30 m)
                     (baixo = a rota não seguiu o trajeto oficial)
    """
    c = _load()
    from core.geometry import meters_per_deg_lon as _mpl
    rows = []
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sid, start, end = c["cmap"][(linha, sentido)]
        shape = c["sm"][(linha, str(sid).strip())]
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        r = best_row(sub)
        nt = int(r["n_trips"])

        gps, _, _, _ = _gps_for(linha, sentido, nt)
        if gps.empty:
            continue
        d = distance_matrix_to_polyline(gps, shape)
        v = d[np.isfinite(d)]
        gps_med = float(np.median(v))
        gps_p90 = float(np.percentile(v, 90))
        frac_far = 100.0 * float(np.mean(v > 100))

        sel = select_trips(c["ts"], linha, start, end, max_vehicles=None,
                           start_radius_m=cfg.START_RADIUS_M)
        sel = validate_trips(c["dfpl"], sel, c["sm"], linha, sid,
                             max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        M = cfg.M_PER_DEG_LAT
        if not sel.empty:
            dA = np.sqrt((sel["lon_start"] - start[0])**2
                         + (sel["lat_start"] - start[1])**2) * M
            dAe = np.sqrt((sel["lon_end"] - start[0])**2
                          + (sel["lat_end"] - start[1])**2) * M
            start_gap = float(np.median(np.minimum(dA, dAe)))
        else:
            start_gap = np.nan

        rp = cfg.ROUTES_DIR / f"{linha}_{sentido}_v{nt}_melhor.csv"
        shape_cov30 = np.nan
        if rp.exists():
            route = pd.read_csv(rp)
            mlon = _mpl(float(route["lat"].mean()))
            model = {
                "ax": route["lon"].to_numpy()[:-1] * mlon,
                "ay": route["lat"].to_numpy()[:-1] * M,
                "bx": route["lon"].to_numpy()[1:] * mlon,
                "by": route["lat"].to_numpy()[1:] * M,
                "m_per_deg_lon": mlon,
            }
            sp = pd.DataFrame({"lon": shape["lons"], "lat": shape["lats"]})
            ds = distance_matrix_to_polyline(sp, model)
            vs = ds[np.isfinite(ds)]
            shape_cov30 = 100.0 * float(np.mean(vs <= 30))

        rows.append(dict(
            config=f"{linha}/{sentido}", cov15=round(r["coverage_15m"], 1),
            gps_med=round(gps_med, 1), gps_p90=round(gps_p90, 1),
            frac_far=round(frac_far, 1), start_gap=round(start_gap, 0),
            shape_cov30=round(shape_cov30, 1),
            ratio=round(r["route_length_km"] / shape["length_km"], 2),
            mae=round(r["mae_shape_m"], 0), ok=int(r["completed"]),
            stop=r["stop_reason"]))

    out = pd.DataFrame(rows).sort_values("cov15")
    out.to_csv(out_dir / "diagnostico_quantitativo.csv", index=False)
    print("[diag] diagnostico_quantitativo.csv")
    return out


if __name__ == "__main__":
    dfr = load_results()
    make_all_diag(dfr)
    perf_panel(dfr)
    runtime_boxplots(dfr)
    runtime_by_line(dfr)
    quantitative_diagnosis(dfr)
