"""
Figuras em mapa — experimento v8.

Para cada par (linha, sentido) desenha, sobre mapa de fundo:
  - a nuvem GPS de entrada (cinza),
  - a rota reconstruída (azul),
  - o shape oficial (vermelho tracejado),
  - terminais A (verde) e B (laranja).

Também monta uma galeria global com as melhores rotas.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj

import config as cfg
from analysis_common import (
    load_results, linhas_circulares, linhas_ida_volta, linhas_validas,
    LINE_COLORS,
)
from core.shapes import load_shape_models
from core.trips import (
    load_gps_data, build_trip_summary, select_trips,
    validate_trips, fetch_trip_points, build_configs,
)
from core.metrics import gps_adherence_metrics

warnings.filterwarnings("ignore")

ANALYSIS_DIR = Path(cfg.ANALYSIS_DIR)
PER_LINE_DIR = ANALYSIS_DIR / "per_line"

FWD = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

COLOR_ROUTE = "#1F77B4"
COLOR_SHAPE = "#D62728"
COLOR_GPS = "#B0B0B0"
COLOR_START = "#2CA02C"
COLOR_END = "#FF7F0E"

_WEB_MERCATOR_Z0 = 156543.03392


def _basemap_zoom(ax, fig_pixels=1080):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    extent_m = max(xmax - xmin, ymax - ymin, 1.0)
    m_per_px = extent_m / float(fig_pixels)
    if m_per_px <= 0:
        return 14
    return int(np.clip(round(np.log2(_WEB_MERCATOR_Z0 / m_per_px)), 8, 19))


def _add_basemap_safe(ax):
    try:
        ctx.add_basemap(
            ax, source=ctx.providers.Esri.WorldStreetMap,
            attribution=False, zoom=_basemap_zoom(ax),
        )
        return True
    except Exception:
        ax.set_facecolor("#F4F4F4")
        return False


# ------------------------------------------------------------
# PIPELINE (reproduz o run_experiment)
# ------------------------------------------------------------
_DATA = {}
_SHAPE_MODELS = None
_CONFIG_MAP = None


def get_shape_models():
    global _SHAPE_MODELS
    if _SHAPE_MODELS is None:
        _SHAPE_MODELS = load_shape_models(cfg.SHAPE_XZ_PATH)
    return _SHAPE_MODELS


def get_config_map():
    global _CONFIG_MAP
    if _CONFIG_MAP is None:
        configs = build_configs(
            get_shape_models(), linhas_circulares(), linhas_ida_volta()
        )
        _CONFIG_MAP = {
            (linha, sentido): (shape_id, start, end)
            for (linha, sentido, shape_id, start, end) in configs
        }
    return _CONFIG_MAP


def _get_pipeline():
    if "df_pl" not in _DATA:
        tmap = None
        if getattr(cfg, "SEGMENT_AT_TERMINALS", False):
            tmap = {}
            for linha in linhas_ida_volta():
                _, st_ida, en_ida = get_config_map()[(linha, "ida")]
                _, st_volta, en_volta = get_config_map()[(linha, "volta")]
                pts = [tuple(st_ida), tuple(en_ida),
                       tuple(st_volta), tuple(en_volta)]
                tmap[linha] = list(dict.fromkeys(pts))
        _DATA["df_pl"] = load_gps_data(
            cfg.PATH_POSITIONS, linhas=linhas_validas(),
            split_terminals=tmap, max_gap_s=cfg.MAX_GAP_S,
            max_speed_kmh=(cfg.MAX_SPEED_KMH
                           if getattr(cfg, "USE_SPEED_FILTER", False) else None),
            min_dt_s=cfg.MIN_DT_S, terminal_near_m=cfg.TERMINAL_NEAR_M,
            terminal_far_m=cfg.TERMINAL_FAR_M)
        _DATA["trip_summary"] = build_trip_summary(
            _DATA["df_pl"], min_trip_points=cfg.MIN_TRIP_POINTS)
    return _DATA["df_pl"], _DATA["trip_summary"]


def _fetch_gps(linha, sentido, n_trips):
    shape_id, start, end = get_config_map()[(linha, sentido)]
    df_pl, ts = _get_pipeline()

    sel = select_trips(ts, linha, start, end,
                       max_vehicles=None, start_radius_m=cfg.START_RADIUS_M)
    if sel.empty:
        return pd.DataFrame(columns=["lon", "lat"])
    sel = validate_trips(df_pl, sel, get_shape_models(), linha, shape_id,
                         max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
    if sel.empty:
        return pd.DataFrame(columns=["lon", "lat"])

    if n_trips is not None:
        sel = sel.sort_values("n_points", ascending=False).reset_index(
            drop=True).iloc[:n_trips]

    gps = fetch_trip_points(
        df_pl, sel, start=start, end=end,
        per_point_filter=getattr(cfg, "PER_POINT_FILTER", True)
        and (end is not None),
        max_points=cfg.MAX_INPUT_POINTS,
    ).dropna(subset=["lon", "lat"]).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)
    return gps[["lon", "lat", "dx", "dy"]]


# ------------------------------------------------------------
# DESENHO
# ------------------------------------------------------------
def _plot_map(ax, gps, route, shape_model, linha, sentido, subtitle,
              show_legend=False):
    if not gps.empty:
        gx, gy = FWD.transform(gps["lon"].to_numpy(), gps["lat"].to_numpy())
        ax.scatter(gx, gy, s=1.0, c=COLOR_GPS, alpha=0.35, linewidths=0,
                   label="amostras de GPS", rasterized=True)

    if shape_model is not None:
        sx, sy = FWD.transform(np.asarray(shape_model["lons"]),
                               np.asarray(shape_model["lats"]))
        ax.plot(sx, sy, "--", color=COLOR_SHAPE, lw=1.1, alpha=0.9,
                label="shape oficial")

    if route is not None and len(route):
        rx, ry = FWD.transform(route["lon"].to_numpy(),
                               route["lat"].to_numpy())
        ax.plot(rx, ry, "-", color=COLOR_ROUTE, lw=1.7, label="rota")

    shape_id, start, end = get_config_map()[(linha, sentido)]
    ax_start = FWD.transform(*start)
    ax.plot(*ax_start, "o", color=COLOR_START, ms=7, mec="white", mew=1.0,
            zorder=6, label="partida A")
    if end is not None:
        ax_end = FWD.transform(*end)
        ax.plot(*ax_end, "o", color=COLOR_END, ms=7, mec="white", mew=1.0,
                zorder=6, label="chegada B")

    ax.set_title(f"Linha {linha} — {sentido}\n{subtitle}", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    _add_basemap_safe(ax)
    if show_legend:
        ax.legend(fontsize=5.5, loc="lower right", frameon=True, ncol=2)


def make_line_maps(df: pd.DataFrame, out_dir: Path = PER_LINE_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_models = get_shape_models()
    n = 0
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        best = sub[sub["completed"]] if sub["completed"].any() else sub
        row = best.loc[best["coverage_pct"].idxmax()]
        nt = int(row["n_trips"])

        shape_id = get_config_map()[(linha, sentido)][0]
        shape_model = shape_models.get((linha, str(shape_id).strip()))
        gps = _fetch_gps(linha, sentido, nt)

        route_path = cfg.ROUTES_DIR / f"{linha}_{sentido}_v{nt}_melhor.csv"
        route = pd.read_csv(route_path) if route_path.exists() else None

        fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=True)
        _plot_map(
            ax, gps, route, shape_model, linha, sentido,
            f"n={nt} viagens | \u2264 5 m: {row['coverage_5m']:.0f}% | "
            f"\u2264 10 m: {row['coverage_10m']:.0f}% | "
            f"\u2264 15 m: {row['coverage_15m']:.0f}%",
            show_legend=True,
        )
        fig.savefig(out_dir / f"map_{linha}_{sentido}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        n += 1
        print(f"  [map] {linha}/{sentido} ok")
    print(f"[maps] {n} mapas -> {out_dir}")
    return n


def make_gallery(df: pd.DataFrame, out_dir: Path = ANALYSIS_DIR,
                 ncols: int = 4):
    """Galeria global: melhor rota de cada config."""
    configs = list(df[["linha_s", "sentido"]].drop_duplicates()
                   .itertuples(index=False))
    shape_models = get_shape_models()
    n = len(configs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (linha, sentido) in zip(axes, configs):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        best = sub[sub["completed"]] if sub["completed"].any() else sub
        row = best.loc[best["coverage_pct"].idxmax()]
        nt = int(row["n_trips"])
        shape_id = get_config_map()[(linha, sentido)][0]
        shape_model = shape_models.get((linha, str(shape_id).strip()))
        gps = _fetch_gps(linha, sentido, nt)
        route_path = cfg.ROUTES_DIR / f"{linha}_{sentido}_v{nt}_melhor.csv"
        route = pd.read_csv(route_path) if route_path.exists() else None
        _plot_map(
            ax, gps, route, shape_model, linha, sentido,
            f"n={nt} | \u2264 15 m: {row['coverage_15m']:.0f}%",
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Galeria — melhor rota por linha (dados multi-dia)",
                 fontsize=11)
    fig.savefig(out_dir / "fig_gallery_best_routes.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[gallery] {n} rotas -> fig_gallery_best_routes.png")


if __name__ == "__main__":
    dfr = load_results()
    make_line_maps(dfr)
    make_gallery(dfr)


# ============================================================
# GRID DE SENSIBILIDADE ESPACIAL (raio mínimo × fator de escala)
# ============================================================
def _snap_local(gps, start, end):
    M = cfg.M_PER_DEG_LAT
    d = np.sqrt((gps["lon"] - start[0]) ** 2
                + (gps["lat"] - start[1]) ** 2) * M
    i = int(d.values.argmin())
    ss = ([float(gps["lon"].median()), float(gps["lat"].median())]
          if d.min() > 500 else
          [float(gps["lon"].iloc[i]), float(gps["lat"].iloc[i])])
    es = end
    if end is not None:
        d2 = np.sqrt((gps["lon"] - end[0]) ** 2
                     + (gps["lat"] - end[1]) ** 2) * M
        j = int(d2.values.argmin())
        es = ([float(gps["lon"].median()), float(gps["lat"].median())]
              if d2.min() > 500 else
              [float(gps["lon"].iloc[j]), float(gps["lat"].iloc[j])])
    return ss, es


def _best_cell(cell: pd.DataFrame) -> pd.Series:
    done = cell[cell["completed"]]
    pool = done if not done.empty else cell
    return pool.loc[pool["coverage_pct"].idxmax()]


def make_sensitivity_grid(df, shape_models, out_path, linha, sentido,
                          n_trips=None, suptitle=None, min_list=None,
                          inc_list=None):
    """Matriz de mapas: raio mínimo (linhas) × fator de escala (colunas).

    Cada célula mostra a melhor execução daquela combinação de parâmetros
    (qualquer N angular) — GPS (cinza), shape oficial (vermelho) e rota.

    min_list / inc_list permitem fixar um subconjunto de raios e fatores.
    Fatores fora do grid do experimento (por exemplo 1,25) são executados
    diretamente, usando o melhor N angular disponível para o raio.
    """
    from core.model import discover_route
    from analysis_common import best_row as _best_row

    sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
    if sub.empty:
        print(f"[grid {linha}/{sentido}] sem dados")
        return
    if n_trips is None:
        n_trips = int(_best_cell(sub)["n_trips"])
    nt_eff = n_trips

    sid, start, end = get_config_map()[(linha, sentido)]
    shape = shape_models.get((linha, str(sid).strip()))
    if shape is None:
        print(f"[grid {linha}/{sentido}] shape nao encontrado")
        return
    gps = _fetch_gps(linha, sentido, nt_eff)
    if gps.empty:
        print(f"[grid {linha}/{sentido}] nuvem vazia")
        return
    ss, es = _snap_local(gps, start, end)
    m_lon = shape["m_per_deg_lon"]
    gx = gps["lon"].to_numpy() * m_lon
    gy = gps["lat"].to_numpy() * cfg.M_PER_DEG_LAT

    pad = 400.0
    x_min = min(gx.min(), shape["xs"].min()) - pad
    x_max = max(gx.max(), shape["xs"].max()) + pad
    y_min = min(gy.min(), shape["ys"].min()) - pad
    y_max = max(gy.max(), shape["ys"].max()) + pad

    min_list = (sorted(sub["meters"].unique()) if min_list is None
                else list(min_list))
    inc_list = (sorted(sub["increase_meters"].unique()) if inc_list is None
                else list(inc_list))

    box_ar = (y_max - y_min) / max(x_max - x_min, 1.0)
    fig_h = len(min_list) * (7.16 / len(inc_list)) * box_ar + 1.1
    fig, axes = plt.subplots(
        len(min_list), len(inc_list),
        figsize=(7.16, float(np.clip(fig_h, 2.6, 11.0))),
        constrained_layout=True, squeeze=False,
    )

    ref_h = route_h = start_h = None
    for r, min_m in enumerate(min_list):
        for c, inc_m in enumerate(inc_list):
            ax = axes[r, c]
            cell = sub[(sub["meters"] == min_m)
                       & (sub["increase_meters"] == inc_m)]
            ax.scatter(gx, gy, s=1.2, color=COLOR_GPS, alpha=0.28,
                       linewidths=0, rasterized=True, zorder=1)
            ref_h, = ax.plot(shape["xs"], shape["ys"], color=COLOR_SHAPE,
                             linewidth=0.9, linestyle="--", alpha=0.8,
                             zorder=2)
            row = None
            if not cell.empty:
                row = _best_cell(cell)
                ams = int(row["angular_samples"])
                mmax = float(row["max_meters"])
            else:
                alt = sub[sub["meters"] == min_m]
                if not alt.empty:
                    rr = _best_cell(alt)
                    ams = int(rr["angular_samples"])
                    mmax = float(rr["max_meters"])
                else:
                    ams, mmax = 180, float(cfg.MAX_METERS_LIST[0])
            route, info = discover_route(
                df=gps, initial_point=ss, end_point=es,
                meters=float(min_m),
                angular_samples=ams,
                increase_meters=float(inc_m),
                decrease_meters=1.0 / float(inc_m),
                max_meters=mmax,
                loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
                loop_min_steps=cfg.LOOP_MIN_STEPS,
                arrival_radius_m=cfg.ARRIVAL_RADIUS_M,
                consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
                smooth_iterations=cfg.SMOOTH_ITERATIONS,
            )
            rx = route["lon"].to_numpy() * m_lon
            ry = route["lat"].to_numpy() * cfg.M_PER_DEG_LAT
            if len(rx):
                route_h, = ax.plot(rx, ry, color=COLOR_ROUTE,
                                   linewidth=1.1, zorder=3)
                start_h = ax.scatter(
                    rx[0], ry[0], s=12, color=COLOR_START,
                    edgecolors="white", linewidths=0.4, zorder=4)
            ok = bool(info["reached_end"] or info["closed_loop"])
            adh = gps_adherence_metrics(route, gps)
            tag = "OK" if ok else str(info["stop_reason"])[:11]
            ax.text(
                0.04, 0.05,
                f"cov {adh['coverage_15m']:.0f}% | N{ams} | {tag}",
                transform=ax.transAxes, fontsize=5.5, ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.8},
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_box_aspect(box_ar)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#CCCCCC")

    for c, inc_m in enumerate(inc_list):
        axes[0, c].set_title(f"fator = {inc_m:g}", fontsize=7.5, pad=4)
    for r, min_m in enumerate(min_list):
        axes[r, 0].set_ylabel(f"raio {min_m:g} m", fontsize=7.5, rotation=90,
                              labelpad=4)

    if suptitle is None:
        tag = f", {nt_eff} viagens" if n_trips is not None else ""
        suptitle = (f"Sensibilidade espacial — Linha {linha} "
                    f"({sentido}{tag})")

    handles = [h for h in [ref_h, route_h, start_h] if h is not None]
    if handles:
        fig.legend(
            handles, ["Shape oficial", "Rota reconstruida", "Inicio"],
            loc="lower center", ncol=3, frameon=False, fontsize=7,
            bbox_to_anchor=(0.5, 0.005),
        )

    fig.suptitle(suptitle, fontsize=9.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid {linha}/{sentido}] ok -> {out_path.name}")


def make_all_sensitivity_grids(df, shape_models, out_dir=None):
    from analysis_common import best_row as _best_row
    if out_dir is None:
        out_dir = PER_LINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        nt = int(_best_row(sub)["n_trips"])
        make_sensitivity_grid(
            df, shape_models,
            out_dir / f"sensitivity_grid_{linha}_{sentido}.png",
            linha, sentido, n_trips=nt,
        )


def make_all_sensitivity_grids_subset(df, shape_models, min_list, inc_list,
                                      out_dir=None):
    """Grade reduzida (raio × fator) para cada par linha/sentido."""
    from analysis_common import best_row as _best_row
    if out_dir is None:
        out_dir = PER_LINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (linha, sentido) in df[["linha_s", "sentido"]].drop_duplicates() \
            .itertuples(index=False):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        nt = int(_best_row(sub)["n_trips"])
        make_sensitivity_grid(
            df, shape_models,
            out_dir / f"sensitivity_grid_{linha}_{sentido}.png",
            linha, sentido, n_trips=nt, min_list=min_list, inc_list=inc_list,
        )
        n += 1
    print(f"[grids] {n} grades -> {out_dir}")
    return n
