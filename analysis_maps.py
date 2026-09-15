"""
Figuras em mapa do experimento v4.

Reaproveita o pipeline de dados do experimento (core/trips.py) para
reconstruir a nuvem GPS de entrada de cada config e desenhar as rotas
sobre mapa de fundo (Esri WorldStreetMap) ou em coordenadas metricas.

Figuras produzidas em outputs/analysis/:
- fig_04_sensitivity_grid_*.png : grid espacial de destaque
- per_line/sensitivity_grid_*.png : grid espacial de cada config
- fig_05_success_gallery.png    : galeria de melhores execucoes
- fig_success_*.png             : imagens de sucesso classicas
- per_line/best_route_*.png     : melhor execucao de cada config
- fig_14_direction_separation_303.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import contextily as ctx
import pyproj

import config as cfg
from analysis_common import load_results, ordered_configs
from core.shapes import load_shape_models
from core.trips import (
    load_gps_data, build_trip_summary, select_trips,
    validate_trips, fetch_trip_points, build_configs,
)

warnings.filterwarnings("ignore")

M_PER_DEG_LAT = cfg.M_PER_DEG_LAT

ANALYSIS_DIR = cfg.ANALYSIS_DIR
PER_LINE_DIR = ANALYSIS_DIR / "per_line"

FWD = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
INV = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

COLOR_ROUTE = "#1F77B4"
COLOR_SHAPE = "#D62728"
COLOR_GPS = "#A0A0A0"
COLOR_START = "#2CA02C"
COLOR_END = "#FF7F0E"

_WEB_MERCATOR_Z0 = 156543.03392


# ============================================================
# MAPA DE FUNDO
# ============================================================
def _basemap_zoom(ax, fig_pixels=1080):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    extent_m = max(xmax - xmin, ymax - ymin, 1.0)
    m_per_px = extent_m / float(fig_pixels)
    if m_per_px <= 0:
        return 14
    zoom = int(round(np.log2(_WEB_MERCATOR_Z0 / m_per_px)))
    return int(np.clip(zoom, 8, 19))


def _add_basemap_safe(ax) -> bool:
    try:
        ctx.add_basemap(
            ax, source=ctx.providers.Esri.WorldStreetMap,
            attribution=False, zoom=_basemap_zoom(ax),
        )
        return True
    except Exception:
        ax.set_facecolor("#F4F4F4")
        return False


# ============================================================
# PIPELINE DE DADOS (reproduz o run_experiment)
# ============================================================
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
            get_shape_models(), cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA
        )
        _CONFIG_MAP = {
            (linha, sentido): (shape_id, start, end)
            for (linha, sentido, shape_id, start, end) in configs
        }
    return _CONFIG_MAP


def _get_pipeline():
    if "df_pl" not in _DATA:
        _DATA["df_pl"] = load_gps_data(cfg.PATH_POSITIONS, linhas=cfg.LINHAS)
        _DATA["trip_summary"] = build_trip_summary(_DATA["df_pl"])
    return _DATA["df_pl"], _DATA["trip_summary"]


def _empty_gps():
    return pd.DataFrame(columns=["lon", "lat", "dx", "dy"])


def _fetch(linha, sentido, n_buses, per_point):
    shape_id, start, end = get_config_map()[(linha, sentido)]
    df_pl, ts = _get_pipeline()

    max_veh = None if end is not None else max(cfg.BUS_COUNTS)
    sel = select_trips(
        ts, linha, start, end, max_vehicles=max_veh,
        start_radius_m=cfg.START_RADIUS_M,
    )
    if sel.empty:
        return _empty_gps()

    sel = validate_trips(
        df_pl, sel, get_shape_models(), linha, shape_id,
        max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M,
    )
    if sel.empty:
        return _empty_gps()

    if end is None and n_buses is not None:
        veh_order = (sel.groupby("veiculo")["n_points"].sum()
                     .sort_values(ascending=False).index.tolist())
        if len(veh_order) < n_buses:
            return _empty_gps()
        sel = sel[sel["veiculo"].isin(veh_order[:n_buses])]

    gps = fetch_trip_points(
        df_pl, sel, start=start, end=end,
        per_point_filter=per_point, max_points=cfg.MAX_INPUT_POINTS,
    )
    return (gps.dropna(subset=["lon", "lat"])
            .drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True))


def get_input_cloud(linha, sentido, n_buses=None):
    """Nuvem GPS de entrada de um config (mesma do experimento)."""
    key = ("in", linha, sentido, n_buses)
    if key not in _DATA:
        _DATA[key] = _fetch(
            linha, sentido, n_buses, per_point=(sentido != "circular")
        )
    return _DATA[key]


def get_raw_cloud(linha, n_vehicles=6):
    """Nuvem GPS bruta da linha (todos os sentidos, sem filtro direcional)."""
    key = ("raw", linha, n_vehicles)
    if key not in _DATA:
        shape_id, start, _ = get_config_map()[(linha, "circular")] \
            if (linha, "circular") in get_config_map() \
            else get_config_map()[(linha, "ida")]
        df_pl, ts = _get_pipeline()
        sel = select_trips(
            ts, linha, start, None, max_vehicles=n_vehicles,
            start_radius_m=cfg.START_RADIUS_M,
        )
        gps = fetch_trip_points(
            df_pl, sel, start=start, end=None,
            per_point_filter=False, max_points=cfg.MAX_INPUT_POINTS,
        )
        _DATA[key] = (gps.dropna(subset=["lon", "lat"])
                      .reset_index(drop=True))
    return _DATA[key]


def load_route(run_id) -> pd.DataFrame:
    return pd.read_csv(cfg.ROUTES_DIR / f"route_{run_id}.csv")


def shape_length_km(shape_models, linha: str, shape_id) -> float:
    m = shape_models.get((str(linha).strip(), str(shape_id).strip()))
    if m is None:
        return np.nan
    return float(m["length_km"])


# ============================================================
# HELPERS DE EIXOS
# ============================================================
def _lon_formatter(x, pos):
    lon, _ = INV.transform(x, 0)
    return f"{lon:.2f}\u00b0"


def _lat_formatter(y, pos):
    _, lat = INV.transform(0, y)
    return f"{lat:.2f}\u00b0"


def _style_map_axis(ax, col_idx, n_cols):
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_lon_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_lat_formatter))
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Longitude", fontsize=7, labelpad=1)
    if col_idx == 0:
        ax.set_ylabel("Latitude", fontsize=7, labelpad=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#333333")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# PAINEL DE ROTA EM MAPA
# ============================================================
def plot_route_map_panel(
    ax, route, shape_model, gps=None,
    title="", metrics="", max_gps=9000, pad_m=250.0,
):
    sx, sy = FWD.transform(
        shape_model["xs"] / shape_model["m_per_deg_lon"],
        shape_model["ys"] / M_PER_DEG_LAT,
    )
    rx, ry = FWD.transform(route["lon"].to_numpy(), route["lat"].to_numpy())

    if gps is not None and len(gps) > 0:
        g = gps if len(gps) <= max_gps else gps.sample(n=max_gps, random_state=42)
        gx, gy = FWD.transform(g["lon"].to_numpy(), g["lat"].to_numpy())
        ax.scatter(gx, gy, s=1.5, color=COLOR_GPS, alpha=0.35,
                   linewidths=0, rasterized=True, zorder=2)

    ax.plot(sx, sy, color=COLOR_SHAPE, linewidth=1.1, linestyle="--",
            alpha=0.85, label="Shape oficial", zorder=3)
    ax.plot(rx, ry, color=COLOR_ROUTE, linewidth=1.5, alpha=0.95,
            label="Rota reconstruida", zorder=4)
    ax.scatter(rx[0], ry[0], s=28, color=COLOR_START, edgecolor="black",
               linewidth=0.5, zorder=5, label="Inicio")
    ax.scatter(rx[-1], ry[-1], s=34, color=COLOR_END, marker="X",
               edgecolor="black", linewidth=0.5, zorder=5, label="Fim")

    xmin = min(sx.min(), rx.min())
    xmax = max(sx.max(), rx.max())
    ymin = min(sy.min(), ry.min())
    ymax = max(sy.max(), ry.max())
    pad = max((xmax - xmin) * 0.05, pad_m)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    _add_basemap_safe(ax)

    if title:
        ax.set_title(title, fontsize=8, pad=3)
    if metrics:
        ax.text(
            0.03, 0.04, metrics, transform=ax.transAxes,
            fontsize=6.2, ha="left", va="bottom", color="#222222",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": "#BBBBBB", "alpha": 0.88},
        )


# ============================================================
# GRID DE SENSIBILIDADE ESPACIAL
# ============================================================
def make_sensitivity_grid(
    df, shape_models, out_path, linha, sentido, n_buses=None,
    samples=360, suptitle=None,
):
    df = df.copy()
    df["linha_s"] = df["linha"].astype(str).str.zfill(3)

    sub = df[
        (df["linha_s"] == linha)
        & (df["sentido"] == sentido)
        & (df["angular_samples"] == samples)
    ].copy()
    if n_buses is not None:
        sub = sub[sub["n_buses"] == n_buses]

    if sub.empty:
        print(f"[grid {linha}/{sentido}] sem dados")
        return

    shape_id = str(sub.iloc[0]["shape_id"]).strip()
    shape = shape_models.get((linha, shape_id))
    if shape is None:
        print(f"[grid {linha}/{sentido}] shape {shape_id} nao encontrado")
        return

    gps = get_input_cloud(
        linha, sentido, n_buses if sentido == "circular" else None
    )
    m_lon = shape["m_per_deg_lon"]

    gx = gps["lon"].to_numpy() * m_lon
    gy = gps["lat"].to_numpy() * M_PER_DEG_LAT

    pad = 400.0
    x_min = min(gx.min(), shape["xs"].min()) - pad
    x_max = max(gx.max(), shape["xs"].max()) + pad
    y_min = min(gy.min(), shape["ys"].min()) - pad
    y_max = max(gy.max(), shape["ys"].max()) + pad

    min_list = sorted(sub["meters"].unique())
    inc_list = sorted(sub["increase_meters"].unique())

    fig, axes = plt.subplots(
        len(min_list), len(inc_list),
        figsize=(7.16, 5.8), sharex=True, sharey=True, squeeze=False,
    )

    ref_h = route_h = start_h = None
    for r, min_m in enumerate(min_list):
        for c, inc_m in enumerate(inc_list):
            ax = axes[r, c]
            cell = sub[(sub["meters"] == min_m)
                       & (sub["increase_meters"] == inc_m)]
            ax.scatter(gx, gy, s=1.2, color=COLOR_GPS, alpha=0.28,
                       linewidths=0, rasterized=True, zorder=1)
            ref_h, = ax.plot(
                shape["xs"], shape["ys"], color=COLOR_SHAPE,
                linewidth=0.9, linestyle="--", alpha=0.8, zorder=2,
            )

            if not cell.empty:
                row = cell.iloc[0]
                route = load_route(row["run_id"])
                rx = route["lon"].to_numpy() * m_lon
                ry = route["lat"].to_numpy() * M_PER_DEG_LAT
                route_h, = ax.plot(rx, ry, color=COLOR_ROUTE, linewidth=1.1,
                                   zorder=3)
                start_h = ax.scatter(
                    rx[0], ry[0], s=12, color=COLOR_START,
                    edgecolors="white", linewidths=0.4, zorder=4,
                )
                ok = bool(row["reached_end"] or row["closed_loop"])
                tag = "OK" if ok else str(row["stop_reason"])[:12]
                ax.text(
                    0.04, 0.06, f"cov {row['coverage_pct']:.0f}% | {tag}",
                    transform=ax.transAxes, fontsize=6, ha="left", va="bottom",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.8},
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#CCCCCC")

    for c, inc_m in enumerate(inc_list):
        axes[0, c].set_title(f"fator de aumento = {inc_m:g}", fontsize=8, pad=5)
    for r, min_m in enumerate(min_list):
        axes[r, 0].set_ylabel(f"raio min = {min_m:g} m", fontsize=8,
                              rotation=90, labelpad=4)

    if suptitle is None:
        done = sub[sub["reached_end"] + sub["closed_loop"] > 0]
        incs_ok = sorted(done["increase_meters"].unique())
        if len(incs_ok) == 0:
            nota = "nenhuma combinacao completa o trajeto"
        elif len(incs_ok) < len(inc_list):
            nota = "completa somente com fator " + " ou ".join(
                f"{v:g}" for v in incs_ok
            )
        else:
            nota = "completa com todos os fatores"
        tag = f", {n_buses} onibus" if n_buses is not None else ""
        suptitle = (f"Sensibilidade espacial — Linha {linha} "
                    f"({sentido}{tag}): {nota}")

    handles = [h for h in [ref_h, route_h, start_h] if h is not None]
    if handles:
        fig.legend(
            handles, ["Shape oficial", "Rota reconstruida", "Inicio"],
            loc="lower center", ncol=3, frameon=False, fontsize=7,
            bbox_to_anchor=(0.5, 0.012),
        )

    fig.suptitle(suptitle, fontsize=9.5, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.13,
                        wspace=0.04, hspace=0.08)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid {linha}/{sentido}"
          f"{' b' + str(n_buses) if n_buses is not None else ''}] ok")


HIGHLIGHT_GRID = ("607", "ida", None)


def make_fig_04_sensitivity_grid(df, shape_models, out_dir: Path):
    linha, sentido, nb = HIGHLIGHT_GRID
    make_sensitivity_grid(
        df, shape_models,
        out_dir / f"fig_04_sensitivity_grid_{linha}_{sentido}.png",
        linha, sentido, n_buses=nb, samples=360,
    )


def make_all_sensitivity_grids(df, shape_models, out_dir: Path = None):
    if out_dir is None:
        out_dir = PER_LINE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_configs = []
    for linha in cfg.LINHAS_IDA_VOLTA:
        grid_configs += [(linha, "ida", None), (linha, "volta", None)]
    for linha in cfg.LINHAS_CIRCULARES:
        grid_configs += [(linha, "circular", 2), (linha, "circular", 5)]

    for linha, sentido, nb in grid_configs:
        name = f"sensitivity_grid_{linha}_{sentido}"
        if nb is not None:
            name += f"_b{nb}"
        make_sensitivity_grid(
            df, shape_models, out_dir / f"{name}.png",
            linha, sentido, n_buses=nb, samples=360,
        )


# ============================================================
# GALERIA DE SUCESSO
# ============================================================
GALLERY = [
    ("303", "ida", None), ("303", "volta", None),
    ("607", "ida", None), ("607", "volta", None),
    ("203", "ida", None), ("924", "volta", None),
    ("020", "circular", 2), ("021", "circular", 2),
]


def _best_row(df, linha, sentido, nb):
    sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
    if nb is not None:
        sub = sub[sub["n_buses"] == nb]
    done = sub[(sub["reached_end"] == 1) | (sub["closed_loop"] == 1)]
    pool = done if not done.empty else sub
    return pool.loc[pool["coverage_pct"].idxmax()]


def make_fig_05_success_gallery(df, shape_models, out_dir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.8))

    for k, (linha, sentido, nb) in enumerate(GALLERY):
        ax = axes[k // 4, k % 4]
        row = _best_row(df, linha, sentido, nb)
        route = load_route(row["run_id"])
        shape = shape_models[(linha, str(row["shape_id"]).strip())]
        gps = get_input_cloud(linha, sentido, nb)

        plot_route_map_panel(
            ax, route, shape, gps=gps,
            title=f"Linha {linha} ({sentido})",
            metrics=(f"cov {row['coverage_pct']:.1f}% | "
                     f"d {row['mean_gps_dist_m']:.1f} m | "
                     f"{row['route_length_km']:.1f} km"),
        )
        _style_map_axis(ax, k % 4, 4)

    fig.legend(
        handles=[
            plt.Line2D([], [], color=COLOR_SHAPE, lw=1.1, ls="--",
                       label="Shape oficial"),
            plt.Line2D([], [], color=COLOR_ROUTE, lw=1.5,
                       label="Rota reconstruida"),
            plt.Line2D([], [], marker="o", ls="", color=COLOR_START,
                       markersize=6, label="Inicio"),
            plt.Line2D([], [], marker="X", ls="", color=COLOR_END,
                       markersize=6, label="Fim"),
        ],
        loc="lower center", ncol=4, frameon=True, fontsize=7,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "Casos de sucesso — melhor execucao por linha/sentido",
        fontsize=9.5, y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17,
                        wspace=0.12, hspace=0.24)
    fig.savefig(out_dir / "fig_05_success_gallery.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_05] ok")


# ============================================================
# IMAGENS INDIVIDUAIS — melhor execucao de CADA config
# ============================================================
def _big_route_image(df, shape_models, row, out_path: Path):
    completed = bool(row["reached_end"] or row["closed_loop"])
    linha = str(row["linha"]).zfill(3)
    sentido = row["sentido"]
    nb = int(row["n_buses"])

    route = load_route(row["run_id"])
    shape = shape_models[(linha, str(row["shape_id"]).strip())]
    gps = get_input_cloud(
        linha, sentido, nb if sentido == "circular" else None
    )

    if completed:
        head = f"Caso de sucesso — Linha {linha} ({sentido}"
        head += f", {nb} onibus)" if sentido == "circular" else ")"
        status = ("chegada ao destino OK" if row["reached_end"]
                  else "loop fechado OK")
    else:
        head = f"Melhor resultado (parcial) — Linha {linha} ({sentido}"
        head += f", {nb} onibus)" if sentido == "circular" else ")"
        status = f"parou: {row['stop_reason']}"

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    plot_route_map_panel(
        ax, route, shape, gps=gps,
        title=(head + "\n"
               f"aderencia {row['coverage_pct']:.1f}% | "
               f"dist. media GPS {row['mean_gps_dist_m']:.1f} m | "
               f"{row['route_length_km']:.1f} km | {status}"),
        metrics=(f"smp={int(row['angular_samples'])} "
                 f"min={row['meters']:.0f}m inc={row['increase_meters']} | "
                 f"MAE shape {row['mae_shape_m']:.1f} m"),
        max_gps=14000,
    )
    _style_map_axis(ax, 0, 1)
    ax.tick_params(labelsize=8)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        frameon=True, fontsize=8, bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(bottom=0.09, top=0.93)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_best_route_images(df, shape_models, out_dir: Path):
    PER_LINE_DIR.mkdir(parents=True, exist_ok=True)
    for linha, sentido, nb in ordered_configs(df, with_buses=True):
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        if nb is not None:
            sub = sub[sub["n_buses"] == nb]
        if sub.empty:
            continue
        done = sub[(sub["reached_end"] == 1) | (sub["closed_loop"] == 1)]
        pool = done if not done.empty else sub
        row = pool.loc[pool["coverage_pct"].idxmax()]
        name = f"best_route_{linha}_{sentido}"
        if sentido == "circular":
            name += f"_b{int(nb)}"
        _big_route_image(
            df, shape_models, row, PER_LINE_DIR / f"{name}.png"
        )
        print(f"[best_route {name}] ok")


def make_success_images(df, shape_models, out_dir: Path):
    cases = [
        ("303", "ida", None, "fig_success_303_ida"),
        ("020", "circular", 2, "fig_success_020_circular"),
        ("021", "circular", 2, "fig_success_021_circular"),
    ]
    for linha, sentido, nb, name in cases:
        row = _best_row(df, linha, sentido, nb)
        _big_route_image(
            df, shape_models, row, out_dir / f"{name}.png"
        )
        print(f"[{name}] ok")


# ============================================================
# SEPARACAO DE SENTIDO (linha 303)
# ============================================================
def make_fig_14_direction_separation(df, shape_models, out_dir: Path):
    linha = "303"
    raw = get_raw_cloud(linha, n_vehicles=6)
    ida = get_input_cloud(linha, "ida")

    row = _best_row(df, linha, "ida", None)
    route = load_route(row["run_id"])
    shape = shape_models[(linha, str(row["shape_id"]).strip())]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95))

    ax = axes[0]
    g = raw if len(raw) <= 12000 else raw.sample(n=12000, random_state=42)
    gx, gy = FWD.transform(g["lon"].to_numpy(), g["lat"].to_numpy())
    ax.scatter(gx, gy, s=1.2, color="#C44E52", alpha=0.3,
               linewidths=0, rasterized=True)
    ax.set_title("(1) Nuvem bruta\nida + volta misturadas", fontsize=7.5)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    g2 = ida if len(ida) <= 12000 else ida.sample(n=12000, random_state=42)
    gx2, gy2 = FWD.transform(g2["lon"].to_numpy(), g2["lat"].to_numpy())
    ax.scatter(gx2, gy2, s=1.2, color=COLOR_ROUTE, alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_title("(2) Apos separacao\napenas sentido ida", fontsize=7.5)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    allx = np.concatenate([gx, gx2])
    ally = np.concatenate([gy, gy2])
    for ax in (axes[0], axes[1]):
        ax.set_xlim(allx.min() - 300, allx.max() + 300)
        ax.set_ylim(ally.min() - 300, ally.max() + 300)

    ax = axes[2]
    plot_route_map_panel(
        ax, route, shape, gps=ida,
        title="(3) Rota reconstruida\nsobre o sentido ida",
        metrics=f"cov {row['coverage_pct']:.1f}% | chegada OK",
        max_gps=9000,
    )
    _style_map_axis(ax, 2, 3)

    fig.suptitle(
        "Separacao geometrica de sentido — Linha 303: "
        "a nuvem misturada vira um corredor de mao unica",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.07,
                        wspace=0.1)
    fig.savefig(out_dir / "fig_14_direction_separation_303.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_14] ok")


# ============================================================
# ANTES/DEPOIS (linha 020) — modelo original vs v4
# ============================================================
def make_fig_before_after(df, shape_models, out_dir: Path):
    old_summary = cfg.BASE / "outputs_original_v2" / "grid_summary.csv"
    old_row = None
    old_route = None
    if old_summary.exists():
        old = pd.read_csv(old_summary)
        old["linha_s"] = old["linha"].astype(str).str.zfill(3)
        cand = old[old["linha_s"] == "020"]
        if not cand.empty:
            old_row = cand.loc[cand["n_route_points"].idxmax()]
            route_path = Path(str(old_row["route_file"]))
            if not route_path.exists():
                route_path = cfg.BASE / str(old_row["route_file"])
            if route_path.exists():
                old_route = pd.read_csv(route_path)
    if old_row is None or old_route is None:
        print("[fig_09] rota antiga indisponivel, pulando")
        return

    new_row = _best_row(df, "020", "circular", 5)
    new_route = load_route(new_row["run_id"])
    shape = shape_models[("020", str(new_row["shape_id"]).strip())]
    gps = get_input_cloud("020", "circular", 5)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 4.3))

    ax = axes[0]
    plot_route_map_panel(
        ax, old_route, shape, gps=None,
        title="Antes — modelo original",
        metrics=(f"MAE shape {old_row['mae_m']:.1f} m | "
                 f"{int(old_row['n_route_points'])} pts | "
                 f"{old_row['route_length_km']:.1f} km"),
    )
    _style_map_axis(ax, 0, 2)

    ax = axes[1]
    plot_route_map_panel(
        ax, new_route, shape, gps=gps,
        title="Depois — modelo v4 (5 onibus)",
        metrics=(f"cov {new_row['coverage_pct']:.1f}% | "
                 f"MAE shape {new_row['mae_shape_m']:.1f} m | "
                 f"{int(new_row['n_route_points'])} pts"),
        max_gps=9000,
    )
    _style_map_axis(ax, 1, 2)

    fig.legend(
        handles=[
            plt.Line2D([], [], color=COLOR_SHAPE, lw=1.1, ls="--",
                       label="Shape oficial"),
            plt.Line2D([], [], color=COLOR_ROUTE, lw=1.5,
                       label="Rota reconstruida"),
            plt.Line2D([], [], marker="o", ls="", color=COLOR_START,
                       markersize=6, label="Inicio"),
            plt.Line2D([], [], marker="X", ls="", color=COLOR_END,
                       markersize=6, label="Fim"),
        ],
        loc="lower center", ncol=4, frameon=True, fontsize=7,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle("Linha 020 (circular) — evolucao do modelo",
                 fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.15,
                        wspace=0.18)
    fig.savefig(out_dir / "fig_09_before_after_020.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig_09] ok")
