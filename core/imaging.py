"""
Imagens das rotas com mapa de fundo (Esri WorldStreetMap).

- Provedor sem chave de API e sem marca d'água.
- Zoom explícito calculado da extensão, limitado a [8, 19]
  (evita o erro de zoom inválido do modo automático).
- Fallback para fundo cinza claro se os tiles falharem (rede).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import pyproj

from config import M_PER_DEG_LAT

_FWD = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_WEB_MERCATOR_Z0 = 156543.03392


def _zoom_for_extent(ax, fig_pixels=1080):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    extent_m = max(xmax - xmin, ymax - ymin, 1.0)
    m_per_px = extent_m / float(fig_pixels)
    z = int(round(np.log2(_WEB_MERCATOR_Z0 / m_per_px))) if m_per_px > 0 else 14
    return int(np.clip(z, 8, 19))


def _basemap(ax):
    try:
        ctx.add_basemap(
            ax, source=ctx.providers.Esri.WorldStreetMap,
            attribution=False, zoom=_zoom_for_extent(ax),
        )
    except Exception:
        ax.set_facecolor("#F4F4F4")


def save_route_image(route: pd.DataFrame, info: dict, output_path):
    """Desenha a rota sobre o mapa e salva em PNG."""
    if route is None or len(route) < 2:
        return

    rx, ry = _FWD.transform(route["lon"].to_numpy(), route["lat"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.plot(rx, ry, color="#1F77B4", linewidth=1.8, label="Rota reconstruida",
            zorder=3)
    ax.scatter(rx[0], ry[0], s=70, color="#2CA02C", edgecolor="black",
               linewidth=0.6, zorder=5, label="Inicio")
    ax.scatter(rx[-1], ry[-1], s=80, color="#FF7F0E", marker="X",
               edgecolor="black", linewidth=0.6, zorder=5, label="Fim")

    pad = max((rx.max() - rx.min()) * 0.05, 250.0)
    ax.set_xlim(rx.min() - pad, rx.max() + pad)
    ax.set_ylim(ry.min() - pad, ry.max() + pad)
    ax.set_aspect("equal")
    _basemap(ax)

    ax.set_title(
        f"Linha {info['linha']} ({info['sentido']}) | "
        f"cov {info['coverage_pct']:.1f}% | "
        f"{info['route_length_km']:.1f} km | "
        f"{'OK' if info['completed'] else info['stop_reason']}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
