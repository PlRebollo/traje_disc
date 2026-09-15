"""
Elementos compartilhados da analise do experimento v4.

Carrega o sumario do grid, padroniza a linha (3 digitos), define a
ordem dos configs e a paleta de cores por linha.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg

# ------------------------------------------------------------
# PALETA POR LINHA (16 linhas) e ORDEM DOS CONFIGS
# ------------------------------------------------------------
_TAB20 = plt.get_cmap("tab20")

_LINHAS_ORD = (
    list(cfg.LINHAS_IDA_VOLTA) + list(cfg.LINHAS_CIRCULARES)
)

LINE_COLORS = {
    linha: to_hex(_TAB20(i % 20))
    for i, linha in enumerate(sorted(set(_LINHAS_ORD)))
}

ORDEM_CONFIGS = []
for linha in cfg.LINHAS_IDA_VOLTA:
    ORDEM_CONFIGS += [f"{linha} ida", f"{linha} volta"]
for linha in cfg.LINHAS_CIRCULARES:
    ORDEM_CONFIGS.append(f"{linha} circular")


def load_results() -> pd.DataFrame:
    """Le o sumario do grid e adiciona colunas auxiliares."""
    df = pd.read_csv(cfg.SUMMARY_CSV)
    df["linha_s"] = df["linha"].astype(str).str.zfill(3)
    df["completed"] = (
        (df["reached_end"] == 1) | (df["closed_loop"] == 1)
    ).astype(bool)
    df["tipo"] = np.where(df["sentido"] == "circular", "circular", "ida/volta")
    df["config"] = df["linha_s"] + " " + df["sentido"]
    return df


def config_tag(linha: str, sentido: str, n_buses=None) -> str:
    tag = f"{linha} {sentido}"
    if sentido == "circular" and n_buses is not None:
        tag += f" b{int(n_buses)}"
    return tag


def ordered_configs(df: pd.DataFrame, with_buses: bool = False) -> list:
    """Configs presentes no sumario, na ordem canonica."""
    out = []
    for cfg_name in ORDEM_CONFIGS:
        linha, sentido = cfg_name.split()
        sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
        if sub.empty:
            continue
        if with_buses and sentido == "circular":
            for nb in sorted(sub["n_buses"].unique()):
                out.append((linha, sentido, int(nb)))
        else:
            out.append((linha, sentido, None))
    return out
