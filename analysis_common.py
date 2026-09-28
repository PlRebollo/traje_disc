"""
Elementos compartilhados da análise do experimento v8.

Carrega o sumário do grid (multi-dia, 13 linhas, 1/2/5/10/20/50/100/200/500
viagens), padroniza a linha (3 dígitos), define a ordem dos configs e a
paleta.

Mudança em relação ao v7: o eixo de volume de dados passa a ser o NÚMERO
DE VIAGENS (n_trips) em vez do número de ônibus (n_buses). A aderência é
reportada em três bandas: 5 m, 10 m e 15 m.
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
# BANDAS DE ADERÊNCIA
# ------------------------------------------------------------
BANDS = [int(t) for t in cfg.ADHERENCE_THRESHOLDS_M]      # [5, 10, 15]
BAND_COL = {b: f"coverage_{b}m" for b in BANDS}
BAND_COLORS = {5: "#D62728", 10: "#FF7F0E", 15: "#2CA02C"}


def linhas_circulares() -> list:
    return list(cfg.LINHAS_CIRCULARES)


def linhas_ida_volta() -> list:
    return list(cfg.LINHAS_IDA_VOLTA)


def linhas_validas() -> list:
    return linhas_circulares() + linhas_ida_volta()


# ------------------------------------------------------------
# PALETA POR LINHA e ORDEM DOS CONFIGS
# ------------------------------------------------------------
_TAB20 = plt.get_cmap("tab20")

LINE_COLORS = {
    linha: to_hex(_TAB20(i % 20))
    for i, linha in enumerate(sorted(set(linhas_validas())))
}

# ordem canônica: ida/volta primeiro, circulares depois
ORDEM_LINHAS = linhas_ida_volta() + linhas_circulares()


def config_label(linha, sentido, n_trips):
    return f"{linha} {sentido} v{int(n_trips)}"


def load_results() -> pd.DataFrame:
    """Lê o sumário do grid e cria colunas auxiliares."""
    df = pd.read_csv(cfg.SUMMARY_CSV)
    df["linha_s"] = df["linha"].astype(str).str.zfill(3)
    df["completed"] = (
        (df["reached_end"] == 1) | (df["closed_loop"] == 1)
    ).astype(bool)
    df["tipo"] = np.where(df["sentido"] == "circular", "circular", "ida/volta")
    df["n_trips"] = df["n_trips"].astype(int)
    df["config"] = [
        config_label(l, s, t)
        for l, s, t in zip(df["linha_s"], df["sentido"], df["n_trips"])
    ]
    return df


def ordered_configs(df: pd.DataFrame) -> list:
    """Lista (linha, sentido, n_trips) na ordem canônica."""
    out = []
    for linha in ORDEM_LINHAS:
        for sentido in ("ida", "volta") if linha in linhas_ida_volta() \
                else ("circular",):
            sub = df[(df["linha_s"] == linha) & (df["sentido"] == sentido)]
            if sub.empty:
                continue
            for nt in sorted(sub["n_trips"].unique()):
                out.append((linha, sentido, int(nt)))
    return out


def best_row(sub: pd.DataFrame, band: str = "coverage_pct") -> pd.Series:
    """Melhor execução de um subconjunto (prefere completas)."""
    done = sub[sub["completed"]]
    pool = done if not done.empty else sub
    return pool.loc[pool[band].idxmax()]
