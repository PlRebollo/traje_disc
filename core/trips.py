"""
Pipeline de dados GPS: carregamento, segmentação em viagens,
separação de sentido (ida/volta), seleção e validação de trips.

Separação ida/volta (5 camadas, sem depender do shape como referência
de trajeto):
  1. Segmentação: gap > 15 min inicia nova viagem
  2. Classificação por trip: deslocamento líquido projetado no eixo A->B
  3. Trips genuínas: começam a <= 2 km do terminal de partida
  4. Filtro por ponto: cada ping só entra se seu movimento local aponta
     no sentido do trajeto
  5. (no modelo) pontuação direcional impede atração por sentido oposto
"""

import numpy as np
import pandas as pd
import polars as pl

from config import (
    MAX_GAP_S, MAX_SPEED_KMH, MIN_DT_S, MIN_TRIP_POINTS,
    M_PER_DEG_LAT, START_RADIUS_M, MAX_INPUT_POINTS,
)


# ============================================================
# CARREGAMENTO E SEGMENTAÇÃO
# ============================================================
def load_gps_data(path, linhas=None) -> pl.DataFrame:
    """
    Lê os pings GPS, filtra por qualidade (velocidade e intervalo) e
    segmenta em viagens (trip_id) por quebra de gap temporal.
    """
    df = pl.read_parquet(path)
    if linhas is not None:
        df = df.filter(pl.col("linha").is_in(linhas))

    df = (
        df.with_columns(pl.col("timestamp").cast(pl.Datetime))
        .sort(["linha", "veiculo", "timestamp"])
    )

    df = df.with_columns([
        pl.col("lat").shift(1).over(["linha", "veiculo"]).alias("lat_prev"),
        pl.col("lon").shift(1).over(["linha", "veiculo"]).alias("lon_prev"),
        pl.col("timestamp").shift(1).over(["linha", "veiculo"]).alias("ts_prev"),
    ])
    df = df.with_columns(
        (pl.col("timestamp") - pl.col("ts_prev"))
        .dt.total_seconds().cast(pl.Float64).alias("dt_s")
    )

    # nova viagem quando o gap temporal excede o limite
    is_new = pl.col("ts_prev").is_null() | (pl.col("dt_s") > MAX_GAP_S)
    df = df.with_columns(
        is_new.cast(pl.Int64).cum_sum().over(["linha", "veiculo"]).alias("trip_id")
    )

    # velocidade via Haversine e filtro de plausibilidade
    d = np.pi / 180.0
    lat1, lon1 = pl.col("lat_prev") * d, pl.col("lon_prev") * d
    lat2, lon2 = pl.col("lat") * d, pl.col("lon") * d
    a = ((lat2 - lat1) / 2).sin() ** 2 + lat1.cos() * lat2.cos() * \
        (((lon2 - lon1) / 2).sin() ** 2)
    dist_km = (2.0 * 6371.0088) * a.sqrt().arcsin()
    df = df.with_columns(
        pl.when(pl.col("lat_prev").is_null())
        .then(None).otherwise(dist_km).alias("dist_km")
    )
    df = df.with_columns(
        (pl.col("dist_km") / (pl.col("dt_s") / 3600.0)).alias("speed_kmh")
    )
    df = df.filter(
        pl.col("speed_kmh").is_null()
        | ((pl.col("dt_s") >= MIN_DT_S)
           & (pl.col("dt_s") <= MAX_GAP_S)
           & (pl.col("speed_kmh") <= MAX_SPEED_KMH)
           & pl.col("speed_kmh").is_finite())
    )
    return df


def build_trip_summary(df_pl: pl.DataFrame) -> pd.DataFrame:
    """
    Resumo por viagem: nº de pings + primeiro/último ponto
    (para classificação geométrica de sentido).
    """
    return (
        df_pl.group_by(["linha", "veiculo", "trip_id"])
        .agg([
            pl.len().alias("n_points"),
            pl.col("lon").first().alias("lon_start"),
            pl.col("lat").first().alias("lat_start"),
            pl.col("lon").last().alias("lon_end"),
            pl.col("lat").last().alias("lat_end"),
        ])
        .filter(pl.col("n_points") >= MIN_TRIP_POINTS)
        .to_pandas()
    )


# ============================================================
# SELEÇÃO DE TRIPS
# ============================================================
def classify_trip_direction(first_lon, first_lat, last_lon, last_lat,
                            start, end) -> str:
    """
    Classifica a trip pelo deslocamento líquido projetado no eixo start->end:
      prog = (último - primeiro) · (fim - início)
      prog > 0 -> sentido 'ida' (correto); < 0 -> 'volta'.
    """
    prog = ((last_lon - first_lon) * (end[0] - start[0])
            + (last_lat - first_lat) * (end[1] - start[1]))
    if prog > 0:
        return "ida"
    if prog < 0:
        return "volta"
    return "indefinido"


def select_trips(trip_summary, linha, start, end, max_vehicles,
                 start_radius_m=START_RADIUS_M):
    """
    Seleciona trips para um config de linha/sentido.

    - ida/volta: mantém trips do sentido correto que começam perto do
      terminal; usa todas as trips genuínas dos veículos com mais dados.
    - circular (end=None): maiores trips, 1 por veículo.
    """
    sub = trip_summary[trip_summary["linha"] == linha].copy()
    if sub.empty:
        return sub

    if end is None:
        sub = (sub.sort_values("n_points", ascending=False)
               .drop_duplicates(subset=["veiculo"], keep="first"))
        return sub.head(max_vehicles).reset_index(drop=True)

    prog = ((sub["lon_end"] - sub["lon_start"]) * (end[0] - start[0])
            + (sub["lat_end"] - sub["lat_start"]) * (end[1] - start[1]))
    sub = sub[prog > 0].copy()
    if sub.empty:
        return sub

    d_start = np.sqrt((sub["lon_start"] - start[0]) ** 2
                      + (sub["lat_start"] - start[1]) ** 2) * M_PER_DEG_LAT
    genuine = sub[d_start <= start_radius_m]

    if max_vehicles is None:
        pool = genuine if not genuine.empty else sub
        return pool.sort_values("n_points", ascending=False).reset_index(drop=True)

    pool = genuine if genuine["veiculo"].nunique() >= max_vehicles else sub
    veh_totals = (pool.groupby("veiculo")["n_points"].sum()
                  .sort_values(ascending=False))
    top = set(veh_totals.head(max_vehicles).index)
    pool = pool[pool["veiculo"].isin(top)]
    return pool.sort_values("n_points", ascending=False).reset_index(drop=True)


def validate_trips(df_pl, sel, shape_models, linha, shape_id,
                   max_median_m=200.0):
    """
    Descarta trips cuja mediana de distância ao shape oficial excede
    max_median_m — indício de erro de etiquetagem (viagem de outra linha).
    """
    if sel.empty:
        return sel
    model = shape_models.get((linha.strip(), str(shape_id).strip()))
    if model is None:
        return sel

    from core.geometry import distance_matrix_to_polyline

    keep = []
    for _, row in sel.iterrows():
        pts = (df_pl.filter(
            (pl.col("linha") == row["linha"])
            & (pl.col("veiculo") == row["veiculo"])
            & (pl.col("trip_id") == int(row["trip_id"]))
        ).select(["lon", "lat"]).to_pandas())
        if pts.empty:
            continue
        if len(pts) > 500:
            pts = pts.sample(n=500, random_state=42)
        d = distance_matrix_to_polyline(pts, model)
        valid = d[np.isfinite(d)]
        if len(valid) and float(np.median(valid)) <= max_median_m:
            keep.append(row)

    if not keep:
        return sel
    return pd.DataFrame(keep).reset_index(drop=True)


def fetch_trip_points(df_pl, sel, start=None, end=None,
                      per_point_filter=False, max_points=MAX_INPUT_POINTS):
    """
    Retorna os pings das trips selecionadas com vetores de deslocamento
    (dx, dy) calculados dentro de cada trip.

    per_point_filter (ida/volta): mantém apenas pings cujo movimento
    local aponta no sentido do trajeto (v·(B-A) >= 0).
    """
    if sel.empty:
        return pd.DataFrame(columns=["lon", "lat", "dx", "dy"])

    keys = pl.DataFrame({
        "linha": [str(v) for v in sel["linha"]],
        "veiculo": [str(v) for v in sel["veiculo"]],
        "trip_id": [int(v) for v in sel["trip_id"]],
    }).with_columns([
        pl.col("linha").cast(pl.String),
        pl.col("veiculo").cast(pl.String),
        pl.col("trip_id").cast(pl.Int64),
    ])

    pts = (df_pl.join(keys, on=["linha", "veiculo", "trip_id"], how="inner")
           .select(["lon", "lat", "timestamp", "veiculo", "trip_id"])
           .to_pandas()
           .sort_values(["veiculo", "trip_id", "timestamp"])
           .reset_index(drop=True))
    if pts.empty:
        return pd.DataFrame(columns=["lon", "lat", "dx", "dy"])

    grp = pts.groupby(["veiculo", "trip_id"], sort=False)
    pts["dx"] = grp["lon"].diff()
    pts["dy"] = grp["lat"].diff()

    if per_point_filter and end is not None:
        proj = pts["dx"] * (end[0] - start[0]) + pts["dy"] * (end[1] - start[1])
        pts = pts[(proj >= 0) | proj.isna()]

    if len(pts) > max_points:
        pts = pts.sample(n=max_points, random_state=42)

    return pts[["lon", "lat", "dx", "dy"]].reset_index(drop=True)


# ============================================================
# MONTAGEM DOS CONFIGS (terminais a partir dos shapes)
# ============================================================
def build_configs(shape_models, linhas_circulares, linhas_ida_volta):
    """
    Monta a lista de configs a partir das linhas e shapes.
    Retorna tuplas (linha, sentido, shape_id, start, end).
    """
    from core.shapes import line_shapes, shape_terminal_points

    configs = []
    for linha in linhas_circulares:
        sids = line_shapes(shape_models, linha)
        if not sids:
            continue
        model = shape_models[(linha, sids[0])]
        start, _ = shape_terminal_points(model)
        configs.append((linha, "circular", sids[0], start, None))

    for linha in linhas_ida_volta:
        sids = line_shapes(shape_models, linha)
        if len(sids) < 2:
            continue
        for sentido, sid in zip(["ida", "volta"], sids[:2]):
            model = shape_models[(linha, sid)]
            start, end = shape_terminal_points(model)
            configs.append((linha, sentido, sid, start, end))

    return configs
