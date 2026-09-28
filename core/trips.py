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
def load_gps_data(path, linhas=None, split_terminals=None) -> pl.DataFrame:
    """
    Lê os pings GPS, filtra por qualidade (velocidade e intervalo) e
    segmenta em viagens (trip_id) por quebra de gap temporal.

    split_terminals: mapa {linha: [(lon, lat), ...]} com os terminais da
    linha. As viagens do dia inteiro do veículo (ida e volta mescladas,
    pois a parada no terminal raramente passa de 15 min) são divididas em
    cada chegada a terminal: a corrida volta a ter um único sentido.
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

    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Re-segmentação por chegada a terminal (ida/volta): a viagem do dia
    # inteiro do veículo (ida e volta mescladas, pois a parada no
    # terminal raramente passa de 15 min) é cortada em cada chegada a
    # terminal, de modo que cada segmento corresponda a uma corrida de
    # sentido único.
    # ------------------------------------------------------------
    if split_terminals:
        rows_t = []
        for k_, pts_ in split_terminals.items():
            for p in pts_:
                rows_t.append({"linha": str(k_), "t_lon": float(p[0]),
                               "t_lat": float(p[1])})
        term = pl.DataFrame(rows_t)
        alvo = [str(k_) for k_ in split_terminals.keys()]

        idx = df.with_row_index("_rid")
        j = (idx.filter(pl.col("linha").is_in(alvo))
             .join(term, on="linha", how="inner"))
        j = j.with_columns(
            (
                ((pl.col("lon") - pl.col("t_lon")) * M_PER_DEG_LAT
                 * (pl.col("lat") * np.pi / 180).cos()) ** 2
                + ((pl.col("lat") - pl.col("t_lat")) * M_PER_DEG_LAT) ** 2
            ).sqrt().alias("d_t")
        )
        mind = j.group_by("_rid").agg(pl.col("d_t").min().alias("d_min"))
        df = idx.join(mind, on="_rid", how="left").drop("_rid")
        df = df.sort(["linha", "veiculo", "timestamp"])

        df = df.with_columns([
            (pl.col("d_min") < 300.0).alias("_near"),
            (pl.col("d_min") > 1000.0)
            .cum_max().over(["linha", "veiculo", "trip_id"]).alias("_far"),
        ])
        df = df.with_columns(
            (
                (pl.col("d_min") < 300.0)
                & ~pl.col("_near").shift(1).over(["linha", "veiculo"])
                .fill_null(False)
                & pl.col("_far")
            ).fill_null(False).alias("_split")
        )
        df = df.with_columns(
            (pl.col("trip_id")
             + pl.col("_split").cast(pl.Int64)
             .cum_sum().over(["linha", "veiculo"])).alias("trip_id")
        )
        df = df.drop(["d_min", "_near", "_far", "_split"])
        df = df.sort(["linha", "veiculo", "timestamp"])

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
                      per_point_filter=False, max_points=MAX_INPUT_POINTS,
                      zone_inner_m=None, zone_outer_m=None):
    """
    Retorna os pings das trips selecionadas com vetores de deslocamento
    (dx, dy) calculados dentro de cada trip.

    per_point_filter (ida/volta): mantém apenas pings cujo movimento
    local aponta no sentido do trajeto (v·(B-A) >= 0).

    zona de terminal (zone_inner_m/zone_outer_m): quando definidos, os
    pings a zone_inner_m..zone_outer_m metros do terminal de PARTIDA
    escapam do filtro direcional — cobre rotas cuja perna de saída
    segue em direção oposta ao destino (cabeça de linha em curva).
    Fora do anel, o filtro direcional segue valendo.
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
        keep = (proj >= 0) | proj.isna()
        if zone_inner_m is not None and zone_outer_m is not None:
            dA = np.sqrt((pts["lon"] - start[0]) ** 2
                         + (pts["lat"] - start[1]) ** 2) * M_PER_DEG_LAT
            keep |= (dA >= zone_inner_m) & (dA <= zone_outer_m)
        pts = pts[keep]

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
