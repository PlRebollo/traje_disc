"""
Benchmark de separação ida/volta — experimento v10.

Mede, para cada estratégia de separação, a pureza do sentido e o efeito a
jusante no modelo. Métricas:

  back        fração de amostras de GPS cujo deslocamento aponta contra o eixo A->B
              (baixo = nuvem pura, sem mistura do sentido oposto)
  dstart/dend distância mediana do início/fim das viagens aos terminais
              A/B (baixo = viagem bem ancorada nos terminais)
  cov15/concl aderência e conclusão do modelo (fixo) sobre a nuvem

Estratégias de seleção:
  prog     atual: prog > 0 (deslocamento líquido) + início perto de A
  ancorado início perto de A E fim perto de B (independe do formato do
           trajeto; mais robusto a ruído nas pontas)
  per_point atual + filtro direcional por ponto (re-liga PER_POINT_FILTER)

Limiares do corte por terminal (near/far): 150/500, 300/1000, 500/1500.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402
from core.trips import (  # noqa: E402
    load_gps_data, build_trip_summary, select_trips, validate_trips,
    fetch_trip_points, build_configs,
)
from core.model import discover_route  # noqa: E402
from core.metrics import gps_adherence_metrics  # noqa: E402
from run_experiment import snap_points  # noqa: E402

M = cfg.M_PER_DEG_LAT
CONFIGS = ["303/ida", "303/volta", "603/ida", "607/ida", "338/ida",
           "203/ida", "307/ida", "505/ida"]
N_TRIPS = 200
PARAMS = dict(meters=10.0, samples=180, inc=1.5, fail=10, arrival=25.0,
              floor=0.0)


def _dist(lon, lat, p):
    return np.sqrt((lon - p[0]) ** 2 + (lat - p[1]) ** 2) * M


def select_ancorado(ts, linha, start, end, radius=2000, end_radius=1500):
    sub = ts[ts["linha"] == linha].copy()
    dstart = _dist(sub["lon_start"], sub["lat_start"], start)
    dend = _dist(sub["lon_end"], sub["lat_end"], end)
    return sub[(dstart <= radius) & (dend <= end_radius)].reset_index(drop=True)


def back_frac(gps, start, end):
    ax = end[0] - start[0]
    ay = end[1] - start[1]
    dx = gps["dx"].to_numpy()
    dy = gps["dy"].to_numpy()
    proj = dx * ax + dy * ay
    v = proj[np.isfinite(proj)]
    return float(np.mean(v < 0)) if len(v) else np.nan


def run_model(gps, start, end):
    ss, es = snap_points(gps, start, end)
    route, info = discover_route(
        df=gps, initial_point=ss, end_point=es,
        meters=PARAMS["meters"], angular_samples=PARAMS["samples"],
        increase_meters=PARAMS["inc"], decrease_meters=1.0 / PARAMS["inc"],
        max_meters=cfg.MAX_METERS_LIST[0],
        loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
        loop_min_steps=cfg.LOOP_MIN_STEPS, arrival_radius_m=PARAMS["arrival"],
        consecutive_fail_stop=PARAMS["fail"], dist_floor=PARAMS["floor"],
        smooth_iterations=cfg.SMOOTH_ITERATIONS)
    ok = bool(info.get("reached_end") or info.get("closed_loop"))
    adh = gps_adherence_metrics(route, gps)
    return adh.get("coverage_15m", np.nan), ok


def build_pipeline(tmap, near, far):
    df = load_gps_data(cfg.PATH_POSITIONS, linhas=cfg.LINHAS,
                       split_terminals=tmap, max_gap_s=900, max_speed_kmh=None,
                       min_dt_s=1, terminal_near_m=near, terminal_far_m=far)
    ts = build_trip_summary(df, min_trip_points=1)
    return df, ts


def evaluate(cmap, sm, df, ts, estrategia, near, far):
    rows = []
    for (linha, sentido) in cmap:
        if f"{linha}/{sentido}" not in CONFIGS:
            continue
        sid, start, end = cmap[(linha, sentido)]
        if estrategia == "ancorado":
            sel = select_ancorado(ts, linha, start, end)
        else:
            sel = select_trips(ts, linha, start, end, max_vehicles=None,
                               start_radius_m=cfg.START_RADIUS_M)
        sel = validate_trips(df, sel, sm, linha, sid,
                             max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        if sel.empty:
            continue
        s = sel.sort_values("n_points", ascending=False).iloc[:N_TRIPS]
        gps = fetch_trip_points(
            df, s, start=start, end=end,
            per_point_filter=(estrategia == "per_point"),
            max_points=cfg.MAX_INPUT_POINTS,
        ).dropna(subset=["lon", "lat"]).drop_duplicates(
            subset=["lon", "lat"]).reset_index(drop=True)
        if len(gps) < 50:
            continue
        back = back_frac(gps, start, end)
        dstart = _dist(sel["lon_start"], sel["lat_start"], start).median()
        dend = _dist(sel["lon_end"], sel["lat_end"], end).median()
        cov, ok = run_model(gps, start, end)
        rows.append(dict(estrategia=estrategia, near=near, far=far,
                         cfg=f"{linha}/{sentido}", n_trips=len(sel),
                         n_amostras=len(gps), back=back, dstart=dstart, dend=dend,
                         cov15=cov, ok=ok))
    return rows


def main():
    sm = load_shape_models(cfg.SHAPE_XZ_PATH)
    configs = build_configs(sm, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
    cmap = {(l, s): (sid, st, en) for (l, s, sid, st, en) in configs}
    tmap = {}
    for (linha, _s, _sid, st, en) in configs:
        if linha not in cfg.LINHAS_IDA_VOLTA:
            continue
        tmap.setdefault(linha, [])
        tmap[linha].append(tuple(st))
        if en is not None:
            tmap[linha].append(tuple(en))

    all_rows = []
    # limiares do corte por terminal (seleção prog)
    for near, far in [(150, 500), (300, 1000), (500, 1500)]:
        df, ts = build_pipeline(tmap, near, far)
        all_rows += evaluate(cmap, sm, df, ts, "prog", near, far)
        print(f"terminal {near}/{far}: trips={len(ts):,}", flush=True)

    # estratégias de seleção (terminal 300/1000)
    df, ts = build_pipeline(tmap, 300, 1000)
    for estr in ("ancorado", "per_point"):
        all_rows += evaluate(cmap, sm, df, ts, estr, 300, 1000)

    out = pd.DataFrame(all_rows)
    OUT = Path(cfg.OUT_ROOT)
    out.to_csv(OUT / "benchmark_separacao.csv", index=False)

    g = out.groupby(["estrategia", "near", "far"]).agg(
        back=("back", "median"), dstart=("dstart", "median"),
        dend=("dend", "median"), cov15=("cov15", "median"),
        concl=("ok", "mean"), n_amostras_gps=("n_amostras", "median"),
    ).reset_index()
    g["concl"] = (100 * g["concl"]).round(1)
    for c in ("back", "dstart", "dend", "cov15"):
        g[c] = g[c].round(1)
    g["n_amostras_gps"] = g["n_amostras_gps"].round(0)
    g.to_csv(OUT / "benchmark_separacao_summary.csv", index=False)
    print("\n=== RESUMO (back | dstart | dend | cov15 | concl | amostras) ===")
    print(g.to_string(index=False))
    return out, g


if __name__ == "__main__":
    main()
