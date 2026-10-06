"""Isola o efeito do raio de chegada (25 m vs 80 m) no pipeline refinado."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402
from core.trips import (load_gps_data, build_trip_summary, select_trips,  # noqa
                        validate_trips, fetch_trip_points, build_configs)
from core.model import discover_route  # noqa: E402
from core.metrics import gps_adherence_metrics  # noqa: E402
from run_experiment import snap_points  # noqa: E402

CONFIGS = ["303/ida", "303/volta", "603/ida", "607/ida", "338/ida",
           "203/ida", "307/ida", "505/ida", "658/volta", "506/ida"]
N_TRIPS = 200
P = dict(meters=10.0, samples=180, inc=1.5)


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
    df = load_gps_data(cfg.PATH_POSITIONS, linhas=cfg.LINHAS,
                       split_terminals=tmap, max_gap_s=cfg.MAX_GAP_S,
                       max_speed_kmh=None, min_dt_s=1,
                       terminal_near_m=cfg.TERMINAL_NEAR_M,
                       terminal_far_m=cfg.TERMINAL_FAR_M)
    ts = build_trip_summary(df, min_trip_points=cfg.MIN_TRIP_POINTS)

    rows = []
    for (linha, sentido) in cmap:
        if f"{linha}/{sentido}" not in CONFIGS:
            continue
        sid, start, end = cmap[(linha, sentido)]
        sel = select_trips(ts, linha, start, end, max_vehicles=None,
                           start_radius_m=cfg.START_RADIUS_M)
        sel = validate_trips(df, sel, sm, linha, sid,
                             max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        if sel.empty:
            continue
        s = sel.sort_values("n_points", ascending=False).iloc[:N_TRIPS]
        gps = fetch_trip_points(df, s, start=start, end=end,
                                per_point_filter=False,
                                max_points=cfg.MAX_INPUT_POINTS
                                ).dropna(subset=["lon", "lat"]).drop_duplicates(
            subset=["lon", "lat"]).reset_index(drop=True)
        ss, es = snap_points(gps, start, end)
        for arr in (25.0, 40.0, 80.0):
            route, info = discover_route(
                df=gps, initial_point=ss, end_point=es, meters=P["meters"],
                angular_samples=P["samples"], increase_meters=P["inc"],
                decrease_meters=1.0 / P["inc"], max_meters=cfg.MAX_METERS_LIST[0],
                loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
                loop_min_steps=cfg.LOOP_MIN_STEPS, arrival_radius_m=arr,
                consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
                dist_floor=cfg.DIST_FLOOR,
                smooth_iterations=cfg.SMOOTH_ITERATIONS)
            ok = bool(info.get("reached_end") or info.get("closed_loop"))
            adh = gps_adherence_metrics(route, gps)
            rows.append(dict(cfg=f"{linha}/{sentido}", arrival=arr,
                             cov15=adh.get("coverage_15m", np.nan), ok=ok))
    d = pd.DataFrame(rows)
    print(d.groupby("arrival").agg(cov15=("cov15", "median"),
                                   concl=("ok", "mean")).round(3).to_string())
    d.to_csv(Path(cfg.OUT_ROOT) / "teste_chegada.csv", index=False)


if __name__ == "__main__":
    main()
