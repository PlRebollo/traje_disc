"""Filtro de retilinearidade em baixa densidade (near/far = 300/1000)."""

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

CONFIGS = ["303/ida", "303/volta", "203/ida", "603/ida", "338/ida",
           "607/ida", "307/ida", "505/ida", "658/ida", "506/ida"]
N_TRIPS = [1, 2, 5, 20]
THRS = [0.0, 0.10, 0.15, 0.20]
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
                       split_terminals=tmap, max_gap_s=900, max_speed_kmh=None,
                       min_dt_s=1, terminal_near_m=300, terminal_far_m=1000)
    ts = build_trip_summary(df, min_trip_points=1)

    rows = []
    for (linha, sentido) in cmap:
        if f"{linha}/{sentido}" not in CONFIGS:
            continue
        sid, start, end = cmap[(linha, sentido)]
        base = select_trips(ts, linha, start, end, max_vehicles=None,
                            start_radius_m=cfg.START_RADIUS_M)
        base = validate_trips(df, base, sm, linha, sid,
                              max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        if base.empty:
            continue
        for thr in THRS:
            sel = base[base["straight"] > thr] if thr > 0 else base
            if sel.empty:
                continue
            for nt in N_TRIPS:
                s = sel.sort_values("n_points", ascending=False).iloc[:nt]
                gps = fetch_trip_points(df, s, start=start, end=end,
                                        per_point_filter=False,
                                        max_points=cfg.MAX_INPUT_POINTS
                                        ).dropna(subset=["lon", "lat"]
                                                 ).drop_duplicates(
                    subset=["lon", "lat"]).reset_index(drop=True)
                if len(gps) < 50:
                    continue
                ax, ay = end[0] - start[0], end[1] - start[1]
                proj = gps["dx"].to_numpy() * ax + gps["dy"].to_numpy() * ay
                proj = proj[np.isfinite(proj)]
                back = float(np.mean(proj < 0)) if len(proj) else np.nan
                ss, es = snap_points(gps, start, end)
                route, info = discover_route(
                    df=gps, initial_point=ss, end_point=es, meters=P["meters"],
                    angular_samples=P["samples"], increase_meters=P["inc"],
                    decrease_meters=1.0 / P["inc"],
                    max_meters=cfg.MAX_METERS_LIST[0],
                    loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
                    loop_min_steps=cfg.LOOP_MIN_STEPS, arrival_radius_m=25.0,
                    consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
                    dist_floor=cfg.DIST_FLOOR,
                    smooth_iterations=cfg.SMOOTH_ITERATIONS)
                ok = bool(info.get("reached_end") or info.get("closed_loop"))
                adh = gps_adherence_metrics(route, gps)
                rows.append(dict(straight_min=thr, cfg=f"{linha}/{sentido}",
                                 n_trips=nt, back=back,
                                 cov15=adh.get("coverage_15m", np.nan), ok=ok))
    d = pd.DataFrame(rows)
    d.to_csv(Path(cfg.OUT_ROOT) / "teste_retilinearidade.csv", index=False)
    print("conclusão por retilinearidade mínima x n_trips:")
    print(d.pivot_table(index="straight_min", columns="n_trips", values="ok",
                        aggfunc="mean").round(2).to_string())
    print("\nback (mediana) por retilinearidade mínima x n_trips:")
    print(d.pivot_table(index="straight_min", columns="n_trips", values="back",
                        aggfunc="median").round(2).to_string())
    print("\ncov15 (mediana) por retilinearidade mínima x n_trips:")
    print(d.pivot_table(index="straight_min", columns="n_trips", values="cov15",
                        aggfunc="median").round(1).to_string())


if __name__ == "__main__":
    main()
