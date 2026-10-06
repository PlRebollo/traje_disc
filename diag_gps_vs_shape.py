"""Diagnóstico: as amostras de GPS seguem o traçado oficial? (v10)"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402
from core.shapes import load_shape_models  # noqa: E402
from core.trips import (load_gps_data, build_trip_summary, select_trips,  # noqa
                        validate_trips, fetch_trip_points, build_configs)
from core.geometry import distance_matrix_to_polyline  # noqa: E402

N = 500


def main():
    sm = load_shape_models(cfg.SHAPE_XZ_PATH)
    configs = build_configs(sm, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA)
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
    for (linha, sentido, sid, start, end) in configs:
        sel = select_trips(ts, linha, start, end, max_vehicles=None,
                           start_radius_m=cfg.START_RADIUS_M)
        if sel.empty:
            continue
        sel = validate_trips(df, sel, sm, linha, sid,
                             max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        if sel.empty:
            continue
        sel_n = sel.sort_values("n_points", ascending=False).iloc[:N]
        gps = fetch_trip_points(df, sel_n, start=start, end=end,
                                per_point_filter=False,
                                max_points=cfg.MAX_INPUT_POINTS
                                ).dropna(subset=["lon", "lat"]).drop_duplicates(
            subset=["lon", "lat"]).reset_index(drop=True)
        model = sm.get((linha, str(sid).strip()))
        d = distance_matrix_to_polyline(gps[["lon", "lat"]], model)
        v = d[np.isfinite(d)]
        rows.append(dict(cfg=f"{linha}/{sentido}", sentido=sentido,
                         n_gps=len(gps), n_veh=int(sel_n["veiculo"].nunique()),
                         gps_med=round(float(np.median(v)), 1),
                         gps_p90=round(float(np.percentile(v, 90)), 1),
                         frac_far=round(100 * float(np.mean(v > 100)), 1)))
    out = pd.DataFrame(rows).sort_values("gps_med")
    dest = Path(cfg.ANALYSIS_DIR) / "regimes"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "diag_gps_vs_shape.csv", index=False)
    print(out.to_string(index=False))
    print("salvo em", dest / "diag_gps_vs_shape.csv")


if __name__ == "__main__":
    main()
