"""
Benchmark de ablação — experimento v10 (refino).

Compara, sobre um subconjunto representativo, o pipeline e o modelo atuais
com as simplificações propostas. Cada variante é uma combinação de
pipeline (filtros/segmentação) e modelo (parada/raio/kernel).

Pipeline (como os dados são preparados):
  baseline    : filtro de velocidade (120 km/h) + min 150 amostras de GPS + gap 900 s
  sem_filtros : sem filtro de velocidade, sem corte de 150 amostras de GPS, gap 900 s
  so_terminal : idem, mas sem segmentação por gap (só corte por terminal)

Modelo (como a caminhada para e pontua):
  M0 fail=10, chegada=80 m, kernel 1/d
  M1 fail=1,  chegada=80 m, kernel 1/d
  M2 fail=1,  chegada=40 m, kernel 1/d
  M3 fail=1,  chegada=25 m, kernel 1/d   (unifica com o laço)
  M4 fail=1,  chegada=25 m, kernel 1/(d+1)
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

# ------------------------------------------------------------
# Variantes
# ------------------------------------------------------------
PIPELINES = {
    "baseline": dict(gap=900, speed=120, min_pts=150),
    "sem_filtros": dict(gap=900, speed=None, min_pts=1),
    "so_terminal": dict(gap=None, speed=None, min_pts=1),
}

MODELS = {
    "M0_fail10_arr80_1d": dict(fail=10, arrival=80, floor=0.0),
    "M1_fail1_arr80_1d": dict(fail=1, arrival=80, floor=0.0),
    "M2_fail1_arr40_1d": dict(fail=1, arrival=40, floor=0.0),
    "M3_fail1_arr25_1d": dict(fail=1, arrival=25, floor=0.0),
    "M4_fail1_arr25_d1": dict(fail=1, arrival=25, floor=1.0),
}

CONFIGS = ["020/circular", "303/ida", "303/volta", "603/ida",
           "607/ida", "338/ida"]
N_TRIPS = [20, 200]
PARAMS = dict(meters=10.0, samples=180, inc=1.5)


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

    rows = []
    for pname, p in PIPELINES.items():
        t0 = time.perf_counter()
        print(f"\n=== pipeline {pname} (gap={p['gap']}, speed={p['speed']}, "
              f"min_pts={p['min_pts']}) ===", flush=True)
        df = load_gps_data(
            cfg.PATH_POSITIONS, linhas=cfg.LINHAS, split_terminals=tmap,
            max_gap_s=p["gap"], max_speed_kmh=p["speed"], min_dt_s=1,
        )
        ts = build_trip_summary(df, min_trip_points=p["min_pts"])
        print(f"  trips={len(ts):,} | amostras de GPS={df.height:,} | "
              f"load={time.perf_counter()-t0:.0f}s", flush=True)

        for (linha, sentido) in cmap:
            if f"{linha}/{sentido}" not in CONFIGS:
                continue
            sid, start, end = cmap[(linha, sentido)]
            sel = select_trips(ts, linha, start, end, max_vehicles=None,
                               start_radius_m=cfg.START_RADIUS_M)
            sel = validate_trips(df, sel, sm, linha, sid,
                                 max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
            if sel.empty:
                print(f"  [skip] {linha}/{sentido}: sem trips", flush=True)
                continue
            for nt in N_TRIPS:
                s = sel.sort_values("n_points", ascending=False).iloc[:nt]
                gps = fetch_trip_points(
                    df, s, start=start, end=end,
                    per_point_filter=False, max_points=cfg.MAX_INPUT_POINTS,
                ).dropna(subset=["lon", "lat"]).drop_duplicates(
                    subset=["lon", "lat"]).reset_index(drop=True)
                if len(gps) < 50:
                    continue
                ss, es = snap_points(gps, start, end)
                for mname, m in MODELS.items():
                    t1 = time.perf_counter()
                    route, info = discover_route(
                        df=gps, initial_point=ss, end_point=es,
                        meters=PARAMS["meters"],
                        angular_samples=PARAMS["samples"],
                        increase_meters=PARAMS["inc"],
                        decrease_meters=1.0 / PARAMS["inc"],
                        max_meters=cfg.MAX_METERS_LIST[0],
                        loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
                        loop_min_steps=cfg.LOOP_MIN_STEPS,
                        arrival_radius_m=m["arrival"],
                        consecutive_fail_stop=m["fail"],
                        dist_floor=m["floor"],
                        smooth_iterations=cfg.SMOOTH_ITERATIONS,
                    )
                    dt = time.perf_counter() - t1
                    ok = bool(info.get("reached_end") or info.get("closed_loop"))
                    adh = gps_adherence_metrics(route, gps)
                    rows.append(dict(
                        pipeline=pname, modelo=mname,
                        cfg=f"{linha}/{sentido}", n_trips=nt,
                        n_amostras=len(gps),
                        cov15=adh.get("coverage_15m", np.nan),
                        cov10=adh.get("coverage_10m", np.nan),
                        cov5=adh.get("coverage_5m", np.nan),
                        ok=ok, stop=info.get("stop_reason"),
                        n_steps=int(info.get("n_steps", 0)),
                        runtime_s=dt,
                    ))

    out = pd.DataFrame(rows)
    OUT = Path(cfg.OUT_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "benchmark.csv", index=False)

    # agregado por variante
    g = out.groupby(["pipeline", "modelo"]).agg(
        cov15=("cov15", "median"), cov10=("cov10", "median"),
        cov5=("cov5", "median"), concl=("ok", "mean"),
        runtime=("runtime_s", "median"), n=("cfg", "size"),
    ).reset_index()
    g["concl"] = (100 * g["concl"]).round(1)
    for c in ("cov15", "cov10", "cov5"):
        g[c] = g[c].round(1)
    g["runtime"] = g["runtime"].round(2)
    g.to_csv(OUT / "benchmark_summary.csv", index=False)
    print("\n=== RESUMO (mediana cov15 | conclusão | tempo med) ===")
    print(g.to_string(index=False))
    return out, g


if __name__ == "__main__":
    main()
