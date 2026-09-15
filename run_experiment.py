"""
Experimento v4 — executa o grid de descoberta de rotas.

16 linhas (4 circulares + 12 ida/volta), fonte: um dia completo de
operação (2026-02-03). Saídas organizadas em outputs/.

Uso:
    cd experimento_v4
    python run_experiment.py
"""

import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from core.shapes import load_shape_models
from core.trips import (
    load_gps_data, build_trip_summary, select_trips,
    validate_trips, fetch_trip_points, build_configs,
)
from core.model import discover_route
from core.metrics import gps_adherence_metrics, turns_per_km
from core.geometry import route_length_km, run_id_from_params, route_mae_to_shape_m
from core.imaging import save_route_image

warnings.filterwarnings("ignore")


def main():
    print("=" * 64)
    print("Experimento v4 — descoberta de rotas (16 linhas)")
    print("=" * 64)

    for d in (cfg.OUT_ROOT, cfg.ROUTES_DIR, cfg.IMAGES_DIR, cfg.ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("\nCarregando shapes...")
    shape_models = load_shape_models(cfg.SHAPE_XZ_PATH)
    configs = build_configs(
        shape_models, cfg.LINHAS_CIRCULARES, cfg.LINHAS_IDA_VOLTA
    )
    print(f"Shapes: {len(shape_models):,} | configs: {len(configs)}")

    print("\nCarregando dados GPS...")
    df_pl = load_gps_data(cfg.PATH_POSITIONS, linhas=cfg.LINHAS)
    trip_summary = build_trip_summary(df_pl)
    print(f"Pings: {df_pl.height:,} | trips validas: {len(trip_summary):,}")

    param_combos = list(product(
        cfg.MIN_METERS_LIST, cfg.MAX_METERS_LIST,
        cfg.SAMPLES_LIST, cfg.INCREASE_METERS_LIST,
    ))
    print(f"Combos de parâmetros: {len(param_combos)}")

    # itens de trabalho: ida/volta = 1 execução; circular = 2 (nº ônibus)
    work = []
    for (linha, sentido, shape_id, start, end) in configs:
        if end is None:
            for b in cfg.BUS_COUNTS:
                work.append((linha, sentido, shape_id, start, end, b))
        else:
            work.append((linha, sentido, shape_id, start, end, None))
    total = len(work) * len(param_combos)
    print(f"Execuções previstas: {total}")

    rows = []
    done = 0

    for (linha, sentido, shape_id, start, end, n_buses) in work:
        max_veh = None if end is not None else max(cfg.BUS_COUNTS)
        sel = select_trips(
            trip_summary, linha, start, end,
            max_vehicles=max_veh, start_radius_m=cfg.START_RADIUS_M,
        )
        if sel.empty:
            print(f"[skip] {linha}/{sentido}: sem trips")
            continue

        n_before = len(sel)
        sel = validate_trips(
            df_pl, sel, shape_models, linha, shape_id,
            max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M,
        )
        if len(sel) < n_before:
            print(f"[valida] {linha}/{sentido}: "
                  f"{n_before - len(sel)} trips descartadas")
        if sel.empty:
            continue

        if n_buses is not None:
            veh_order = (sel.groupby("veiculo")["n_points"].sum()
                         .sort_values(ascending=False).index.tolist())
            if len(veh_order) < n_buses:
                continue
            sel = sel[sel["veiculo"].isin(veh_order[:n_buses])]

        vehs = sorted(sel["veiculo"].unique().tolist())
        gps = fetch_trip_points(
            df_pl, sel, start=start, end=end,
            per_point_filter=(end is not None),
            max_points=cfg.MAX_INPUT_POINTS,
        ).dropna(subset=["lon", "lat"]).drop_duplicates(
            subset=["lon", "lat"]).reset_index(drop=True)
        if len(gps) < 50:
            print(f"[skip] {linha}/{sentido}: poucos pontos ({len(gps)})")
            continue

        # Snapar ponto inicial para o ponto GPS mais proximo
        d_start = np.sqrt((gps['lon']-start[0])**2 + (gps['lat']-start[1])**2) * 111132
        if d_start.min() > 500:
            start_snap = [float(gps['lon'].median()), float(gps['lat'].median())]
        else:
            idx = d_start.idxmin()
            start_snap = [float(gps['lon'].iloc[idx]), float(gps['lat'].iloc[idx])]

        # Snapar ponto final (ida/volta)
        end_snap = end
        if end is not None:
            d_end = np.sqrt((gps['lon']-end[0])**2 + (gps['lat']-end[1])**2) * 111132
            if d_end.min() > 500:
                end_snap = [float(gps['lon'].median()), float(gps['lat'].median())]
            else:
                idx = d_end.idxmin()
                end_snap = [float(gps['lon'].iloc[idx]), float(gps['lat'].iloc[idx])]

        n_input_points = len(gps)
        tag = f"{linha}/{sentido}" + (f" b{n_buses}" if n_buses else "")
        print(f"\n=== {tag} | {len(sel)} trips | {len(gps):,} pings ===")

        shape_model = shape_models.get((linha, str(shape_id).strip()))
        config_rows = []

        for (min_m, max_m, samples, inc) in param_combos:
            min_m, max_m, inc = float(min_m), float(max_m), float(inc)
            samples = int(samples)
            dec = 1.0 / inc

            params = dict(linha=linha, sentido=sentido, n_buses=n_buses or len(vehs),
                          meters=min_m, max_meters=max_m, samples=samples, inc=inc)
            rid = run_id_from_params(params)

            t0 = time.perf_counter()
            route, info = discover_route(
                df=gps, initial_point=start_snap, end_point=end_snap,
                meters=min_m, angular_samples=samples,
                increase_meters=inc, decrease_meters=dec, max_meters=max_m,
                loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
                loop_min_steps=cfg.LOOP_MIN_STEPS,
                arrival_radius_m=cfg.ARRIVAL_RADIUS_M,
                max_length_factor=cfg.MAX_LENGTH_FACTOR,
                max_length_km_circular=cfg.MAX_LENGTH_KM_CIRCULAR,
                consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
                smooth_iterations=cfg.SMOOTH_ITERATIONS,
            )
            runtime = time.perf_counter() - t0

            adh = gps_adherence_metrics(route, gps)
            km = route_length_km(route)
            mae = route_mae_to_shape_m(route, shape_model)
            completed = bool(info["reached_end"] or info["closed_loop"])

            route_path = cfg.ROUTES_DIR / f"route_{rid}.csv"
            route.to_csv(route_path, index=False, float_format="%.7f")

            row = dict(
                run_id=rid, linha=linha, sentido=sentido, shape_id=shape_id,
                n_buses=n_buses if n_buses else len(vehs),
                n_vehicles=len(vehs), n_trips=len(sel),
                n_input_points=len(gps),
                angular_samples=samples, meters=min_m, max_meters=max_m,
                increase_meters=inc, n_route_points=len(route),
                route_length_km=km, reached_end=int(info["reached_end"]),
                closed_loop=int(info["closed_loop"]),
                stop_reason=info["stop_reason"], completed=int(completed),
                coverage_pct=adh["coverage_pct"],
                mean_gps_dist_m=adh["mean_gps_dist_m"],
                p95_gps_dist_m=adh["p95_gps_dist_m"],
                mae_shape_m=mae, turns_per_km=turns_per_km(route),
                runtime_s=runtime, route_file=str(route_path),
            )
            rows.append(row)
            config_rows.append(row)
            done += 1
            print(f"  [{done}/{total}] N={samples} r={min_m:.0f} G={inc} | "
                  f"cov={adh['coverage_pct']:.1f}% dm={adh['mean_gps_dist_m']:.1f}m"
                  f" | {'OK' if completed else info['stop_reason'][:14]}")

        # imagem da melhor execução do config
        if config_rows:
            cdf = pd.DataFrame(config_rows)
            best = cdf[cdf["completed"] == 1] if (cdf["completed"] == 1).any() \
                else cdf
            best = best.loc[best["coverage_pct"].idxmax()]
            try:
                save_route_image(
                    pd.read_csv(best["route_file"]),
                    {"linha": linha, "sentido": sentido,
                     "coverage_pct": best["coverage_pct"],
                     "route_length_km": best["route_length_km"],
                     "completed": bool(best["completed"]),
                     "stop_reason": best["stop_reason"]},
                    cfg.IMAGES_DIR / (
                        f"{linha}_{sentido}"
                        + (f"_b{best['n_buses']}" if n_buses else "")
                        + ".png"
                    ),
                )
            except Exception as exc:
                print(f"  [aviso] imagem falhou: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.SUMMARY_CSV, index=False, float_format="%.6f")

    print("\n" + "=" * 64)
    print(f"Concluído: {len(df)} execuções")
    if len(df):
        print(f"Completas: {df['completed'].sum()} "
              f"({100 * df['completed'].mean():.1f}%)")
        print(f"Configs com sucesso: "
              f"{df[df['completed'] == 1].groupby(['linha', 'sentido']).ngroups}"
              f"/{df.groupby(['linha', 'sentido']).ngroups}")
    print(f"Sumário: {cfg.SUMMARY_CSV}")
    print(f"Rotas:   {cfg.ROUTES_DIR}")
    print(f"Imagens: {cfg.IMAGES_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
