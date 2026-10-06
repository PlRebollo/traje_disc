"""
Experimento v8 — grid com eixo de dados por NÚMERO DE VIAGENS.

Modelo final do v7 (sem filtro direcional de dados + segmentação por
terminal). Grid: 3 ângulos × 5 raios × 10 fatores × 9 tamanhos
(1–500 viagens genuínas, misturando veículos).

Cada item de trabalho (linha, sentido, nº de viagens) executa suas 150
combinações e salva:
  outputs/rotas/{tag}.parquet      : todas as rotas (formato longo, zstd)
  outputs/rotas/{tag}_melhor.csv   : rota da melhor combinação
  outputs/images/{tag}.png         : imagem da melhor combinação

Uso:
    cd experimento_v8
    python run_experiment.py [--procs N] [--limite N]
"""

import argparse
import gc
import multiprocessing as mp
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

_STATE = None


def _discover(st, min_m, max_m, samples, inc):
    return discover_route(
        df=st["gps"], initial_point=st["start_snap"],
        end_point=st["end_snap"], meters=min_m, angular_samples=samples,
        increase_meters=inc, decrease_meters=1.0 / inc, max_meters=max_m,
        loop_close_radius=cfg.LOOP_CLOSE_RADIUS_M,
        loop_min_steps=cfg.LOOP_MIN_STEPS,
        arrival_radius_m=cfg.ARRIVAL_RADIUS_M,
        consecutive_fail_stop=cfg.CONSECUTIVE_FAIL_STOP,
        dist_floor=cfg.DIST_FLOOR,
        smooth_iterations=cfg.SMOOTH_ITERATIONS,
    )


def _save_best_png(route, gps, shape_model, path, meta, row):
    """Imagem compacta da melhor rota (coordenadas métricas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = cfg.M_PER_DEG_LAT
    mlon = M * np.cos(np.radians(float(gps["lat"].mean())))
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=120)
    ax.scatter(gps["lon"] * mlon, gps["lat"] * M, s=0.8, c="#BBBBBB",
               alpha=0.4, linewidths=0, rasterized=True)
    if shape_model is not None:
        ax.plot(np.asarray(shape_model["lons"]) * mlon,
                np.asarray(shape_model["lats"]) * M, "--", c="#D62728",
                lw=0.9)
    ax.plot(route["lon"] * mlon, route["lat"] * M, "-", c="#1F77B4", lw=1.5)
    ax.set_title(
        f"{meta['linha']}/{meta['sentido']} · {meta['n_trips']} viagens | "
        f"N={int(row['angular_samples'])} r={float(row['meters']):g} "
        f"G={float(row['increase_meters']):g} | "
        f"cov15={row['coverage_15m']:.0f}%",
        fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_work(wid):
    """Executa as 150 combinações de um item e salva rotas (parquet)."""
    st = _STATE[wid]
    meta = st["meta"]
    tag = f"{meta['linha']}_{meta['sentido']}_v{meta['n_trips']}"
    rows, chunks = [], []
    best = None

    for (min_m, max_m, samples, inc) in st["combos"]:
        params = dict(linha=meta["linha"], sentido=meta["sentido"],
                      n_trips=meta["n_trips"], meters=min_m,
                      max_meters=max_m, samples=samples, inc=inc)
        rid = run_id_from_params(params)
        t0 = time.perf_counter()
        try:
            route, info = _discover(st, min_m, max_m, samples, inc)
            err = ""
        except Exception as exc:
            route, info = pd.DataFrame(), {"stop_reason": "erro",
                                           "n_steps": 0}
            err = str(exc)
        runtime = time.perf_counter() - t0

        if err:
            adh, km, mae = {}, np.nan, np.nan
            completed = False
        else:
            adh = gps_adherence_metrics(route, st["gps"])
            km = route_length_km(route)
            mae = route_mae_to_shape_m(route, st["shape_model"])
            completed = bool(info["reached_end"] or info["closed_loop"])

        rows.append(dict(
            run_id=rid, linha=meta["linha"], sentido=meta["sentido"],
            shape_id=meta["shape_id"], n_trips=meta["n_trips"],
            n_trips_used=meta["n_trips_used"], n_vehicles=meta["n_vehicles"],
            n_input_points=meta["n_input_points"],
            angular_samples=int(samples), meters=float(min_m),
            max_meters=float(max_m), increase_meters=float(inc),
            n_route_points=len(route) if err == "" else 0,
            route_length_km=km,
            reached_end=int(info.get("reached_end", 0)),
            closed_loop=int(info.get("closed_loop", 0)),
            completed=int(completed), stop_reason=info.get("stop_reason", "erro"),
            coverage_pct=adh.get("coverage_pct", np.nan),
            coverage_5m=adh.get("coverage_5m", np.nan),
            coverage_10m=adh.get("coverage_10m", np.nan),
            coverage_15m=adh.get("coverage_15m", np.nan),
            mean_gps_dist_m=adh.get("mean_gps_dist_m", np.nan),
            p95_gps_dist_m=adh.get("p95_gps_dist_m", np.nan),
            mae_shape_m=mae,
            turns_per_km=turns_per_km(route) if err == "" else np.nan,
            runtime_s=runtime, erro=err,
        ))

        if err == "" and len(route):
            chunks.append(pd.DataFrame({
                "run_id": rid,
                "n_trips": meta["n_trips"],
                "meters": float(min_m),
                "angular_samples": int(samples),
                "increase_meters": float(inc),
                "seq": np.arange(len(route), dtype=np.int32),
                "lon": route["lon"].to_numpy(dtype=np.float64),
                "lat": route["lat"].to_numpy(dtype=np.float64),
            }))

        cov15 = rows[-1]["coverage_15m"]
        key = (completed, cov15 if np.isfinite(cov15) else -1)
        if best is None or key > best[0]:
            best = (key, route, rows[-1])

    # ------------------------------------------------------------
    # salvar: todas as rotas (parquet zstd) + melhor (csv/png)
    # ------------------------------------------------------------
    try:
        if chunks:
            pd.concat(chunks, ignore_index=True).to_csv(
                cfg.ROUTES_DIR / f"{tag}.csv.gz", index=False,
                compression="gzip")
    except Exception as exc:
        print(f"  [aviso] rotas {tag}: {exc}", flush=True)

    try:
        if best is not None and len(best[1]):
            best[1].to_csv(cfg.ROUTES_DIR / f"{tag}_melhor.csv",
                           index=False, float_format="%.7f")
            _save_best_png(best[1], st["gps"], st["shape_model"],
                           cfg.IMAGES_DIR / f"{tag}.png", meta, best[2])
    except Exception as exc:
        print(f"  [aviso] melhor {tag}: {exc}", flush=True)

    done = sum(1 for r_ in rows if r_["completed"])
    print(f"  [{tag}] {len(rows)} execs | completas {done}/{len(rows)} | "
          f"melhor cov15={max((r_['coverage_15m'] for r_ in rows), default=np.nan):.0f}%",
          flush=True)
    return rows


def snap_points(gps, start, end):
    d_start = np.sqrt((gps["lon"] - start[0]) ** 2
                      + (gps["lat"] - start[1]) ** 2) * cfg.M_PER_DEG_LAT
    if d_start.min() > 500:
        ss = [float(gps["lon"].median()), float(gps["lat"].median())]
    else:
        i = int(d_start.values.argmin())
        ss = [float(gps["lon"].iloc[i]), float(gps["lat"].iloc[i])]
    es = end
    if end is not None:
        d_end = np.sqrt((gps["lon"] - end[0]) ** 2
                        + (gps["lat"] - end[1]) ** 2) * cfg.M_PER_DEG_LAT
        if d_end.min() > 500:
            es = [float(gps["lon"].median()), float(gps["lat"].median())]
        else:
            j = int(d_end.values.argmin())
            es = [float(gps["lon"].iloc[j]), float(gps["lat"].iloc[j])]
    return ss, es


def build_state(shape_models, configs, df_pl, trip_summary):
    """Um item de trabalho por (linha, sentido, nº de viagens)."""
    state = {}
    for (linha, sentido, shape_id, start, end) in configs:
        sel_all = select_trips(trip_summary, linha, start, end,
                               max_vehicles=None,
                               start_radius_m=cfg.START_RADIUS_M)
        if sel_all.empty:
            print(f"[skip] {linha}/{sentido}: sem trips")
            continue
        n_before = len(sel_all)
        sel_all = validate_trips(df_pl, sel_all, shape_models, linha,
                                 shape_id,
                                 max_median_m=cfg.TRIP_SHAPE_MAX_MEDIAN_M)
        if len(sel_all) < n_before:
            print(f"[valida] {linha}/{sentido}: "
                  f"{n_before - len(sel_all)} descartadas")
        if sel_all.empty:
            continue

        # ordena as viagens (maiores primeiro) — volume de dados crescente
        sel_sorted = sel_all.sort_values("n_points",
                                         ascending=False).reset_index(drop=True)
        n_avail = len(sel_sorted)

        for N in cfg.TRIP_COUNTS:
            eff = min(N, n_avail)
            sel_n = sel_sorted.iloc[:eff]
            gps = fetch_trip_points(
                df_pl, sel_n, start=start, end=end,
                per_point_filter=getattr(cfg, "PER_POINT_FILTER", True)
                and (end is not None),
                max_points=cfg.MAX_INPUT_POINTS,
            ).dropna(subset=["lon", "lat"]).drop_duplicates(
                subset=["lon", "lat"]).reset_index(drop=True)
            if len(gps) < 50:
                print(f"[skip] {linha}/{sentido} v{N}: "
                      f"poucos pontos ({len(gps)})")
                continue
            ss, es = snap_points(gps, start, end)
            wid = len(state)
            state[wid] = dict(
                gps=gps, start_snap=ss, end_snap=es,
                shape_model=shape_models.get((linha, str(shape_id).strip())),
                combos=list(product(cfg.MIN_METERS_LIST, cfg.MAX_METERS_LIST,
                                    cfg.SAMPLES_LIST,
                                    cfg.INCREASE_METERS_LIST)),
                meta=dict(linha=linha, sentido=sentido, shape_id=shape_id,
                          n_trips=N, n_trips_used=eff,
                          n_vehicles=int(sel_n["veiculo"].nunique()),
                          n_input_points=len(gps)),
            )
            print(f"  [{wid}] {linha}/{sentido} v{N}: {eff}/{n_avail} viagens | "
                  f"{sel_n['veiculo'].nunique()} veic | {len(gps):,} amostras de GPS",
                  flush=True)
    return state


def _progress(done, total, t0):
    el = time.perf_counter() - t0
    eta = el / done * (total - done)
    print(f"  [{done}/{total}] {el / 60:.1f} min | ETA {eta / 60:.1f} min",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--limite", type=int, default=None)
    args = ap.parse_args()

    print("=" * 66)
    print("Experimento v8 — eixo de dados por Nº DE VIAGENS")
    print("fator 1.1–2.0 (10 valores) · 13 linhas · sem filtro direcional")
    print("=" * 66)

    for d in (cfg.OUT_ROOT, cfg.ROUTES_DIR, cfg.IMAGES_DIR, cfg.ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("\nCarregando shapes...")
    shape_models = load_shape_models(cfg.SHAPE_XZ_PATH)
    configs = build_configs(shape_models, cfg.LINHAS_CIRCULARES,
                            cfg.LINHAS_IDA_VOLTA)
    print(f"Shapes: {len(shape_models):,} | configs: {len(configs)}")

    print("\nCarregando dados GPS multi-dia...")
    tmap = None
    if getattr(cfg, "SEGMENT_AT_TERMINALS", False):
        tmap = {}
        for (linha, _s, _sid, st, en) in configs:
            if linha not in cfg.LINHAS_IDA_VOLTA:
                continue
            tmap.setdefault(linha, [])
            tmap[linha].append(tuple(st))
            if en is not None:
                tmap[linha].append(tuple(en))
    speed = cfg.MAX_SPEED_KMH if getattr(cfg, "USE_SPEED_FILTER", False) else None
    df_pl = load_gps_data(
        cfg.PATH_POSITIONS, linhas=cfg.LINHAS, split_terminals=tmap,
        max_gap_s=cfg.MAX_GAP_S, max_speed_kmh=speed, min_dt_s=cfg.MIN_DT_S,
        terminal_near_m=cfg.TERMINAL_NEAR_M, terminal_far_m=cfg.TERMINAL_FAR_M,
    )
    trip_summary = build_trip_summary(df_pl,
                                      min_trip_points=cfg.MIN_TRIP_POINTS)
    print(f"Amostras de GPS: {df_pl.height:,} | trips válidas: "
          f"{len(trip_summary):,}")

    print("\nMontando itens de trabalho (por nº de viagens)...")
    state = build_state(shape_models, configs, df_pl, trip_summary)
    del df_pl, trip_summary
    gc.collect()
    work_ids = sorted(state.keys())
    if args.limite:
        work_ids = work_ids[:args.limite]
        state = {k: state[k] for k in work_ids}
    n_exec = sum(len(state[w]["combos"]) for w in work_ids)
    print(f"Itens de trabalho: {len(work_ids)} | execuções: {n_exec:,}")

    global _STATE
    _STATE = state

    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    rows, done, t0 = [], 0, time.perf_counter()
    n_procs = max(1, args.procs)
    if n_procs == 1:
        for wid in work_ids:
            rows.extend(run_work(wid))
            done += 1
            _progress(done, len(work_ids), t0)
    else:
        with mp.Pool(processes=n_procs) as pool:
            for out_rows in pool.imap_unordered(run_work, work_ids,
                                                chunksize=1):
                rows.extend(out_rows)
                done += 1
                _progress(done, len(work_ids), t0)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.SUMMARY_CSV, index=False, float_format="%.6f")

    print("\n" + "=" * 66)
    print(f"Concluído: {len(df):,} execuções em "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    if len(df):
        print(f"Completas: {df['completed'].sum():,} "
              f"({100 * df['completed'].mean():.1f}%)")
    print(f"Sumário: {cfg.SUMMARY_CSV}")
    print(f"Rotas:   {cfg.ROUTES_DIR} (parquet, uma por item)")
    print("=" * 66)


if __name__ == "__main__":
    main()
