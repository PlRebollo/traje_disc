"""
Modelo de descoberta de rota (caminhada gulosa guiada por destino).

Ideia: caminhar por uma nuvem de pontos GPS escolhendo, a cada passo, o
candidato mais apoiado pelos dados. A pontuação de cada candidato combina
densidade direcional (pontos que se movem na direção do candidato) e uma
guia proporcional ao progresso em direção ao destino.

Mudanças em relação ao modelo original:
1. Pontuação direcional: cada ponto GPS só pontua candidatos alinhados ao
   seu próprio vetor de deslocamento (evita recuo por depleção de densidade).
2. Guia ao destino g_i = 1 + progresso * max(0, cos(alinhamento)),
   sem constantes arbitrárias; progresso = fração do eixo A->B já percorrida.
3. Direção inicial: vetor resultante dos deslocamentos de TODOS os pontos
   ativos dentro do raio mínimo (não apenas os N mais próximos).
4. Captura apenas em torno do candidato escolhido.
5. Raio adaptativo: crescimento serve de ponte sobre buracos de dados.
6. Otimização exata: pré-filtro por desigualdade triangular (só pontos a
   distância <= 2r podem pontuar algum candidato).
"""

import numpy as np
import pandas as pd

from config import M_PER_DEG_LAT
from core.geometry import route_length_km


# ============================================================
# SUAVIZAÇÃO (Chaikin corner-cutting)
# ============================================================
def chaikin_smooth(lons, lats, iterations=2, cut=0.25):
    """
    Cada segmento (Pk, Pk+1) é substituído por dois pontos a 25% e 75%
    do comprimento. Extremidades preservadas. Em N iterações converge
    para uma B-spline quadrática dos pontos de controle.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    for _ in range(iterations):
        if len(lons) < 3:
            break
        q_lon = lons[:-1] + cut * (lons[1:] - lons[:-1])
        q_lat = lats[:-1] + cut * (lats[1:] - lats[:-1])
        r_lon = lons[:-1] + (1 - cut) * (lons[1:] - lons[:-1])
        r_lat = lats[:-1] + (1 - cut) * (lats[1:] - lats[:-1])
        pairs_lon = np.stack([q_lon, r_lon], axis=1).ravel()
        pairs_lat = np.stack([q_lat, r_lat], axis=1).ravel()
        lons = np.concatenate([[lons[0]], pairs_lon, [lons[-1]]])
        lats = np.concatenate([[lats[0]], pairs_lat, [lats[-1]]])
    return lons, lats


# ============================================================
# NÚCLEO DA CAMINHADA
# ============================================================
def _discover_core(
    df: pd.DataFrame,
    initial_point,
    end_point,
    meters: float,
    angular_samples: int,
    iterations: int,
    increase_meters: float,
    decrease_meters: float,
    max_meters: float,
    loop_close_radius: float,
    loop_min_steps: int,
    arrival_radius_m: float,
    max_length_km: float,
    consecutive_fail_stop: int,
    verbose: bool,
):
    """
    Executa a caminhada. Retorna (lons, lats, dist_meters, info).

    Componentes do score de cada candidato i:
      w(p,i) = max(0, cos(ângulo entre o deslocamento do ponto p e a
                          direção do candidato i))
      S_i    = [ Σ_p w(p,i)/d(p,C_i) ] · g_i
      g_i    = 1 + progresso(P) · max(0, cos(ângulo(C_i, P->B)))
    """
    # vetores de deslocamento por ponto (dentro de cada trip)
    if "dx" in df.columns and "dy" in df.columns:
        d_lon = df["dx"].to_numpy(dtype=float)
        d_lat = df["dy"].to_numpy(dtype=float)
    else:
        d_lon = df["lon"].diff().to_numpy()
        d_lat = df["lat"].diff().to_numpy()
    lons_all = df["lon"].to_numpy()
    lats_all = df["lat"].to_numpy()

    meters_rad = meters / M_PER_DEG_LAT
    meters_min_rad = meters_rad
    max_meters_rad = max_meters / M_PER_DEG_LAT
    loop_close_rad = loop_close_radius / M_PER_DEG_LAT
    arrival_rad = arrival_radius_m / M_PER_DEG_LAT
    max_length_rad = (max_length_km * 1000.0) / M_PER_DEG_LAT

    start = np.array(initial_point, dtype=float)
    pos = start.copy()

    end = None
    axis = None
    axis_norm2 = None
    if end_point is not None:
        end = np.array(end_point, dtype=float)
        axis = end - start
        axis_norm2 = float(np.dot(axis, axis))

    route_lons, route_lats, dist_meters = [], [], []
    cumulative = 0.0

    angles = np.linspace(0, 2 * np.pi, angular_samples, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    info = {
        "reached_end": False,
        "closed_loop": False,
        "stop_reason": "max_iterations",
        "n_steps": 0,
    }

    fails = 0
    n = 0

    # arrays de trabalho: pontos ativos compactados
    alive = np.ones(len(df), dtype=bool)
    alive[0] = False
    w_lon = lons_all[alive]
    w_lat = lats_all[alive]
    w_dlon = d_lon[alive]
    w_dlat = d_lat[alive]
    w_norm = np.sqrt(w_dlon ** 2 + w_dlat ** 2)
    w_norm_safe = np.where(w_norm > 0, w_norm, 1e-9)

    # ------------------------------------------------------------
    # DIREÇÃO INICIAL: vetor resultante dos deslocamentos de todos os
    # pontos ativos dentro do raio mínimo.
    #   d0 = (Σ v_j) / |Σ v_j|,  para j com d(x_j, P0) <= r_min
    # Direções opostas se cancelam; a resultante indica o sentido
    # dominante de movimento no terminal.
    # ------------------------------------------------------------
    init_dir = None
    if len(w_lon) >= 3:
        d_start = np.sqrt((w_lon - start[0]) ** 2 + (w_lat - start[1]) ** 2)
        idx_near = np.where(d_start <= meters_min_rad)[0]
        if len(idx_near) >= 3:
            vlon = w_dlon[idx_near]
            vlat = w_dlat[idx_near]
            valid = np.isfinite(vlon) & np.isfinite(vlat)
            if valid.sum() >= 3:
                sx = float(np.sum(vlon[valid]))
                sy = float(np.sum(vlat[valid]))
                norm = np.sqrt(sx ** 2 + sy ** 2)
                if norm > 0:
                    init_dir = np.array([sx / norm, sy / norm])

    while (n < iterations) and (meters_rad <= max_meters_rad):
        n += 1
        info["n_steps"] = n

        cand_lon = pos[0] + meters_rad * cos_a
        cand_lat = pos[1] + meters_rad * sin_a

        # ---------------- fator de guia ----------------
        if end is not None:
            to_target = end - pos
            t_norm = np.linalg.norm(to_target)
            align = (
                cos_a * to_target[0] / t_norm + sin_a * to_target[1] / t_norm
                if t_norm > 0 else np.ones(angular_samples)
            )
            progress = (
                float(np.clip(np.dot(pos - start, axis) / axis_norm2, 0.0, 1.0))
                if axis_norm2 > 0 else 0.0
            )
            guide = 1.0 + progress * np.maximum(align, 0.0)
        else:
            guide = np.ones(angular_samples)

        # primeiros passos: restringe à direção inicial estimada
        if len(route_lons) < 2 and init_dir is not None:
            align_init = cos_a * init_dir[0] + sin_a * init_dir[1]
            guide = guide * np.where(align_init > 0, 1.0, 0.0)

        if len(w_lon) == 0:
            info["stop_reason"] = "sem_pontos_ativos"
            break

        # pré-filtro exato: só pontos a <= 2r podem pontuar algum candidato
        d_now = np.sqrt((w_lon - pos[0]) ** 2 + (w_lat - pos[1]) ** 2)
        near = np.where(d_now <= 2.0 * meters_rad)[0]

        best_score = -1.0
        best_point = None
        best_captured = np.array([], dtype=int)

        if len(near) > 0:
            nl, nlt = w_lon[near], w_lat[near]
            ndl, ndt = w_dlon[near], w_dlat[near]
            nns = w_norm_safe[near]

            for i in range(angular_samples):
                dist2 = np.sqrt((nl - cand_lon[i]) ** 2 + (nlt - cand_lat[i]) ** 2)
                mask = dist2 <= meters_rad
                if not np.any(mask):
                    continue

                dot = ndl * cos_a[i] + ndt * sin_a[i]
                with np.errstate(invalid="ignore"):
                    dir_w = np.maximum(dot / nns, 0.0)
                dir_w = np.nan_to_num(dir_w, nan=0.0, posinf=0.0, neginf=0.0)

                w = mask & (dir_w > 0)
                if not np.any(w):
                    continue

                density = float(np.sum(dir_w[w] / dist2[w]))
                score = density * guide[i]
                if score > best_score:
                    best_score = score
                    best_point = (cand_lon[i], cand_lat[i])
                    best_captured = near[w]

        if best_score <= 0:
            meters_rad *= increase_meters
            if meters_rad >= max_meters_rad:
                meters_rad = max_meters_rad
                fails += 1
                if fails >= consecutive_fail_stop:
                    info["stop_reason"] = "passos_sem_dados"
                    break
            continue

        fails = 0
        route_lons.append(best_point[0])
        route_lats.append(best_point[1])
        dist_meters.append(meters_rad * M_PER_DEG_LAT)
        cumulative += meters_rad

        keep = np.ones(len(w_lon), dtype=bool)
        keep[best_captured] = False
        w_lon, w_lat = w_lon[keep], w_lat[keep]
        w_dlon, w_dlat = w_dlon[keep], w_dlat[keep]
        w_norm_safe = w_norm_safe[keep]

        pos = np.array(best_point)
        meters_rad = max(meters_min_rad, meters_rad * decrease_meters)

        # ---------------- critérios de parada ----------------
        if end is not None:
            if np.sqrt((pos[0] - end[0]) ** 2 + (pos[1] - end[1]) ** 2) <= arrival_rad:
                info["reached_end"] = True
                info["stop_reason"] = "chegada_destino"
                break

        if end is None and loop_close_rad > 0 and len(route_lons) >= loop_min_steps:
            if np.sqrt((route_lons[-1] - start[0]) ** 2 +
                       (route_lats[-1] - start[1]) ** 2) <= loop_close_rad:
                info["closed_loop"] = True
                info["stop_reason"] = "loop_fechado"
                break

        if cumulative >= max_length_rad:
            info["stop_reason"] = "limite_comprimento"
            break

    if verbose:
        print(f"\n  parada: {info['stop_reason']} | passos={info['n_steps']}")

    return route_lons, route_lats, dist_meters, info


def discover_route(
    df: pd.DataFrame,
    initial_point: list,
    end_point: list = None,
    meters: float = 10.0,
    angular_samples: int = 180,
    iterations: int = 200000,
    increase_meters: float = 1.25,
    decrease_meters: float = None,
    max_meters: float = 250.0,
    loop_close_radius: float = 25.0,
    loop_min_steps: int = 80,
    arrival_radius_m: float = 80.0,
    max_length_factor: float = 2.5,
    max_length_km_circular: float = 120.0,
    consecutive_fail_stop: int = 10,
    smooth_iterations: int = 2,
    retry_relaxed: bool = True,
    verbose: bool = False,
):
    """
    Descobre a rota entre initial_point e end_point (ou loop circular se
    end_point=None). Retorna (route_df, info).

    Re-tenta com parâmetros relaxados se a rota ficar muito curta, e
    aplica suavização Chaikin no final.
    """
    if decrease_meters is None:
        decrease_meters = 1.0 / increase_meters

    if end_point is not None:
        straight = route_length_km(_two_points(initial_point, end_point))
        max_length_km = max(max_length_factor * straight, 5.0)
    else:
        max_length_km = max_length_km_circular

    def run(m, ams, inc, dec, mmax, fail_stop):
        return _discover_core(
            df=df, initial_point=initial_point, end_point=end_point,
            meters=m, angular_samples=ams, iterations=iterations,
            increase_meters=inc, decrease_meters=dec, max_meters=mmax,
            loop_close_radius=loop_close_radius, loop_min_steps=loop_min_steps,
            arrival_radius_m=arrival_radius_m, max_length_km=max_length_km,
            consecutive_fail_stop=fail_stop, verbose=verbose,
        )

    lons, lats, dists, info = run(
        meters, angular_samples, increase_meters, decrease_meters,
        max_meters, consecutive_fail_stop,
    )
    info["retry_used"] = False

    if retry_relaxed and not (info["reached_end"] or info["closed_loop"]) \
            and len(lons) < 30:
        lons2, lats2, dists2, info2 = run(
            meters * 2.0, max(angular_samples, 180), increase_meters * 1.5,
            decrease_meters * 0.85, max_meters * 1.5,
            consecutive_fail_stop * 2,
        )
        info2["retry_used"] = True
        if len(lons2) > len(lons):
            lons, lats, dists, info = lons2, lats2, dists2, info2

    if len(lons) >= 3:
        sl, sa = chaikin_smooth(lons, lats, iterations=smooth_iterations)
        if len(dists) > 0:
            sd = np.interp(
                np.linspace(0, 1, len(sl)),
                np.linspace(0, 1, len(dists)),
                np.asarray(dists, dtype=float),
            )
        else:
            sd = np.zeros(len(sl))
        route = pd.DataFrame({"lon": sl, "lat": sa, "_meters_": sd})
    else:
        route = pd.DataFrame({"lon": lons, "lat": lats, "_meters_": dists})

    return route, info


def _two_points(a, b) -> pd.DataFrame:
    return pd.DataFrame({"lon": [a[0], b[0]], "lat": [a[1], b[1]]})
