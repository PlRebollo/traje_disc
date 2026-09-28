"""
Configuração do modelo de reconstrução de rotas.

Modelo final: segmentação por terminal e sem filtro direcional de dados.
O trabalho caminha sobre a nuvem de pings e reconstrói a rota de cada
sentido.

Dois pontos definem o experimento:

  1) Fator de escala do raio em 10 valores: 1.1 a 2.0 (passos de 0.1);
  2) O eixo de volume de dados é o NÚMERO DE VIAGENS (corridas completas de
     sentido único, resultado da segmentação por terminal). Para cada
     tamanho N, usam-se as N maiores viagens genuínas da linha (misturando
     veículos quando necessário).

Grid: 3 ângulos × 5 raios × 10 fatores × 9 tamanhos (1–500 viagens).
"""

from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================
BASE = Path(__file__).resolve().parent
HERE = BASE

# Dados de entrada (não versionados; coloque os arquivos em data/)
PATH_POSITIONS = BASE / "data" / "multiday_positions.parquet"
SHAPE_XZ_PATH = BASE / "data" / "2026_07_01_shapeLinha.json.xz"

OUT_ROOT = HERE / "outputs"
ROUTES_DIR = OUT_ROOT / "rotas"
IMAGES_DIR = OUT_ROOT / "images"
ANALYSIS_DIR = OUT_ROOT / "analysis"
SUMMARY_CSV = OUT_ROOT / "experiment_summary.csv"

# ============================================================
# LINHAS (13 — sem 545, 707, 924)
# ============================================================
LINHAS_CIRCULARES = ["020", "021", "022", "023"]
LINHAS_IDA_VOLTA = [
    "303", "203", "603", "338", "658",
    "506", "505", "307", "607",
]
LINHAS = LINHAS_CIRCULARES + LINHAS_IDA_VOLTA

# ============================================================
# FILTRO DE QUALIDADE DOS DADOS GPS
# ============================================================
MIN_TRIP_POINTS = 150
MAX_GAP_S = 900
MAX_SPEED_KMH = 120
MIN_DT_S = 1

# ============================================================
# GRID DE HIPERPARÂMETROS
# ============================================================
SAMPLES_LIST = [90, 180, 360]                    # 4°, 2°, 1°
MIN_METERS_LIST = [5, 10, 15, 20, 25]            # raio mínimo (m)
MAX_METERS_LIST = [250]                          # raio máximo (m)
INCREASE_METERS_LIST = [1.1, 1.2, 1.3, 1.4, 1.5,
                        1.6, 1.7, 1.8, 1.9, 2.0]  # fator de escala
TRIP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # nº de viagens (dados)

# ============================================================
# MODELO
# ============================================================
ARRIVAL_RADIUS_M = 80.0
LOOP_CLOSE_RADIUS_M = 25.0
LOOP_MIN_STEPS = 80
CONSECUTIVE_FAIL_STOP = 10
SMOOTH_ITERATIONS = 2

# ============================================================
# PIPELINE DE DADOS
# ============================================================
START_RADIUS_M = 2000.0
MAX_INPUT_POINTS = 60000
TRIP_SHAPE_MAX_MEDIAN_M = 200.0

# segmentação por terminal + SEM filtro direcional de dados
SEGMENT_AT_TERMINALS = True
PER_POINT_FILTER = False

# ============================================================
# ADERÊNCIA (3 bandas)
# ============================================================
ADHERENCE_THRESHOLDS_M = (5.0, 10.0, 15.0)
ADHERENCE_THRESHOLD_M = 15.0

# ============================================================
# CONSTANTES FÍSICAS
# ============================================================
M_PER_DEG_LAT = 111_132.0
SEGMENT_CHUNK = 8000
