"""
Experimento v4 — Configuração.

Modelo de descoberta de rotas de ônibus a partir de dados GPS.
Organização completa: código em core/, saídas em outputs/.

Linhas testadas: 16 (4 circulares + 12 ida/volta).
Fonte de dados: um dia completo de operação (2026-02-03), todas as linhas.
Terminais extraídos automaticamente dos shapes oficiais.
"""

from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================
BASE = Path(__file__).resolve().parent.parent

PATH_POSITIONS = BASE / "urbs_data - cópia" / "2026_02_03_vehicle_positions.parquet"
SHAPE_XZ_PATH = BASE / "urbs_data - cópia" / "2026_07_01_shapeLinha.json.xz"

OUT_ROOT = Path(__file__).resolve().parent / "outputs"
ROUTES_DIR = OUT_ROOT / "routes"
IMAGES_DIR = OUT_ROOT / "images"
ANALYSIS_DIR = OUT_ROOT / "analysis"
SUMMARY_CSV = OUT_ROOT / "experiment_summary.csv"

# ============================================================
# LINHAS (16)
#   tipo "circular"  -> 1 shape  (start = shape start)
#   tipo "ida_volta" -> 2 shapes (ida = shape[0], volta = shape[1])
# Terminais extraídos dos shapes em tempo de execução.
# ============================================================
LINHAS_CIRCULARES = ["020", "021", "022", "023"]
LINHAS_IDA_VOLTA = [
    "303", "203", "603", "338", "924", "658",
    "506", "707", "505", "545", "307", "607",
]
LINHAS = LINHAS_CIRCULARES + LINHAS_IDA_VOLTA

# ============================================================
# FILTRO DE QUALIDADE DOS DADOS GPS
# ============================================================
MIN_TRIP_POINTS = 150          # mínimo de pings por viagem
MAX_GAP_S = 900                # gap que separa viagens (15 min)
MAX_SPEED_KMH = 120            # velocidade máxima plausível
MIN_DT_S = 1                   # intervalo mínimo entre pings

# ============================================================
# GRID DE HIPERPARÂMETROS (expandido)
# ============================================================
BUS_COUNTS = [2, 5]            # circulares: nº de veículos amostrados
SAMPLES_LIST = [90, 180, 360]  # amostras angulares N
MIN_METERS_LIST = [5, 10, 20]  # raio mínimo r_min (m)
MAX_METERS_LIST = [250]        # raio máximo r_max (m)
INCREASE_METERS_LIST = [1.1, 1.25, 1.5, 2.0]  # taxa de crescimento G

# ============================================================
# MODELO
# ============================================================
ARRIVAL_RADIUS_M = 80.0        # raio de chegada ao destino B
LOOP_CLOSE_RADIUS_M = 25.0     # raio de fechamento de loop (circular)
LOOP_MIN_STEPS = 80            # mínimo de passos antes de fechar loop
MAX_LENGTH_FACTOR = 2.5        # comprimento máx = 2.5 x distância reta A-B
MAX_LENGTH_KM_CIRCULAR = 120.0
CONSECUTIVE_FAIL_STOP = 10     # falhas consecutivas no raio máx -> para
SMOOTH_ITERATIONS = 2          # Chaikin

# ============================================================
# SELEÇÃO DE TRIPS
# ============================================================
START_RADIUS_M = 2000.0        # trip genuína: parte a <= 2 km do terminal
MAX_INPUT_POINTS = 60000       # limite de pontos de entrada (amostragem)
TRIP_SHAPE_MAX_MEDIAN_M = 200.0  # validação: descarta trip cuja mediana
                                 # de distância ao shape excede este valor

# ============================================================
# MÉTRICA DE ADERÊNCIA
#
# Threshold = 15 m, justificado pela precisão horizontal 95% de
# receptores GPS single-frequency em ambiente urbano:
#   Kaplan & Hegarty (2017), "Understanding GPS/GNSS" — 10-15 m (95%)
#   GPS SPS Performance Standard (DoD, 2020) — 7.8 m (sinal) + ruído
# Aderência = % de pings dentro desta banda da rota reconstruída.
# ============================================================
ADHERENCE_THRESHOLD_M = 15.0

# ============================================================
# CONSTANTES FÍSICAS
# ============================================================
M_PER_DEG_LAT = 111_132.0
SEGMENT_CHUNK = 8000
