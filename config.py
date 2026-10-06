"""
Experimento v10 (refino) — Configuração final do modelo refinado.

Mudanças em relação ao modelo anterior, todas testadas por ablação:

  1) Filtro de velocidade e corte de 150 amostras: REMOVIDOS. Não agregavam
     e o filtro de velocidade descartava amostras de GPS válidas.
  2) Segmentação temporal por gap de 900 s: MANTIDA. É ela que limpa as
     corridas antes do corte por terminal; sem ela a separação ida/volta
     piora.
  3) Corte por terminal: limiares definidos a partir dos dados
     (analise_limiares_terminal.py). A velocidade sobe de ~7 km/h na zona
     de parada e atinge 90% do cruzeiro (~24 km/h) a ~175 m; o cruzeiro
     (~27 km/h) se estabiliza por volta de 300-500 m. Adota-se
     near = 150 m (zona de manobra) e far = 500 m (veículo saiu de fato).
  4) Persistência de 10 falhas consecutivas: MANTIDA. Reduzir para 1
     piorava a conclusão (a caminhada precisa insistir em buracos de dados).
  5) Raio de chegada: unificado com o laço em 25 m. Frente à precisão do
     GPS (~15 m) e ao espaçamento do traçado, 80 m era grande e sem efeito
     mensurável; 25 m simplifica e não altera a qualidade.
  6) Kernel de distância: mantido 1/d puro. O 1/(d+1) piorou.

Grid: raios 5, 10 e 15 m (os valores 20 e 25 não se mostraram necessários).
"""

from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================
BASE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

PATH_POSITIONS = BASE / "experimento_v7" / "data" / "multiday_positions.parquet"
SHAPE_XZ_PATH = BASE / "urbs_data - cópia" / "2026_07_01_shapeLinha.json.xz"

OUT_ROOT = HERE / "outputs"
ROUTES_DIR = OUT_ROOT / "rotas"
IMAGES_DIR = OUT_ROOT / "images"
ANALYSIS_DIR = OUT_ROOT / "analysis"
SUMMARY_CSV = OUT_ROOT / "experiment_summary.csv"

# ============================================================
# LINHAS
# ============================================================
LINHAS_CIRCULARES = ["020", "021", "022", "023"]
LINHAS_IDA_VOLTA = [
    "303", "203", "603", "338", "658",
    "506", "505", "307", "607",
]
LINHAS = LINHAS_CIRCULARES + LINHAS_IDA_VOLTA

# ============================================================
# FILTROS DE QUALIDADE
# ============================================================
USE_SPEED_FILTER = False      # filtro de velocidade removido
MAX_SPEED_KMH = 120           # usado só se USE_SPEED_FILTER = True
MIN_DT_S = 1
MIN_TRIP_POINTS = 1           # <=2 = sem corte de tamanho (mínimo 2 amostras)
MAX_GAP_S = 900               # segmentação temporal por gap (15 min)

# ============================================================
# CORTE POR TERMINAL (definidos a partir dos dados)
# ============================================================
TERMINAL_NEAR_M = 150.0       # zona de manobra/parada
TERMINAL_FAR_M = 500.0        # veículo saiu de fato do terminal

# Filtro de retilinearidade: descarta viagens que na verdade são voltas
# completas ida+volta (deslocamento líquido quase nulo). Viagens de sentido
# único ficam acima de ~0,2 e as voltas completas abaixo de ~0,1, então
# qualquer corte nesse intervalo é equivalente.
STRAIGHT_MIN = 0.15

# ============================================================
# GRID (raios 5, 10 e 15)
# ============================================================
SAMPLES_LIST = [90, 180, 360]
MIN_METERS_LIST = [5, 10, 15]
MAX_METERS_LIST = [250]
INCREASE_METERS_LIST = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
TRIP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# ============================================================
# MODELO
# ============================================================
ARRIVAL_RADIUS_M = 25.0       # unificado com o laço
LOOP_CLOSE_RADIUS_M = 25.0
LOOP_MIN_STEPS = 80
CONSECUTIVE_FAIL_STOP = 10
DIST_FLOOR = 0.0              # 1/d puro
SMOOTH_ITERATIONS = 2

# ============================================================
# PIPELINE DE DADOS
# ============================================================
START_RADIUS_M = 2000.0
MAX_INPUT_POINTS = 60000
TRIP_SHAPE_MAX_MEDIAN_M = 200.0
SEGMENT_AT_TERMINALS = True
PER_POINT_FILTER = False

# ============================================================
# ADERÊNCIA
# ============================================================
ADHERENCE_THRESHOLDS_M = (5.0, 10.0, 15.0)
ADHERENCE_THRESHOLD_M = 15.0

# ============================================================
# CONSTANTES FÍSICAS
# ============================================================
M_PER_DEG_LAT = 111_132.0
SEGMENT_CHUNK = 8000
