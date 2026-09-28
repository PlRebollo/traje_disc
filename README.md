# Descoberta de rotas de ônibus a partir de dados GPS

Reconstrução automática do traçado de linhas de ônibus a partir de pings
GPS brutos. A rota é descoberta diretamente dos dados observados. O traçado
oficial é usado apenas para definir os terminais de partida e chegada e,
depois, como referência de avaliação. Ele nunca guia a reconstrução, o que
permite detectar divergências entre o traçado cadastrado e a operação real.

## O modelo

Cada viagem de um veículo é cortada ao passar pelos terminais, de modo que
cada segmento tenha um único sentido. Sobre a nuvem de pings do sentido, uma
caminhada avança em passos curtos. Em cada passo, gera candidatos em um anel
de raio adaptativo e escolhe o candidato com maior apoio nos dados, somando
o inverso das distâncias aos pings que apontam na mesma direção. Um fator de
guia favorece candidatos que avançam rumo ao destino. O raio cresce para
atravessar trechos sem pings e encolhe quando há dados. Ao final, o traçado
é suavizado (Chaikin).

Os detalhes de cada etapa, com equações e ilustrações, estão descritos no
relatório interno do projeto (não versionado neste repositório).

## Estrutura

```
.
├── config.py                  # Linhas, grid de hiperparâmetros e métricas
├── run_experiment.py          # Executa o grid completo
├── analysis.py                # Figuras globais e por linha
├── analysis_common.py         # Carregamento do sumário e utilidades
├── analysis_maps.py           # Mapas, grids espaciais e galeria
├── analysis_perline.py        # Sensibilidade e efeito de dados por linha
├── analysis_data_volume.py    # Análise do volume de dados
├── analysis_regimes.py        # Análise por faixas de dados
├── analysis_diagnostics.py    # Mapas de diagnóstico e painéis de desempenho
├── core/
│   ├── shapes.py              # Traçados oficiais e terminais
│   ├── trips.py               # Segmentação, sentido e seleção de viagens
│   ├── model.py               # Caminhada guiada
│   ├── geometry.py            # Distâncias e geometria
│   ├── metrics.py             # Métricas de aderência
│   └── imaging.py             # Imagens de rota
└── outputs/
    ├── experiment_summary.csv # Resultado de todas as execuções
    ├── run_v9.log             # Log da execução do grid
    ├── images/                # Melhor rota de cada par linha/sentido
    └── analysis/              # Figuras e tabelas de análise
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Dados de entrada

Os dados não são versionados (arquivos grandes). Coloque os dois arquivos
em uma pasta `data/` na raiz do projeto:

| Constante        | Arquivo esperado                          |
|------------------|-------------------------------------------|
| `PATH_POSITIONS` | `data/multiday_positions.parquet`         |
| `SHAPE_XZ_PATH`  | `data/2026_07_01_shapeLinha.json.xz`      |

Os caminhos ficam em `config.py` e podem ser ajustados.

## Como executar

```bash
# 1. Grid completo (gera rotas, sumário e imagens das melhores rotas)
python run_experiment.py

# 2. Figuras e tabelas de análise
python analysis.py
python analysis_data_volume.py
python analysis_regimes.py

# 3. Diagnósticos (mapas de falha e painéis de desempenho)
python analysis_diagnostics.py
```

## Grid do experimento

```python
SAMPLES_LIST         = [90, 180, 360]                  # resolução angular N
MIN_METERS_LIST      = [5, 10, 15, 20, 25]             # raio mínimo r_min
MAX_METERS_LIST      = [250]                           # raio máximo r_max
INCREASE_METERS_LIST = [1.1, 1.2, ..., 2.0]            # fator de crescimento g
TRIP_COUNTS          = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # nº de viagens
```

São 1350 execuções por par linha/sentido. Com 13 linhas (4 circulares e 9 de
ida e volta), o total é de 29.700 execuções.

## Métrica de aderência

A qualidade é medida diretamente contra os dados observados, porque o
traçado oficial pode estar desatualizado:

- `coverage_5m`, `coverage_10m`, `coverage_15m` — fração de pings a até 5,
  10 e 15 m da rota reconstruída;
- `mean_gps_dist_m` — distância média entre ping e rota;
- `p95_gps_dist_m` — percentil 95 dessa distância.

A banda de 15 m é a referência. O erro médio contra o traçado oficial é
mantido apenas como comparação.

## Resultados

Grid completo: **29.700 execuções**.

- Conclusão global: 63,2%;
- Aderência mediana na banda de 15 m: 86,5%;
- Com 500 viagens: aderência mediana 93,8% e conclusão 94,9%;
- 16 dos 22 pares linha/sentido atingem ao menos 90% de aderência com 500
  viagens.

As figuras ficam em `outputs/analysis/` (globais, `per_line/`,
`data_volume/` e `regimes/`) e as imagens das melhores rotas em
`outputs/images/`.
