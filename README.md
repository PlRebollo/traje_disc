# Descoberta de rotas de ônibus a partir de dados GPS

Modelo de reconstrução automática de trajetos de linhas de ônibus a partir
de pings GPS brutos. A rota é descoberta diretamente dos dados observados,
usando os terminais oficiais apenas como sementes (ponto de partida e
destino), sem depender do shape oficial como referência de trajeto.

Este repositório contém o **experimento v4**: 16 linhas de uma cidade
brasileira (4 circulares + 12 ida/volta), com um dia completo de operação.
O shape oficial é usado somente como referência de avaliação (MAE), nunca
como guia da reconstrução — o que permite detectar divergências entre o
shape cadastrado e a operação real.

## Estrutura

```
.
├── config.py              # Configuração: linhas, grid de hiperparâmetros, métricas
├── run_experiment.py      # Executa o grid completo de experimentos
├── analysis.py            # Orquestrador das figuras de análise
├── analysis_common.py     # Carregamento do sumário, ordem e cores dos configs
├── analysis_maps.py       # Figuras em mapa, grids espaciais, melhores rotas
├── analysis_perline.py    # Sensibilidade e heatmaps por linha
├── core/
│   ├── shapes.py          # Carregamento e indexação dos shapes oficiais
│   ├── trips.py           # Segmentação, separação de sentido, seleção de trips
│   ├── model.py           # Modelo de descoberta de rota
│   ├── geometry.py        # Geometria, distâncias, identificador de execução
│   ├── metrics.py         # Métricas de aderência aos dados GPS
│   └── imaging.py         # Imagens de rota com mapa de fundo
└── outputs/
    ├── experiment_summary.csv   # Resultado de todas as execuções
    ├── run.log                  # Log da execução do grid
    └── analysis/                # Figuras e tabelas de análise
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Dados de entrada

O experimento espera dois arquivos de dados, referenciados em `config.py`:

| Constante          | Arquivo esperado                                        |
|--------------------|---------------------------------------------------------|
| `PATH_POSITIONS`   | `urbs_data - cópia/2026_02_03_vehicle_positions.parquet` |
| `SHAPE_XZ_PATH`    | `urbs_data - cópia/2026_07_01_shapeLinha.json.xz`         |

Por padrão, `config.py` procura esses arquivos no diretório **pai** do
repositório (`.parent.parent`), na pasta `urbs_data - cópia/`. Os dados
não são versionados (arquivos grandes / privados).

## Como executar

```bash
# 1. Rodar o grid de experimentos (gera rotas, sumário e imagens)
python run_experiment.py

# 2. Gerar as figuras e tabelas de análise
python analysis.py
```

O grid é definido em `config.py`:

```python
BUS_COUNTS           = [2, 5]                     # circulares: nº de veículos
SAMPLES_LIST         = [90, 180, 360]             # amostras angulares N
MIN_METERS_LIST      = [5, 10, 20]                # raio mínimo r_min (m)
MAX_METERS_LIST      = [250]                      # raio máximo r_max (m)
INCREASE_METERS_LIST = [1.1, 1.25, 1.5, 2.0]      # taxa de crescimento G
```

## Métrica de aderência

Como o shape oficial pode estar desatualizado, a qualidade da reconstrução
é medida **diretamente contra os dados observados**:

- `coverage_pct` (%) — fração de pings a ≤ 15 m da rota reconstruída;
- `mean_gps_dist_m` — distância média ping → rota;
- `p95_gps_dist_m` — percentil 95 (captura fugas localizadas).

O limiar de 15 m segue a precisão horizontal de 95% de receptores GPS
single-frequency em ambiente urbano (Kaplan & Hegarty, 2017; GPS SPS
Performance Standard, DoD, 2020). O MAE contra o shape oficial é mantido
apenas como referência.

## Resultados

Grid completo: **1.152 execuções** (28 configs × 36 combinações, com
circulares variando entre 2 e 5 veículos).

- **620 execuções completas** (53,8%);
- **18 dos 28 configs** atingiram o destino/loop fechado em pelo menos
  uma combinação.

Melhores resultados por linha/sentido (tabela completa em
`outputs/analysis/best_configs_v4.csv`):

| Linha | Sentido  | Aderência (%) | r_min (m) | Fator |
|-------|----------|---------------|-----------|-------|
| 021   | circular | 98,3          | 5         | 1,25  |
| 020   | circular | 98,2          | 5         | 1,25  |
| 022   | circular | 97,1          | 5         | 1,5   |
| 506   | ida      | 94,4          | 10        | 1,5   |
| 203   | ida      | 94,7          | 10        | 1,1   |
| 303   | ida      | 89,1          | 10        | 1,5   |

## Figuras de análise

Geradas em `outputs/analysis/`:

- `fig_01` – sensibilidade dos parâmetros;
- `fig_02` – heatmaps de aderência mediana (ida/volta vs circular);
- `fig_03` – matriz de taxa de conclusão por config × fator;
- `fig_05` – galeria de casos de sucesso;
- `fig_06` – aderência vs divergência do shape;
- `fig_07` – comprimento reconstruído vs shape oficial;
- `fig_08` – escalabilidade do tempo de execução;
- `fig_10` – motivos de parada por config;
- `fig_11` – p95 vs distância média;
- `fig_12` – aderência vs volume de dados;
- `fig_13` – compressão de pontos;
- `fig_14` – separação de sentido (linha 303);
- `per_line/` – sensibilidade, heatmaps, grids espaciais e melhores rotas
  de cada config.

## Licença

Uso acadêmico / de pesquisa.
