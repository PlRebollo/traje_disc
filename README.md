# Descoberta de rotas de ônibus a partir de amostras de GPS

Reconstrução automática do traçado de linhas de ônibus a partir de amostras
de GPS brutas. A rota é descoberta diretamente dos dados observados. O
traçado oficial é usado apenas para definir os terminais de partida e
chegada e como referência de avaliação, nunca como guia da reconstrução.

## O modelo

Cada viagem de um veículo é cortada ao passar pelos terminais, de modo que
cada segmento tenha um único sentido. Sobre a nuvem de amostras do sentido,
uma caminhada avança em passos curtos. Em cada passo, gera candidatos em um
anel de raio adaptativo e escolhe o candidato com maior apoio nos dados,
somando o inverso das distâncias às amostras que apontam na mesma direção.
Um fator de guia favorece candidatos que avançam rumo ao destino. O raio
cresce para atravessar trechos sem dados e encolhe quando há dados. Ao
final, o traçado é suavizado (Chaikin).

## Refinamentos desta versão

Todos os ajustes foram avaliados por ablação (`benchmark.py`,
`benchmark_separacao.py`, `teste_*.py`):

- **Filtro de velocidade e corte de 150 amostras por viagem: removidos.**
  Não agregavam; o filtro de velocidade descartava amostras válidas.
- **Segmentação temporal por gap de 900 s: mantida.** É ela que limpa as
  corridas antes do corte por terminal; sem ela a separação ida/volta piora.
- **Limiares do corte por terminal definidos a partir dos dados**
  (`analise_limiares_terminal.py`): a velocidade sobe de ~7 km/h na zona de
  parada e atinge 90% do cruzeiro (~24 km/h) a ~175 m; adota-se
  `near = 150 m` e `far = 500 m`.
- **Filtro de retilinearidade** (`straight > 0,15`): descarta viagens que na
  verdade são voltas completas ida+volta, com deslocamento líquido quase
  nulo, que passavam no teste de sentido por ruído. Foi o ajuste que
  eliminou a contaminação em baixa densidade de dados.
- **Persistência de 10 falhas consecutivas: mantida.** Reduzir para 1
  piorava a conclusão.
- **Raio de chegada unificado com o laço em 25 m.** Frente à precisão do
  GPS (~15 m), 80 m era grande e sem efeito mensurável.
- **Kernel de distância 1/d mantido.** O 1/(d+1) piorou.

Grid: raios de 5, 10 e 15 m.

## Estrutura

```
.
├── config.py                     # Linhas, grid e parâmetros do modelo
├── run_experiment.py             # Executa o grid completo
├── core/
│   ├── trips.py                  # Segmentação, sentido, seleção, retilinearidade
│   ├── model.py                  # Caminhada guiada
│   ├── shapes.py                 # Traçados oficiais e terminais
│   ├── geometry.py               # Distâncias e geometria
│   ├── metrics.py                # Métricas de aderência
│   └── imaging.py                # Imagens de rota
├── analysis.py                   # Figuras globais e por linha
├── analysis_perline.py           # Sensibilidade e efeito de dados por linha
├── analysis_maps.py              # Mapas e grades espaciais
├── analysis_data_volume.py       # Análise do volume de dados
├── analysis_regimes.py           # Análise por faixas de dados
├── analysis_diagnostics.py       # Mapas de diagnóstico e painéis
├── analise_limiares_terminal.py  # Justificativa dos limiares de terminal
├── diag_gps_vs_shape.py          # Alinhamento GPS x traçado oficial
├── benchmark.py                  # Ablação de filtros e do modelo
├── benchmark_separacao.py        # Ablação da separação ida/volta
├── teste_retilinearidade.py      # Efeito do filtro de retilinearidade
└── outputs/
    ├── experiment_summary.csv    # Resultado de todas as execuções
    ├── images/                   # Melhor rota de cada par linha/sentido
    └── analysis/                 # Figuras e tabelas de análise
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Dados de entrada

Os dados não são versionados. Coloque os arquivos em `data/`:

| Constante        | Arquivo esperado                        |
|------------------|-----------------------------------------|
| `PATH_POSITIONS` | `data/multiday_positions.parquet`       |
| `SHAPE_XZ_PATH`  | `data/2026_07_01_shapeLinha.json.xz`    |

## Como executar

```bash
python run_experiment.py        # grid completo
python analysis.py              # figuras globais e por linha
python analysis_data_volume.py  # figuras de volume de dados
python analysis_regimes.py      # figuras por faixas de dados
python analysis_diagnostics.py  # diagnósticos
```

## Grid

```python
SAMPLES_LIST         = [90, 180, 360]                    # resolução angular
MIN_METERS_LIST      = [5, 10, 15]                       # raio mínimo (m)
INCREASE_METERS_LIST = [1.1, 1.2, ..., 2.0]              # fator de crescimento
TRIP_COUNTS          = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # nº de viagens
```

São 810 execuções por par linha/sentido e 17.820 no total (13 linhas,
22 pares).

## Resultado

No mesmo grid, o modelo refinado melhora a aderência na banda de 15 m de
88,3% para 92,6% (91,1% em ida/volta, contra 83,6% antes) e eleva a
conclusão de 66,4% para 68,1%, com ganho maior no regime de 20 a 200
viagens.
