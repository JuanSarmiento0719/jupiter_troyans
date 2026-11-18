# Los troyanos de Júpiter: ¿hasta dónde sirve el análisis lineal?
Mecánica Clásica, UIS, Reto III.1. Paola Andrea Ríos Campos y Juan Diego Sarmiento Villarreal.

## Reproducir todo
Opción A, Google Colab: abrir `Troyanos_Jupiter.ipynb`, ejecutar todas las celdas y subir `datos/sbdb_query_results2.csv` cuando lo pida.

Opción B, local (Python ≥ 3.9 con numpy, scipy, pandas y matplotlib):
```bash
python run_barrido.py            # barrido en amplitud (PR3C plano)          ~1 min
python run_real.py circular      # 14 809 troyanos, Júpiter circular          ~4 min
python run_real.py eliptico      # 14 809 troyanos, Júpiter elíptico          ~4 min
python run_verificacion.py       # pruebas independientes                     ~5 min
python figuras.py                # figuras en figs/ y números en resultados/resumen.json
```

## Contenido
| Archivo | Qué hace |
|---|---|
| `troyanos.py` | Modelo: constantes JPL, teoría lineal, péndulo promediado, PR3C rotante, primarios keplerianos, RK4 vectorizado |
| `run_barrido.py` | Período y forma de la libración frente a la amplitud |
| `run_real.py` | Integración 3D de los troyanos reales |
| `run_verificacion.py` | Jacobi, convergencia, dos integradores |
| `figuras.py` | Figuras y resumen numérico |
| `datos/` | CSV crudo del JPL SBDB (sin editar); se usa la época 2026-06-09 (14 809 de 16 453) |
| `informe/` | Informe en LaTeX (XeLaTeX) |
| `presentacion/` | Diapositivas Beamer y guion |

## Fuentes de datos
- JPL Small-Body Database Query, troyanos de Júpiter (clase TJN), elementos osculantes heliocéntricos, eclíptica J2000.
- Elementos de Júpiter: JPL Horizons, 2026-Jun-09 00:00 TDB (JD 2461200.5), centro: Sol.
