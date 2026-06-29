# Transformada de Hough — rectas y circunferencias

Implementación **desde cero** de la transformada de Hough para detección de
rectas y de circunferencias de radio conocido, desarrollada para un trabajo
práctico de Inteligencia Artificial (detección de figuras en imágenes).

Toda la transformada (acumulador, votación y detección de picos) está
programada con `numpy`. **No se utilizan bibliotecas de visión por computador**
(OpenCV u otras "cajas negras"): el objetivo es implementar el algoritmo, no
invocarlo. `matplotlib` se usa únicamente para graficar los resultados.

## Contenido

```
hough/
├── hough_rectas.py            # Hough de rectas (acumulador theta-rho)
├── hough_circunferencias.py   # Hough de circunferencias de radio conocido
└── README.md                  # detalle de cada script
```

## Requisitos

- Python 3.8 o superior
- `numpy` y `matplotlib`

```bash
python3 -m pip install numpy matplotlib
```

## Ejecución

```bash
cd hough
python3 hough_rectas.py
python3 hough_circunferencias.py
```

Cada script imprime sus resultados por consola y guarda las figuras en una
carpeta `hough/salidas/`, que se crea automáticamente junto a los scripts
(funciona sin importar el directorio desde donde se ejecute).

## Resultados esperados

**Rectas** — detecta las 3 rectas presentes en una imagen con ruido:

```
rho =  20.0 px , theta =   0 deg , votos = 111
rho =  85.0 px , theta =  48 deg , votos = 105
rho =  10.0 px , theta =  90 deg , votos = 102
```

**Circunferencias** — localiza el centro del aro con radio conocido,
ignorando distractores y ruido:

```
Centro real      = (82, 60)
Centro detectado = (82, 60)   votos = 330
Error de localizacion = 0.00 px
```

## Cómo funciona (resumen)

La transformada de Hough es un método de **votación** en un espacio de
parámetros. Cada punto de borde vota por todas las figuras que podrían pasar
por él; la acumulación de votos delata las figuras reales presentes en la
imagen.

- **Rectas:** se usa la parametrización polar `rho = x·cos(theta) +
  y·sin(theta)` (evita el problema de las rectas verticales). Cada punto se
  transforma en una sinusoide en el plano `(theta, rho)`; las sinusoides de
  puntos alineados se cruzan en un punto, que son los parámetros de la recta.
- **Circunferencias:** con la ecuación `(x − x0)² + (y − y0)² = R²` y radio `R`
  conocido, el acumulador es 2D `(x0, y0)`. Cada punto vota por los centros
  posibles; el máximo del acumulador es el centro buscado.

## Licencia

MIT.
