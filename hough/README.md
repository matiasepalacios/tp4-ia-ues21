# Transformada de Hough — Prototipos

Implementación **desde cero** (solo `numpy`; `matplotlib` para graficar) de la
transformada de Hough para **rectas** y para **circunferencias de radio
conocido**. No se utilizan bibliotecas de visión por computador (OpenCV u otras
"cajas negras"), según lo exige la consigna.

## Archivos

- `hough_rectas.py` — Hough de rectas en parametrización polar
  `rho = x·cos(theta) + y·sin(theta)`. Genera una imagen binaria con tres
  segmentos y ruido, vota en el plano `(theta, rho)`, detecta los picos
  (máximos locales sobre umbral) y suprime duplicados antipodales del borde
  angular.
- `hough_circunferencias.py` — Hough de circunferencias con radio `R` conocido.
  Acumulador 2D `(xo, yo)`. Genera el aro "C" (dos bordes concéntricos), dos
  circunferencias distractoras y ruido; el máximo del acumulador es el centro
  "A" buscado.

## Requisitos

- Python 3.8+
- `numpy`, `matplotlib`

## Ejecución

```bash
python3 hough_rectas.py
python3 hough_circunferencias.py
```

Cada script imprime el resultado por consola y guarda las figuras en una
carpeta `salidas/` ubicada junto a los scripts. Esa ruta es independiente del
directorio desde donde se ejecute (se calcula a partir de la ubicacion del
archivo) y se crea automaticamente, por lo que los scripts funcionan
correctamente sin importar desde donde se los invoque.

## Salida esperada

Rectas: detecta las 3 rectas presentes
`(rho=20, theta=0°)`, `(rho=85, theta=48°)`, `(rho=10, theta=90°)`.

Circunferencias: detecta el centro `A = (82, 60)` con error de localización
de `0.00 px` respecto del centro real, ignorando distractores y ruido.
