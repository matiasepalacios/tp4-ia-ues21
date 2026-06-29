"""
Transformada de Hough para RECTAS - implementacion desde cero.

No se utilizan bibliotecas de vision por computador (OpenCV u otras
"cajas negras"): toda la transformada se implementa con numpy. matplotlib
se usa unicamente para guardar las figuras de salida.

Parametrizacion polar de la recta:
    rho = x * cos(theta) + y * sin(theta)
Cada punto de borde (x, y) vota por todas las rectas (rho, theta) que pasan
por el. La acumulacion de votos en el plano (theta, rho) delata las rectas
presentes en la imagen.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Carpeta de salida: siempre junto a este script, sin importar el directorio
# desde donde se ejecute. Se crea automaticamente si no existe.
DIR_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salidas")


# --------------------------------------------------------------------------
# 1) Generacion del caso sencillo: una imagen binaria de bordes
# --------------------------------------------------------------------------
def trazar_recta(imagen, p0, p1):
    """Dibuja un segmento de recta (valor 1) entre p0 y p1 sobre la imagen."""
    (x0, y0), (x1, y1) = p0, p1
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)
    h, w = imagen.shape
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    imagen[ys[ok], xs[ok]] = 1
    return imagen


def generar_imagen_rectas(alto=120, ancho=120, ruido=0.02, semilla=7):
    """Imagen binaria con tres segmentos de recta mas ruido sal y pimienta."""
    rng = np.random.default_rng(semilla)
    img = np.zeros((alto, ancho), dtype=np.uint8)
    # Tres rectas con orientaciones distintas (bordes tipicos de un block).
    trazar_recta(img, (10, 10), (110, 10))    # horizontal
    trazar_recta(img, (20, 5), (20, 115))     # vertical
    trazar_recta(img, (5, 110), (110, 15))    # oblicua
    # Ruido: pixeles de borde espurios.
    n_ruido = int(ruido * alto * ancho)
    ys = rng.integers(0, alto, n_ruido)
    xs = rng.integers(0, ancho, n_ruido)
    img[ys, xs] = 1
    return img


# --------------------------------------------------------------------------
# 2) Nucleo de la transformada de Hough para rectas
# --------------------------------------------------------------------------
def hough_rectas(bordes, paso_theta=1.0):
    """
    Calcula el acumulador de Hough en el plano (theta, rho).

    Parametros
    ----------
    bordes : ndarray (H, W) binaria; los pixeles != 0 son puntos de borde.
    paso_theta : resolucion angular en grados.

    Devuelve
    --------
    acc      : ndarray (n_rho, n_theta) con los votos.
    thetas   : ndarray de angulos (radianes).
    rhos     : ndarray de distancias.
    """
    alto, ancho = bordes.shape
    thetas = np.deg2rad(np.arange(0.0, 180.0, paso_theta))
    diag = int(np.ceil(np.hypot(alto, ancho)))
    rhos = np.arange(-diag, diag + 1)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(bordes)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    # Para cada punto de borde se vota por todos los theta.
    for x, y in zip(xs, ys):
        rho = x * cos_t + y * sin_t                  # vector de rho por theta
        idx = np.round(rho).astype(int) + diag       # indice en el acumulador
        acc[idx, np.arange(len(thetas))] += 1
    return acc, thetas, rhos


def detectar_picos(acc, umbral, vecindad=10):
    """Devuelve (fila, columna) de los maximos locales que superan el umbral."""
    picos = []
    n_rho, n_theta = acc.shape
    for i in range(n_rho):
        for j in range(n_theta):
            v = acc[i, j]
            if v < umbral:
                continue
            i0, i1 = max(0, i - vecindad), min(n_rho, i + vecindad + 1)
            j0, j1 = max(0, j - vecindad), min(n_theta, j + vecindad + 1)
            if v == acc[i0:i1, j0:j1].max():
                picos.append((i, j))
    return picos


def suprimir_antipodales(picos, acc, thetas, rhos, tol_theta=3.0, tol_rho=6):
    """
    Elimina picos duplicados por la topologia cilindrica del plano de Hough:
    una recta cercana al borde angular (theta ~ 0) reaparece cerca de
    theta ~ 180 con rho de signo opuesto, ya que (rho, theta) y
    (-rho, theta + 180) describen la misma recta. Se conserva el de mas votos.
    """
    ordenados = sorted(picos, key=lambda p: acc[p[0], p[1]], reverse=True)
    conservados = []
    for (i, j) in ordenados:
        rho, th = rhos[i], thetas[j]
        pie = np.array([rho * np.cos(th), rho * np.sin(th)])  # pie de la perp.
        dup = any(np.hypot(*(pie - pc)) <= tol_rho for (_, _, pc) in conservados)
        if not dup:
            conservados.append((i, j, pie))
    return [(i, j) for (i, j, _) in conservados]


# --------------------------------------------------------------------------
# 3) Ejecucion del caso y generacion de figuras
# --------------------------------------------------------------------------
def main(dir_figuras=DIR_SALIDA):
    os.makedirs(dir_figuras, exist_ok=True)
    img = generar_imagen_rectas()
    acc, thetas, rhos = hough_rectas(img, paso_theta=1.0)

    umbral = int(0.5 * acc.max())
    picos = detectar_picos(acc, umbral=umbral, vecindad=12)
    picos = suprimir_antipodales(picos, acc, thetas, rhos)

    print("Imagen:", img.shape, "| puntos de borde:", int(img.sum()))
    print("Acumulador:", acc.shape, "| voto maximo:", int(acc.max()))
    print("Umbral de deteccion:", umbral)
    print("Rectas detectadas:", len(picos))
    for (i, j) in picos:
        rho, th = rhos[i], np.rad2deg(thetas[j])
        print(f"   rho = {rho:6.1f} px , theta = {th:6.1f} deg , votos = {acc[i, j]}")

    # --- Figura: escena de entrada ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="gray_r", origin="upper")
    ax.set_title("Imagen de bordes (entrada)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-rectas-escena.png", dpi=150)
    plt.close(fig)

    # --- Figura: acumulador de Hough ---
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ext = [np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    im = ax.imshow(acc, cmap="inferno", aspect="auto", extent=ext)
    for (i, j) in picos:
        ax.plot(np.rad2deg(thetas[j]), rhos[i], "c+", markersize=12, markeredgewidth=2)
    ax.set_title("Acumulador de Hough  (plano theta-rho)")
    ax.set_xlabel("theta (grados)"); ax.set_ylabel("rho (pixeles)")
    fig.colorbar(im, ax=ax, label="votos")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-rectas-acumulador.png", dpi=150)
    plt.close(fig)

    # --- Figura: rectas detectadas sobre la imagen ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="gray_r", origin="upper")
    alto, ancho = img.shape
    for (i, j) in picos:
        rho, th = rhos[i], thetas[j]
        c, s = np.cos(th), np.sin(th)
        if abs(s) > 1e-6:        # recta no vertical: y = (rho - x c) / s
            xx = np.array([0, ancho - 1])
            yy = (rho - xx * c) / s
        else:                    # recta vertical: x = rho / c
            yy = np.array([0, alto - 1])
            xx = np.full_like(yy, rho / c, dtype=float)
        ax.plot(xx, yy, "r-", linewidth=1.5)
    ax.set_xlim(0, ancho - 1); ax.set_ylim(alto - 1, 0)
    ax.set_title("Rectas detectadas")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-rectas-deteccion.png", dpi=150)
    plt.close(fig)

    print("Figuras guardadas en", dir_figuras)
    return picos


if __name__ == "__main__":
    main()
