"""
Transformada de Hough para CIRCUNFERENCIAS de radio conocido - desde cero.

Caso del problema: localizar el centro "A" del aro "C" del block.
Como el radio R es conocido, el espacio de parametros se reduce a un plano
2D (xo, yo). Cada punto de borde (x, y) vota por todos los centros posibles:
    xo = x - R * cos(phi)
    yo = y - R * sin(phi)   con phi en [0, 2*pi)
es decir, traza una circunferencia de radio R en el plano (xo, yo). La
interseccion de esas circunferencias -el maximo del acumulador- delata el
centro buscado "A".

No se utilizan bibliotecas de vision por computador.
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
# 1) Generacion del caso sencillo
# --------------------------------------------------------------------------
def trazar_circunferencia(imagen, centro, radio):
    """Dibuja el borde (valor 1) de una circunferencia sobre la imagen."""
    cx, cy = centro
    h, w = imagen.shape
    phi = np.linspace(0, 2 * np.pi, int(2 * np.pi * radio) + 1)
    xs = np.round(cx + radio * np.cos(phi)).astype(int)
    ys = np.round(cy + radio * np.sin(phi)).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    imagen[ys[ok], xs[ok]] = 1
    return imagen


def generar_imagen_aro(alto=140, ancho=140, centro_A=(82, 60),
                       r_ext=34, r_int=27, ruido=0.015, semilla=11):
    """
    Imagen binaria con el aro 'C' (dos bordes concentricos: interior y
    exterior), dos circunferencias menores que actuan de distractores y
    ruido sal y pimienta. El centro real del aro es 'centro_A'.
    """
    rng = np.random.default_rng(semilla)
    img = np.zeros((alto, ancho), dtype=np.uint8)
    trazar_circunferencia(img, centro_A, r_ext)     # borde exterior del aro
    trazar_circunferencia(img, centro_A, r_int)     # borde interior del aro
    # Distractores (otras circunferencias menores presentes en el block).
    trazar_circunferencia(img, (28, 30), 12)
    trazar_circunferencia(img, (35, 105), 9)
    # Ruido.
    n_ruido = int(ruido * alto * ancho)
    ys = rng.integers(0, alto, n_ruido)
    xs = rng.integers(0, ancho, n_ruido)
    img[ys, xs] = 1
    return img, centro_A, (r_ext, r_int)


# --------------------------------------------------------------------------
# 2) Nucleo de la transformada de Hough para circunferencias (R conocido)
# --------------------------------------------------------------------------
def hough_circunferencias(bordes, radios, paso_phi=2.0):
    """
    Acumulador 2D (yo, xo) para uno o varios radios conocidos.

    Parametros
    ----------
    bordes : ndarray (H, W) binaria; pixeles != 0 son puntos de borde.
    radios : lista de radios conocidos por los que votar (p. ej. los dos
             bordes del aro).
    paso_phi : resolucion angular en grados para barrer la circunferencia.

    Devuelve
    --------
    acc : ndarray (H, W) con los votos por cada centro posible (xo, yo).
    """
    alto, ancho = bordes.shape
    acc = np.zeros((alto, ancho), dtype=np.int32)
    phis = np.deg2rad(np.arange(0.0, 360.0, paso_phi))
    cos_p, sin_p = np.cos(phis), np.sin(phis)
    ys, xs = np.nonzero(bordes)
    for R in radios:
        dx = np.round(R * cos_p).astype(int)
        dy = np.round(R * sin_p).astype(int)
        for x, y in zip(xs, ys):
            xo = x - dx
            yo = y - dy
            ok = (xo >= 0) & (xo < ancho) & (yo >= 0) & (yo < alto)
            np.add.at(acc, (yo[ok], xo[ok]), 1)
    return acc


def centro_detectado(acc):
    """Devuelve (xo, yo) del maximo del acumulador = centro 'A'."""
    yo, xo = np.unravel_index(np.argmax(acc), acc.shape)
    return int(xo), int(yo), int(acc[yo, xo])


# --------------------------------------------------------------------------
# 3) Ejecucion del caso y generacion de figuras
# --------------------------------------------------------------------------
def main(dir_figuras=DIR_SALIDA):
    os.makedirs(dir_figuras, exist_ok=True)
    img, centro_real, (r_ext, r_int) = generar_imagen_aro()
    # El radio es conocido: se vota por ambos bordes del aro.
    acc = hough_circunferencias(img, radios=[r_ext, r_int], paso_phi=2.0)
    xo, yo, votos = centro_detectado(acc)

    print("Imagen:", img.shape, "| puntos de borde:", int(img.sum()))
    print("Radios conocidos del aro:", (r_ext, r_int))
    print("Centro real A   = ", centro_real)
    print("Centro detectado= ", (xo, yo), " votos =", votos)
    err = np.hypot(xo - centro_real[0], yo - centro_real[1])
    print(f"Error de localizacion = {err:.2f} px")

    # --- Figura: escena de entrada ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="gray_r", origin="upper")
    ax.set_title("Imagen de bordes: aro C + distractores + ruido")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-circ-escena.png", dpi=150)
    plt.close(fig)

    # --- Figura: acumulador 2D ---
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(acc, cmap="inferno", origin="upper")
    ax.plot(xo, yo, "c+", markersize=14, markeredgewidth=2)
    ax.set_title("Acumulador de Hough  (plano xo-yo)")
    ax.set_xlabel("xo"); ax.set_ylabel("yo")
    fig.colorbar(im, ax=ax, label="votos")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-circ-acumulador.png", dpi=150)
    plt.close(fig)

    # --- Figura: deteccion sobre la imagen ---
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(img, cmap="gray_r", origin="upper")
    th = np.linspace(0, 2 * np.pi, 200)
    for R in (r_ext, r_int):
        ax.plot(xo + R * np.cos(th), yo + R * np.sin(th), "r-", linewidth=1.3)
    ax.plot(xo, yo, "bx", markersize=12, markeredgewidth=2.5,
            label=f"A = ({xo}, {yo})")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Aro detectado y centro A")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(f"{dir_figuras}/fig-circ-deteccion.png", dpi=150)
    plt.close(fig)

    print("Figuras guardadas en", dir_figuras)
    return (xo, yo)


if __name__ == "__main__":
    main()
