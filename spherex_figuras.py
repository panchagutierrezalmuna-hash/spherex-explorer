"""Figuras adicionales a partir de las observaciones REALES escogidas por el motor.

La comparación inicial muestra los recortes FITS sin reproyectar; la comparación
alineada utiliza los datos y máscaras que ya seleccionó spherex_explorer.
No interpola ni reconstruye píxeles perdidos.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, ImageNormalize

from spherex_core import leer_recorte


def _rejilla(n):
    columnas = min(3, n)
    filas = int(np.ceil(n / columnas))
    return filas, columnas


def _normalizacion(datos, mascara=None):
    valores = np.asarray(datos, dtype=float)
    if mascara is not None:
        valores = valores[mascara]
    valores = valores[np.isfinite(valores)]
    if valores.size < 10:
        return None
    minimo, maximo = np.percentile(valores, [1, 99.5])
    if not np.isfinite(minimo) or not np.isfinite(maximo) or minimo >= maximo:
        return None
    return ImageNormalize(vmin=minimo, vmax=maximo, stretch=AsinhStretch())


def crear_comparaciones(grupo, comun, wcs_salida, ruta_gif, detector):
    """Genera PNGs para las mismas épocas del GIF. Retorna rutas de ambos."""
    n = len(grupo)
    filas, columnas = _rejilla(n)
    base = Path(ruta_gif)
    base.parent.mkdir(parents=True, exist_ok=True)
    ruta_inicial = base.with_name(base.stem + '_inicial.png')
    ruta_alineada = base.with_name(base.stem + '_alineada.png')

    # Imágenes originales: cada observación conserva su propia orientación WCS.
    fig = plt.figure(figsize=(4.8 * columnas, 4.35 * filas))
    mapa = plt.colormaps['magma'].copy()
    mapa.set_bad('#555555')
    for i, d in enumerate(grupo, 1):
        imagen, wcs, _, _, _ = leer_recorte(d['ruta'],
                                            wcs_salida.wcs.crval[0],
                                            wcs_salida.wcs.crval[1])
        ax = fig.add_subplot(filas, columnas, i, projection=wcs)
        norma = _normalizacion(imagen)
        ax.imshow(np.ma.masked_invalid(imagen), origin='lower', cmap=mapa,
                  norm=norma, interpolation='nearest')
        ax.set_title(f"{d['fecha_utc'][:10]} · λ {d['meta']['lambda_um']:.3f} µm", fontsize=11)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.coords.grid(color='white', alpha=.18, linestyle=':')
    fig.suptitle(f'SPHEREx {detector} · Imágenes originales', fontsize=15, y=.99)
    fig.subplots_adjust(hspace=.36, wspace=.28, top=.91, bottom=.10)
    try:
        fig.savefig(ruta_inicial, dpi=125, facecolor='#f7f7fb')
    finally:
        plt.close(fig)

    # Imágenes reproyectadas: escala y píxeles disponibles comunes en todas.
    valores = np.concatenate([d['imagen'][comun] for d in grupo])
    norma = _normalizacion(valores)
    fig = plt.figure(figsize=(4.8 * columnas, 4.35 * filas))
    for i, d in enumerate(grupo, 1):
        ax = fig.add_subplot(filas, columnas, i, projection=wcs_salida)
        imagen_alineada = np.ma.masked_where(~comun, d['imagen'])
        ax.imshow(imagen_alineada, origin='lower', cmap=mapa,
                  norm=norma, interpolation='nearest')
        ax.set_title(f"{d['fecha_utc'][:10]} · λ {d['meta']['lambda_um']:.3f} µm", fontsize=11)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.coords.grid(color='white', alpha=.18, linestyle=':')
    fig.suptitle(f'SPHEREx {detector} · Comparación alineada', fontsize=15, y=.99)
    fig.subplots_adjust(hspace=.36, wspace=.28, top=.91, bottom=.10)
    try:
        fig.savefig(ruta_alineada, dpi=125, facecolor='#f7f7fb')
    finally:
        plt.close(fig)

    return str(ruta_inicial), str(ruta_alineada)
