"""SPHEREx Motion Explorer: comparación exploratoria de dos recortes FITS.

Basado en el tratamiento WCS del tutorial de IRSA/SPHEREx:
https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts/

Este módulo es una demostración visual, no un detector validado de variabilidad.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
import astropy.units as u
from reproject import reproject_interp


def leer_recorte(ruta, ra, dec):
    """Lee un FITS y calcula la banda espectral en (RA, Dec), en grados."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    with fits.open(ruta) as hdul:
        hdu = hdul["IMAGE"]
        imagen = np.asarray(hdu.data, dtype=float).copy()
        wcs = WCS(hdu.header).celestial
        x, y = wcs.world_to_pixel(coord)
        if not (0 <= x < imagen.shape[1] and 0 <= y < imagen.shape[0]):
            raise ValueError(f"{ruta.name}: las coordenadas no caen en el recorte.")

        # WCS-WAVE es alternativo y no debe aplicar la distorsión SIP espacial.
        wcs_espectral = WCS(hdu.header, fobj=hdul, key="W")
        wcs_espectral.sip = None
        longitud_onda, ancho_banda = wcs_espectral.pixel_to_world(x, y)
        metadatos = {
            "archivo": ruta.name,
            "lambda_um": float(longitud_onda.to_value(u.um)),
            "ancho_um": float(ancho_banda.to_value(u.um)),
            "x": float(x),
            "y": float(y),
            "unidad": str(hdu.header.get("BUNIT", "sin especificar")),
            "version": str(hdul[0].header.get("VERSION", "no indicada")),
        }
        # Conservar información para la validación de calidad posterior.
        flags = np.array(hdul["FLAGS"].data, copy=True)
        varianza = np.array(hdul["VARIANCE"].data, copy=True)
    return imagen, wcs, flags, varianza, metadatos


def crear_blink(ruta_1, ruta_2, ra, dec, salida="05_Analisis/M101_blink_automatico.gif"):
    """Alinea dos recortes FITS y genera un GIF de inspección visual.

    NO iguala las PSF, respuesta espectral, fondos ni calibraciones relativas.
    """
    im1, wcs1, flags1, var1, meta1 = leer_recorte(ruta_1, ra, dec)
    im2, wcs2, flags2, var2, meta2 = leer_recorte(ruta_2, ra, dec)

    if meta1["unidad"] != meta2["unidad"]:
        raise ValueError("Las imágenes tienen unidades BUNIT distintas; revisa la calibración.")

    alineada2, huella = reproject_interp(
        (im2, wcs2), wcs1, shape_out=im1.shape, order="bilinear"
    )
    comunes = (np.isfinite(im1) & np.isfinite(alineada2) & (huella > 0.99))
    if comunes.sum() < 10:
        raise ValueError("No hay suficientes píxeles válidos en la cobertura común.")

    # Ambas épocas usan el mismo rango visual, sin normalizar sus intensidades por separado.
    intensidades = np.concatenate([im1[comunes], alineada2[comunes]])
    vmin, vmax = np.percentile(intensidades, [1, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError("No se pudo establecer una escala de intensidad válida.")
    norma = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    mapa = plt.colormaps["magma"].copy()
    mapa.set_bad("black")
    cuadros = [
        np.ma.masked_where(~comunes, im1),
        np.ma.masked_where(~comunes, alineada2),
    ]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=wcs1)
    vista = ax.imshow(
        cuadros[0], origin="lower", cmap=mapa, norm=norma,
        interpolation="nearest"
    )
    ax.set_xlabel("Ascensión recta (RA)")
    ax.set_ylabel("Declinación (Dec)")
    titulo = ax.set_title("Época 1")

    def actualizar(frame):
        vista.set_data(cuadros[frame])
        titulo.set_text(f"Época {frame + 1} · λ = { [meta1, meta2][frame]['lambda_um']:.4f} μm")
        return vista, titulo

    animacion = FuncAnimation(fig, actualizar, frames=2, interval=1000, blit=False)
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    try:
        animacion.save(str(salida), writer=PillowWriter(fps=1), dpi=130)
    finally:
        plt.close(fig)

    diferencia_lam = abs(meta1["lambda_um"] - meta2["lambda_um"])
    informe = {
        "archivo_gif": str(salida),
        "cobertura_comun_pct": float(comunes.mean() * 100),
        "delta_lambda_um": float(diferencia_lam),
        "lambda_1_um": meta1["lambda_um"],
        "lambda_2_um": meta2["lambda_um"],
        "ancho_1_um": meta1["ancho_um"],
        "ancho_2_um": meta2["ancho_um"],
        "unidad": meta1["unidad"],
    }
    return informe
