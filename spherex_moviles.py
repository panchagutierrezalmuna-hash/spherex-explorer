"""Posiciones previstas de objetos móviles (NO detecciones medidas).

Barnard: propagación del movimiento propio J2000 de SIMBAD hasta la fecha
requerida. Asteroides: IRSA MOST consulta efemérides JPL y exposiciones SPHEREx.
"""
from __future__ import annotations

import io
from urllib.parse import urlsplit

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from astroquery.ipac.irsa.most import Most
from matplotlib import pyplot as plt


# SIMBAD: ICRS/J2000, proper motions in mas/yr (pm_ra includes cos(dec)).
# https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Barnard%27s+Star
BARNARD_J2000 = SkyCoord(
    ra="17h57m48.4984700685s",
    dec="+04d41m36.113879676s",
    frame="icrs",
    obstime=Time("J2000"),
    pm_ra_cosdec=-801.551 * u.mas / u.yr,
    pm_dec=10362.394 * u.mas / u.yr,
    distance=Distance(parallax=546.9759 * u.mas),
    radial_velocity=-110.11 * u.km / u.s,
)


def barnard_en(fecha: str | Time) -> SkyCoord:
    """Posición astrométrica propagada a la época, no observación medida."""
    momento = fecha if isinstance(fecha, Time) else Time(fecha)
    return BARNARD_J2000.apply_space_motion(new_obstime=momento)


def consultar_asteroide(naif_id: str = "20000004", inicio: str = "2025-08-01",
                       fin: str = "2025-08-31") -> pd.DataFrame:
    """Exposiciones SPHEREx que MOST relaciona con la efeméride prevista."""
    datos = Most.query_object(
        output_mode="Regular", obj_name=str(naif_id),
        obs_begin=str(inicio), obs_end=str(fin), catalog="spherex",
    )
    resultado = datos.get("results")
    if resultado is None or len(resultado) == 0:
        return pd.DataFrame(columns=["fecha", "mjd", "ra_deg", "dec_deg", "imagen", "imagen_id"])
    tabla = resultado.to_pandas()
    columnas = {col.lower(): col for col in tabla.columns}
    obligatorias = ("ra_obj", "dec_obj", "image_url")
    faltantes = [c for c in obligatorias if c not in columnas]
    if faltantes:
        raise ValueError(f"MOST no devolvió las columnas esperadas: {', '.join(faltantes)}")
    def texto(valor):
        if isinstance(valor, (bytes, bytearray)):
            return valor.decode("utf-8", errors="replace")
        return str(valor)

    fecha_col = columnas.get("date_obs")
    tiempo_col = columnas.get("time_obs")
    fecha = tabla[fecha_col].map(texto) if fecha_col else pd.Series([""]*len(tabla))
    if tiempo_col:
        fecha = fecha.str.cat(tabla[tiempo_col].map(texto), sep=" ")
    mjd_col = columnas.get("mjd_obs")
    id_col = columnas.get("image_id")
    df = pd.DataFrame({
        "fecha": fecha,
        "mjd": pd.to_numeric(tabla[mjd_col], errors="coerce") if mjd_col else np.nan,
        "ra_deg": pd.to_numeric(tabla[columnas["ra_obj"]], errors="coerce"),
        "dec_deg": pd.to_numeric(tabla[columnas["dec_obj"]], errors="coerce"),
        "imagen": tabla[columnas["image_url"]].map(texto),
        "imagen_id": tabla[id_col].map(texto) if id_col else "sin ID",
    })
    return df.dropna(subset=["ra_deg", "dec_deg"]).sort_values(
        ["mjd", "fecha"], na_position="last").reset_index(drop=True)


def figura_posiciones(df: pd.DataFrame):
    """Puntos de las posiciones previstas. No interpola trayectoria orbital."""
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    secuencia = np.arange(len(df))
    p = ax.scatter(df["ra_deg"], df["dec_deg"], c=secuencia,
                   cmap="plasma", s=85, zorder=3, edgecolors="white", linewidths=.5)
    ax.scatter(df["ra_deg"].iloc[0], df["dec_deg"].iloc[0],
               marker="*", s=230, c="cyan", zorder=4, label="Primera observación")
    ax.set_xlabel("Ascensión recta RA (grados)")
    ax.set_ylabel("Declinación Dec (grados)")
    ax.set_title("Posiciones previstas en las exposiciones SPHEREx")
    ax.grid(alpha=.25, linestyle=":")
    ax.invert_xaxis()  # Convención habitual de imágenes astronómicas.
    fig.colorbar(p, ax=ax, label="Orden cronológico de las exposiciones")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def vista_recorte_asteroide(fila: dict, fov_arcmin: float = 4):
    """Muestra un recorte FITS REAL y un marcador efemérides; no detecta el objeto.

    Descarga controlada (max 25 MB) desde IRSA, para evitar agotar memoria de Streamlit.
    """
    # Esta función no presupone que todas las URLs de MOST tengan el mismo esquema.
    from spherex_explorer import _url_recorte
    url = _url_recorte(str(fila["imagen"]), float(fila["ra_deg"]),
                       float(fila["dec_deg"]), float(fov_arcmin))
    partes = urlsplit(url)
    if partes.scheme != "https" or not partes.netloc.endswith("ipac.caltech.edu"):
        raise ValueError("MOST devolvió una URL que no es de IRSA; no la descargamos.")
    buf = io.BytesIO()
    total = 0
    with requests.get(url, stream=True, timeout=(15, 80)) as r:
        r.raise_for_status()
        for bloque in r.iter_content(1024 * 256):
            if bloque:
                total += len(bloque)
                if total > 25 * 1024 * 1024:
                    raise ValueError("El recorte supera el límite de tamaño para la web.")
                buf.write(bloque)
    buf.seek(0)
    with fits.open(buf, memmap=False) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else hdul[1]
        imagen = np.asarray(hdu.data, dtype=float)
        wcs = WCS(hdu.header).celestial
    if imagen.ndim != 2:
        raise ValueError("El recorte no es una imagen bidimensional.")
    validos = imagen[np.isfinite(imagen)]
    if validos.size < 10:
        raise ValueError("El recorte no contiene suficientes píxeles válidos.")
    vmin, vmax = np.percentile(validos, [1, 99.5])
    if vmin >= vmax:
        raise ValueError("El recorte no presenta contraste suficiente.")
    pos = SkyCoord(ra=float(fila["ra_deg"]) * u.deg,
                   dec=float(fila["dec_deg"]) * u.deg)
    px, py = wcs.world_to_pixel(pos)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection=wcs)
    mapa = plt.colormaps["magma"].copy()
    mapa.set_bad("#555555")
    ax.imshow(np.ma.masked_invalid(imagen), origin="lower", cmap=mapa,
              norm=ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
              interpolation="nearest")
    ax.scatter([px], [py], s=250, linewidths=1.5, edgecolors="#71f0f3",
               facecolors="none", label="Posición prevista (MOST)")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("SPHEREx · recorte del asteroide (predicción)")
    fig.tight_layout()
    return fig
