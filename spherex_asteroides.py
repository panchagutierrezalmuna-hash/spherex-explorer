"""Seguimiento EXPLORATORIO con recortes FITS reales de SPHEREx.

MOST encuentra exposiciones que contienen la posicion prevista del objeto.
Las imagenes se reproyectan alrededor de ESA posicion (centro movil), por lo
que el fondo se desplaza entre cuadros. No es una deteccion automatica.

Referencias:
https://caltech-ipac.github.io/irsa-tutorials/most-queries/
https://irsa.ipac.caltech.edu/spherex-archive-documentation/spherex-data-access/
"""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from astroquery.ipac.irsa.most import Most
from matplotlib import pyplot as plt
from PIL import Image
from reproject import reproject_interp

OBJETOS = {
    "(4) Vesta": "20000004",
    "(1) Ceres": "20000001",
    "(2) Pallas": "20000002",
}


def _texto(v):
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


def _detector_de_url(url: str) -> str | None:
    # Ejemplo IRSA SPHEREx: level2_2025W32_1A_0105_1D6_spx_...fits
    nombre = PurePosixPath(urlsplit(url).path).name
    m = re.search(r"\dD([1-6])(?=_|\.fits|$)", nombre, re.IGNORECASE)
    return f"D{m.group(1)}" if m else None


def buscar_imagenes(identificador: str, desde: str, hasta: str) -> pd.DataFrame:
    """Consulta MOST para exposiciones SPHEREx de un objeto solar-sistemico."""
    if identificador not in OBJETOS.values():
        raise ValueError("Identificador no disponible en esta demostracion.")
    if desde > hasta:
        raise ValueError("La fecha inicial debe ser anterior a la final.")
    out = Most.query_object(
        output_mode="Regular", obj_name=identificador,
        obs_begin=desde, obs_end=hasta, catalog="spherex",
    )
    tabla = out.get("results")
    columnas_salida = ["fecha", "mjd", "ra_deg", "dec_deg", "imagen", "imagen_id", "detector"]
    if tabla is None or len(tabla) == 0:
        return pd.DataFrame(columns=columnas_salida)
    df = tabla.to_pandas()
    cols = {str(c).lower(): c for c in df.columns}
    obligatorias = ["ra_obj", "dec_obj", "image_url"]
    if any(k not in cols for k in obligatorias):
        raise ValueError("MOST cambio el formato de la tabla: faltan RA, Dec o image_url.")
    def col(name, valor=""):
        return df[cols[name]].map(_texto) if name in cols else pd.Series([valor] * len(df))
    fecha = col("date_obs")
    if "time_obs" in cols:
        fecha = fecha.str.cat(col("time_obs"), sep=" ").str.strip()
    fechas_mjd = (pd.to_numeric(df[cols["mjd_obs"]], errors="coerce")
                  if "mjd_obs" in cols else pd.Series([np.nan] * len(df)))
    links = col("image_url")
    resultado = pd.DataFrame({
        "fecha": fecha,
        "mjd": fechas_mjd,
        "ra_deg": pd.to_numeric(df[cols["ra_obj"]], errors="coerce"),
        "dec_deg": pd.to_numeric(df[cols["dec_obj"]], errors="coerce"),
        "imagen": links,
        "imagen_id": col("image_id", "sin ID"),
        "detector": links.map(_detector_de_url),
    })
    resultado = (resultado.replace([np.inf, -np.inf], np.nan)
                 .dropna(subset=["ra_deg", "dec_deg"])
                 .drop_duplicates(subset=["imagen", "mjd", "ra_deg", "dec_deg"])
                 .sort_values(["mjd", "fecha"], na_position="last")
                 .reset_index(drop=True))
    return resultado


def _url_cutout(url: str, ra: float, dec: float, campo_arcmin: float) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host != "irsa.ipac.caltech.edu":
        raise ValueError("URL de imagen no reconocida como el archivo HTTPS de IRSA.")
    if not (2 <= campo_arcmin <= 8):
        raise ValueError("Campo fuera del intervalo permitido de 2 a 8 minutos.")
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs.update(center=f"{ra:.8f},{dec:.8f}", size=f"{campo_arcmin/60:.7f}")
    return urlunsplit(parsed._replace(query=urlencode(qs)))


def _descargar_fits(url: str, max_mb: int = 32) -> BytesIO:
    if urlsplit(url).hostname != "irsa.ipac.caltech.edu":
        raise ValueError("La descarga no apunta a IRSA.")
    datos = BytesIO()
    tam = 0
    with requests.get(url, stream=True, timeout=(12, 70)) as res:
        res.raise_for_status()
        for bloque in res.iter_content(chunk_size=256 * 1024):
            if not bloque:
                continue
            tam += len(bloque)
            if tam > max_mb * 1024 * 1024:
                raise ValueError(f"El recorte supera {max_mb} MB. Prueba un campo menor.")
            datos.write(bloque)
    datos.seek(0)
    return datos


def _wcs_centrado(ra: float, dec: float, campo_arcmin: float):
    # Escala aproximada para visualizacion; WCS celestial norte arriba.
    n = max(23, int(round(campo_arcmin * 60 / 6.2)))
    n += (n + 1) % 2  # impar
    w = WCS(naxis=2)
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-campo_arcmin / (60 * n), campo_arcmin / (60 * n)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w, (n, n)


def _leer_proyectar(fila: dict, campo_arcmin: float) -> dict:
    ra, dec = float(fila["ra_deg"]), float(fila["dec_deg"])
    url = _url_cutout(str(fila["imagen"]), ra, dec, campo_arcmin)
    archivo = _descargar_fits(url)
    with fits.open(archivo, memmap=False) as h:
        if "IMAGE" in h:
            hdu = h["IMAGE"]
        else:
            hdu = next((x for x in h[1:] if x.data is not None
                        and np.ndim(x.data) == 2), None)
            if hdu is None:
                raise ValueError("El FITS no tiene una imagen 2D.")
        img = np.asarray(hdu.data, dtype=float).copy()
        wcs_in = WCS(hdu.header).celestial
        unidad = str(hdu.header.get("BUNIT", "no indicada"))
        coord = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
        x, y = wcs_in.world_to_pixel(coord)
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            raise ValueError("La posicion prevista no aparece dentro del recorte.")
        # Metadatos espectrales opcionales: no presentarlos si no se pueden validar.
        lon, banda = np.nan, np.nan
        try:
            w_w = WCS(hdu.header, fobj=h, key="W")
            w_w.sip = None
            l, b = w_w.pixel_to_world(x, y)
            lon, banda = float(l.to_value(u.um)), float(b.to_value(u.um))
        except Exception:
            pass
        valid = np.isfinite(img)
        if "VARIANCE" in h:
            var = np.asarray(h["VARIANCE"].data, dtype=float)
            if var.shape == img.shape:
                valid &= np.isfinite(var) & (var > 0)
        if "FLAGS" in h:
            flags = np.asarray(h["FLAGS"].data)
            if flags.shape == img.shape:
                bits = np.zeros_like(valid)
                fu = flags.astype(np.uint32)
                for bit in (1, 2, 6, 9):
                    bits |= (fu & np.uint32(1 << bit)) != 0
                valid &= ~bits
    destino, forma = _wcs_centrado(ra, dec, campo_arcmin)
    proyectada, huella = reproject_interp((img, wcs_in), destino,
                                          shape_out=forma, order="bilinear")
    m, _ = reproject_interp((valid.astype(float), wcs_in), destino,
                             shape_out=forma, order="nearest-neighbor")
    buena = np.isfinite(proyectada) & (huella > .99) & (m > .5)
    if np.count_nonzero(buena) < 30:
        raise ValueError("El recorte no tiene suficiente cobertura util.")
    return dict(img=np.where(buena, proyectada, np.nan), wcs=destino,
                lambda_um=lon, ancho_um=banda, unidad=unidad,
                fecha=str(fila["fecha"]), detector=fila.get("detector") or "—",
                id=str(fila["imagen_id"]), ra=ra, dec=dec,
                cobertura=float(np.mean(buena) * 100))


def _seleccionar_cercanas(df: pd.DataFrame, detector: str, max_imagenes: int,
                           intervalo_horas: float = 36.) -> pd.DataFrame:
    """Elige bloque de exposiciones cercanas en el tiempo del mismo detector."""
    if detector != "Automático":
        df = df[df.detector == detector]
    else:
        df = df[df.detector.notna()]
    if df.empty:
        return df
    candidatos = []
    for det, grupo in df.groupby("detector", dropna=True):
        grupo = grupo.sort_values("mjd", na_position="last").reset_index(drop=True)
        if grupo["mjd"].notna().all():
            for j in range(len(grupo)):
                bloque = grupo[(grupo["mjd"] >= grupo.iloc[j]["mjd"])
                               & (grupo["mjd"] <= grupo.iloc[j]["mjd"] + intervalo_horas / 24)]
                candidatos.append(bloque.head(max_imagenes))
        else:
            candidatos.append(grupo.head(max_imagenes))
    if not candidatos:
        return df.iloc[0:0]
    mejor = max(candidatos, key=lambda g: (len(g),
                                           -float(g["mjd"].max()-g["mjd"].min())
                                           if g["mjd"].notna().all() else 0))
    return mejor.reset_index(drop=True)


def _png_fotograma(recorte: dict, normalizacion, cmap, nombre: str, numero: int, total: int) -> bytes:
    fig = plt.figure(figsize=(5.8, 5.5))
    try:
        ax = fig.add_subplot(111, projection=recorte["wcs"])
        ax.imshow(np.ma.masked_invalid(recorte["img"]), origin="lower",
                  cmap=cmap, norm=normalizacion, interpolation="nearest")
        # Centro del recorte: posicion CALCULADA por MOST (no fotometria).
        px, py = recorte["wcs"].world_to_pixel(
            SkyCoord(recorte["ra"]*u.deg, recorte["dec"]*u.deg))
        ax.scatter([px], [py], s=190, edgecolors="#7df9ee", facecolors="none",
                   linewidths=1.4)
        ax.set_xlabel("Ascensión recta (RA)")
        ax.set_ylabel("Declinación (Dec)")
        l = recorte["lambda_um"]
        lam = f" · λ {l:.3f} μm" if np.isfinite(l) else ""
        ax.set_title(f"{nombre} · SPHEREx {recorte['detector']}\n"
                     f"{recorte['fecha']}{lam} · {numero}/{total}", fontsize=10)
        fig.tight_layout()
        out = BytesIO()
        fig.savefig(out, format="png", dpi=115, facecolor="#f7f8fc")
        return out.getvalue()
    finally:
        plt.close(fig)


def _gif(pngs: list[bytes], fps: float = 1.0) -> bytes:
    imagenes = [Image.open(BytesIO(b)).convert("RGB") for b in pngs]
    with Image.new("RGB", (imagenes[0].width, imagenes[0].height * len(imagenes))) as mosaico:
        for k, imagen in enumerate(imagenes):
            mosaico.paste(imagen, (0, k * imagenes[0].height))
        paleta = mosaico.quantize(256)
        cuadros = [im.quantize(palette=paleta, dither=Image.Dither.NONE) for im in imagenes]
        salida = BytesIO()
        cuadros[0].save(salida, format="GIF", save_all=True,
                        append_images=cuadros[1:], duration=int(1000 / fps),
                        loop=0, optimize=False)
    return salida.getvalue()


def procesar_imagenes(df: pd.DataFrame, objeto: str = "(4) Vesta",
                      detector: str = "Automático", max_imagenes: int = 4,
                      campo_arcmin: float = 5.) -> dict:
    """Descarga exposiciones reales y genera GIF de seguimiento centrado en efemerides."""
    if df is None or df.empty:
        raise ValueError("No hay exposiciones para procesar.")
    if detector != "Automático" and detector not in {f"D{i}" for i in range(1, 7)}:
        raise ValueError("Detector desconocido.")
    if not 2 <= int(max_imagenes) <= 6:
        raise ValueError("Cantidad de exposiciones fuera de intervalo.")
    seleccion = _seleccionar_cercanas(df, detector, int(max_imagenes))
    if seleccion.empty:
        raise ValueError("No hay exposiciones identificadas para ese detector.")
    resultados, fallas = [], []
    for _, fila in seleccion.iterrows():
        try:
            resultados.append(_leer_proyectar(fila.to_dict(), campo_arcmin))
        except Exception as exc:
            fallas.append(f"{fila['imagen_id']}: {type(exc).__name__}: {exc}")
    if not resultados:
        raise RuntimeError("No se pudieron descargar recortes válidos. " + "; ".join(fallas[:2]))
    valores = np.concatenate([r["img"][np.isfinite(r["img"])] for r in resultados])
    lo, hi = np.percentile(valores, [1, 99.5])
    if not np.isfinite(lo + hi) or hi <= lo:
        raise ValueError("Las imagenes no tienen rango de intensidad valido.")
    norma = ImageNormalize(vmin=lo, vmax=hi, stretch=AsinhStretch())
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#535b6d")
    pngs = [_png_fotograma(r, norma, cmap, objeto, i+1, len(resultados))
            for i, r in enumerate(resultados)]
    detalle = pd.DataFrame([{
        "Fecha UTC": r["fecha"], "Detector": r["detector"],
        "λ (µm)": round(r["lambda_um"], 4) if np.isfinite(r["lambda_um"]) else np.nan,
        "Ancho de banda (µm)": round(r["ancho_um"], 4) if np.isfinite(r["ancho_um"]) else np.nan,
        "RA prevista (°)": round(r["ra"], 6),
        "Dec prevista (°)": round(r["dec"], 6),
        "Cobertura (%)": round(r["cobertura"], 1),
        "Imagen": r["id"],
    } for r in resultados])
    band_valid = detail_l = detail_w = None
    lam = np.array([r["lambda_um"] for r in resultados], dtype=float)
    wide = np.array([r["ancho_um"] for r in resultados], dtype=float)
    if len(resultados) >= 2 and np.all(np.isfinite(lam)) and np.all(np.isfinite(wide)) and np.nanmean(wide) > 0:
        band_valid = (np.max(lam)-np.min(lam))/np.mean(wide) <= .7
        detail_l = float(np.ptp(lam))
        detail_w = float(np.mean(wide))
    return {
        "frames_png": pngs,
        "gif_bytes": _gif(pngs) if len(pngs) >= 2 else None,
        "tabla": detalle,
        "errores": fallas,
        "n_encontradas": len(df),
        "n_seleccionadas": len(seleccion),
        "n_descargadas": len(resultados),
        "banda_comparable": band_valid,
        "delta_lambda_um": detail_l,
        "ancho_medio_um": detail_w,
    }
