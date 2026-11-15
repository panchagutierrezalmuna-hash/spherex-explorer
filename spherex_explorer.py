"""SPHEREx Motion Explorer: prototipo de consulta a GIF temporal.

Usa SIA2 (DR1 y QR3), servicio IBE de recortes y el modulo
`spherex_core.py` para leer FITS y recuperar WCS-WAVE.

ADVERTENCIA: visualizacion exploratoria. NO es deteccion fotometrica de
variabilidad ni se ha homogeneizado la PSF, fondo o respuesta espectral.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import time

import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from reproject import reproject_interp

from spherex_core import leer_recorte

COLECCIONES = ["spherex_dr1", "spherex_qr3"]
# Solo bits relevantes para una mascara instrumental preliminar.
# SOURCE (21) no se enmascara.
BITS_CRITICOS = (1, 2, 6, 9)  # OVERFLOW, SUR_ERROR, NONFUNC, MISSING_DATA


def _validar_entrada(ra: float, dec: float, fov_arcmin: float, detector: str):
    ra, dec, fov_arcmin = float(ra), float(dec), float(fov_arcmin)
    detector = str(detector).upper()
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA debe ser [0, 360) grados; Dec, [-90, 90] grados.")
    if not (1 <= fov_arcmin <= 12):
        raise ValueError("En este prototipo, el campo debe estar entre 1 y 12 arcmin.")
    if detector not in [f"D{i}" for i in range(1, 7)]:
        raise ValueError("Selecciona un detector desde D1 hasta D6.")
    return ra, dec, fov_arcmin, detector


def buscar_observaciones(ra: float, dec: float, detector: str = "D2", maxrec: int = 5000) -> pd.DataFrame:
    """Consulta imagenes que cubren el punto (ra, dec), no necesariamente todo el FOV."""
    ra, dec, _, detector = _validar_entrada(ra, dec, 6.0, detector)
    centro = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    # Un 502/503 de IRSA no implica que la región carezca de datos.
    for intento in range(2):
        try:
            tabla_vo = Irsa.query_sia(
                pos=(centro, 1 * u.arcsec),
                collection=COLECCIONES,
                maxrec=maxrec,
            )
            break
        except Exception as exc:
            mensaje = str(exc).lower()
            recuperable = any(token in mensaje for token in
                              ("502", "503", "504", "timeout", "timed out", "connection"))
            if intento == 1 or not recuperable:
                raise
            time.sleep(2.0)

    df = tabla_vo.to_pandas()
    if df.empty:
        return df
    for col in ("energy_bandpassname", "obs_collection", "obs_id", "access_url", "t_min", "s_ra", "s_dec"):
        if col not in df.columns:
            raise ValueError(f"La respuesta SIA2 no contiene la columna {col!r}.")
    df = df.loc[df["energy_bandpassname"] == f"SPHEREx-{detector}"].copy()
    if df.empty:
        return df
    for c in ("t_min", "s_ra", "s_dec"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t_min", "s_ra", "s_dec", "access_url"])
    # El mismo obs_id puede existir con detectores distintos.
    df = df.drop_duplicates(subset=["obs_collection", "obs_id", "energy_bandpassname", "access_url"])
    df = df.sort_values("t_min").reset_index(drop=True)
    if df.empty:
        return df
    df["fecha_utc"] = [str(x) for x in Time(df["t_min"].to_numpy(), format="mjd").isot]
    posiciones = SkyCoord(df["s_ra"].to_numpy() * u.deg,
                          df["s_dec"].to_numpy() * u.deg)
    df["distancia_centro_arcmin"] = posiciones.separation(centro).arcmin
    df.attrs["consulta_podria_estar_limitada"] = len(tabla_vo) >= maxrec
    return df


def preseleccionar_candidatas(catalogo: pd.DataFrame, max_por_ventana: int = 5,
                               max_ventanas: int = 5, separacion_ventanas_dias: float = 45) -> pd.DataFrame:
    """Preseleccion economica: ventanas temporales y proximidad del centro de imagen.

    No es una seleccion espectral definitiva; esa se hace DESPUES de leer FITS.
    """
    if catalogo.empty:
        return catalogo.copy()
    df = catalogo.sort_values("t_min").copy()
    df["ventana"] = (df["t_min"].diff().fillna(separacion_ventanas_dias + 1)
                     > separacion_ventanas_dias).cumsum()
    ventanas = sorted(df["ventana"].unique())
    if len(ventanas) > max_ventanas:
        indices = np.unique(np.linspace(0, len(ventanas) - 1, max_ventanas).round().astype(int))
        ventanas = [ventanas[i] for i in indices]
    partes = []
    for ventana in ventanas:
        grupo = df[df["ventana"] == ventana]
        partes.append(grupo.nsmallest(max_por_ventana, "distancia_centro_arcmin"))
    return pd.concat(partes, ignore_index=True).sort_values("t_min").reset_index(drop=True)


def _url_recorte(access_url: str, ra: float, dec: float, fov_arcmin: float) -> str:
    partes = urlsplit(str(access_url))
    if partes.scheme != "https" or parts_host_irsa(partes.netloc) is False:
        raise ValueError("URL de archivo no reconocida como dirección HTTPS de IRSA.")
    params = dict(parse_qsl(partes.query, keep_blank_values=True))
    params.update({"center": f"{ra:.8f},{dec:.8f}", "size": f"{fov_arcmin / 60:.8f}"})
    return urlunsplit(partes._replace(query=urlencode(params)))


def parts_host_irsa(host: str) -> bool:
    h = host.split(":")[0].lower()
    return h == "irsa.ipac.caltech.edu" or h.endswith(".ipac.caltech.edu")


def descargar_recorte(fila, ra: float, dec: float, fov_arcmin: float,
                       carpeta: str | Path = "datos_spherex", reintentos: int = 2) -> Path:
    """Descarga o recupera de cache un recorte valido, sin dejar archivos parciales."""
    url = _url_recorte(fila["access_url"], ra, dec, fov_arcmin)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    identidad = sha256(url.encode("utf-8")).hexdigest()[:14]
    nombre = f"{fila['obs_collection']}_{fila['obs_id']}_{fila['energy_bandpassname']}_{identidad}.fits"
    destino = carpeta / nombre
    if destino.exists():
        try:
            with fits.open(destino, memmap=False) as hdus:
                if all(x in hdus for x in ("IMAGE", "VARIANCE", "FLAGS", "WCS-WAVE")):
                    return destino
        except Exception:
            pass
        destino.unlink(missing_ok=True)

    temporal = destino.with_suffix(".part")
    ultimo_error = None
    for intento in range(reintentos + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 75)) as resp:
                resp.raise_for_status()
                with temporal.open("wb") as handle:
                    for bloque in resp.iter_content(chunk_size=256 * 1024):
                        if bloque:
                            handle.write(bloque)
            with fits.open(temporal, memmap=False) as hdus:
                for ext in ("IMAGE", "FLAGS", "VARIANCE", "WCS-WAVE"):
                    if ext not in hdus:
                        raise ValueError(f"La descarga carece de {ext}.")
            temporal.replace(destino)
            return destino
        except Exception as exc:
            ultimo_error = exc
            temporal.unlink(missing_ok=True)
            if intento < reintentos:
                time.sleep(2 * (intento + 1))
    raise RuntimeError(f"No fue posible descargar {fila['obs_id']}: {ultimo_error}")


def _wcs_salida(ra: float, dec: float, fov_arcmin: float):
    # Paso de ~6 segundos de arco por pixel (visualizacion, no escala nativa garantizada).
    n = max(21, int(round(fov_arcmin * 60 / 6)))
    if n % 2 == 0:
        n += 1
    w = WCS(naxis=2)
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-fov_arcmin / (60 * n), fov_arcmin / (60 * n)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w, (n, n)


def _proyectar(ruta: Path, ra: float, dec: float, wcs_salida: WCS, forma):
    imagen, wcs, flags, varianza, meta = leer_recorte(ruta, ra, dec)
    imagen_proj, footprint = reproject_interp((imagen, wcs), wcs_salida,
                                              shape_out=forma, order="bilinear")
    flags_u = np.asarray(flags, dtype=np.uint32)
    malos = np.zeros(flags_u.shape, dtype=bool)
    for bit in BITS_CRITICOS:
        malos |= (flags_u & np.uint32(1 << bit)) != 0
    numericamente_validos = (np.isfinite(imagen) & np.isfinite(varianza)
                            & (varianza > 0) & ~malos)
    reproy_mascara, _ = reproject_interp((numericamente_validos.astype(float), wcs),
                                         wcs_salida, shape_out=forma,
                                         order="nearest-neighbor")
    mascara = (footprint > 0.99) & (reproy_mascara > 0.5) & np.isfinite(imagen_proj)
    return imagen_proj, mascara, meta


def _seleccionar_secuencia(datos: list[dict], max_epocas: int = 5,
                            separacion_dias: float = 15,
                            max_fraccion_banda: float = 0.70,
                            cobertura_minima: float = 0.65):
    """Busca combinacion con mas epocas, condicionada por banda y cobertura.

    max_fraccion_banda y cobertura_minima son criterios exploratorios configurables.
    """
    if len(datos) < 2:
        raise ValueError("Se necesitan al menos dos recortes utilizables.")
    datos = sorted(datos, key=lambda d: d["t_min"])
    mejor = None
    for numero in range(2, min(max_epocas, len(datos)) + 1):
        for indices in combinations(range(len(datos)), numero):
            grupo = [datos[i] for i in indices]
            tiempos = [d["t_min"] for d in grupo]
            if any((b - a) < separacion_dias for a, b in zip(tiempos, tiempos[1:])):
                continue
            unidades = {d["meta"]["unidad"] for d in grupo}
            if len(unidades) != 1:
                continue
            lambdas = [d["meta"]["lambda_um"] for d in grupo]
            anchuras = [d["meta"]["ancho_um"] for d in grupo]
            if not np.all(np.isfinite(lambdas + anchuras)) or min(anchuras) <= 0:
                continue
            fraccion = (max(lambdas) - min(lambdas)) / np.mean(anchuras)
            if fraccion > max_fraccion_banda:
                continue
            comun = np.logical_and.reduce([d["mascara"] for d in grupo])
            cobertura = float(100 * comun.mean())
            if cobertura < 100 * cobertura_minima:
                continue
            # Prioridad: cantidad de epocas; cobertura comun; similitud espectral; lapso temporal.
            puntaje = (numero, cobertura, -fraccion, tiempos[-1] - tiempos[0])
            if mejor is None or puntaje > mejor[0]:
                mejor = (puntaje, grupo, comun, fraccion, cobertura)
    if mejor is None:
        raise ValueError("No hay dos epocas con compatibilidad y cobertura suficientes. "
                         "Prueba un campo menor, otro detector o reduce los umbrales exploratorios.")
    _, grupo, comun, fraccion, cobertura = mejor
    return grupo, comun, float(fraccion), cobertura


def _crear_gif(grupo: list[dict], comun: np.ndarray, wcs_salida: WCS,
                ruta_gif: str | Path, detector: str, fps: float = 1.0,
                objeto_movil: str | None = None):
    pixeles = np.concatenate([d["imagen"][comun] for d in grupo])
    pixeles = pixeles[np.isfinite(pixeles)]
    if len(pixeles) < 10:
        raise ValueError("Muy pocos pixeles utiles para representar el campo.")
    vmin, vmax = np.percentile(pixeles, [1, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Las intensidades no permiten establecer una escala compartida.")
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#555555")
    cuadros = [np.where(comun, d["imagen"], np.nan) for d in grupo]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=wcs_salida)
    im = ax.imshow(cuadros[0], origin="lower", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.coords[0].set_format_unit(u.deg)
    ax.coords[1].set_format_unit(u.deg)
    ax.coords[0].set_axislabel("Ascensión recta (RA)")
    ax.coords[1].set_axislabel("Declinación (Dec)")
    ax.coords.grid(color="white", alpha=0.2, linestyle=":")
    titulo = ax.set_title("")
    marcador = None
    if objeto_movil == "Barnard":
        from spherex_moviles import barnard_en
        primero = barnard_en(Time(grupo[0]["t_min"], format="mjd"))
        x, y = wcs_salida.world_to_pixel(primero)
        marcador = ax.scatter([x], [y], s=270, facecolors="none",
                             edgecolors="#60edff", linewidths=1.7, zorder=10)
        ax.text(.02, .03, "○ Posición prevista de Barnard · catálogo",
                color="#60edff", fontsize=9, transform=ax.transAxes,
                bbox={"facecolor": "#0b152b", "alpha": .75, "edgecolor": "none"})
    fig.colorbar(im, ax=ax, shrink=0.78, label=f"Intensidad ({grupo[0]['meta']['unidad']})")

    def actualizar(i):
        d = grupo[i]
        im.set_data(cuadros[i])
        if marcador is not None:
            estrella = barnard_en(Time(d["t_min"], format="mjd"))
            x, y = wcs_salida.world_to_pixel(estrella)
            marcador.set_offsets(np.array([[x, y]]))
        titulo.set_text(f"SPHEREx {detector} · {d['fecha_utc'][:10]} UTC\n"
                       f"λ ≈ {d['meta']['lambda_um']:.4f} µm")
        return im, titulo

    ani = FuncAnimation(fig, actualizar, frames=len(grupo),
                        interval=1000 / fps, blit=False, repeat=True)
    ruta_gif = Path(ruta_gif)
    ruta_gif.parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(str(ruta_gif), writer=PillowWriter(fps=fps), dpi=120)
    finally:
        plt.close(fig)
    return ruta_gif


def explorar_region(ra: float, dec: float, fov_arcmin: float = 6,
                    detector: str = "D2", max_epocas: int = 5,
                    max_por_ventana: int = 5,
                    separacion_dias: float = 15,
                    carpeta: str | Path = "datos_spherex",
                    salida: str | Path | None = None,
                    mostrar_progreso: bool = True,
                    objeto_movil: str | None = None) -> dict:
    """Ejecuta consulta SIA2, recortes FITS, compatibilidad y GIF RA/Dec.

    La salida contiene: archivo_gif, observaciones (DataFrame),
    catalogo_total, candidatas, procesadas, errores, cobertura_comun_pct,
    fraccion_ancho, rango_lambda_um, detector y advertencia.
    """
    ra, dec, fov_arcmin, detector = _validar_entrada(ra, dec, fov_arcmin, detector)
    if objeto_movil not in (None, "Barnard"):
        raise ValueError("El seguimiento móvil solo está habilitado para Barnard.")
    max_epocas = int(max_epocas)
    if not (2 <= max_epocas <= 12):
        raise ValueError("max_epocas debe estar entre 2 y 12.")
    catalogo = buscar_observaciones(ra, dec, detector)
    if catalogo.empty:
        raise ValueError(f"IRSA no devolvió observaciones {detector} en esa posición.")
    candidatas = preseleccionar_candidatas(catalogo, max_por_ventana=max_por_ventana)
    if mostrar_progreso:
        print(f"{len(catalogo)} observaciones {detector} encontradas; "
              f"se examinarán hasta {len(candidatas)} recortes.")
    wcs_salida, forma = _wcs_salida(ra, dec, fov_arcmin)
    datos, errores = [], []
    for _, fila in candidatas.iterrows():
        obs_id = fila["obs_id"]
        try:
            ruta = descargar_recorte(fila, ra, dec, fov_arcmin, carpeta=carpeta)
            imagen, mascara, meta = _proyectar(ruta, ra, dec, wcs_salida, forma)
            datos.append({
                "obs_id": obs_id,
                "fecha_utc": fila["fecha_utc"],
                "t_min": float(fila["t_min"]),
                "ruta": str(ruta),
                "imagen": imagen,
                "mascara": mascara,
                "meta": meta,
            })
            if mostrar_progreso:
                print(f"OK {obs_id} · λ={meta['lambda_um']:.4f} µm")
        except Exception as exc:
            errores.append({"obs_id": obs_id, "error": str(exc)})
            if mostrar_progreso:
                print(f"Sin procesar {obs_id}: {exc}")
    grupo, comun, frac, cobertura = _seleccionar_secuencia(
        datos, max_epocas=max_epocas, separacion_dias=separacion_dias
    )
    if salida is None:
        identificador = sha256(f"{ra:.8f}|{dec:.8f}|{fov_arcmin:.4f}|{detector}|{max_epocas}|{max_por_ventana}|{objeto_movil}".encode()).hexdigest()[:12]
        salida = Path("resultados_spherex") / f"blink_{detector}_{identificador}.gif"
    gif = _crear_gif(grupo, comun, wcs_salida, salida, detector,
                     objeto_movil=objeto_movil)
    # Las figuras son una ayuda visual opcional: un fallo en ellas no debe
    # descartar una animación científica ya generada correctamente.
    ruta_inicial = ruta_alineada = None
    error_figuras = None
    try:
        from spherex_figuras import crear_comparaciones
        ruta_inicial, ruta_alineada = crear_comparaciones(
            grupo, comun, wcs_salida, gif, detector,
            objeto_movil=objeto_movil
        )
    except Exception as exc:
        error_figuras = str(exc)
    tabla = pd.DataFrame([{
        "obs_id": d["obs_id"],
        "fecha_utc": d["fecha_utc"],
        "lambda_um": d["meta"]["lambda_um"],
        "ancho_um": d["meta"]["ancho_um"],
        "archivo": d["ruta"],
    } for d in grupo])
    return {
        "archivo_gif": str(gif),
        "archivo_comparacion_inicial": ruta_inicial,
        "archivo_comparacion_alineada": ruta_alineada,
        "error_figuras": error_figuras,
        "observaciones": tabla,
        "catalogo_total": len(catalogo),
        "candidatas": len(candidatas),
        "procesadas": len(datos),
        "errores": pd.DataFrame(errores),
        "cobertura_comun_pct": cobertura,
        "fraccion_ancho": frac,
        "rango_lambda_um": float(tabla["lambda_um"].max() - tabla["lambda_um"].min()),
        "detector": detector,
        "objeto_movil": objeto_movil,
        "advertencia": ("Las imágenes provienen de observaciones reales tomadas en distintas fechas. "
                    "Aunque elegimos bandas parecidas, las diferencias visuales por sí solas "
                    "no permiten confirmar cambios físicos en los objetos."),
    }
