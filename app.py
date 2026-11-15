"""SPHEREx Motion Explorer — interfaz Streamlit + motor científico del equipo.

La búsqueda libre usa exclusivamente spherex_explorer.explorar_region.
Las demos precalculadas, si existen, se leen de archivos reales en demos/
(no se generan datos sintéticos ni se hacen pasar por consultas en vivo).
"""
from __future__ import annotations

from pathlib import Path
import threading

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

from spherex_explorer import explorar_region

st.set_page_config(page_title="SPHEREx Motion Explorer", page_icon="🔭", layout="centered")

st.markdown("""
<style>
.stApp { background: radial-gradient(ellipse at top, #121a3a 0%, #05070f 60%); color: #e8ecff; }
h1, h2, h3 { color: #ffffff; letter-spacing: .3px; }
.hero { padding: 1.2rem 0 .4rem 0; }
.hero h1 { font-size: 2.1rem; margin-bottom: .1rem; }
.hero p { color: #9fb0ff; margin-top: 0; }
.real-badge { background:#052e22; border:1px solid #22c993; color:#9ff0d2;
              padding:.55rem .8rem; border-radius:10px; font-size:.9rem; margin:.4rem 0 1rem 0; }
.card { background: rgba(255,255,255,.04); border:1px solid rgba(140,160,255,.25);
        border-radius:14px; padding:.9rem 1rem; margin:.4rem 0; }
.muted { color:#9fb0ff; font-size:.9rem; }
div.stButton > button { background:#5b6dff; color:white; border:0; border-radius:10px;
                         padding:.6rem 1rem; font-weight:600; width:100%; }
div.stButton > button:hover { background:#7a89ff; color:white; }
</style>
""", unsafe_allow_html=True)

# Coordenadas de referencia utilizadas en los notebooks del equipo.
# Barnard: posición de referencia, NO corregida automáticamente por movimiento propio
# en el buscador general. Para mostrar movimiento, usar su GIF validado en demos/.
EJEMPLOS = {
    "M101 (galaxia espiral)": ("M101", 210.80227, 54.34895,
                             "Campo de prueba: galaxia espiral; comparación de distintas épocas."),
    "Estrella de Barnard": ("Barnard", 269.452075, 4.693391,
                            "Movimiento propio alto. El buscador general NO sigue su movimiento; utiliza la demo validada para ello."),
    "R Draconis (estrella variable)": ("R_Draconis", 248.16792, 66.75500,
                                       "Estrella variable. Coordenadas aproximadas: verificar contra el catálogo del equipo."),
}

# Solo se mostrarán demos que el equipo suba efectivamente; se evita representar
# gráficos inventados como si fueran resultados observacionales.
DEMO_ARCHIVOS = {
    "M101": ["demos/M101.gif", "05_Analisis/M101_blink.gif"],
    "Barnard": ["demos/Barnard.gif", "05_Analisis/Barnard_blink.gif"],
    "R_Draconis": ["demos/R_Draconis.gif", "05_Analisis/R_Draconis_blink.gif"],
}
RAIZ = Path(__file__).resolve().parent
CUSTOM = "— Coordenadas propias —"


def _cambiar_ejemplo():
    ex = EJEMPLOS.get(st.session_state.ejemplo)
    if ex:
        st.session_state.ra = ex[1]
        st.session_state.dec = ex[2]
        st.session_state.resultado = None


def _coordenadas_manuales():
    # Si se editan RA/Dec a mano, no mantener seleccionado un ejemplo anterior.
    st.session_state.ejemplo = CUSTOM


def _archivo_demo(nombre):
    for rel in DEMO_ARCHIVOS.get(nombre, []):
        p = RAIZ / rel
        if p.is_file() and p.stat().st_size:
            return p
    return None


@st.cache_resource

def _bloqueo_graficos():
    return threading.Lock()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def _consultar(ra, dec, fov, detector, max_epocas):
    # Matplotlib pyplot comparte estado global: serializamos la generación de GIFs.
    with _bloqueo_graficos():
        return explorar_region(
            ra=ra, dec=dec, fov_arcmin=fov,
            detector=detector, max_epocas=max_epocas,
            max_por_ventana=3, separacion_dias=15,
            mostrar_progreso=False,
        )


st.session_state.setdefault("ra", 210.80227)
st.session_state.setdefault("dec", 54.34895)
st.session_state.setdefault("fov", 4)
st.session_state.setdefault("detector", "D2")
st.session_state.setdefault("ejemplo", "M101 (galaxia espiral)")
st.session_state.setdefault("resultado", None)

st.markdown(
    '<div class="hero"><h1>🔭 SPHEREx Motion Explorer</h1>'
    '<p>Mira cómo cambia el cielo infrarrojo a lo largo del tiempo.</p></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="real-badge">🛰️ <b>Datos reales de NASA/IPAC IRSA.</b> '
    'Comparación visual exploratoria; los cambios observados no son por sí solos '
    'detecciones físicas confirmadas.</div>', unsafe_allow_html=True,
)

with st.expander("¿Qué hace esta herramienta?"):
    st.write(
        "Consulta observaciones públicas de SPHEREx en la región indicada, "
        "preselecciona recortes, compara sus bandas infrarrojas y su cobertura, "
        "alinea las imágenes mediante WCS y crea una animación temporal con ejes RA/Dec. "
        "El procesamiento no realiza una calibración fotométrica completa."
    )

st.header("1 · Buscar en el cielo")
st.selectbox(
    "Objeto de ejemplo", [CUSTOM, *EJEMPLOS.keys()],
    key="ejemplo", on_change=_cambiar_ejemplo,
    help="También puedes introducir libremente otras coordenadas RA/Dec.",
)
ex = EJEMPLOS.get(st.session_state.ejemplo)
if ex:
    st.caption(ex[3])

c1, c2 = st.columns(2)
c1.number_input("Ascensión recta RA (°)", min_value=0.0, max_value=359.99999,
                key="ra", format="%.5f", step=0.01, on_change=_coordenadas_manuales)
c2.number_input("Declinación Dec (°)", min_value=-90.0, max_value=90.0,
                key="dec", format="%.5f", step=0.01, on_change=_coordenadas_manuales)
c3, c4 = st.columns(2)
c3.select_slider("Campo de visión (arcmin)", options=[2, 3, 4, 5, 6, 8, 10, 12], key="fov")
c4.selectbox("Detector", [f"D{i}" for i in range(1, 7)], key="detector")
max_epocas = st.slider("Máximo de épocas (si existen compatibles)", 2, 5, 5)

b1, b2 = st.columns(2)
consulta = b1.button("🔍 Consultar IRSA y generar GIF")
demo_p = _archivo_demo(ex[0]) if ex else None
ver_demo = b2.button("▶ Ver demostración preparada", disabled=(demo_p is None))
if ex and demo_p is None:
    st.caption("Para habilitar esta demostración sin conexión, coloca el GIF real del equipo en demos/.")

if consulta:
    try:
        with st.spinner("Consultando IRSA, seleccionando observaciones y procesando recortes FITS…"):
            data = _consultar(
                float(st.session_state.ra), float(st.session_state.dec),
                float(st.session_state.fov), st.session_state.detector, max_epocas,
            )
        st.session_state.resultado = {"tipo": "consulta", "data": data}
    except Exception as exc:
        st.session_state.resultado = None
        st.error(f"No se pudo completar esta consulta: {type(exc).__name__}: {exc}")
        st.info("IRSA puede devolver 502/503 temporalmente. Intenta nuevamente o utiliza una demo preparada.")
if ver_demo and demo_p:
    st.session_state.resultado = {"tipo": "demo", "ruta": str(demo_p), "nombre": ex[0]}

resultado = st.session_state.resultado
if resultado is None:
    st.info("Introduce RA/Dec o selecciona un ejemplo y pulsa Consultar. También puedes mostrar un GIF ya validado.")
    st.stop()

st.header("2 · Animación temporal")
if resultado["tipo"] == "demo":
    ruta = Path(resultado["ruta"])
    if ruta.is_file():
        st.image(str(ruta), caption=f"{resultado['nombre']} · GIF real preparado por el equipo", width="stretch")
        st.download_button("Descargar GIF", ruta.read_bytes(), file_name=ruta.name, mime="image/gif")
        st.caption("Este GIF se preparó previamente; al mostrarlo no se vuelve a consultar IRSA.")
    else:
        st.warning("No se encuentra el GIF de demostración en el repositorio.")
else:
    datos = resultado["data"]
    gif = Path(datos["archivo_gif"])
    if not gif.is_file():
        st.error("La búsqueda terminó, pero el archivo del GIF ya no está disponible en el servidor.")
        st.stop()
    st.image(str(gif), caption=(
        f"RA {st.session_state.ra:.5f}°, Dec {st.session_state.dec:+.5f}° · "
        f"{datos['detector']} · {len(datos['observaciones'])} épocas seleccionadas"
    ), width="stretch")
    st.download_button("Descargar GIF", gif.read_bytes(), file_name=gif.name, mime="image/gif")

    st.header("3 · Información científica")
    m1, m2, m3 = st.columns(3)
    m1.metric("Encontradas", datos["catalogo_total"])
    m2.metric("FITS procesados", datos["procesadas"])
    m3.metric("Épocas seleccionadas", len(datos["observaciones"]))
    m4, m5, m6 = st.columns(3)
    m4.metric("Cobertura común", f"{datos['cobertura_comun_pct']:.1f} %")
    m5.metric("Fracción espectral", f"{datos['fraccion_ancho']:.3f}")
    m6.metric("Rango de λ", f"{datos['rango_lambda_um']:.4f} µm")

    obs = datos["observaciones"].copy()
    st.dataframe(obs.drop(columns=["archivo"], errors="ignore"), hide_index=True, width="stretch")
    st.download_button(
        "Descargar tabla CSV", obs.to_csv(index=False).encode("utf-8"),
        file_name="spherex_observaciones.csv", mime="text/csv",
    )
    with st.expander("Detalles de descarga y advertencias"):
        st.info(datos["advertencia"])
        st.write(f"Candidatas evaluadas: {datos['candidatas']}. "
                 "La selección es heurística y no garantiza el mejor conjunto global del archivo.")
        if not datos["errores"].empty:
            st.warning("Algunos recortes no pudieron procesarse:")
            st.dataframe(datos["errores"], hide_index=True, width="stretch")
        st.caption("Los píxeles sin cobertura común se muestran en gris; no se inventan valores.")

st.caption("NASA Space Apps Challenge 2026 · Fuente: NASA/IPAC IRSA · SPHEREx DR1 + QR3")
