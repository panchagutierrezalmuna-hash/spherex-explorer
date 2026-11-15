"""Planeta 9 | SPHEREx Motion Explorer.

Interfaz Streamlit. Las consultas nuevas SIEMPRE ejecutan el motor científico
spherex_explorer.explorar_region; las demos son archivos de datos reales preparados
por el equipo, y no se presentan como búsquedas en vivo.
"""
from __future__ import annotations

import base64
import json
import zlib
import threading
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import streamlit as st

from spherex_explorer import explorar_region
from spherex_moviles import barnard_en, consultar_asteroide, figura_posiciones, vista_recorte_asteroide

st.set_page_config(page_title="Planeta 9 · SPHEREx Motion Explorer",
                   page_icon="🌌", layout="wide",
                   initial_sidebar_state="expanded")
RAIZ = Path(__file__).resolve().parent

# Coordenadas de referencia en grados (ICRS); su disponibilidad en SPHEREx
# depende de los datos públicos, la banda y los criterios de selección.
OBJETOS = {
    "M101 · Galaxia del Molinete": {
        "id": "M101", "ra": 210.80227, "dec": 54.34895,
        "tipo": "Galaxia", "descripcion": "Nuestro primer campo de prueba: galaxia espiral y tres épocas compatibles.",
    },
    "M51 · Galaxia del Remolino": {
        "id": "M51", "ra": 202.46960, "dec": 47.19520,
        "tipo": "Galaxia", "descripcion": "Segunda validación del buscador: otras coordenadas, mismo procesamiento.",
    },
    "M82 · Galaxia del Cigarro": {
        "id": "M82", "ra": 148.96969, "dec": 69.67938,
        "tipo": "Galaxia", "descripcion": "Galaxia conocida por su intensa formación estelar.",
    },
    "M31 · Andrómeda": {
        "id": "M31", "ra": 10.68479, "dec": 41.26907,
        "tipo": "Galaxia", "descripcion": "Nuestro gran vecino galáctico; con 4′ verás solo una pequeña región.",
    },
    "M33 · Galaxia del Triángulo": {
        "id": "M33", "ra": 23.46204, "dec": 30.66022,
        "tipo": "Galaxia", "descripcion": "Otra galaxia espiral cercana.",
    },
    "M104 · Galaxia del Sombrero": {
        "id": "M104", "ra": 189.99763, "dec": -11.62305,
        "tipo": "Galaxia", "descripcion": "Galaxia con un núcleo muy brillante y una franja de polvo.",
    },
    "Estrella de Barnard": {
        "id": "Barnard", "ra": 269.452075, "dec": 4.693391,
        "tipo": "Estrella cercana", "descripcion": (
            "Como esta estrella se desplaza rápidamente por el cielo, usamos su "
            "posición prevista para finales de 2025 y la marcamos en cada observación."
        ),
    },
    "R Draconis · Estrella variable": {
        "id": "R_Draconis", "ra": 248.16792, "dec": 66.75500,
        "tipo": "Estrella variable", "descripcion": (
            "Gigante roja variable. Las coordenadas de esta referencia son aproximadas "
            "y deben verificarse si se realiza un análisis detallado."
        ),
    },
}

DEMO_GIF = {
    "M101": ["demos/M101.gif", "05_Analisis/M101_blink.gif"],
    "Barnard": ["demos/Barnard.gif", "05_Analisis/Barnard_blink.gif"],
    "R_Draconis": ["demos/R_Draconis.gif", "05_Analisis/R_Draconis_blink.gif"],
}
# Los notebooks se pueden LEER en la web, sin ejecutar consultas ni depender de IRSA.
DEMO_NOTEBOOKS = {
    "M101": ["demos/M101_SPHEREx_multiples.ipynb"],
    "Barnard": ["demos/Barnard_SPHEREx_autonomo.ipynb"],
    "R_Draconis": ["demos/R_Draconis.ipynb"],
}

# Copias integradas de los notebooks originales del equipo.
# Esto permite leer los análisis aunque los .ipynb no se hayan subido a demos/.
# Si se sube un notebook a demos/ con el nombre esperado, se usa esa versión.
NOTEBOOKS_INCLUIDOS = {
    'M101': (
        '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'
        'ASCLdqo='
    ),
    'Barnard': (
        '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'
    ),
}

DEMO_FIGURAS = {
    "M101": {
        "Comparación inicial": ["demos/M101_inicial.png", "M101_D2_comparacion_inicial.png"],
        "Comparación alineada": ["demos/M101_alineada.png", "M101_comparacion_alineada.png"],
    },
    "Barnard": {
        "Comparación inicial": ["demos/Barnard_inicial.png", "Barnard_D2_comparacion_inicial.png"],
        "Comparación alineada": ["demos/Barnard_alineada.png", "Barnard_comparacion_alineada.png"],
        "Movimiento medido y esperado": ["demos/Barnard_movimiento.png", "Barnard_movimiento_medido.png"],
    },
    "R_Draconis": {
        "Comparación inicial": ["demos/R_Draconis_inicial.png"],
        "Comparación alineada": ["demos/R_Draconis_alineada.png"],
    },
}
# Barnard no está en 2025/2026 donde figuraba en el catálogo J2000.
# Propagamos su astrometría a la mitad del período SPHEREx explorado.
_BARNARD_REFERENCIA = barnard_en("2025-12-01")
OBJETOS["Estrella de Barnard"]["ra"] = _BARNARD_REFERENCIA.ra.deg
OBJETOS["Estrella de Barnard"]["dec"] = _BARNARD_REFERENCIA.dec.deg

CUSTOM = "— Introducir coordenadas —"


def _buscar_archivo(opciones):
    for nombre in opciones:
        ruta = RAIZ / nombre
        if ruta.is_file() and ruta.stat().st_size:
            return ruta
    return None


def _notebook_disponible(clave):
    return (_buscar_archivo(DEMO_NOTEBOOKS.get(clave, [])) is not None
            or clave in NOTEBOOKS_INCLUIDOS)


def _leer_notebook(clave):
    ruta = _buscar_archivo(DEMO_NOTEBOOKS.get(clave, []))
    if ruta is not None:
        return ruta.read_bytes(), ruta.name
    comprimido = NOTEBOOKS_INCLUIDOS.get(clave)
    if comprimido is None:
        return None, None
    return zlib.decompress(base64.b64decode(comprimido)), (
        'M101_SPHEREx_multiples.ipynb' if clave == 'M101'
        else 'Barnard_SPHEREx_autonomo.ipynb'
    )


def _limpiar():
    st.session_state.resultado = None


def _actualizar_ejemplo():
    objeto = OBJETOS.get(st.session_state.ejemplo)
    if objeto:
        st.session_state.ra = objeto["ra"]
        st.session_state.dec = objeto["dec"]
    _limpiar()


def _editar_coordenadas():
    st.session_state.ejemplo = CUSTOM
    _limpiar()


def _abrir_objeto(nombre):
    objeto = OBJETOS[nombre]
    st.session_state.ejemplo = nombre
    st.session_state.ra = objeto["ra"]
    st.session_state.dec = objeto["dec"]
    st.session_state.pagina = "Explorar el cielo"
    _limpiar()


@st.cache_resource
def _bloqueo_graficos():
    return threading.Lock()


@st.cache_data(ttl=1800, max_entries=6, show_spinner=False)
def _consultar(ra, dec, fov, detector, max_epocas, es_barnard=False):
    """Cachea bytes, no solo rutas efímeras de Streamlit Cloud."""
    with _bloqueo_graficos():
        res = explorar_region(ra=ra, dec=dec, fov_arcmin=fov,
                              detector=detector, max_epocas=max_epocas,
                              max_por_ventana=5 if es_barnard else 3,
                              separacion_dias=15,
                              mostrar_progreso=False
                          )

        for clave, destino in [
            ("archivo_gif", "gif_bytes"),
            ("archivo_comparacion_inicial", "inicial_bytes"),
            ("archivo_comparacion_alineada", "alineada_bytes"),
        ]:
            ruta = res.get(clave)
            res[destino] = (Path(ruta).read_bytes() if ruta and Path(ruta).is_file() else None)
    return res


# Fondo: ilustración astronómica generada específicamente para esta web.
# Comprimida a WebP para reducir el peso de la página.
imagen_fondo = RAIZ / "firmamento.webp"
fondo_css = ""
if imagen_fondo.is_file():
    datos_webp = base64.b64encode(imagen_fondo.read_bytes()).decode("ascii")
    fondo_css = f'url("data:image/webp;base64,{datos_webp}")'

st.markdown("""
<style>
:root { --violet: #9d8df9; --cyan: #78d8e6; --line: rgba(180,195,250,.22); }
.stApp { background: #070b1c !important; color: #eaf0ff; }
.stApp::before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
   background-image: linear-gradient(180deg, rgba(7,10,27,.46) 0%, rgba(5,8,22,.88) 72%, #060917 100%), __FONDO__;
   background-position: center top; background-size: cover; background-attachment: fixed; }
[data-testid="stAppViewContainer"] > .main { position: relative; z-index:1; }
[data-testid="stSidebar"] { background:rgba(8,12,31,.96) !important;
    border-right:1px solid rgba(160,180,255,.15); }
[data-testid="stSidebar"] * { color: #e7ebff; }
.block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { color: #f4f7ff !important; letter-spacing:-.025em; }
p, label { color: #dae3fa; }
.hero-panel { position: relative; padding: 2rem 2rem 2.15rem; margin-bottom:1.6rem;
    border:1px solid rgba(175,192,255,.26); border-radius:24px;
    background: linear-gradient(125deg, rgba(11,17,45,.87), rgba(20,23,62,.55));
    box-shadow: 0 24px 70px rgba(0,0,0,.3); backdrop-filter: blur(12px); }
.hero-eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.25em;
    text-transform:uppercase; color:#abbeff; margin-bottom:1rem; }
.hero-title { font-size: clamp(2.6rem, 5.5vw, 4.7rem); line-height:1.02;
    letter-spacing:-.055em; font-weight:760; color:#fff; max-width:730px; }
.hero-title em { font-style:normal; background:linear-gradient(100deg,#e0d2ff,#8be5f3);
    -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-desc { max-width:670px; font-size:1.08rem; color:#becbed; line-height:1.65;
    margin-top:1.1rem; }
.team-tag { position:absolute; right:1.5rem; top:1.4rem; border:1px solid rgba(160,180,255,.36);
    border-radius:99px; padding:.45rem .85rem; font-weight:750; font-size:.82rem;
    letter-spacing:.03em; background:rgba(60,55,135,.55); color:#fff; }
.sci-tag { display:inline-block; border-radius:99px; border:1px solid rgba(110,225,215,.3);
    color:#b9fff2; background:rgba(20,86,85,.24); font-size:.8rem;
    padding:.34rem .7rem; margin-top:.8rem; }
.small-note { border-left:3px solid #80d3ef; color:#d6e2fa; padding:.7rem .9rem;
   border-radius:0 8px 8px 0; background:rgba(13,30,56,.65); margin:.7rem 0 1.2rem; }
.team-sidebar { padding:.4rem 0 1.2rem; font-weight:850; font-size:1.27rem;
    letter-spacing:-.02em; color:#f3f0ff; }
.subheading { text-transform:uppercase; color:#a5bcf0; font-size:.72rem; letter-spacing:.17em;
    font-weight:800; padding:.5rem 0 .2rem; }
[data-testid="stMetric"] { background:rgba(15,20,46,.74); padding:1rem;
    border-radius:14px; border:1px solid var(--line); min-height:112px; }
.stButton > button[kind="primary"], .stButton > button {
    border:1px solid rgba(155,180,255,.4); border-radius:11px;
    background:linear-gradient(100deg,#6d5ce5,#376eab); color:#fff;
    font-weight:750; min-height:2.9rem; }
.stButton > button:hover { background:linear-gradient(100deg,#897dff,#477fc2); color:white; }
[data-testid="stExpander"] { background:rgba(11,17,42,.5); border:1px solid var(--line); border-radius:12px; }
[data-testid="stTabs"] { background:rgba(8,14,34,.28); border-radius:10px; }
[data-testid="stDataFrame"] { border-radius:12px; }
@media(max-width:700px) { .hero-panel{padding:2rem 1.15rem 1.4rem;}
 .team-tag{position:static;display:inline-block;margin-bottom:1rem;}
 .hero-title{font-size:2.8rem;} }
</style>
""".replace("__FONDO__", fondo_css or "radial-gradient(ellipse at top, #16264d, #070b1c 75%)"),
unsafe_allow_html=True)

st.session_state.setdefault("ra", OBJETOS["M101 · Galaxia del Molinete"]["ra"])
st.session_state.setdefault("dec", OBJETOS["M101 · Galaxia del Molinete"]["dec"])
st.session_state.setdefault("ejemplo", "M101 · Galaxia del Molinete")
st.session_state.setdefault("detector", "D2")
st.session_state.setdefault("fov", 4)
st.session_state.setdefault("resultado", None)
st.session_state.setdefault("pagina", "Explorar el cielo")
# Actualizar el nombre seleccionado en una sesion previa, sin perder la navegacion.
if st.session_state.pagina == "Galería del equipo":
    st.session_state.pagina = "Exploraciones destacadas"

with st.sidebar:
    st.markdown('<div class="team-sidebar">✦ Planeta 9</div>', unsafe_allow_html=True)
    st.caption("NASA Space Apps Challenge 2026")
    st.radio("Menú", ["Explorar el cielo", "Objetos para probar",
                       "Exploraciones destacadas", "Asteroides", "El proyecto"],
             key="pagina", label_visibility="collapsed")
    st.divider()
    st.caption("Imágenes reales de SPHEREx · NASA/IPAC IRSA")
    st.caption("Este proyecto explora observaciones; no anuncia detecciones confirmadas.")

st.markdown('''<section class="hero-panel">
    <span class="team-tag">✦ PLANETA 9</span>
    <div class="hero-eyebrow">NASA SPACE APPS CHALLENGE · 2026</div>
    <div class="hero-title">El cielo siempre está cambiando.<br><em>¡Vamos a explorarlo!</em></div>
    <div class="hero-desc">Elige una región del cielo y compárala en diferentes fechas.
       Usamos imágenes infrarrojas reales del telescopio espacial SPHEREx para crear
       una pequeña película del universo.</div>
    <div class="sci-tag">◉ DATOS REALES · NASA / IPAC IRSA</div>
</section>''', unsafe_allow_html=True)

pagina = st.session_state.pagina
if pagina == "Objetos para probar":
    st.header("Un punto de partida")
    st.write("Puedes empezar por una galaxia conocida o escribir tus propias coordenadas. "
             "No todas las regiones tendrán suficientes observaciones compatibles, ¡y eso también es parte de explorar datos reales!")
    for nombre, o in OBJETOS.items():
        with st.container(border=True):
            izq, der = st.columns([4, 1], vertical_alignment="center")
            with izq:
                st.markdown(f"**{nombre}** · {o['tipo']}")
                st.caption(o["descripcion"])
                st.caption(f"RA {o['ra']:.5f}° · Dec {o['dec']:+.5f}°")
            with der:
                st.button("Usar", key="usar_" + o["id"],
                          on_click=_abrir_objeto, args=(nombre,), use_container_width=True)
    st.info("Las coordenadas son referencias para la búsqueda. Un recorte no equivale a una detección del objeto.")
    st.stop()

if pagina == "Exploraciones destacadas":
    st.header("✦ Exploraciones destacadas")
    st.write(
        "Estas son las exploraciones que preparamos como equipo antes de abrir el "
        "buscador a cualquier punto del cielo. Puedes conocer el código del análisis "
        "y, cuando estén subidas, ver las animaciones originales y sus comparaciones."
    )
    st.caption("Esta sección funciona sin consultar IRSA. Los notebooks se muestran, no se ejecutan.")

    nombres = ["M101", "Barnard", "R_Draconis"]
    etiquetas = {
        "M101": "M101 · Galaxia del Molinete",
        "Barnard": "Estrella de Barnard",
        "R_Draconis": "R Draconis · Estrella variable",
    }
    descripciones = {
        "M101": "Una galaxia espiral observada en distintas fechas.",
        "Barnard": "Una estrella cercana con un movimiento propio excepcional.",
        "R_Draconis": "Una estrella variable que podemos comparar en el infrarrojo.",
    }
    presentaciones = st.columns(3)
    for columna, clave in zip(presentaciones, nombres):
        with columna:
            with st.container(border=True):
                st.markdown(f"**{etiquetas[clave]}**")
                st.caption(descripciones[clave])
                gif_existe = _buscar_archivo(DEMO_GIF.get(clave, [])) is not None
                nb_existe = _notebook_disponible(clave)
                if gif_existe:
                    st.caption("🎞️ Animación disponible")
                elif nb_existe:
                    st.caption("📓 Análisis disponible · GIF pendiente")
                else:
                    st.caption("Próximamente")

    disponibles = [
        k for k in nombres
        if (_buscar_archivo(DEMO_GIF.get(k, [])) is not None
            or _notebook_disponible(k))
    ]

    if not disponibles:
        st.info("Aún faltan los archivos de las exploraciones. Sube los notebooks o GIF a `demos/` en GitHub.")
    else:
        clave = st.selectbox("Elige una exploración", disponibles,
                             format_func=lambda k: etiquetas[k])
        gif = _buscar_archivo(DEMO_GIF.get(clave, []))
        notebook_data, notebook_nombre = _leer_notebook(clave)

        # El usuario puede leer ya sus notebooks aun cuando el archivo GIF no exista.
        opciones = ["🎞️ Ver animación", "📓 Ver notebook"]
        elegir = st.radio("¿Qué quieres ver?", opciones, horizontal=True,
                          index=0 if gif is not None else 1,
                          key=f"vista_destacada_{clave}")

        if elegir == "🎞️ Ver animación":
            if gif is None:
                st.info("Todavía no tenemos el GIF exportado de esta exploración. "
                        "El archivo `.ipynb` contiene el código, pero no la animación guardada.")
                st.caption(f"Cuando esté disponible, sube `demos/{clave}.gif` a GitHub. "
                           "La galería lo reconocerá automáticamente.")
            else:
                st.image(gif.read_bytes(), caption=etiquetas[clave] + " · Animación del equipo",
                         use_container_width=True)
                st.download_button("⬇️ Descargar GIF", gif.read_bytes(), file_name=gif.name,
                                   mime="image/gif", key=f"gif_{clave}")
                figuras = DEMO_FIGURAS.get(clave, {})
                disponibles_fig = [
                    (titulo, ruta) for titulo, opciones_fig in figuras.items()
                    if (ruta := _buscar_archivo(opciones_fig)) is not None
                ]
                if disponibles_fig:
                    st.subheader("Comparaciones del análisis original")
                    pestanas = st.tabs([titulo for titulo, _ in disponibles_fig])
                    for pestana, (titulo, ruta) in zip(pestanas, disponibles_fig):
                        with pestana:
                            st.image(ruta.read_bytes(), caption=titulo, use_container_width=True)
                st.caption("Animación preparada con datos reales de SPHEREx; no es una consulta en vivo.")

        else:  # notebook
            if notebook_data is None:
                st.info("Todavía no está disponible el notebook de este objeto.")
            else:
                try:
                    contenido = json.loads(notebook_data.decode("utf-8"))
                    celdas = contenido.get("cells", [])
                    st.subheader(f"Cuaderno de análisis · {etiquetas[clave]}")
                    st.caption("Puedes leer el procedimiento y el código. "
                               "La web no ejecuta estas celdas ni descarga imágenes nuevas.")
                    st.download_button("⬇️ Descargar notebook para Google Colab",
                                       data=notebook_data, file_name=notebook_nombre,
                                       mime="application/x-ipynb+json", key=f"nb_{clave}")
                    st.link_button("Abrir Google Colab", "https://colab.research.google.com/")
                    st.caption("Para ejecutarlo: descarga el notebook, abre Colab y selecciona "
                               "Archivo → Subir notebook. Si IRSA sigue con errores 502, "
                               "la ejecución puede fallar.")
                    for i, celda in enumerate(celdas, 1):
                        texto_celda = "".join(celda.get("source", []))
                        if not texto_celda.strip():
                            continue
                        if celda.get("cell_type") == "markdown":
                            st.markdown(texto_celda)
                        elif celda.get("cell_type") == "code":
                            with st.expander(f"Celda {i} · Código Python", expanded=False):
                                st.code(texto_celda, language="python")
                    if not any(c.get("outputs") for c in celdas):
                        st.caption("Este notebook no tiene figuras ni resultados guardados dentro "
                                   "del propio archivo; se mostrarán cuando añadamos los GIF/PNG.")
                except (OSError, ValueError, UnicodeError) as error:
                    st.error(f"No se pudo leer el notebook: {error}")

    st.stop()

if pagina == "Asteroides":
    st.header("Tras la pista de un asteroide")
    st.write("Los asteroides se desplazan respecto al fondo de estrellas. A diferencia "
             "de las galaxias, no basta con buscar una única RA y Dec: necesitamos "
             "saber dónde se esperaba el objeto en cada fecha.")
    st.write("Usamos **MOST**, el servicio de NASA/IPAC que combina efemérides "
             "de objetos del sistema solar con el archivo SPHEREx.")
    st.caption("Primera demostración experimental: (4) Vesta. "
               "No estamos midiendo su movimiento directamente desde los píxeles.")
    c1, c2 = st.columns(2)
    with c1:
        inicio = st.date_input("Desde", date(2025, 8, 1),
                               min_value=date(2025, 1, 1), max_value=date.today())
    with c2:
        fin = st.date_input("Hasta", date(2025, 8, 31),
                            min_value=date(2025, 1, 1), max_value=date.today())
    if inicio > fin:
        st.error("La fecha de inicio debe ser anterior a la fecha final.")
        st.stop()
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
    def _most_vesta(a, b):
        return consultar_asteroide("20000004", a, b)

    if st.button("☄️ Buscar observaciones de Vesta", type="primary"):
        try:
            with st.spinner("Consultando efemérides y observaciones de Vesta en IRSA MOST…"):
                st.session_state["most_vesta"] = _most_vesta(inicio.isoformat(), fin.isoformat())
                st.session_state["most_vesta_periodo"] = (inicio.isoformat(), fin.isoformat())
        except Exception as error:
            st.session_state.pop("most_vesta", None)
            st.error("MOST no respondió o no pudo completar la búsqueda. "
                     "Puedes volver a intentarlo.")
            with st.expander("Detalles técnicos"):
                st.code(f"{type(error).__name__}: {error}")
    resultados = st.session_state.get("most_vesta")
    if st.session_state.get("most_vesta_periodo") != (inicio.isoformat(), fin.isoformat()):
        resultados = None
    if resultados is not None:
        if resultados.empty:
            st.info("MOST no encontró imágenes coincidentes en ese período.")
        else:
            st.metric("Exposiciones que coinciden con la efeméride", len(resultados))
            st.caption("Cada punto del gráfico es una posición **prevista**, no un "
                       "asteroide detectado automáticamente en la imagen.")
            fig = figura_posiciones(resultados)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            vista = resultados[["fecha", "ra_deg", "dec_deg", "imagen_id"]].copy()
            vista.columns = ["Fecha UTC", "RA prevista (°)", "Dec prevista (°)", "Imagen"]
            st.dataframe(vista, hide_index=True, use_container_width=True)
            with st.expander("Ver un recorte real de una exposición (experimental)"):
                indice = st.selectbox("Observación", list(range(len(resultados))),
                                     format_func=lambda i: f"{i + 1} · {resultados.iloc[i]['fecha']}")
                if st.button("Mostrar recorte FITS de Vesta"):
                    try:
                        with st.spinner("Descargando recorte desde IRSA…"):
                            fig = vista_recorte_asteroide(resultados.iloc[indice].to_dict(), 4)
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                        st.caption("El círculo marca la posición prevista de Vesta "
                                   "por la efeméride; no constituye una detección del asteroide.")
                    except Exception as error:
                        st.warning("No se pudo mostrar este recorte. Puede que IRSA "
                                   "esté ocupado o que la URL no admita recortes.")
                        with st.expander("Detalles técnicos"):
                            st.code(f"{type(error).__name__}: {error}")
    st.link_button("Abrir MOST directamente en NASA/IPAC",
                   "https://irsa.ipac.caltech.edu/applications/MOST/")
    st.info("Para generar un GIF de un asteroide sobre el **mismo campo fijo**, "
            "todavía debemos seleccionar exposiciones suficientemente cercanas en "
            "tiempo y longitud de onda, y comprobar la fuente contra el fondo.")
    st.stop()

if pagina == "El proyecto":
    st.header("¿Cómo funciona?")
    st.write("SPHEREx observa el cielo en el infrarrojo. Aquí puedes seleccionar un punto "
             "(RA y Dec), elegir uno de sus seis detectores y comparar observaciones de distintas fechas.")
    st.markdown("**De las coordenadas a la animación**")
    st.write("① Buscamos observaciones públicas en IRSA. ② Descargamos recortes FITS. "
             "③ Buscamos épocas separadas y bandas parecidas. ④ Alineamos las imágenes "
             "mediante sus coordenadas WCS. ⑤ Creamos la animación y las comparaciones.")
    st.markdown("**¿Qué muestran los huecos grises?**")
    st.write("Son píxeles que no tienen datos válidos en todas las imágenes. "
             "Preferimos dejarlos visibles y no completar con valores estimados.")
    st.markdown("**¿Se puede detectar el Planeta 9 aquí?**")
    st.write("Todavía no. Esta versión permite explorar imágenes y posibles cambios, pero no hace "
             "una búsqueda automatizada ni valida candidatos de planetas lejanos.")
    st.markdown("**¿Y los asteroides?**")
    st.write("Los asteroides cambian rápidamente de coordenadas. Para estudiarlos, "
             "usamos la herramienta MOST de IRSA, que consulta las posiciones previstas "
             "y busca imágenes de SPHEREx que cubren esos lugares. Puedes probarlo "
             "en la sección Asteroides; aún no es un detector de objetos en movimiento.")
    st.link_button("Explorar objetos móviles en IRSA MOST", "https://irsa.ipac.caltech.edu/applications/MOST/")
    st.caption("Realizado por el equipo Planeta 9 para NASA Space Apps Challenge 2026. "
               "Datos: NASA/IPAC IRSA, colecciones SPHEREx DR1 + QR3.")
    st.stop()

st.header("01 / Elige una región")
st.write("Elige un objeto conocido o escribe otras coordenadas. "
         "Con el botón de búsqueda procesamos imágenes reales de SPHEREx. "
         "Los GIF preparados por el equipo están en la **Exploraciones destacadas**.")
st.selectbox("Objeto de referencia", [CUSTOM, *OBJETOS.keys()], key="ejemplo",
             on_change=_actualizar_ejemplo,
             help="Los nombres sirven para rellenar las coordenadas; la búsqueda siempre usa RA y Dec.")
seleccionado = OBJETOS.get(st.session_state.ejemplo)
if seleccionado:
    st.caption(seleccionado["descripcion"])
    if seleccionado["id"] == "Barnard":
        st.info("Barnard se desplaza unos 10,4″ por año. "
                "Las coordenadas del buscador están propagadas al 01-12-2025 "
                "(no son las originales J2000). El círculo cian en las figuras "
                "indica su posición esperada en cada fecha, no una detección medida.")

izq, der = st.columns(2)
with izq:
    st.number_input("Ascensión recta · RA (°)", min_value=0.0,
                    max_value=359.99999, step=.01, format="%.5f", key="ra",
                    on_change=_editar_coordenadas)
with der:
    st.number_input("Declinación · Dec (°)", min_value=-90.0,
                    max_value=90.0, step=.01, format="%.5f", key="dec",
                    on_change=_editar_coordenadas)
c3, c4, c5 = st.columns([1, 1, 1.3])
with c3:
    st.select_slider("Campo de visión", options=[2, 3, 4, 5, 6, 8, 10, 12],
                     format_func=lambda v: f"{v}′", key="fov", on_change=_limpiar)
with c4:
    st.selectbox("Detector", [f"D{i}" for i in range(1, 7)],
                 key="detector", on_change=_limpiar)
with c5:
    max_epocas = st.slider("Máximo de épocas", 2, 5, 5,
                           help="Si hay menos imágenes compatibles, se usarán menos.",
                           on_change=_limpiar)

st.markdown('<div class="small-note">Las consultas son reales y pueden tardar. '
            'Si aparece un error 502 o 503, IRSA podría estar ocupado: vuelve a intentar. '
            'Si la aplicación completa se detiene, vuelve a abrirla.</div>',
            unsafe_allow_html=True)

# El buscador consulta siempre IRSA; las demos viven en la galería independiente.
consultar = st.button("🔭 Explorar esta región en IRSA", type="primary", use_container_width=True)

if consultar:
    ra_elegida, dec_elegida = float(st.session_state.ra), float(st.session_state.dec)
    try:
        with st.spinner("Buscando observaciones en IRSA y preparando la comparación…"):
            resultado = _consultar(ra_elegida, dec_elegida,
                                  float(st.session_state.fov), st.session_state.detector,
                                  max_epocas,
                                  es_barnard=bool(seleccionado and seleccionado["id"] == "Barnard"))
        st.session_state.resultado = {
            "tipo": "consulta", "data": resultado, "ra": ra_elegida, "dec": dec_elegida,
            "fov": float(st.session_state.fov),
        }
    except Exception as exc:
        st.session_state.resultado = None
        mensaje = str(exc)
        if any(x in mensaje.lower() for x in ("502", "503", "504", "proxy", "timeout", "timed out")):
            st.warning("IRSA no respondió correctamente esta vez. "
                       "No significa que no existan datos: vuelve a pulsar «Explorar esta región».")
        elif "no hay dos" in mensaje.lower() or "al menos dos" in mensaje.lower():
            st.warning("No encontramos suficientes épocas compatibles. "
                       "Prueba otro detector, unas coordenadas distintas o un campo más pequeño.")
        else:
            st.error("No se pudo completar la consulta. Prueba de nuevo o revisa los detalles.")
        with st.expander("Detalles técnicos del error"):
            st.code(f"{type(exc).__name__}: {mensaje}")

resultado_actual = st.session_state.resultado
if resultado_actual is None:
    st.divider()
    st.info("Introduce unas coordenadas y pulsa «Explorar esta región en IRSA». "
            "Para ver los GIF originales del equipo, abre **Exploraciones destacadas** en el menú.")
    st.stop()

st.divider()
st.header("02 / El cielo en movimiento")
res = resultado_actual["data"]
if res.get("gif_bytes"):
    st.image(res["gif_bytes"],
             caption=(f"RA {resultado_actual['ra']:.5f}°, Dec {resultado_actual['dec']:+.5f}° · "
                      f"{res['detector']} · {len(res['observaciones'])} épocas · "
                      f"campo de {resultado_actual['fov']:g}′"),
             use_container_width=True)
    st.download_button("Descargar GIF", res["gif_bytes"],
                       file_name="planeta9_spherex.gif", mime="image/gif")
else:
    st.warning("No se pudo recuperar el GIF de esta consulta.")

if res.get("objeto_movil") == "Barnard":
    st.caption("⭕ Círculo cian: posición esperada de la estrella de Barnard "
               "a partir de su movimiento propio catalogado; "
               "no equivale a una detección ni a un movimiento medido.")

# Ambas imágenes provienen de las mismas épocas seleccionadas, no de otro algoritmo.
if res.get("inicial_bytes") and res.get("alineada_bytes"):
    st.header("03 / Mira la diferencia")
    tabs = st.tabs(["Comparación alineada", "Imágenes originales"])
    with tabs[0]:
        st.image(res["alineada_bytes"], use_container_width=True)
        st.caption("Misma región del cielo, misma escala de intensidad. "
                   "Los huecos grises son zonas sin datos válidos compartidos.")
    with tabs[1]:
        st.image(res["inicial_bytes"], use_container_width=True)
        st.caption("Los recortes originales, cada uno con su orientación y escala de visualización.")

st.header("04 / Qué encontramos")
met1, met2, met3 = st.columns(3)
met1.metric("Observaciones en el catálogo", res["catalogo_total"])
met2.metric("Recortes examinados", res["procesadas"])
met3.metric("Épocas del GIF", len(res["observaciones"]))
met4, met5, met6 = st.columns(3)
met4.metric("Cobertura común", f"{res['cobertura_comun_pct']:.1f} %")
met5.metric("Rango de longitudes de onda", f"{res['rango_lambda_um']:.4f} µm")
met6.metric("Detector", res["detector"])

obs = res["observaciones"].copy()
obs = obs.rename(columns={"obs_id": "ID de observación", "fecha_utc": "Fecha UTC",
                          "lambda_um": "λ (µm)", "ancho_um": "Ancho de banda (µm)"})
st.dataframe(obs.drop(columns=["archivo"], errors="ignore"), hide_index=True,
             use_container_width=True)
st.download_button("Descargar tabla CSV", res["observaciones"].to_csv(index=False).encode("utf-8"),
                   file_name="planeta9_observaciones.csv", mime="text/csv")

with st.expander("Sobre la selección y la calidad de los datos"):
    st.write("**¿Cómo se eligieron las imágenes?**")
    st.write("Revisamos hasta varias observaciones de cada período y elegimos "
             "una secuencia con fechas separadas, bandas parecidas y cobertura suficiente.")
    st.write(f"Se preseleccionaron {res['candidatas']} recortes; "
             "{res['procesadas']} pudieron procesarse. No revisamos todas las imágenes "
             "del archivo, así que podrían existir otras combinaciones.")
    st.write("**¿Qué podemos concluir?**")
    st.write("Esta película permite comparar el cielo, pero para afirmar que una fuente "
             "cambió de brillo o posición se necesitarían más controles de calibración "
             "y análisis cuantitativo.")
    st.caption(f"Fracción de separación espectral relativa al ancho medio de banda: "
               f"{res['fraccion_ancho']:.3f}.")
    if res.get("error_figuras"):
        st.warning("El GIF se generó, pero no pudimos crear todas las comparaciones adicionales.")
    if not res["errores"].empty:
        st.warning("Algunos recortes no pudieron procesarse.")
        st.dataframe(res["errores"], hide_index=True, use_container_width=True)
    st.caption("Las zonas grises representan ausencia de cobertura común válida. "
               "No se rellenaron con valores inventados.")

st.caption("Planeta 9 · NASA Space Apps Challenge 2026 · Datos SPHEREx DR1 + QR3 · NASA/IPAC IRSA")