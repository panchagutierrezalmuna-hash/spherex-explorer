"""Planeta 9 | SPHEREx Motion Explorer.

Interfaz Streamlit. Las consultas nuevas SIEMPRE ejecutan el motor científico
spherex_explorer.explorar_region; las demos son archivos de datos reales preparados
por el equipo, y no se presentan como búsquedas en vivo.
"""
from __future__ import annotations

import base64
import json
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

with st.sidebar:
    st.markdown('<div class="team-sidebar">✦ Planeta 9</div>', unsafe_allow_html=True)
    st.caption("NASA Space Apps Challenge 2026")
    st.radio("Menú", ["Explorar el cielo", "Objetos para probar",
                       "Galería del equipo", "Asteroides", "El proyecto"],
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

if pagina == "Galería del equipo":
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
                nb_existe = _buscar_archivo(DEMO_NOTEBOOKS.get(clave, [])) is not None
                if gif_existe:
                    st.caption("🎞️ Animación disponible")
                elif nb_existe:
                    st.caption("📓 Notebook disponible · falta subir GIF")
                else:
                    st.caption("Próximamente")

    disponibles = [
        k for k in nombres
        if (_buscar_archivo(DEMO_GIF.get(k, [])) is not None
            or _buscar_archivo(DEMO_NOTEBOOKS.get(k, [])) is not None)
    ]

    if not disponibles:
        st.info("Aún faltan los archivos de las exploraciones. Sube los notebooks o GIF a `demos/` en GitHub.")
    else:
        clave = st.selectbox("Elige una exploración", disponibles,
                             format_func=lambda k: etiquetas[k])
        gif = _buscar_archivo(DEMO_GIF.get(clave, []))
        notebook = _buscar_archivo(DEMO_NOTEBOOKS.get(clave, []))

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
            if notebook is None:
                st.info("Todavía no está disponible el notebook de este objeto.")
            else:
                try:
                    contenido = json.loads(notebook.read_text(encoding="utf-8"))
                    celdas = contenido.get("cells", [])
                    st.subheader(f"Cuaderno de análisis · {etiquetas[clave]}")
                    st.caption("Puedes leer el procedimiento y el código. "
                               "La web no ejecuta estas celdas ni descarga imágenes nuevas.")
                    st.download_button("⬇️ Descargar notebook para Google Colab",
                                       data=notebook.read_bytes(), file_name=notebook.name,
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
         "Los GIF preparados por el equipo están en la **Galería del equipo**.")
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
            "Para ver los GIF originales del equipo, abre **Galería del equipo** en el menú.")
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