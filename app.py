"""Planeta 9 | SPHEREx Motion Explorer.

Interfaz Streamlit. Las consultas nuevas SIEMPRE ejecutan el motor científico
spherex_explorer.explorar_region; las demos son archivos de datos reales preparados
por el equipo, y no se presentan como búsquedas en vivo.
"""
from __future__ import annotations

import base64
import threading
from datetime import date

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence, ImageDraw
import io
from pathlib import Path

import streamlit as st

from spherex_explorer import explorar_region
from spherex_moviles import barnard_en, figura_posiciones

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
    "M101": ["demo/M101.gif"],
    "Barnard": ["demo/Barnard.gif"],
    "R_Draconis": ["demo/R_Draconis.gif"],
}
# Buscamos los GIF originales en la carpeta demos/.
# Otras rutas se conservan por compatibilidad con la página original.
DEMO_TABLAS = {
    "M101": ["demos/M101_observaciones.csv", "demos/M101_tabla.csv"],
    "Barnard": ["demos/Barnard_observaciones.csv", "demos/Barnard_tabla.csv"],
    "R_Draconis": ["demos/R_Draconis_observaciones.csv", "demos/R_Draconis_tabla.csv"],
}

DEMO_FIGURAS = {
    "M101": {
        "Comparación inicial": ["demos/M101_inicial.png", "demos/M101_D2_comparacion_inicial.png", "M101_D2_comparacion_inicial.png"],
        "Comparación alineada": ["demos/M101_alineada.png", "demos/M101_comparacion_alineada.png", "M101_comparacion_alineada.png"],
    },
    "Barnard": {
        "Comparación inicial": ["demos/Barnard_inicial.png", "demos/Barnard_D2_comparacion_inicial.png", "Barnard_D2_comparacion_inicial.png"],
        "Comparación alineada": ["demos/Barnard_alineada.png", "demos/Barnard_comparacion_alineada.png", "Barnard_comparacion_alineada.png"],
        "Movimiento medido y esperado": ["demos/Barnard_movimiento.png", "demos/Barnard_movimiento_medido.png", "Barnard_movimiento_medido.png"],
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


@st.cache_data(show_spinner=False, max_entries=30)
def _datos_gif(ruta, modificacion):
    """Información técnica extraída del archivo GIF, sin interpretar datos científicos."""
    with Image.open(ruta) as imagen:
        total = imagen.n_frames
        ancho, alto = imagen.size
        duraciones = []
        for cuadro in ImageSequence.Iterator(imagen):
            duraciones.append(max(10, int(cuadro.info.get("duration", 500))))
    return {"cuadros": total, "ancho": ancho, "alto": alto,
            "duraciones_ms": duraciones, "segundos": sum(duraciones) / 1000.0}


@st.cache_data(show_spinner=False, max_entries=15)
def _gif_a_velocidad(ruta, modificacion, fps):
    """Reproduce los mismos fotogramas a otra velocidad; nunca sobrescribe el GIF original."""
    with Image.open(ruta) as imagen:
        fotogramas = [cuadro.convert("RGB") for cuadro in ImageSequence.Iterator(imagen)]
    if not fotogramas:
        raise ValueError("El GIF no contiene fotogramas.")
    # Paleta compartida entre todos los cuadros: minimiza el parpadeo cromático.
    paleta = fotogramas[0].quantize(colors=256)
    cuadros = [f.quantize(palette=paleta, dither=Image.Dither.NONE)
               for f in fotogramas]
    salida = io.BytesIO()
    cuadros[0].save(salida, format="GIF", save_all=True,
                    append_images=cuadros[1:],
                    duration=int(1000 / fps), loop=0, disposal=2,
                    optimize=False)
    return salida.getvalue()


@st.cache_data(show_spinner=False, max_entries=15)
def _fotograma_gif(ruta, modificacion, indice):
    with Image.open(ruta) as imagen:
        imagen.seek(indice)
        salida = io.BytesIO()
        imagen.convert("RGB").save(salida, format="PNG")
    return salida.getvalue()


@st.cache_data(show_spinner=False, max_entries=10)
def _mosaico_fotogramas(ruta, modificacion):
    """Comparación visual de los cuadros del GIF, no de los FITS originales."""
    with Image.open(ruta) as imagen:
        n = imagen.n_frames
        indices = list(range(n)) if n <= 12 else sorted(set(round(x * (n-1) / 11) for x in range(12)))
        elementos = []
        for i in indices:
            imagen.seek(i)
            mini = imagen.convert("RGB")
            mini.thumbnail((420, 370), Image.Resampling.LANCZOS)
            elementos.append((i, mini.copy()))
    columnas = min(3, len(elementos))
    filas = (len(elementos)+columnas-1)//columnas
    ancho_celda, alto_celda = 450, 415
    lienzo = Image.new("RGB", (columnas*ancho_celda, filas*alto_celda), "#090f25")
    dibujo = ImageDraw.Draw(lienzo)
    for j,(numero,mini) in enumerate(elementos):
        col,fil = j%columnas, j//columnas
        x = col*ancho_celda + (ancho_celda-mini.width)//2
        y = fil*alto_celda + 14
        lienzo.paste(mini,(x,y))
        dibujo.text((col*ancho_celda+18, fil*alto_celda+390),
                    f"Fotograma {numero + 1}", fill="#d7e7ff")
    out=io.BytesIO()
    lienzo.save(out, format="PNG")
    return out.getvalue()


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
    st.write("Tres historias del cielo infrarrojo que trabajamos como equipo. "
             "Aquí reunimos sus animaciones originales, las comparaciones y la información "
             "de cada exploración. Puedes verlas incluso cuando IRSA no responda.")

    etiquetas = {
        "M101": "M101 · Galaxia del Molinete",
        "Barnard": "Estrella de Barnard",
        "R_Draconis": "R Draconis · Estrella variable",
    }
    descripciones = {
        "M101": "Una galaxia espiral vista en distintas épocas del infrarrojo.",
        "Barnard": "Una estrella cercana que se desplaza rápidamente respecto al fondo.",
        "R_Draconis": "Una estrella variable: su luminosidad cambia con el tiempo.",
    }
    nombres = list(etiquetas)
    cols = st.columns(3)
    for col, clave in zip(cols, nombres):
        gif_encontrado = _buscar_archivo(DEMO_GIF[clave])
        with col:
            with st.container(border=True):
                st.markdown(f"**{etiquetas[clave]}**")
                st.caption(descripciones[clave])
                st.caption("✅ Animación lista" if gif_encontrado else "⏳ Esperando GIF")

    disponibles = [k for k in nombres if _buscar_archivo(DEMO_GIF[k]) is not None]
    if not disponibles:
        st.info("Aún faltan los GIF. Súbelos a GitHub dentro de `demos/` con los nombres "
                "`M101.gif`, `Barnard.gif` y `R_Draconis.gif`. Aparecerán automáticamente.")
        st.stop()

    clave = st.selectbox("Elige una exploración", disponibles,
                         format_func=lambda k: etiquetas[k], key="destacada_visual")
    archivo = _buscar_archivo(DEMO_GIF[clave])
    modificado = archivo.stat().st_mtime_ns
    st.subheader(etiquetas[clave])
    st.caption(descripciones[clave])

    try:
        info = _datos_gif(str(archivo), modificado)
    except Exception as err:
        st.error("No se pudo abrir la animación guardada.")
        with st.expander("Detalles técnicos"):
            st.code(f"{type(err).__name__}: {err}")
        st.stop()

    st.markdown("#### Animación temporal")
    izq, der = st.columns([2.2, 1], vertical_alignment="bottom")
    with izq:
        velocidad = st.slider("Velocidad de reproducción (fotogramas por segundo)",
                              min_value=0.5, max_value=5.0, value=2.0,
                              step=0.5, key=f"velocidad_{clave}")
    with der:
        ver_original = st.checkbox("Ver GIF original sin modificar", value=False,
                                   key=f"original_{clave}")
    try:
        if ver_original:
            datos_animacion = archivo.read_bytes()
            leyenda = "Archivo original del equipo (velocidad original)"
        else:
            datos_animacion = _gif_a_velocidad(str(archivo), modificado, velocidad)
            leyenda = f"Mismos fotogramas, reproducción a {velocidad:g} fotogramas/s"
        st.image(datos_animacion, caption=f"{etiquetas[clave]} · {leyenda}",
                 use_container_width=True)
    except Exception as err:
        st.warning("No pudimos cambiar la velocidad; mostramos el GIF original.")
        st.image(archivo.read_bytes(), use_container_width=True)
        with st.expander("Detalles técnicos"):
            st.code(f"{type(err).__name__}: {err}")

    st.download_button("⬇️ Descargar GIF original del equipo", archivo.read_bytes(),
                       file_name=archivo.name, mime="image/gif", key=f"descarga_original_{clave}")
    st.caption("El GIF original queda guardado tal como se subió a GitHub. "
               "Cambiar la velocidad solo modifica la reproducción; no altera los datos fuente.")

    st.markdown("#### Comparar imágenes")
    figuras = DEMO_FIGURAS.get(clave, {})
    figuras_existentes = [(titulo, ruta) for titulo, opciones in figuras.items()
                         if (ruta := _buscar_archivo(opciones)) is not None]
    etiquetas_tabs = [nombre for nombre, _ in figuras_existentes] + ["Fotogramas del GIF"]
    tabs = st.tabs(etiquetas_tabs)
    for tab, (nombre, ruta) in zip(tabs, figuras_existentes):
        with tab:
            st.image(ruta.read_bytes(), caption=nombre + " · figura original del equipo",
                     use_container_width=True)
    with tabs[-1]:
        st.caption("Vista conjunta de los fotogramas **del GIF guardado**. "
                   "No son los recortes FITS originales sin alinear.")
        st.image(_mosaico_fotogramas(str(archivo), modificado),
                 use_container_width=True)
        indice = st.slider("Inspeccionar un fotograma", 1, info["cuadros"], 1,
                           key=f"fotograma_{clave}")
        st.image(_fotograma_gif(str(archivo), modificado, indice - 1),
                 caption=f"Fotograma {indice} de {info['cuadros']}",
                 use_container_width=True)
    if not figuras_existentes:
        st.caption("Para recuperar las figuras **Comparación inicial**, "
                   "**Comparación alineada** y **Movimiento medido/esperado** de tu compañero, "
                   "sube también sus PNG originales a `demos/`. El GIF no permite reconstruir "
                   "de forma fiable las imágenes FITS sin alinear.")

    st.markdown("#### Información de la exploración")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fotogramas", info["cuadros"])
    c2.metric("Duración original", f"{info['segundos']:.1f} s")
    c3.metric("Resolución del GIF", f"{info['ancho']} × {info['alto']}")
    referencia = next((o for o in OBJETOS.values() if o["id"] == clave), None)
    if referencia is not None:
        st.caption(f"Coordenadas de referencia: RA {referencia['ra']:.5f}° · "
                   f"Dec {referencia['dec']:+.5f}°. "
                   "Son coordenadas del objeto, no una medición nueva del GIF.")

    csv = _buscar_archivo(DEMO_TABLAS.get(clave, []))
    if csv is not None:
        try:
            tabla = pd.read_csv(csv)
            if len(tabla):
                st.markdown("**Observaciones utilizadas en el análisis original**")
                st.dataframe(tabla, hide_index=True, use_container_width=True)
                st.download_button("Descargar tabla original (CSV)", csv.read_bytes(),
                                   file_name=csv.name, mime="text/csv", key=f"csv_{clave}")
            else:
                csv = None
        except Exception:
            st.warning("El archivo CSV de observaciones no pudo leerse.")
            csv = None
    if csv is None:
        st.markdown("**Información disponible en el GIF**")
        st.dataframe(pd.DataFrame({
            "Fotograma": list(range(1, info["cuadros"] + 1)),
            "Duración original (s)": [round(ms / 1000, 2)
                                      for ms in info["duraciones_ms"]],
            "Duración a esta velocidad (s)": [round(1 / velocidad, 2)] * info["cuadros"],
        }), hide_index=True, use_container_width=True)
        st.caption("Para mostrar la tabla científica con **fechas, ID de observación "
                   "y longitudes de onda**, necesitamos el CSV exportado por tu compañero. "
                   f"Puede subirlo como `demos/{clave}_observaciones.csv`. "
                   "No es posible recuperar esos metadatos con certeza solo desde el GIF.")

    st.info("Estas animaciones son resultados **ya procesados con observaciones reales de "
            "SPHEREx**. No se consulta IRSA para reproducirlas ni se afirma que "
            "constituyan una detección confirmada de variabilidad o movimiento.")
    st.stop()

if pagina == "Asteroides":
    # Modulo independiente: no modifica las busquedas del explorador RA/Dec.
    from spherex_asteroides import OBJETOS as ASTEROIDES, buscar_imagenes, procesar_imagenes, _gif

    st.header("☄️ Sigue un asteroide en imágenes de SPHEREx")
    st.write("Podemos buscar imágenes que hayan observado la zona por la que pasó "
             "un asteroide. NASA/IPAC MOST calcula su **posición prevista** en cada "
             "exposición; después descargamos los recortes FITS reales y los "
             "convertimos en una secuencia que puedes explorar.")
    st.caption("Esta es una inspección visual: el círculo indica la posición calculada, "
               "no que hayamos detectado automáticamente al asteroide.")

    a1, a2, a3 = st.columns([1.4, 1, 1])
    with a1:
        elegido = st.selectbox("Asteroide", list(ASTEROIDES), key="ast_objeto",
                               help="Vesta es el ejemplo oficial utilizado en el tutorial MOST de IRSA.")
    with a2:
        inicio_ast = st.date_input("Desde", date(2025, 8, 1),
                                   min_value=date(2025, 1, 1), max_value=date.today(),
                                   key="ast_inicio")
    with a3:
        fin_ast = st.date_input("Hasta", date(2025, 8, 31),
                                min_value=date(2025, 1, 1), max_value=date.today(),
                                key="ast_fin")
    if inicio_ast > fin_ast:
        st.error("La fecha inicial debe ser anterior a la final.")
        st.stop()
    if (fin_ast - inicio_ast).days > 90:
        st.warning("Para esta demostración, limita la búsqueda a un máximo de 90 días.")
        st.stop()

    @st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
    def _buscar_most_seguro(identificador, desde, hasta):
        return buscar_imagenes(identificador, desde, hasta)

    if st.button("🔎 Buscar imágenes del asteroide en IRSA", type="primary",
                 use_container_width=True):
        try:
            with st.spinner("Buscando exposiciones reales mediante IRSA MOST…"):
                df = _buscar_most_seguro(ASTEROIDES[elegido],
                                        inicio_ast.isoformat(), fin_ast.isoformat())
            st.session_state["ast_consulta"] = {
                "objeto": elegido, "periodo": (inicio_ast.isoformat(), fin_ast.isoformat()),
                "datos": df,
            }
            st.session_state.pop("ast_imagenes", None)
        except Exception as error:
            st.session_state.pop("ast_consulta", None)
            st.session_state.pop("ast_imagenes", None)
            st.warning("IRSA MOST no respondió a la búsqueda. Es posible que el servidor "
                       "esté temporalmente ocupado; puedes volver a intentarlo.")
            with st.expander("Ver detalle del error"):
                st.code(f"{type(error).__name__}: {error}")

    consulta_ast = st.session_state.get("ast_consulta")
    es_actual = (consulta_ast is not None
                 and consulta_ast["objeto"] == elegido
                 and consulta_ast["periodo"] == (inicio_ast.isoformat(), fin_ast.isoformat()))
    if not es_actual:
        st.info("Para empezar, prueba **(4) Vesta** entre el 1 y el 31 de agosto de 2025 "
                "y pulsa «Buscar imágenes». Si IRSA está caído, no se pueden "
                "fabricar resultados: la página mostrará el aviso correspondiente.")
        st.link_button("Abrir MOST en NASA/IPAC", "https://irsa.ipac.caltech.edu/applications/MOST/")
        st.stop()

    df_ast = consulta_ast["datos"]
    if df_ast.empty:
        st.info("MOST no encontró exposiciones que coincidan con la trayectoria "
                "prevista de este objeto en las fechas seleccionadas.")
        st.stop()
    st.metric("Exposiciones encontradas por MOST", len(df_ast))
    st.write("Estas exposiciones cubren la posición prevista del asteroide. "
             "Ahora puedes descargar algunas imágenes y compararlas.")

    disponibles_ast = sorted(set(df_ast["detector"].dropna().astype(str)))
    c1, c2, c3 = st.columns(3)
    with c1:
        det_ast = st.selectbox("Detector", ["Automático"] + disponibles_ast,
                               key="ast_detector",
                               help="Usamos un mismo detector para la secuencia.")
    with c2:
        n_ast = st.slider("Máximo de imágenes", 2, 6, 4, key="ast_numero")
    with c3:
        campo_ast = st.select_slider("Campo por imagen", options=[2, 3, 4, 5, 6, 8],
                                     value=5, format_func=lambda x: f"{x}′",
                                     key="ast_campo")

    @st.cache_data(ttl=2400, show_spinner=False, max_entries=5)
    def _obtener_secuencia(datos, nombre, detector, n, campo):
        return procesar_imagenes(datos, objeto=nombre, detector=detector,
                                max_imagenes=n, campo_arcmin=campo)

    firma_ast = (elegido, inicio_ast.isoformat(), fin_ast.isoformat(),
                 det_ast, int(n_ast), int(campo_ast))
    if st.button("🖼️ Mostrar imágenes y crear animación", type="primary",
                 use_container_width=True):
        try:
            with st.spinner("Descargando recortes FITS y preparando imágenes reales de SPHEREx…"):
                salida_ast = _obtener_secuencia(df_ast, elegido, det_ast, n_ast, campo_ast)
            st.session_state["ast_imagenes"] = {"firma": firma_ast, "resultado": salida_ast}
        except Exception as error:
            st.session_state.pop("ast_imagenes", None)
            st.warning("No pudimos recuperar los recortes de IRSA esta vez. "
                       "Puedes reintentar cuando el servidor esté disponible.")
            with st.expander("Ver detalle del error"):
                st.code(f"{type(error).__name__}: {error}")

    guardado_ast = st.session_state.get("ast_imagenes")
    if guardado_ast is not None and guardado_ast["firma"] == firma_ast:
        r = guardado_ast["resultado"]
        st.subheader("Así se ve el recorrido en SPHEREx")
        st.caption("En esta vista cada recorte se centra en **la posición prevista del asteroide**. "
                   "Por eso las estrellas del fondo pueden cambiar de posición. "
                   "El círculo cian no es una detección medida.")
        a, b, c = st.columns(3)
        a.metric("Imágenes descargadas", r["n_descargadas"])
        b.metric("Detector", str(r["tabla"]["Detector"].iloc[0]))
        c.metric("Exposiciones examinadas", r["n_seleccionadas"])

        pestañas = st.tabs(["Animación", "Imágenes individuales", "Trayectoria prevista"])
        with pestañas[0]:
            if r["gif_bytes"] is not None:
                velocidad_ast = st.slider("Velocidad (fotogramas por segundo)",
                                          0.5, 3.0, 1.0, 0.5, key="ast_velocidad")
                gif_ast = (r["gif_bytes"] if velocidad_ast == 1.0
                           else _gif(r["frames_png"], velocidad_ast))
                st.image(gif_ast, caption=(f"{elegido} · recortes FITS reales "
                                                "centrados en las posiciones previstas"),
                         use_container_width=True)
                st.download_button("⬇️ Descargar GIF de seguimiento", gif_ast,
                                   file_name="planeta9_asteroide_seguimiento.gif",
                                   mime="image/gif")
            else:
                st.info("Se recuperó una sola imagen válida: se muestra en la pestaña "
                        "de imágenes individuales. Necesitamos al menos dos para un GIF.")
        with pestañas[1]:
            indice_ast = st.slider("Observación", 1, len(r["frames_png"]), 1,
                                   key="ast_frame")
            st.image(r["frames_png"][indice_ast - 1],
                     caption=f"Exposición {indice_ast} de {len(r['frames_png'])}",
                     use_container_width=True)
            with st.expander("Ver todas las imágenes"):
                for k, png in enumerate(r["frames_png"], 1):
                    st.image(png, caption=f"Fotograma {k}", use_container_width=True)
        with pestañas[2]:
            fig = figura_posiciones(df_ast)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            st.caption("Este gráfico utiliza las efemérides de MOST, no centroides "
                       "obtenidos de los píxeles.")

        st.markdown("#### Información de las exposiciones")
        st.dataframe(r["tabla"], hide_index=True, use_container_width=True)
        st.download_button("Descargar tabla (CSV)",
                           r["tabla"].to_csv(index=False).encode("utf-8"),
                           file_name="planeta9_asteroides.csv", mime="text/csv")
        if r["banda_comparable"] is False:
            st.warning("Las longitudes de onda de estas exposiciones difieren demasiado "
                       "para comparar sus intensidades como si fueran la misma banda. "
                       "La secuencia es solo una visualización espacial.")
        elif r["banda_comparable"] is None:
            st.info("No se pudo verificar completamente la compatibilidad espectral; "
                    "la secuencia no demuestra cambios de brillo.")
        if r["errores"]:
            st.warning(f"Se omitieron {len(r['errores'])} recortes que no pudieron procesarse.")
            with st.expander("Detalles de las descargas omitidas"):
                for mensaje in r["errores"]:
                    st.caption(mensaje)

    else:
        with st.expander("Ver exposiciones encontradas (sin descargar FITS)"):
            vista_ast = df_ast[["fecha", "detector", "ra_deg", "dec_deg", "imagen_id"]].copy()
            vista_ast.columns = ["Fecha UTC", "Detector", "RA prevista (°)",
                                  "Dec prevista (°)", "Imagen"]
            st.dataframe(vista_ast, hide_index=True, use_container_width=True)

    st.divider()
    st.caption("Fuente: NASA/IPAC IRSA MOST + imágenes SPHEREx. Esta herramienta "
               "no confirma detecciones de asteroides ni mide sus órbitas. "
               "Los recortes están centrados en efemérides y no en un fondo fijo.")
    st.link_button("Abrir MOST directamente", "https://irsa.ipac.caltech.edu/applications/MOST/")
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
             "usamos MOST de IRSA para buscar las exposiciones de SPHEREx que cubren "
             "sus posiciones previstas y generar recortes reales centrados en ellas. "
             "Es un seguimiento visual por efemérides, no una detección confirmada.")
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
