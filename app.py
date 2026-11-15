"""Planeta 9 | SPHEREx Motion Explorer.

Interfaz Streamlit. Las consultas nuevas SIEMPRE ejecutan el motor científico
spherex_explorer.explorar_region; las demos son archivos de datos reales preparados
por el equipo, y no se presentan como búsquedas en vivo.
"""
from __future__ import annotations

import base64
import threading
from pathlib import Path

import streamlit as st

from spherex_explorer import explorar_region

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
            "Su movimiento propio es notable. La búsqueda libre usa coordenadas fijas: "
            "para seguir la estrella, utiliza la demostración preparada del equipo."
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
def _consultar(ra, dec, fov, detector, max_epocas):
    """Cachea bytes, no solo rutas efímeras de Streamlit Cloud."""
    with _bloqueo_graficos():
        res = explorar_region(ra=ra, dec=dec, fov_arcmin=fov,
                              detector=detector, max_epocas=max_epocas,
                              max_por_ventana=3, separacion_dias=15,
                              mostrar_progreso=False)
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
    st.radio("Menú", ["Explorar el cielo", "Objetos para probar", "El proyecto"],
             key="pagina", label_visibility="collapsed")
    st.divider()
    st.caption("Imágenes reales de SPHEREx · NASA/IPAC IRSA")
    st.caption("Este proyecto explora observaciones; no anuncia detecciones confirmadas.")

st.markdown('''<section class="hero-panel">
    <span class="team-tag">✦ PLANETA 9</span>
    <div class="hero-eyebrow">NASA SPACE APPS CHALLENGE · 2026</div>
    <div class="hero-title">El cielo cambia.<br><em>Vamos a explorarlo.</em></div>
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
    st.write("Para seguir un objeto que se desplaza por el sistema solar hay que calcular su "
             "posición en cada fecha. Es una extensión futura; por ahora las búsquedas libres "
             "utilizan coordenadas fijas.")
    st.link_button("Explorar objetos móviles en IRSA MOST", "https://irsa.ipac.caltech.edu/applications/MOST/")
    st.caption("Realizado por el equipo Planeta 9 para NASA Space Apps Challenge 2026. "
               "Datos: NASA/IPAC IRSA, colecciones SPHEREx DR1 + QR3.")
    st.stop()

st.header("01 / Elige una región")
st.write("Prueba con un objeto conocido o introduce coordenadas nuevas. "
         "Si tenemos una animación preparada, verás el GIF original al elegir el objeto; "
         "para crear una animación nueva, usa el botón de consulta.")
st.selectbox("Objeto de referencia", [CUSTOM, *OBJETOS.keys()], key="ejemplo",
             on_change=_actualizar_ejemplo,
             help="Los nombres sirven para rellenar las coordenadas; la búsqueda siempre usa RA y Dec.")
seleccionado = OBJETOS.get(st.session_state.ejemplo)
if seleccionado:
    st.caption(seleccionado["descripcion"])

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

# Al seleccionar un objeto del catálogo, su GIF original (si fue subido a
# demos/) se muestra automáticamente en la zona de resultados. El botón de
# búsqueda SIEMPRE lanza nuestra consulta en vivo mediante explorar_region().
ruta_demo = _buscar_archivo(DEMO_GIF.get(seleccionado["id"], [])) if seleccionado else None
consultar = st.button("🔭 Explorar esta región en IRSA", type="primary", use_container_width=True)

if consultar:
    ra_elegida, dec_elegida = float(st.session_state.ra), float(st.session_state.dec)
    try:
        with st.spinner("Buscando observaciones en IRSA y preparando la comparación…"):
            resultado = _consultar(ra_elegida, dec_elegida,
                                  float(st.session_state.fov), st.session_state.detector,
                                  max_epocas)
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
# Vista previa automática, igual que en el buscador original: el usuario
# selecciona el objeto y aparece su GIF previamente preparado. No se confunde
# con los resultados de la consulta en vivo ni depende del detector elegido.
if resultado_actual is None and not consultar and ruta_demo is not None and seleccionado:
    resultado_actual = {
        "tipo": "demo", "nombre": seleccionado["id"],
        "titulo": st.session_state.ejemplo,
        "gif": ruta_demo.read_bytes(),
    }
if resultado_actual is None:
    st.divider()
    st.info("Escribe unas coordenadas y pulsa «Explorar esta región en IRSA» para crear una animación. "
            "Si eliges un objeto con un GIF ya preparado por el equipo, aparecerá aquí automáticamente.")
    st.stop()

st.divider()
st.header("02 / El cielo en movimiento")
if resultado_actual["tipo"] == "demo":
    st.image(resultado_actual["gif"], caption=f"{resultado_actual['titulo']} · Trabajo del equipo Planeta 9",
             use_container_width=True)
    st.download_button("Descargar GIF", resultado_actual["gif"],
                       file_name=f"{resultado_actual['nombre']}.gif", mime="image/gif")
    st.caption("Animación preparada por el equipo a partir de datos reales de SPHEREx. "
               "No es una consulta en vivo; si quieres generar otra secuencia para estas "
               "coordenadas, pulsa «Explorar esta región en IRSA» arriba. "
               "El detector y otros controles corresponden a la búsqueda en vivo.")
    figuras = DEMO_FIGURAS.get(resultado_actual["nombre"], {})
    disponibles = [(nombre, ruta) for nombre, opciones in figuras.items()
                   if (ruta := _buscar_archivo(opciones)) is not None]
    if disponibles:
        st.header("03 / Otras vistas de este objeto")
        tabs = st.tabs([nombre for nombre, _ in disponibles])
        for tab, (nombre, ruta) in zip(tabs, disponibles):
            with tab:
                st.image(str(ruta), caption=f"{nombre} · Análisis preparado por el equipo",
                         use_container_width=True)
    st.stop()

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
