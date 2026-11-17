import io
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Análisis Estadístico Descriptivo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Análisis Estadístico Descriptivo")
st.caption("Dashboard web basado en el notebook de Google Colab: descriptivo_pruebas_un_solo_tablero.ipynb")

# -----------------------------
# Carga de datos
# -----------------------------
st.sidebar.header("📁 Datos")
uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo CSV",
    type=["csv"],
    help="Seleccione el archivo CSV que desea analizar."
)

if uploaded_file is None:
    st.info(
        "Cargue un archivo CSV desde la barra lateral para iniciar el análisis."
    )
    st.markdown(
        """
        ### Estructura esperada
        El dashboard reproduce la lógica del notebook original:
        - elimina la columna `NaN` si existe;
        - calcula estadísticas descriptivas;
        - permite seleccionar dos columnas;
        - genera automáticamente una visualización según el tipo de variables.
        """
    )
    st.stop()

# Intento robusto de lectura del CSV
try:
    descriptivo = pd.read_csv(uploaded_file, sep=",")
except Exception as e:
    st.error(f"No fue posible leer el CSV: {e}")
    st.stop()

# Mantener la lógica del notebook
if "NaN" in descriptivo.columns:
    descriptivo = descriptivo.drop(columns=["NaN"])

if descriptivo.empty:
    st.warning("El archivo CSV no contiene registros después de la limpieza.")
    st.stop()

# -----------------------------
# Resumen
# -----------------------------
st.subheader("📌 Resumen del conjunto de datos")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{descriptivo.shape[0]:,}")
c2.metric("Variables", f"{descriptivo.shape[1]:,}")
c3.metric("Valores faltantes", f"{int(descriptivo.isna().sum().sum()):,}")
c4.metric("Memoria aprox.", f"{descriptivo.memory_usage(deep=True).sum()/1024:.1f} KB")

# -----------------------------
# Vista de datos
# -----------------------------
with st.expander("🔎 Ver datos cargados", expanded=False):
    st.dataframe(descriptivo, use_container_width=True, height=320)

# -----------------------------
# Estadísticas descriptivas
# -----------------------------
st.subheader("📋 Estadísticas descriptivas")

descriptive_stats = descriptivo.describe(include="all").T
st.dataframe(descriptive_stats, use_container_width=True)

# Descarga de estadísticas
csv_stats = descriptive_stats.to_csv(index=True).encode("utf-8")
st.download_button(
    "⬇️ Descargar estadísticas en CSV",
    data=csv_stats,
    file_name="descriptivo_estadisticas.csv",
    mime="text/csv",
)

# -----------------------------
# Selección de variables
# -----------------------------
st.subheader("📈 Análisis gráfico")

col1, col2 = st.columns(2)

with col1:
    selected_col1 = st.selectbox(
        "Columna 1:",
        options=descriptivo.columns.tolist(),
        index=0,
    )

with col2:
    selected_col2 = st.selectbox(
        "Columna 2:",
        options=descriptivo.columns.tolist(),
        index=0,
    )

# La lógica sigue la heurística del notebook:
# columnas enteras con <= 20 valores únicos = categóricas.
def is_categorical(series):
    return (
        pd.api.types.is_integer_dtype(series)
        and series.nunique(dropna=True) <= 20
    )

is_categorical_col1 = is_categorical(descriptivo[selected_col1])
is_categorical_col2 = is_categorical(descriptivo[selected_col2])

# -----------------------------
# Visualización automática
# -----------------------------
if selected_col1 == selected_col2:
    if is_categorical_col1:
        counts = (
            descriptivo[selected_col1]
            .value_counts(dropna=False)
            .rename_axis(selected_col1)
            .reset_index(name="Conteo")
        )
        fig = px.bar(
            counts,
            x=selected_col1,
            y="Conteo",
            title=f"Distribución de {selected_col1}",
            text_auto=True,
        )
        fig.update_layout(xaxis_title=selected_col1, yaxis_title="Conteo")
    else:
        fig = px.histogram(
            descriptivo,
            x=selected_col1,
            marginal="box",
            title=f"Distribución de {selected_col1}",
        )
        fig.update_layout(xaxis_title=selected_col1, yaxis_title="Frecuencia")

elif is_categorical_col1 and is_categorical_col2:
    cross_tab = pd.crosstab(
        descriptivo[selected_col1],
        descriptivo[selected_col2]
    )
    cross_tab.index = cross_tab.index.astype(str)
    cross_tab.columns = cross_tab.columns.astype(str)

    plot_df = cross_tab.reset_index().melt(
        id_vars=selected_col1,
        var_name=selected_col2,
        value_name="Conteo",
    )

    fig = px.bar(
        plot_df,
        x=selected_col1,
        y="Conteo",
        color=selected_col2,
        barmode="stack",
        title=f"Conteo de {selected_col2} por {selected_col1}",
        text_auto=True,
    )
    fig.update_layout(xaxis_title=selected_col1, yaxis_title="Conteo")

elif is_categorical_col1 and not is_categorical_col2:
    fig = px.box(
        descriptivo,
        x=selected_col1,
        y=selected_col2,
        points="all",
        title=f"{selected_col2} por {selected_col1}",
    )
    fig.update_layout(xaxis_title=selected_col1, yaxis_title=selected_col2)

elif not is_categorical_col1 and is_categorical_col2:
    fig = px.box(
        descriptivo,
        x=selected_col2,
        y=selected_col1,
        points="all",
        title=f"{selected_col1} por {selected_col2}",
    )
    fig.update_layout(xaxis_title=selected_col2, yaxis_title=selected_col1)

else:
    fig = px.scatter(
        descriptivo,
        x=selected_col1,
        y=selected_col2,
        title=f"Relación entre {selected_col1} y {selected_col2}",
        trendline=None,
    )
    fig.update_layout(xaxis_title=selected_col1, yaxis_title=selected_col2)

fig.update_layout(
    height=560,
    margin=dict(l=30, r=30, t=70, b=30),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Información de la selección
# -----------------------------
with st.expander("ℹ️ Clasificación de variables"):
    st.write(
        f"**{selected_col1}**: "
        f"{'categórica según la heurística del notebook' if is_categorical_col1 else 'numérica según la heurística del notebook'}"
    )
    st.write(
        f"**{selected_col2}**: "
        f"{'categórica según la heurística del notebook' if is_categorical_col2 else 'numérica según la heurística del notebook'}"
    )

st.divider()
st.caption(
    "Versión web adaptada del notebook de análisis descriptivo. "
    "La aplicación no depende de Google Colab ni de Google Drive."
)
