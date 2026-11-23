
import os
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Dashboard EDA - Rotación de Personal (IBM HR)",
    layout="wide",
)

st.title("Análisis Exploratorio: Rotación de Empleados (IBM HR)")
st.write(
    "Dashboard interactivo construido a partir del EDA de la Tarea 3 "
    "para explorar los factores asociados a la rotación (Attrition)."
)


# ==========================================
# CARGA DE DATOS
# ==========================================

@st.cache_data
def cargar_datos(csv_path="datos.csv"):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    path = kagglehub.dataset_download(
        "pavansubhasht/ibm-hr-analytics-attrition-dataset"
    )

    csv_files = list(Path(path).rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No se encontró ningún archivo CSV en el dataset descargado."
        )

    return pd.read_csv(csv_files[0])


try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()


# ==========================================
# PREPARACIÓN DE DATOS
# ==========================================

df["MonthlyIncome"] = pd.to_numeric(
    df["MonthlyIncome"], errors="coerce"
)

df = df.dropna(subset=["MonthlyIncome", "Attrition"])

# ==========================================
# FILTROS INTERACTIVOS
# ==========================================

st.sidebar.title("Filtros del Dashboard")

attrition_opcion = st.sidebar.selectbox(
    "Estado de rotación",
    ["Todos", "Yes", "No"],
    index=0
)

ingreso_min = int(df["MonthlyIncome"].min())
ingreso_max = int(df["MonthlyIncome"].max())

rango_ingresos = st.sidebar.slider(
    "Rango de ingreso mensual",
    min_value=ingreso_min,
    max_value=ingreso_max,
    value=(ingreso_min, ingreso_max)
)

mostrar_correlacion = st.sidebar.checkbox(
    "Mostrar mapa de correlaciones",
    value=True
)

# Aplicar filtros
df_filtrado = df[
    (df["MonthlyIncome"] >= rango_ingresos[0]) &
    (df["MonthlyIncome"] <= rango_ingresos[1])
].copy()

if attrition_opcion != "Todos":
    df_filtrado = df_filtrado[
        df_filtrado["Attrition"] == attrition_opcion
    ]

if df_filtrado.empty:
    st.warning("No hay empleados que coincidan con los filtros seleccionados.")
    st.stop()


# ==========================================
# 1. RESUMEN GENERAL (KPIs)
# ==========================================

st.header("1. Resumen general")

col1, col2, col3, col4 = st.columns(4)

total_emp = len(df_filtrado)
tasa_rotacion = (
    (df_filtrado["Attrition"] == "Yes").mean() * 100
)
edad_prom = df_filtrado["Age"].mean()
ingreso_prom = df_filtrado["MonthlyIncome"].mean()

col1.metric("Total de empleados", f"{total_emp:,}")
col2.metric("Porcentaje de rotación", f"{tasa_rotacion:.2f}%")
col3.metric("Edad promedio", f"{edad_prom:.1f} años")
col4.metric("Ingreso mensual promedio", f"${ingreso_prom:,.0f}")


# ==========================================
# 2. EXPLORACIÓN DE DATOS
# ==========================================

st.header("2. Exploración de datos")

st.write(
    f"Mostrando **{len(df_filtrado):,}** empleados según los filtros "
    f"seleccionados (Rotación: **{attrition_opcion}**, "
    f"Ingreso mensual: **${rango_ingresos[0]:,} - "
    f"${rango_ingresos[1]:,}**)."
)

cols_muestra = [
    "EmployeeNumber",
    "Department",
    "JobRole",
    "Attrition",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "Age",
]

st.dataframe(
    df_filtrado[cols_muestra].head(50),
    use_container_width=True
)


# ==========================================
# 3. VISUALIZACIONES
# ==========================================

st.header("3. Visualizaciones")

# Gráfico 1: Boxplot de ingresos y rotación

st.subheader(
    "3.1 Ingreso mensual frente a rotación"
)

fig1, ax1 = plt.subplots(figsize=(8, 4.2))

sns.boxplot(
    data=df_filtrado,
    x="Attrition",
    y="MonthlyIncome",
    ax=ax1
)

ax1.set_title("Distribución del ingreso mensual según rotación")
ax1.set_xlabel("Rotación (Attrition)")
ax1.set_ylabel("Ingreso mensual")

st.pyplot(fig1)
plt.close(fig1)


# Gráfico 2: Antigüedad e ingresos

st.subheader(
    "3.2 Antigüedad frente a ingreso mensual por rotación"
)

fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))

sns.scatterplot(
    data=df_filtrado,
    x="YearsAtCompany",
    y="MonthlyIncome",
    hue="Attrition",
    alpha=0.7,
    ax=ax2
)

ax2.set_title("Antigüedad vs. ingreso mensual")
ax2.set_xlabel("Años en la empresa")
ax2.set_ylabel("Ingreso mensual")

st.pyplot(fig2)
plt.close(fig2)


# Gráfico 3: Satisfacción por departamento

st.subheader(
    "3.3 Satisfacción laboral por departamento y rotación"
)

df_sat = (
    df_filtrado.groupby(
        ["Department", "Attrition"], as_index=False
    )["JobSatisfaction"]
    .mean()
    .rename(
        columns={"JobSatisfaction": "AvgJobSatisfaction"}
    )
)

fig3, ax3 = plt.subplots(figsize=(9, 4.5))

sns.barplot(
    data=df_sat,
    x="Department",
    y="AvgJobSatisfaction",
    hue="Attrition",
    ax=ax3
)

ax3.set_title(
    "Satisfacción laboral promedio por departamento y rotación"
)
ax3.set_xlabel("Departamento")
ax3.set_ylabel("Satisfacción laboral promedio")

st.pyplot(fig3)
plt.close(fig3)


# Gráfico 4: Mapa de correlaciones

if mostrar_correlacion:

    st.subheader(
        "3.4 Mapa de correlaciones de variables numéricas"
    )

    vars_corr = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "JobSatisfaction",
        "WorkLifeBalance",
        "DistanceFromHome",
    ]

    cols_disponibles = [
        c for c in vars_corr
        if c in df_filtrado.select_dtypes(
            include=np.number
        ).columns
    ]

    if len(cols_disponibles) >= 2:

        corr_matrix = df_filtrado[
            cols_disponibles
        ].corr()

        fig4, ax4 = plt.subplots(figsize=(9, 6))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax4
        )

        ax4.set_title(
            "Matriz de correlación de variables numéricas clave"
        )

        st.pyplot(fig4)
        plt.close(fig4)


# ==========================================
# 4. VALIDACIÓN ESTADÍSTICA
# ==========================================

st.header("4. Validación estadística")

st.caption(
    "Los resultados estadísticos siguientes corresponden "
    "al análisis del conjunto original completo de la Tarea 3, "
    "no a los subconjuntos obtenidos mediante los filtros."
)

st.markdown("""
* **Homogeneidad de varianzas (Levene):** estadístico = **14.8996**,
  p-value = **0.000118**. Se rechaza la igualdad de varianzas.
* **Prueba t de Welch:** estadístico t = **7.4826**,
  p-value = **4.4336 × 10⁻¹³**.
* **Diferencia de medias:** quienes permanecen ganan en promedio
  **$2,045.65** más que quienes abandonan, con IC del 95 %
  entre **$1,508.24 y $2,583.05**.
* **Asimetría:** la media supera a la mediana en ambos grupos,
  evidenciando asimetría positiva en los ingresos.
* **Decisión (α = 0.05):** se rechaza la hipótesis nula de
  igualdad de medias de ingreso mensual.
""")

resumen_stat_df = pd.DataFrame({
    "Grupo": [
        "Permanece (No)",
        "Abandona (Yes)",
        "Diferencia (No - Yes)"
    ],
    "Media ($)": [
        "$6,832.74",
        "$4,787.09",
        "$2,045.65"
    ],
    "DE ($)": [
        "$4,818.21",
        "$3,640.21",
        "Levene p = 0.000118"
    ],
    "Mediana ($)": [
        "$5,204.00",
        "$3,202.00",
        "$2,002.00"
    ],
})

st.dataframe(
    resumen_stat_df,
    use_container_width=True
)


# ==========================================
# 5. HALLAZGOS Y RECOMENDACIONES
# ==========================================

st.header("5. Hallazgos y recomendaciones")

st.markdown("""
### Hallazgos clave

1. **Brecha salarial asociada a la rotación:**
   los empleados que abandonan la organización perciben ingresos
   mensuales inferiores en promedio en $2,045.65.

2. **Distribución sesgada hacia salarios altos:**
   la asimetría positiva de MonthlyIncome indica que un subgrupo
   de altos ingresos eleva los promedios.

3. **Antigüedad y satisfacción laboral:**
   cruzar YearsAtCompany, MonthlyIncome y JobSatisfaction permite
   explorar segmentos donde estos factores coinciden con mayor rotación.

### Recomendaciones para Recursos Humanos

* **Revisar la competitividad salarial:** auditar las bandas salariales
  en los roles y departamentos con mayor rotación.
* **Diseñar planes de retención temprana:** establecer oportunidades
  de crecimiento y revisión salarial durante los primeros años.
* **Activar planes por departamento:** focalizar acciones de liderazgo,
  reconocimiento y satisfacción laboral.
* **Complementar el análisis estadístico:** utilizar pruebas no
  paramétricas como Mann-Whitney U para contrastar medianas.

""")

st.divider()

st.caption(
    "Proyecto académico | Tarea 4 - Comunicación de resultados "
    "mediante dashboard interactivo | IBM HR Analytics | Grupo 29"
)
