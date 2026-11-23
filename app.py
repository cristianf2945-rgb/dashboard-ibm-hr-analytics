import os
import kagglehub
import pandas as pd
import scipy.stats as stats
import streamlit as st

st.set_page_config(
    page_title="Dashboard EDA - Rotación de Personal (IBM HR)",
    layout="wide",
)

st.title("Análisis Exploratorio: Rotación de Empleados (IBM HR)")
st.write(
    "Dashboard interactivo construido a partir del EDA de la Tarea 3 para explorar "
    "los factores asociados a la rotación (`Attrition`) en la organización."
)


@st.cache_data
def cargar_datos(csv_path: str = "datos.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    path = kagglehub.dataset_download(
        "pavansubhasht/ibm-hr-analytics-attrition-dataset"
    )
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            "No se encontró ningún archivo CSV en el dataset descargado."
        )

    df_original = pd.read_csv(os.path.join(path, csv_files[0]))
    df_original.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)


try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()


# ==========================================
# 1. RESUMEN GENERAL (KPIs)
# ==========================================
st.header("1. Resumen general")
col1, col2, col3, col4 = st.columns(4)

total_emp = len(df_filtrado)
tasa_rotacion = (df_filtrado["Attrition"] == "Yes").mean() * 100
edad_prom = df_filtrado["Age"].mean()
ingreso_prom = df_filtrado["MonthlyIncome"].mean()

col1.metric("Total de empleados", f"{total_emp:,}")
col2.metric("Porcentaje de rotación", f"{tasa_rotacion:.2f}%")
col3.metric("Edad promedio", f"{edad_prom:.1f} años")
col4.metric("Ingreso mensual promedio", f"${ingreso_prom:,.0f}")

# ==========================================
# 2. EXPLORACIÓN DE DATOS (TABLA)
# ==========================================
st.header("2. Exploración de datos")
st.write(
    f"Mostrando **{len(df_filtrado):,}** empleados según los filtros seleccionados "
    f"(Rotación: **{attrition_opcion}**, Ingreso mensual: **${rango_ingresos[0]:,} - ${rango_ingresos[1]:,}**)."
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
st.dataframe(df_filtrado[cols_muestra].head(50), use_container_width=True)

# ==========================================
# 3. VISUALIZACIONES (TAREA 3)
# ==========================================
st.header("3. Visualizaciones")

# Gráfico 1: Boxplot MonthlyIncome frente a Attrition
st.subheader("3.1 Ingreso mensual (MonthlyIncome) frente a rotación (Attrition)")
fig1, ax1 = plt.subplots(figsize=(8, 4.2))
sns.boxplot(data=df_filtrado, x="Attrition", y="MonthlyIncome", ax=ax1)
ax1.set_title("Distribución del ingreso mensual según rotación")
ax1.set_xlabel("Rotación (Attrition)")
ax1.set_ylabel("Ingreso mensual (MonthlyIncome)")
st.pyplot(fig1)
plt.close(fig1)

# Gráfico 2: Dispersión YearsAtCompany frente a MonthlyIncome por rotación
st.subheader(
    "3.2 Antigüedad (YearsAtCompany) frente a ingreso mensual por rotación"
)
fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
sns.scatterplot(
    data=df_filtrado,
    x="YearsAtCompany",
    y="MonthlyIncome",
    hue="Attrition",
    alpha=0.7,
    ax=ax2,
)
ax2.set_title("Antigüedad vs. Ingreso mensual diferenciando rotación")
ax2.set_xlabel("Años en la empresa (YearsAtCompany)")
ax2.set_ylabel("Ingreso mensual (MonthlyIncome)")
st.pyplot(fig2)
plt.close(fig2)

# Gráfico 3: Barras JobSatisfaction por Department y rotación
st.subheader("3.3 Satisfacción laboral por departamento y rotación")
df_sat = (
    df_filtrado.groupby(["Department", "Attrition"], as_index=False)[
        "JobSatisfaction"
    ]
    .mean()
    .rename(columns={"JobSatisfaction": "AvgJobSatisfaction"})
)
fig3, ax3 = plt.subplots(figsize=(9, 4.5))
sns.barplot(
    data=df_sat,
    x="Department",
    y="AvgJobSatisfaction",
    hue="Attrition",
    ax=ax3,
)
ax3.set_title("Satisfacción laboral promedio por departamento y rotación")
ax3.set_xlabel("Departamento (Department)")
ax3.set_ylabel("Satisfacción laboral promedio (JobSatisfaction)")
st.pyplot(fig3)
plt.close(fig3)

# Visualización complementaria: Mapa de correlaciones
if mostrar_correlacion:
    st.subheader("3.4 Mapa de correlaciones complementario (variables numéricas)")
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
        c for c in vars_corr if c in df_filtrado.select_dtypes(include=np.number).columns
    ]
    if len(cols_disponibles) >= 2:
        corr_matrix = df_filtrado[cols_disponibles].corr()
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax4,
        )
        ax4.set_title("Matriz de correlación de variables numéricas clave")
        st.pyplot(fig4)
        plt.close(fig4)

# ==========================================
# 4. VALIDACIÓN ESTADÍSTICA (TAREA 3)
# ==========================================
st.header("4. Validación estadística")

st.markdown("""
* **Homogeneidad de varianzas (Prueba de Levene):** estadístico = **14.8996**, *p*-value = **0.000118** ($< 0.05$). Se rechaza la igualdad de varianzas (*DE* = **$4,818.21** en quienes permanecen frente a **$3,640.21** en quienes abandonan), justificando el uso de la **prueba $t$ de Welch**.
* **Prueba $t$ de Welch (`MonthlyIncome` entre `No` y `Yes`):** estadístico $t$ = **7.4826**, *p*-value = **$4.4336 \\times 10^{-13}$** ($< 0.001$).
* **Magnitud e IC del 95 %:** quienes permanecen ganan en promedio **$2,045.65** más que quienes abandonan, con un intervalo de confianza del 95 % entre **$1,508.24** y **$2,583.05**.
* **Asimetría de los datos:** en ambos grupos la media supera claramente a la mediana (por ejemplo, **$6,832.74** vs. **$5,204.00** en `No`, con diferencia de medianas de **$2,002.00**), evidenciando asimetría positiva (cola derecha). Con $n = 1,470$ el Teorema del Límite Central respalda a Welch, y complementar con **Mann-Whitney U** ayuda a reforzar el contraste sobre medianas.
* **Decisión ($\alpha = 0.05$):** dado que $4.4336 \\times 10^{-13} \\ll 0.05$, se **rechaza categóricamente la hipótesis nula ($H_0$)**.
""")

resumen_stat_df = pd.DataFrame({
    "Grupo": ["Permanece (No)", "Abandona (Yes)", "Diferencia (No - Yes)"],
    "Media ($)": ["$6,832.74", "$4,787.09", "$2,045.65 (IC 95%: $1,508.24 - $2,583.05)"],
    "DE ($)": ["$4,818.21", "$3,640.21", "Levene p = 0.000118"],
    "Mediana ($)": ["$5,204.00", "$3,202.00", "$2,002.00"],
})
st.dataframe(resumen_stat_df, use_container_width=True)

# ==========================================
# 5. HALLAZGOS Y RECOMENDACIONES PARA RR. HH.
# ==========================================
st.header("5. Hallazgos y recomendaciones")

st.markdown("""
### Hallazgos clave
1. **Brecha salarial significativa asociada a la rotación:** los empleados que abandonan la organización perciben ingresos mensuales considerablemente menores en promedio (**$2,045.65** menos; *p* = $4.43 \\times 10^{-13}$) y con menor dispersión salarial que quienes permanecen.
2. **Distribución sesgada hacia salarios altos:** la asimetría positiva en `MonthlyIncome` indica que un subgrupo de altos ingresos eleva los promedios, mientras que el grueso de la plantilla —y especialmente quienes rotan— se concentra en rangos salariales más bajos.
3. **Antigüedad y satisfacción como señales de alerta:** al cruzar antigüedad (`YearsAtCompany`), ingreso mensual y satisfacción laboral (`JobSatisfaction`) por departamento (`Department`), es posible identificar segmentos donde la combinación de menor remuneración o menor satisfacción coincide con mayor rotación.

### Recomendaciones para Recursos Humanos
* **Revisar competitividad salarial en tramos bajos y medios:** auditar las bandas de `MonthlyIncome` en los roles y departamentos con mayor rotación frente a rangos salariales más bajos, priorizando ajustes focalizados donde existan brechas injustificadas.
* **Diseñar planes de retención temprana y progresión salarial:** establecer hitos claros de revisión salarial y crecimiento de carrera en los primeros años de antigüedad (`YearsAtCompany`) para reducir el riesgo de salida antes de consolidarse en la empresa.
* **Activar planes de acción por departamento según `JobSatisfaction`:** focalizar intervenciones de gestión, liderazgo y reconocimiento en los departamentos donde `JobSatisfaction` muestre promedios más bajos en el grupo que rota.
* **Complementar el seguimiento con pruebas no paramétricas (Mann-Whitney U):** dado el sesgo a la derecha en los ingresos, mantener el contraste de medianas como indicador de rutina para evitar que valores salariales extremos distorsionen la lectura ejecutiva.
""")
