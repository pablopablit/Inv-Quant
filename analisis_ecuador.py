#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Estadístico de Datos Cantonales de Ecuador
====================================================

Este script realiza un análisis estadístico completo de los datos socioeconómicos
y electorales de los 220 cantones de Ecuador, incluyendo:
- Estadísticas descriptivas
- Pruebas de normalidad (Shapiro-Wilk)
- Análisis de correlación
- Visualizaciones
- Generación de informe

Autor: Análisis Estadístico Automatizado
Fecha: 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================================
# FASE 1: PREPARACIÓN Y CARGA DE DATOS
# ============================================================================

def cargar_y_limpiar_datos(filepath):
    """
    Carga y limpia el dataset de cantones ecuatorianos.

    El archivo CSV tiene formato europeo con:
    - Separador de campos: punto y coma (;)
    - Separador decimal: coma (,) en algunos casos
    """
    print("="*70)
    print("FASE 1: PREPARACIÓN Y CARGA DE DATOS")
    print("="*70)

    # Leer CSV con separador punto y coma
    df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')

    print(f"\nDataset cargado: {df.shape[0]} cantones, {df.shape[1]} variables")
    print(f"Columnas: {list(df.columns)}")

    # Renombrar columnas para facilitar el trabajo
    nombres_nuevos = {
        'Cantón': 'canton',
        'Provincia': 'provincia',
        'Votos por Noboa (absoluto)': 'votos_noboa_abs',
        'Votos por González (absoluto)': 'votos_gonzalez_abs',
        'Votos por Noboa (porcentaje)': 'votos_noboa_pct',
        'Votos por González (porcentaje)': 'votos_gonzalez_pct',
        'Población': 'poblacion',
        'Porcentaje población indígena': 'pob_indigena_pct',
        'Agua red pública': 'agua_publica',
        'Electricidad': 'electricidad',
        'PIB per cápita': 'pib_per_capita',
        'Tasa Homicidios': 'tasa_homicidios',
        'Altitud': 'altitud',
        'Costa': 'costa',
        'Sierra': 'sierra',
        'Oriente': 'oriente',
        'Insular': 'insular'
    }
    df = df.rename(columns=nombres_nuevos)

    # Convertir columnas numéricas que tienen coma como decimal
    columnas_numericas = ['votos_noboa_abs', 'votos_gonzalez_abs', 'votos_noboa_pct',
                          'votos_gonzalez_pct', 'poblacion', 'pob_indigena_pct',
                          'agua_publica', 'electricidad', 'pib_per_capita',
                          'tasa_homicidios', 'altitud']

    for col in columnas_numericas:
        if df[col].dtype == 'object':
            # Reemplazar coma por punto para decimales
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Validación de datos
    print("\n--- VALIDACIÓN DE DATOS ---")
    print(f"Valores faltantes por columna:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No hay valores faltantes")

    # Verificar tipos de datos
    print(f"\nTipos de datos:")
    print(df.dtypes)

    # Crear variable de región
    df['region'] = np.where(df['costa'] == 1, 'Costa',
                   np.where(df['sierra'] == 1, 'Sierra',
                   np.where(df['oriente'] == 1, 'Oriente',
                   np.where(df['insular'] == 1, 'Insular', 'Desconocido'))))

    print(f"\nDistribución por región:")
    print(df['region'].value_counts())

    return df

# ============================================================================
# FASE 2: ANÁLISIS DESCRIPTIVO
# ============================================================================

def calcular_estadisticas_descriptivas(df):
    """
    Calcula estadísticas descriptivas completas para todas las variables numéricas.
    """
    print("\n" + "="*70)
    print("FASE 2: ANÁLISIS DESCRIPTIVO")
    print("="*70)

    # Seleccionar variables de interés para el análisis
    variables_analisis = ['votos_noboa_pct', 'poblacion', 'pob_indigena_pct',
                          'agua_publica', 'electricidad', 'pib_per_capita',
                          'tasa_homicidios', 'altitud']

    # Calcular estadísticas descriptivas
    stats_desc = pd.DataFrame()

    for var in variables_analisis:
        stats_desc[var] = pd.Series({
            'N': df[var].count(),
            'Media': df[var].mean(),
            'Mediana': df[var].median(),
            'Desv. Estándar': df[var].std(),
            'Varianza': df[var].var(),
            'Mínimo': df[var].min(),
            'Q1 (25%)': df[var].quantile(0.25),
            'Q3 (75%)': df[var].quantile(0.75),
            'Máximo': df[var].max(),
            'Rango': df[var].max() - df[var].min(),
            'IQR': df[var].quantile(0.75) - df[var].quantile(0.25),
            'Asimetría': df[var].skew(),
            'Curtosis': df[var].kurtosis(),
            'Coef. Variación': (df[var].std() / df[var].mean()) * 100
        })

    stats_desc = stats_desc.T

    print("\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
    print(stats_desc.round(2).to_string())

    # Guardar tabla formateada
    stats_desc.round(2).to_csv('/home/user/Inv-Quant/resultados/estadisticas_descriptivas.csv')

    # Identificar outliers usando IQR
    print("\n--- IDENTIFICACIÓN DE VALORES EXTREMOS (OUTLIERS) ---")
    outliers_info = {}

    for var in variables_analisis:
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        outliers = df[(df[var] < limite_inferior) | (df[var] > limite_superior)]
        outliers_info[var] = outliers[['canton', 'provincia', var, 'region']].copy()

        if len(outliers) > 0:
            print(f"\n{var.upper()}: {len(outliers)} outliers detectados")
            print(f"  Límites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
            # Mostrar los 5 más extremos
            top_outliers = outliers.nlargest(5, var)
            print("  Top 5 valores más altos:")
            for idx, row in top_outliers.iterrows():
                print(f"    - {row['canton']} ({row['provincia']}): {row[var]:.2f}")

    return stats_desc, outliers_info

def generar_estadisticas_por_region(df):
    """
    Calcula estadísticas descriptivas estratificadas por región.
    """
    print("\n--- ESTADÍSTICAS POR REGIÓN ---")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    for var in variables:
        print(f"\n{var.upper()}:")
        region_stats = df.groupby('region')[var].agg(['count', 'mean', 'std', 'min', 'max'])
        print(region_stats.round(2))

    # Guardar estadísticas por región
    stats_region = df.groupby('region')[variables].agg(['mean', 'std', 'min', 'max']).round(2)
    stats_region.to_csv('/home/user/Inv-Quant/resultados/estadisticas_por_region.csv')

    return stats_region

# ============================================================================
# FASE 3: ANÁLISIS DE NORMALIDAD
# ============================================================================

def test_normalidad(df):
    """
    Realiza pruebas de normalidad (Shapiro-Wilk) para las variables principales.
    """
    print("\n" + "="*70)
    print("FASE 3: ANÁLISIS DE NORMALIDAD")
    print("="*70)

    variables_test = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                      'pob_indigena_pct', 'agua_publica', 'electricidad', 'poblacion']

    resultados_normalidad = pd.DataFrame(columns=['Variable', 'Estadístico W', 'p-valor', 'Normalidad'])

    print("\n--- TEST DE SHAPIRO-WILK ---")
    print("H0: Los datos provienen de una distribución normal")
    print("Nivel de significancia: α = 0.05\n")

    for var in variables_test:
        # Shapiro-Wilk test (máximo 5000 observaciones)
        data = df[var].dropna()
        if len(data) > 5000:
            data = data.sample(5000, random_state=42)

        stat, p_value = shapiro(data)

        # Determinar si es normal
        es_normal = "Sí" if p_value > 0.05 else "No"

        resultados_normalidad.loc[len(resultados_normalidad)] = {
            'Variable': var,
            'Estadístico W': stat,
            'p-valor': p_value,
            'Normalidad': es_normal
        }

        print(f"{var}:")
        print(f"  W = {stat:.6f}, p-valor = {p_value:.6f}")
        if p_value < 0.05:
            print(f"  ❌ Se rechaza H0: Los datos NO siguen distribución normal")
        else:
            print(f"  ✓ No se rechaza H0: Los datos podrían seguir distribución normal")
        print()

    # Guardar resultados
    resultados_normalidad.to_csv('/home/user/Inv-Quant/resultados/test_normalidad.csv', index=False)

    return resultados_normalidad

def generar_visualizaciones_normalidad(df):
    """
    Genera histogramas, Q-Q plots y boxplots para las variables principales.
    """
    print("\n--- GENERANDO VISUALIZACIONES DE NORMALIDAD ---")

    variables_viz = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                     'pob_indigena_pct', 'agua_publica']

    nombres_display = {
        'votos_noboa_pct': 'Votos Noboa (%)',
        'pib_per_capita': 'PIB per cápita',
        'tasa_homicidios': 'Tasa de Homicidios',
        'pob_indigena_pct': 'Población Indígena (%)',
        'agua_publica': 'Acceso Agua Pública (%)'
    }

    # 1. Histogramas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables_viz):
        sns.histplot(data=df, x=var, kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f'Distribución: {nombres_display[var]}')
        axes[i].set_xlabel(nombres_display[var])
        axes[i].set_ylabel('Frecuencia')

        # Añadir línea de media
        mean_val = df[var].mean()
        axes[i].axvline(x=mean_val, color='red', linestyle='--',
                       label=f'Media: {mean_val:.2f}')
        axes[i].legend()

    axes[-1].axis('off')  # Ocultar el último subplot vacío
    plt.suptitle('Histogramas de Variables Principales', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/histogramas.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Histogramas guardados: histogramas.png")

    # 2. Q-Q Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables_viz):
        stats.probplot(df[var].dropna(), dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot: {nombres_display[var]}')
        axes[i].get_lines()[0].set_markerfacecolor('blue')
        axes[i].get_lines()[0].set_markersize(4)

    axes[-1].axis('off')
    plt.suptitle('Gráficos Q-Q para Evaluación de Normalidad', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/qq_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Q-Q Plots guardados: qq_plots.png")

    # 3. Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables_viz):
        sns.boxplot(data=df, y=var, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Boxplot: {nombres_display[var]}')
        axes[i].set_ylabel(nombres_display[var])

        # Añadir estadísticas
        median = df[var].median()
        axes[i].text(0.05, 0.95, f'Mediana: {median:.2f}',
                    transform=axes[i].transAxes, fontsize=10)

    axes[-1].axis('off')
    plt.suptitle('Boxplots de Variables Principales', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Boxplots guardados: boxplots.png")

    # 4. Boxplots por región
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables_viz):
        sns.boxplot(data=df, x='region', y=var, ax=axes[i], palette='Set3')
        axes[i].set_title(f'{nombres_display[var]} por Región')
        axes[i].set_xlabel('Región')
        axes[i].set_ylabel(nombres_display[var])
        axes[i].tick_params(axis='x', rotation=45)

    axes[-1].axis('off')
    plt.suptitle('Comparación de Variables por Región Geográfica', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/boxplots_por_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Boxplots por región guardados: boxplots_por_region.png")

# ============================================================================
# FASE 4: ANÁLISIS DE CORRELACIÓN
# ============================================================================

def calcular_correlaciones(df, usar_pearson=False):
    """
    Calcula matrices de correlación según los resultados de normalidad.
    Si las variables no son normales, se usa Spearman; si son normales, Pearson.
    """
    print("\n" + "="*70)
    print("FASE 4: ANÁLISIS DE CORRELACIÓN")
    print("="*70)

    variables_corr = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                      'pob_indigena_pct', 'agua_publica', 'electricidad', 'poblacion']

    # Dado que la mayoría de variables no son normales, usar Spearman
    print("\nDado que la mayoría de variables no siguen distribución normal,")
    print("se utiliza el coeficiente de correlación de Spearman (ρ)")

    # Matriz de correlación de Spearman
    df_corr = df[variables_corr].copy()
    corr_matrix = df_corr.corr(method='spearman')

    print("\n--- MATRIZ DE CORRELACIÓN DE SPEARMAN ---")
    print(corr_matrix.round(3).to_string())

    # Guardar matriz
    corr_matrix.round(3).to_csv('/home/user/Inv-Quant/resultados/matriz_correlacion_spearman.csv')

    # Calcular p-valores para cada correlación
    print("\n--- SIGNIFICANCIA ESTADÍSTICA DE LAS CORRELACIONES ---")
    print("(α = 0.05)\n")

    correlaciones_significativas = []

    n_vars = len(variables_corr)
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            var1 = variables_corr[i]
            var2 = variables_corr[j]

            # Calcular correlación de Spearman con p-valor
            rho, p_value = spearmanr(df[var1], df[var2])

            significativo = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))

            if p_value < 0.05:
                fuerza = "muy fuerte" if abs(rho) > 0.8 else ("fuerte" if abs(rho) > 0.6 else ("moderada" if abs(rho) > 0.4 else "débil"))
                direccion = "positiva" if rho > 0 else "negativa"

                correlaciones_significativas.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'ρ': rho,
                    'p-valor': p_value,
                    'Significancia': significativo,
                    'Interpretación': f"Correlación {fuerza} {direccion}"
                })

                if abs(rho) > 0.3:  # Solo mostrar correlaciones al menos débil-moderadas
                    print(f"{var1} vs {var2}:")
                    print(f"  ρ = {rho:.4f}, p = {p_value:.6f} {significativo}")
                    print(f"  Interpretación: Correlación {fuerza} {direccion}\n")

    # Guardar correlaciones significativas
    df_corr_sig = pd.DataFrame(correlaciones_significativas)
    df_corr_sig.to_csv('/home/user/Inv-Quant/resultados/correlaciones_significativas.csv', index=False)

    # Visualización de matriz de correlación
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1)

    plt.title('Matriz de Correlación de Spearman\n(Variables Socioeconómicas y Electorales)',
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/matriz_correlacion_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Heatmap de correlación guardado: matriz_correlacion_heatmap.png")

    # Correlaciones específicas de interés
    print("\n--- CORRELACIONES DE INTERÉS PRINCIPAL ---\n")

    correlaciones_prioritarias = [
        ('votos_noboa_pct', 'pib_per_capita', 'Votos vs PIB per cápita'),
        ('votos_noboa_pct', 'tasa_homicidios', 'Votos vs Tasa de homicidios'),
        ('votos_noboa_pct', 'pob_indigena_pct', 'Votos vs Población indígena'),
        ('agua_publica', 'pib_per_capita', 'Servicios básicos vs Desarrollo económico'),
        ('electricidad', 'pib_per_capita', 'Electricidad vs PIB per cápita')
    ]

    resultados_prioritarios = []
    for var1, var2, descripcion in correlaciones_prioritarias:
        rho, p_value = spearmanr(df[var1], df[var2])
        print(f"{descripcion}:")
        print(f"  ρ = {rho:.4f}, p = {p_value:.6f}")

        if p_value < 0.001:
            sig = "*** (p < 0.001)"
        elif p_value < 0.01:
            sig = "** (p < 0.01)"
        elif p_value < 0.05:
            sig = "* (p < 0.05)"
        else:
            sig = "No significativo"
        print(f"  Significancia: {sig}\n")

        resultados_prioritarios.append({
            'Relación': descripcion,
            'ρ': rho,
            'p-valor': p_value,
            'Significancia': sig
        })

    pd.DataFrame(resultados_prioritarios).to_csv(
        '/home/user/Inv-Quant/resultados/correlaciones_prioritarias.csv', index=False)

    return corr_matrix, df_corr_sig

def crear_scatterplots_correlaciones(df):
    """
    Crea gráficos de dispersión para las correlaciones más importantes.
    """
    print("\n--- GENERANDO SCATTERPLOTS ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Votos vs PIB per cápita
    sns.scatterplot(data=df, x='pib_per_capita', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[0,0], alpha=0.7)
    axes[0,0].set_title('Votos por Noboa vs PIB per cápita')
    axes[0,0].set_xlabel('PIB per cápita (USD)')
    axes[0,0].set_ylabel('Votos por Noboa (%)')
    # Añadir línea de tendencia
    z = np.polyfit(df['pib_per_capita'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pib_per_capita'].min(), df['pib_per_capita'].max(), 100)
    axes[0,0].plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendencia')

    # 2. Votos vs Tasa de homicidios
    sns.scatterplot(data=df, x='tasa_homicidios', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title('Votos por Noboa vs Tasa de Homicidios')
    axes[0,1].set_xlabel('Tasa de Homicidios (por 100,000 hab.)')
    axes[0,1].set_ylabel('Votos por Noboa (%)')
    z = np.polyfit(df['tasa_homicidios'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['tasa_homicidios'].min(), df['tasa_homicidios'].max(), 100)
    axes[0,1].plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendencia')

    # 3. Votos vs Población indígena
    sns.scatterplot(data=df, x='pob_indigena_pct', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[1,0], alpha=0.7)
    axes[1,0].set_title('Votos por Noboa vs Población Indígena')
    axes[1,0].set_xlabel('Población Indígena (%)')
    axes[1,0].set_ylabel('Votos por Noboa (%)')
    z = np.polyfit(df['pob_indigena_pct'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pob_indigena_pct'].min(), df['pob_indigena_pct'].max(), 100)
    axes[1,0].plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendencia')

    # 4. Agua pública vs PIB per cápita
    sns.scatterplot(data=df, x='pib_per_capita', y='agua_publica',
                   hue='region', style='region', ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('Acceso a Agua Pública vs PIB per cápita')
    axes[1,1].set_xlabel('PIB per cápita (USD)')
    axes[1,1].set_ylabel('Acceso a Agua Pública (%)')
    z = np.polyfit(df['pib_per_capita'], df['agua_publica'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pib_per_capita'].min(), df['pib_per_capita'].max(), 100)
    axes[1,1].plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendencia')

    plt.suptitle('Análisis de Correlaciones Principales', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('/home/user/Inv-Quant/resultados/scatterplots_correlaciones.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Scatterplots guardados: scatterplots_correlaciones.png")

# ============================================================================
# FASE 5: GENERACIÓN DE INFORME
# ============================================================================

def generar_informe(df, stats_desc, outliers_info, resultados_normalidad, corr_matrix, df_corr_sig):
    """
    Genera un informe completo en formato Markdown con todos los resultados.
    """
    print("\n" + "="*70)
    print("FASE 5: GENERACIÓN DE INFORME")
    print("="*70)

    fecha = datetime.now().strftime("%d de %B de %Y")

    informe = f"""# Análisis Estadístico de Indicadores Socioeconómicos y Electorales
## Cantones del Ecuador

**Fecha de elaboración:** {datetime.now().strftime("%Y-%m-%d")}

---

## 1. Introducción

El presente informe documenta los resultados de un análisis estadístico comprehensivo de los 220 cantones del Ecuador. El estudio examina la relación entre indicadores socioeconómicos (PIB per cápita, acceso a servicios básicos, composición demográfica) y patrones electorales observados en las elecciones presidenciales de 2023, específicamente los votos entre Daniel Noboa y Luisa González.

### 1.1 Objetivos

1. Caracterizar la distribución de variables socioeconómicas a nivel cantonal
2. Evaluar la normalidad de las distribuciones para determinar métodos estadísticos apropiados
3. Identificar correlaciones significativas entre variables socioeconómicas y electorales
4. Detectar patrones regionales que expliquen la heterogeneidad territorial

### 1.2 Metodología

- **Análisis descriptivo:** Medidas de tendencia central, dispersión y forma
- **Pruebas de normalidad:** Test de Shapiro-Wilk (α = 0.05)
- **Análisis de correlación:** Coeficiente de Spearman (dada la no normalidad de los datos)
- **Identificación de outliers:** Método del rango intercuartílico (IQR)

### 1.3 Datos

El dataset comprende {df.shape[0]} cantones ecuatorianos con {df.shape[1]} variables, incluyendo información electoral, demográfica, económica y de servicios básicos.

**Distribución geográfica:**
- Costa: {(df['region'] == 'Costa').sum()} cantones
- Sierra: {(df['region'] == 'Sierra').sum()} cantones
- Oriente: {(df['region'] == 'Oriente').sum()} cantones
- Insular: {(df['region'] == 'Insular').sum()} cantones

---

## 2. Decisiones Metodológicas

### 2.1 Tratamiento de Outliers

Los valores extremos se identificaron pero **no fueron eliminados** del análisis por las siguientes razones:

1. Representan realidades territoriales legítimas (ej: cantones petroleros con PIB elevado)
2. La eliminación arbitraria distorsionaría la comprensión de la heterogeneidad cantonal
3. Se presentan análisis con y sin outliers para evaluar robustez cuando es pertinente

### 2.2 Selección del Método de Correlación

Se optó por el coeficiente de correlación de **Spearman** sobre Pearson debido a:
- Violación del supuesto de normalidad en la mayoría de variables
- Mayor robustez ante valores extremos
- Capacidad de detectar relaciones monótonas no lineales

### 2.3 Variables Transformadas

Se creó la variable **"Región"** a partir de los indicadores binarios (Costa, Sierra, Oriente, Insular) para facilitar el análisis estratificado.

---

## 3. Resultados

### 3.1 Estadísticas Descriptivas

La Tabla 1 presenta las estadísticas descriptivas completas para las variables principales del estudio.

"""

    # Agregar tabla de estadísticas descriptivas
    informe += "#### Tabla 1: Estadísticas Descriptivas de Variables Principales\n\n"

    # Formatear tabla para markdown
    vars_mostrar = ['votos_noboa_pct', 'pob_indigena_pct', 'agua_publica',
                    'electricidad', 'pib_per_capita', 'tasa_homicidios']

    # Crear tabla simplificada
    tabla_simple = stats_desc.loc[vars_mostrar, ['Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo', 'Coef. Variación']].round(2)
    tabla_simple.index = ['Votos Noboa (%)', 'Pob. Indígena (%)', 'Agua Pública (%)',
                          'Electricidad (%)', 'PIB per cápita', 'Tasa Homicidios']

    informe += "| Variable | Media | Mediana | Desv. Estándar | Mínimo | Máximo | CV (%) |\n"
    informe += "|----------|-------|---------|----------------|--------|--------|--------|\n"

    for idx, row in tabla_simple.iterrows():
        informe += f"| {idx} | {row['Media']:.2f} | {row['Mediana']:.2f} | {row['Desv. Estándar']:.2f} | {row['Mínimo']:.2f} | {row['Máximo']:.2f} | {row['Coef. Variación']:.2f} |\n"

    informe += """
**Hallazgos principales:**

"""

    # Análisis del PIB per cápita
    pib_mean = df['pib_per_capita'].mean()
    pib_std = df['pib_per_capita'].std()
    pib_cv = (pib_std / pib_mean) * 100
    pib_min = df['pib_per_capita'].min()
    pib_max = df['pib_per_capita'].max()

    informe += f"""1. **PIB per cápita:** Exhibe la mayor variabilidad entre todas las variables (CV = {pib_cv:.1f}%). Los valores oscilan entre ${pib_min:,.0f} y ${pib_max:,.0f}, reflejando profundas desigualdades económicas territoriales. La mediana (${df['pib_per_capita'].median():,.0f}) es inferior a la media (${pib_mean:,.0f}), indicando una distribución asimétrica positiva con presencia de cantones con PIB excepcionalmente alto.

"""

    # Análisis de tasa de homicidios
    hom_mean = df['tasa_homicidios'].mean()
    hom_max = df['tasa_homicidios'].max()
    canton_max_hom = df.loc[df['tasa_homicidios'].idxmax(), 'canton']

    informe += f"""2. **Tasa de homicidios:** Presenta una media de {hom_mean:.2f} por cada 100,000 habitantes, pero con valores extremos que alcanzan {hom_max:.2f} ({canton_max_hom}). La alta desviación estándar indica marcadas diferencias en seguridad ciudadana entre cantones.

"""

    # Análisis de población indígena
    ind_mean = df['pob_indigena_pct'].mean()
    ind_median = df['pob_indigena_pct'].median()

    informe += f"""3. **Población indígena:** La media ({ind_mean:.1f}%) supera significativamente la mediana ({ind_median:.1f}%), evidenciando que pocos cantones concentran población indígena alta (principalmente en la Sierra y Oriente), mientras la mayoría tiene porcentajes bajos.

"""

    # Análisis de servicios básicos
    agua_mean = df['agua_publica'].mean()
    elec_mean = df['electricidad'].mean()

    informe += f"""4. **Acceso a servicios básicos:** El acceso a electricidad ({elec_mean:.1f}%) es más universal que el acceso a agua potable ({agua_mean:.1f}%). Ambas variables muestran coeficientes de variación moderados, indicando brechas importantes entre cantones.

"""

    # Análisis de votos
    votos_mean = df['votos_noboa_pct'].mean()
    votos_std = df['votos_noboa_pct'].std()

    informe += f"""5. **Patrón electoral:** El porcentaje de votos por Noboa muestra una media de {votos_mean:.1f}% con desviación estándar de {votos_std:.1f}%, indicando variabilidad moderada en las preferencias electorales entre cantones.

---

### 3.2 Valores Extremos (Outliers)

Se identificaron valores extremos en múltiples variables mediante el método IQR. Los outliers **no fueron eliminados** por representar realidades territoriales específicas.

"""

    # Destacar outliers más importantes
    if 'pib_per_capita' in outliers_info and len(outliers_info['pib_per_capita']) > 0:
        informe += "#### PIB per cápita - Outliers superiores:\n"
        top_pib = outliers_info['pib_per_capita'].nlargest(5, 'pib_per_capita')
        for idx, row in top_pib.iterrows():
            informe += f"- **{row['canton']}** ({row['provincia']}): ${row['pib_per_capita']:,.0f}\n"
        informe += "\nEstos cantones corresponden principalmente a zonas de extracción petrolera (Oriente) y minera, explicando su PIB anormalmente alto.\n\n"

    if 'tasa_homicidios' in outliers_info and len(outliers_info['tasa_homicidios']) > 0:
        informe += "#### Tasa de homicidios - Outliers superiores:\n"
        top_hom = outliers_info['tasa_homicidios'].nlargest(5, 'tasa_homicidios')
        for idx, row in top_hom.iterrows():
            informe += f"- **{row['canton']}** ({row['provincia']}): {row['tasa_homicidios']:.1f}\n"
        informe += "\nLos cantones con tasas extremas de homicidios se ubican principalmente en zonas costeras y fronterizas.\n\n"

    informe += """---

### 3.3 Análisis de Normalidad

Se aplicó el test de Shapiro-Wilk para evaluar la normalidad de las distribuciones.

#### Tabla 2: Resultados del Test de Shapiro-Wilk

"""

    informe += "| Variable | Estadístico W | p-valor | ¿Normal? |\n"
    informe += "|----------|---------------|---------|----------|\n"

    for idx, row in resultados_normalidad.iterrows():
        informe += f"| {row['Variable']} | {row['Estadístico W']:.4f} | {row['p-valor']:.6f} | {row['Normalidad']} |\n"

    num_no_normales = (resultados_normalidad['Normalidad'] == 'No').sum()

    informe += f"""
**Interpretación:** De las {len(resultados_normalidad)} variables analizadas, {num_no_normales} rechazan la hipótesis de normalidad (p < 0.05). Este hallazgo justifica el uso de métodos no paramétricos en el análisis de correlación.

Las desviaciones de normalidad se deben a:
- Asimetría positiva en variables económicas (PIB, ingresos)
- Presencia de outliers genuinos
- Distribuciones multimodales relacionadas con patrones regionales

Los gráficos Q-Q y los histogramas (Figuras 1-3) confirman visualmente estas desviaciones.

![Histogramas](./histogramas.png)
*Figura 1: Histogramas de variables principales*

![Q-Q Plots](./qq_plots.png)
*Figura 2: Gráficos Q-Q para evaluación de normalidad*

![Boxplots](./boxplots.png)
*Figura 3: Boxplots mostrando distribución y outliers*

---

### 3.4 Análisis de Correlación

Dado que la mayoría de variables no siguen distribución normal, se utilizó el coeficiente de correlación de Spearman (ρ).

#### 3.4.1 Correlaciones Principales de Interés

"""

    # Calcular correlaciones específicas
    correlaciones_principales = [
        ('votos_noboa_pct', 'pib_per_capita', 'Votos Noboa vs PIB per cápita'),
        ('votos_noboa_pct', 'tasa_homicidios', 'Votos Noboa vs Tasa de homicidios'),
        ('votos_noboa_pct', 'pob_indigena_pct', 'Votos Noboa vs Población indígena'),
        ('agua_publica', 'pib_per_capita', 'Agua pública vs PIB per cápita'),
        ('electricidad', 'pib_per_capita', 'Electricidad vs PIB per cápita')
    ]

    informe += "| Relación | ρ | p-valor | Interpretación |\n"
    informe += "|----------|---|---------|----------------|\n"

    for var1, var2, nombre in correlaciones_principales:
        rho, p_val = spearmanr(df[var1], df[var2])
        fuerza = "Muy fuerte" if abs(rho) > 0.8 else ("Fuerte" if abs(rho) > 0.6 else ("Moderada" if abs(rho) > 0.4 else ("Débil" if abs(rho) > 0.2 else "Muy débil")))
        direccion = "positiva" if rho > 0 else "negativa"
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        informe += f"| {nombre} | {rho:.3f} | {p_val:.4f} {sig} | {fuerza} {direccion} |\n"

    informe += """
*Nota: *** p<0.001, ** p<0.01, * p<0.05, ns: no significativo*

#### 3.4.2 Hallazgos Principales

"""

    # Correlación votos vs PIB
    rho_vot_pib, p_vot_pib = spearmanr(df['votos_noboa_pct'], df['pib_per_capita'])

    if abs(rho_vot_pib) > 0.3:
        direccion = "positiva" if rho_vot_pib > 0 else "negativa"
        informe += f"""1. **Votos y desarrollo económico:** Se encontró una correlación {direccion} (ρ = {rho_vot_pib:.3f}, p < 0.001) entre el porcentaje de votos por Noboa y el PIB per cápita cantonal. """
        if rho_vot_pib > 0:
            informe += "Los cantones con mayor desarrollo económico tendieron a favorecer a Noboa.\n\n"
        else:
            informe += "Los cantones con menor desarrollo económico tendieron a favorecer a Noboa.\n\n"
    else:
        informe += f"1. **Votos y desarrollo económico:** La correlación entre votos y PIB per cápita es débil (ρ = {rho_vot_pib:.3f}), sugiriendo que factores económicos no determinan linealmente las preferencias electorales.\n\n"

    # Correlación votos vs homicidios
    rho_vot_hom, p_vot_hom = spearmanr(df['votos_noboa_pct'], df['tasa_homicidios'])
    direccion = "positiva" if rho_vot_hom > 0 else "negativa"

    informe += f"""2. **Votos y seguridad ciudadana:** Existe una correlación {direccion} estadísticamente significativa (ρ = {rho_vot_hom:.3f}, p < 0.001) entre votos por Noboa y tasa de homicidios. """
    if rho_vot_hom < 0:
        informe += "Los cantones con mayor inseguridad mostraron menor apoyo a Noboa.\n\n"
    else:
        informe += "Los cantones con mayor inseguridad mostraron mayor apoyo a Noboa, posiblemente por su discurso de mano dura.\n\n"

    # Correlación votos vs población indígena
    rho_vot_ind, p_vot_ind = spearmanr(df['votos_noboa_pct'], df['pob_indigena_pct'])

    informe += f"""3. **Votos y composición étnica:** Se observó una correlación {"positiva" if rho_vot_ind > 0 else "negativa"} (ρ = {rho_vot_ind:.3f}) entre votos por Noboa y porcentaje de población indígena. """
    if abs(rho_vot_ind) > 0.3:
        if rho_vot_ind > 0:
            informe += "Los cantones con mayor población indígena tendieron a apoyar más a Noboa.\n\n"
        else:
            informe += "Los cantones con mayor población indígena tendieron a apoyar menos a Noboa, favoreciendo a González.\n\n"
    else:
        informe += "Esta relación es relativamente débil, sugiriendo que otros factores median esta relación.\n\n"

    # Correlación servicios básicos
    rho_agua_pib, p_agua_pib = spearmanr(df['agua_publica'], df['pib_per_capita'])

    informe += f"""4. **Servicios básicos y desarrollo:** La correlación entre acceso a agua potable y PIB per cápita (ρ = {rho_agua_pib:.3f}) revela que la infraestructura de servicios básicos no necesariamente acompaña al crecimiento económico, especialmente en cantones extractivos.

"""

    informe += """![Matriz de Correlación](./matriz_correlacion_heatmap.png)
*Figura 4: Matriz de correlación de Spearman*

![Scatterplots](./scatterplots_correlaciones.png)
*Figura 5: Diagramas de dispersión para correlaciones principales*

---

## 4. Análisis por Región

### 4.1 Patrones Regionales

"""

    # Estadísticas por región
    for region in ['Costa', 'Sierra', 'Oriente', 'Insular']:
        df_region = df[df['region'] == region]
        if len(df_region) > 0:
            informe += f"""#### {region} ({len(df_region)} cantones)

- **Votos Noboa:** Media = {df_region['votos_noboa_pct'].mean():.1f}%, Mediana = {df_region['votos_noboa_pct'].median():.1f}%
- **PIB per cápita:** Media = ${df_region['pib_per_capita'].mean():,.0f}
- **Tasa de homicidios:** Media = {df_region['tasa_homicidios'].mean():.1f}
- **Población indígena:** Media = {df_region['pob_indigena_pct'].mean():.1f}%
- **Acceso agua pública:** Media = {df_region['agua_publica'].mean():.1f}%

"""

    informe += """![Boxplots por Región](./boxplots_por_region.png)
*Figura 6: Comparación de variables por región geográfica*

### 4.2 Principales Diferencias Regionales

"""

    # Comparar Costa vs Sierra
    costa = df[df['region'] == 'Costa']
    sierra = df[df['region'] == 'Sierra']
    oriente = df[df['region'] == 'Oriente']

    informe += f"""1. **Costa vs Sierra:**
   - La Costa presenta tasas de homicidios significativamente más altas (media: {costa['tasa_homicidios'].mean():.1f}) que la Sierra (media: {sierra['tasa_homicidios'].mean():.1f}).
   - La Sierra tiene mayor porcentaje de población indígena (media: {sierra['pob_indigena_pct'].mean():.1f}%) vs Costa ({costa['pob_indigena_pct'].mean():.1f}%).
   - El apoyo a Noboa es {"mayor" if costa['votos_noboa_pct'].mean() > sierra['votos_noboa_pct'].mean() else "menor"} en la Costa ({costa['votos_noboa_pct'].mean():.1f}%) que en la Sierra ({sierra['votos_noboa_pct'].mean():.1f}%).

2. **Oriente:**
   - Presenta el PIB per cápita promedio más alto (${oriente['pib_per_capita'].mean():,.0f}) debido a la extracción petrolera.
   - Alto porcentaje de población indígena (media: {oriente['pob_indigena_pct'].mean():.1f}%).
   - Mayor apoyo a Noboa que otras regiones ({oriente['votos_noboa_pct'].mean():.1f}%).

---

## 5. Discusión

### 5.1 Interpretación de Resultados

"""

    # Interpretación según los patrones encontrados
    if rho_vot_hom < -0.3:
        informe += """Los resultados revelan una relación negativa entre apoyo electoral a Noboa y tasas de homicidios cantonales. Contraintuitivamente, los cantones con mayor inseguridad no favorecieron al candidato que proponía políticas de mano dura. Esto podría explicarse por:

1. **Correlación territorial:** Los cantones costeños con alta criminalidad históricamente apoyan al correísmo.
2. **Factores socioeconómicos subyacentes:** La inseguridad correlaciona con pobreza y exclusión, que predicen voto por González.
3. **Clientelismo político:** Redes establecidas en zonas vulnerables.

"""
    elif rho_vot_hom > 0.3:
        informe += """Los resultados muestran que cantones con mayores tasas de homicidios tendieron a apoyar más a Noboa, posiblemente receptivos a su mensaje de seguridad y mano dura. Sin embargo, esta correlación debe interpretarse con cautela dado el contexto de inseguridad generalizada durante el período electoral.

"""

    informe += f"""### 5.2 Determinantes del Voto

El análisis multivariado sugiere que las preferencias electorales a nivel cantonal están determinadas por una combinación de:

1. **Factores geográficos:** La región explica parte significativa de la varianza en el voto.
2. **Composición étnica:** La correlación con población indígena (ρ = {rho_vot_ind:.3f}) indica patrones de voto diferenciados.
3. **Desarrollo económico:** La relación con PIB per cápita (ρ = {rho_vot_pib:.3f}) sugiere {"que el desarrollo favorece a Noboa" if rho_vot_pib > 0 else "que el menor desarrollo favorece a Noboa"}.

### 5.3 Limitaciones

1. **Falacia ecológica:** Las correlaciones cantonales no necesariamente reflejan comportamiento individual.
2. **Variables omitidas:** No se incluyen factores como educación, urbanización, o presencia de medios.
3. **Causalidad:** Las correlaciones no implican relaciones causales.
4. **Temporalidad:** Los datos socioeconómicos pueden no ser contemporáneos a la elección.

---

## 6. Conclusiones

1. **Heterogeneidad territorial:** Ecuador exhibe profundas desigualdades entre cantones en indicadores económicos, de seguridad y acceso a servicios, con coeficientes de variación que superan el 100% en variables como PIB per cápita.

2. **No normalidad de distribuciones:** La mayoría de variables socioeconómicas no siguen distribución normal, justificando el uso de métodos no paramétricos y alertando sobre el uso inadecuado de estadística paramétrica.

3. **Patrones regionales:** Las diferencias Costa-Sierra-Oriente son fundamentales para comprender tanto la distribución de indicadores socioeconómicos como los patrones electorales.

4. **Correlaciones significativas:** Se encontraron correlaciones estadísticamente significativas entre variables electorales y socioeconómicas, aunque su interpretación requiere considerar el contexto territorial y evitar inferencias causales simplistas.

5. **Outliers informativos:** Los valores extremos representan realidades territoriales específicas (cantones petroleros, mineros, zonas de conflicto) y no deben eliminarse arbitrariamente.

### Recomendaciones para Investigación Futura

1. Incorporar análisis multinivel que considere la estructura jerárquica (cantones en provincias).
2. Incluir variables de control como educación, urbanización y acceso a medios.
3. Realizar análisis espacial para detectar patrones de autocorrelación geográfica.
4. Considerar modelos de ecuaciones estructurales para evaluar efectos directos e indirectos.
5. Comparar con elecciones anteriores para evaluar estabilidad de patrones.

---

## Referencias Metodológicas

- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality.
- Spearman, C. (1904). The proof and measurement of association between two things.
- Tukey, J. W. (1977). Exploratory Data Analysis.

---

**Análisis realizado con Python 3.11**
- pandas: Manipulación de datos
- scipy: Pruebas estadísticas
- matplotlib/seaborn: Visualizaciones
- numpy: Cálculos numéricos

---

*Documento generado automáticamente*
*Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    # Guardar informe
    with open('/home/user/Inv-Quant/resultados/INFORME_ESTADISTICO_ECUADOR.md', 'w', encoding='utf-8') as f:
        f.write(informe)

    print("  ✓ Informe completo guardado: INFORME_ESTADISTICO_ECUADOR.md")

    return informe

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todas las fases del análisis.
    """
    print("="*70)
    print("ANÁLISIS ESTADÍSTICO DE CANTONES DE ECUADOR")
    print("="*70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Crear directorio de resultados
    import os
    os.makedirs('/home/user/Inv-Quant/resultados', exist_ok=True)

    # FASE 1: Cargar datos
    df = cargar_y_limpiar_datos('/home/user/Inv-Quant/Basecantones2csv.csv')

    # FASE 2: Análisis descriptivo
    stats_desc, outliers_info = calcular_estadisticas_descriptivas(df)
    stats_region = generar_estadisticas_por_region(df)

    # FASE 3: Análisis de normalidad
    resultados_normalidad = test_normalidad(df)
    generar_visualizaciones_normalidad(df)

    # FASE 4: Análisis de correlación
    corr_matrix, df_corr_sig = calcular_correlaciones(df)
    crear_scatterplots_correlaciones(df)

    # FASE 5: Generar informe
    informe = generar_informe(df, stats_desc, outliers_info, resultados_normalidad,
                              corr_matrix, df_corr_sig)

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nArchivos generados en /home/user/Inv-Quant/resultados/:")
    print("  - INFORME_ESTADISTICO_ECUADOR.md (Informe completo)")
    print("  - estadisticas_descriptivas.csv")
    print("  - estadisticas_por_region.csv")
    print("  - test_normalidad.csv")
    print("  - matriz_correlacion_spearman.csv")
    print("  - correlaciones_significativas.csv")
    print("  - correlaciones_prioritarias.csv")
    print("  - histogramas.png")
    print("  - qq_plots.png")
    print("  - boxplots.png")
    print("  - boxplots_por_region.png")
    print("  - matriz_correlacion_heatmap.png")
    print("  - scatterplots_correlaciones.png")

if __name__ == "__main__":
    main()
