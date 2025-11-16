#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ANÁLISIS ESTADÍSTICO COMPLETO CON VISUALIZACIONES EXTENSIVAS
Cantones de Ecuador - Versión Python
===============================================================================

Este script genera una amplia colección de gráficos para cada fase del análisis:
- Fase 1: Exploración inicial de datos
- Fase 2: Análisis descriptivo con múltiples visualizaciones
- Fase 3: Análisis de normalidad exhaustivo
- Fase 4: Análisis de correlación detallado
- Fase 5: Análisis por regiones
- Fase 6: Identificación visual de outliers

Requisitos:
    pip install pandas numpy scipy matplotlib seaborn

Uso:
    python analisis_visualizaciones.py

Autor: Análisis Estadístico Cantonal
Fecha: 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, spearmanr, pearsonr, zscore
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuración global
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Rutas
RUTA_DATOS = 'Basecantones2csv.csv'
RUTA_GRAFICOS = 'graficos'

# Colores personalizados
COLORES_REGION = {'Costa': '#FF6B6B', 'Sierra': '#4ECDC4', 'Oriente': '#45B7D1', 'Insular': '#96CEB4'}
COLORES_PALETA = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

def cargar_y_preparar_datos():
    """Carga y prepara el dataset."""
    print("=" * 70)
    print("CARGANDO Y PREPARANDO DATOS")
    print("=" * 70)

    df = pd.read_csv(RUTA_DATOS, sep=';', encoding='utf-8-sig')

    # Renombrar columnas
    nombres = {
        'Cantón': 'canton', 'Provincia': 'provincia',
        'Votos por Noboa (absoluto)': 'votos_noboa_abs',
        'Votos por González (absoluto)': 'votos_gonzalez_abs',
        'Votos por Noboa (porcentaje)': 'votos_noboa_pct',
        'Votos por González (porcentaje)': 'votos_gonzalez_pct',
        'Población': 'poblacion', 'Porcentaje población indígena': 'pob_indigena_pct',
        'Agua red pública': 'agua_publica', 'Electricidad': 'electricidad',
        'PIB per cápita': 'pib_per_capita', 'Tasa Homicidios': 'tasa_homicidios',
        'Altitud': 'altitud', 'Costa': 'costa', 'Sierra': 'sierra',
        'Oriente': 'oriente', 'Insular': 'insular'
    }
    df = df.rename(columns=nombres)

    # Convertir tipos
    for col in ['votos_noboa_pct', 'votos_gonzalez_pct', 'pob_indigena_pct',
                'agua_publica', 'electricidad', 'tasa_homicidios']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    # Crear región
    df['region'] = np.select(
        [df['costa'] == 1, df['sierra'] == 1, df['oriente'] == 1, df['insular'] == 1],
        ['Costa', 'Sierra', 'Oriente', 'Insular'], 'Desconocido'
    )

    print(f"✓ Datos cargados: {len(df)} cantones")
    return df

# =============================================================================
# FASE 1: GRÁFICOS DE EXPLORACIÓN INICIAL
# =============================================================================

def graficos_exploracion(df):
    """Gráficos iniciales para explorar los datos."""
    print("\n" + "=" * 70)
    print("FASE 1: GRÁFICOS DE EXPLORACIÓN INICIAL")
    print("=" * 70)

    # 1.1 Distribución de cantones por provincia
    plt.figure(figsize=(14, 8))
    provincia_counts = df['provincia'].value_counts().head(15)
    bars = plt.barh(range(len(provincia_counts)), provincia_counts.values, color=COLORES_PALETA[0])
    plt.yticks(range(len(provincia_counts)), provincia_counts.index)
    plt.xlabel('Número de Cantones')
    plt.title('Distribución de Cantones por Provincia (Top 15)', fontweight='bold', fontsize=14)
    for i, v in enumerate(provincia_counts.values):
        plt.text(v + 0.2, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/01_cantones_por_provincia.png')
    plt.close()
    print("✓ 01_cantones_por_provincia.png")

    # 1.2 Distribución por región (pie chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    region_counts = df['region'].value_counts()
    colors = [COLORES_REGION[r] for r in region_counts.index]
    wedges, texts, autotexts = ax1.pie(region_counts.values, labels=region_counts.index,
                                        colors=colors, autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 12})
    ax1.set_title('Distribución de Cantones por Región', fontweight='bold', fontsize=14)

    # Gráfico de barras
    bars = ax2.bar(region_counts.index, region_counts.values, color=colors, edgecolor='black')
    ax2.set_ylabel('Número de Cantones')
    ax2.set_title('Cantidad de Cantones por Región', fontweight='bold', fontsize=14)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/02_distribucion_regiones.png')
    plt.close()
    print("✓ 02_distribucion_regiones.png")

    # 1.3 Mapa de calor de valores faltantes
    plt.figure(figsize=(12, 6))
    missing = df.isnull().astype(int)
    sns.heatmap(missing, cmap='YlOrRd', cbar_kws={'label': 'Faltante (1) / Presente (0)'})
    plt.title('Mapa de Valores Faltantes por Variable', fontweight='bold', fontsize=14)
    plt.xlabel('Variables')
    plt.ylabel('Observaciones')
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/03_mapa_valores_faltantes.png')
    plt.close()
    print("✓ 03_mapa_valores_faltantes.png")

    # 1.4 Vista general de todas las variables numéricas
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    vars_numericas = ['votos_noboa_pct', 'votos_gonzalez_pct', 'poblacion',
                      'pob_indigena_pct', 'agua_publica', 'electricidad',
                      'pib_per_capita', 'tasa_homicidios', 'altitud']

    for i, var in enumerate(vars_numericas):
        sns.histplot(df[var], kde=True, ax=axes[i], color=COLORES_PALETA[i % len(COLORES_PALETA)])
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
        axes[i].axvline(df[var].mean(), color='red', linestyle='--', label='Media')
        axes[i].axvline(df[var].median(), color='green', linestyle='--', label='Mediana')
        axes[i].legend(fontsize=8)

    plt.suptitle('Vista General de Distribuciones', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/04_vista_general_distribuciones.png')
    plt.close()
    print("✓ 04_vista_general_distribuciones.png")

# =============================================================================
# FASE 2: GRÁFICOS DE ANÁLISIS DESCRIPTIVO
# =============================================================================

def graficos_descriptivos(df):
    """Gráficos detallados para análisis descriptivo."""
    print("\n" + "=" * 70)
    print("FASE 2: GRÁFICOS DE ANÁLISIS DESCRIPTIVO")
    print("=" * 70)

    vars_principales = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                        'pob_indigena_pct', 'agua_publica']

    # 2.1 Violin plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.violinplot(data=df, y=var, ax=axes[i], color=COLORES_PALETA[i], inner='box')
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        axes[i].set_ylabel('')
    axes[-1].axis('off')

    plt.suptitle('Violin Plots - Distribución y Densidad', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/05_violin_plots.png')
    plt.close()
    print("✓ 05_violin_plots.png")

    # 2.2 Strip plots (jitter)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.stripplot(data=df, y=var, ax=axes[i], color=COLORES_PALETA[i],
                     alpha=0.6, jitter=True, size=5)
        axes[i].axhline(df[var].mean(), color='red', linestyle='--', linewidth=2, label='Media')
        axes[i].axhline(df[var].median(), color='green', linestyle='--', linewidth=2, label='Mediana')
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
        axes[i].legend()
    axes[-1].axis('off')

    plt.suptitle('Strip Plots - Cada Punto es un Cantón', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/06_strip_plots.png')
    plt.close()
    print("✓ 06_strip_plots.png")

    # 2.3 ECDF (Función de Distribución Acumulada Empírica)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.ecdfplot(data=df, x=var, ax=axes[i], color=COLORES_PALETA[i], linewidth=2)
        axes[i].set_title(f'ECDF: {var.replace("_", " ").title()}', fontweight='bold')
        axes[i].set_ylabel('Proporción acumulada')
        axes[i].grid(True, alpha=0.3)
        # Añadir línea en mediana
        mediana = df[var].median()
        axes[i].axvline(mediana, color='red', linestyle='--', alpha=0.7, label=f'Mediana: {mediana:.1f}')
        axes[i].legend()
    axes[-1].axis('off')

    plt.suptitle('Funciones de Distribución Acumulada Empírica (ECDF)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/07_ecdf_plots.png')
    plt.close()
    print("✓ 07_ecdf_plots.png")

    # 2.4 Resumen de estadísticas con barras
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Media por variable
    medias = df[vars_principales].mean()
    ax = axes[0, 0]
    bars = ax.bar(range(len(medias)), medias.values, color=COLORES_PALETA[:len(medias)])
    ax.set_xticks(range(len(medias)))
    ax.set_xticklabels([v.replace('_', '\n') for v in medias.index], fontsize=9)
    ax.set_title('Media por Variable', fontweight='bold')
    ax.set_ylabel('Valor')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.1f}',
               ha='center', va='bottom', fontsize=9)

    # Desviación estándar
    stds = df[vars_principales].std()
    ax = axes[0, 1]
    bars = ax.bar(range(len(stds)), stds.values, color=COLORES_PALETA[:len(stds)])
    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels([v.replace('_', '\n') for v in stds.index], fontsize=9)
    ax.set_title('Desviación Estándar por Variable', fontweight='bold')
    ax.set_ylabel('Valor')

    # Coeficiente de variación
    cvs = (df[vars_principales].std() / df[vars_principales].mean() * 100)
    ax = axes[1, 0]
    bars = ax.bar(range(len(cvs)), cvs.values, color=COLORES_PALETA[:len(cvs)])
    ax.set_xticks(range(len(cvs)))
    ax.set_xticklabels([v.replace('_', '\n') for v in cvs.index], fontsize=9)
    ax.set_title('Coeficiente de Variación (%)', fontweight='bold')
    ax.set_ylabel('CV (%)')
    ax.axhline(100, color='red', linestyle='--', label='CV=100%')
    ax.legend()

    # Rango (Max - Min)
    rangos = df[vars_principales].max() - df[vars_principales].min()
    ax = axes[1, 1]
    bars = ax.bar(range(len(rangos)), rangos.values, color=COLORES_PALETA[:len(rangos)])
    ax.set_xticks(range(len(rangos)))
    ax.set_xticklabels([v.replace('_', '\n') for v in rangos.index], fontsize=9)
    ax.set_title('Rango (Máximo - Mínimo)', fontweight='bold')
    ax.set_ylabel('Rango')

    plt.suptitle('Resumen de Estadísticas Descriptivas', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/08_resumen_estadisticas.png')
    plt.close()
    print("✓ 08_resumen_estadisticas.png")

    # 2.5 Boxplots detallados con swarm
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.boxplot(data=df, y=var, ax=axes[i], color='lightgray', width=0.3)
        sns.swarmplot(data=df, y=var, ax=axes[i], color=COLORES_PALETA[i],
                     alpha=0.6, size=3)
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')

        # Añadir estadísticas
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1
        axes[i].text(0.02, 0.98, f'IQR: {iqr:.1f}', transform=axes[i].transAxes,
                    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    axes[-1].axis('off')

    plt.suptitle('Boxplots con Puntos Individuales (Swarm)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/09_boxplots_swarm.png')
    plt.close()
    print("✓ 09_boxplots_swarm.png")

    # 2.6 Histogramas con múltiples bins
    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    bins_options = [10, 20, 30]

    for i, var in enumerate(vars_principales):
        for j, bins in enumerate(bins_options):
            ax = axes[i, j]
            sns.histplot(df[var], bins=bins, kde=True, ax=ax, color=COLORES_PALETA[i])
            ax.set_title(f'{var.replace("_", " ").title()} - {bins} bins', fontweight='bold')
            if j == 0:
                ax.set_ylabel('Frecuencia')

    plt.suptitle('Comparación de Histogramas con Diferentes Números de Bins',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/10_histogramas_multiples_bins.png')
    plt.close()
    print("✓ 10_histogramas_multiples_bins.png")

# =============================================================================
# FASE 3: GRÁFICOS DE ANÁLISIS DE NORMALIDAD
# =============================================================================

def graficos_normalidad(df):
    """Gráficos exhaustivos para evaluar normalidad."""
    print("\n" + "=" * 70)
    print("FASE 3: GRÁFICOS DE ANÁLISIS DE NORMALIDAD")
    print("=" * 70)

    vars_test = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    # 3.1 Histogramas con curva normal teórica superpuesta
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_test):
        data = df[var].dropna()
        ax = axes[i]

        # Histograma normalizado
        n, bins, patches = ax.hist(data, bins=25, density=True, alpha=0.7,
                                   color=COLORES_PALETA[i], edgecolor='black')

        # Curva normal teórica
        mu, std = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, std)
        ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal teórica')

        ax.set_title(f'{var.replace("_", " ").title()}\nμ={mu:.1f}, σ={std:.1f}', fontweight='bold')
        ax.legend()
        ax.set_ylabel('Densidad')
    axes[-1].axis('off')

    plt.suptitle('Histogramas vs Distribución Normal Teórica', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/11_histogramas_vs_normal.png')
    plt.close()
    print("✓ 11_histogramas_vs_normal.png")

    # 3.2 Q-Q plots detallados
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_test):
        ax = axes[i]
        data = df[var].dropna()

        # Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
        ax.scatter(osm, osr, color=COLORES_PALETA[i], alpha=0.6, s=30)
        ax.plot(osm, slope*osm + intercept, 'r-', linewidth=2, label=f'R² = {r**2:.4f}')

        ax.set_xlabel('Cuantiles Teóricos')
        ax.set_ylabel('Cuantiles Muestrales')
        ax.set_title(f'Q-Q Plot: {var.replace("_", " ").title()}', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    axes[-1].axis('off')

    plt.suptitle('Gráficos Q-Q con R² de Ajuste', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/12_qq_plots_detallados.png')
    plt.close()
    print("✓ 12_qq_plots_detallados.png")

    # 3.3 Gráficos de densidad comparativos
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_test):
        ax = axes[i]
        data = df[var].dropna()

        # Densidad empírica
        sns.kdeplot(data, ax=ax, color=COLORES_PALETA[i], linewidth=2, label='Empírica', fill=True, alpha=0.3)

        # Densidad normal teórica
        x = np.linspace(data.min(), data.max(), 100)
        normal_pdf = stats.norm.pdf(x, data.mean(), data.std())
        ax.plot(x, normal_pdf, 'r--', linewidth=2, label='Normal teórica')

        ax.set_title(var.replace('_', ' ').title(), fontweight='bold')
        ax.legend()
        ax.set_ylabel('Densidad')
    axes[-1].axis('off')

    plt.suptitle('Comparación: Densidad Empírica vs Normal Teórica', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/13_densidad_empirica_vs_normal.png')
    plt.close()
    print("✓ 13_densidad_empirica_vs_normal.png")

    # 3.4 Gráfico de asimetría y curtosis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    asimetrias = [df[var].skew() for var in vars_test]
    curtosis = [df[var].kurtosis() for var in vars_test]

    # Asimetría
    bars1 = ax1.bar(range(len(vars_test)), asimetrias, color=COLORES_PALETA[:len(vars_test)])
    ax1.set_xticks(range(len(vars_test)))
    ax1.set_xticklabels([v.replace('_', '\n') for v in vars_test], fontsize=9)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    ax1.axhspan(-0.5, 0.5, alpha=0.2, color='green', label='Rango aceptable')
    ax1.set_title('Asimetría (Skewness)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Valor')
    ax1.legend()
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}',
                ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=9)

    # Curtosis
    bars2 = ax2.bar(range(len(vars_test)), curtosis, color=COLORES_PALETA[:len(vars_test)])
    ax2.set_xticks(range(len(vars_test)))
    ax2.set_xticklabels([v.replace('_', '\n') for v in vars_test], fontsize=9)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Normal (0)')
    ax2.axhspan(-1, 1, alpha=0.2, color='green', label='Rango aceptable')
    ax2.set_title('Curtosis (Kurtosis)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Valor')
    ax2.legend()
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.2f}',
                ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=9)

    plt.suptitle('Indicadores de Forma de la Distribución', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/14_asimetria_curtosis.png')
    plt.close()
    print("✓ 14_asimetria_curtosis.png")

    # 3.5 Resumen de tests de normalidad
    fig, ax = plt.subplots(figsize=(12, 8))

    resultados = []
    for var in vars_test:
        stat, p = shapiro(df[var].dropna())
        resultados.append({'Variable': var, 'W': stat, 'p-valor': p})

    df_resultados = pd.DataFrame(resultados)

    # Crear tabla visual
    cell_text = []
    colors_cells = []
    for i, row in df_resultados.iterrows():
        es_normal = row['p-valor'] > 0.05
        cell_text.append([row['Variable'].replace('_', ' '), f"{row['W']:.4f}",
                         f"{row['p-valor']:.6f}", 'Sí' if es_normal else 'No'])
        color = ['white', 'white', 'white', '#90EE90' if es_normal else '#FFB6C1']
        colors_cells.append(color)

    table = ax.table(cellText=cell_text,
                     colLabels=['Variable', 'Estadístico W', 'p-valor', '¿Normal? (α=0.05)'],
                     cellLoc='center', loc='center', cellColours=colors_cells)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    ax.axis('off')
    ax.set_title('Resultados del Test de Shapiro-Wilk', fontweight='bold', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/15_tabla_shapiro_wilk.png')
    plt.close()
    print("✓ 15_tabla_shapiro_wilk.png")

    # 3.6 P-P Plots (Probability-Probability)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_test):
        ax = axes[i]
        data = np.sort(df[var].dropna())
        n = len(data)

        # Probabilidades empíricas
        p_empirica = np.arange(1, n+1) / (n+1)

        # Probabilidades teóricas (normal)
        p_teorica = stats.norm.cdf(data, loc=data.mean(), scale=data.std())

        ax.scatter(p_teorica, p_empirica, color=COLORES_PALETA[i], alpha=0.6, s=20)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Línea de referencia')
        ax.set_xlabel('Probabilidad Teórica (Normal)')
        ax.set_ylabel('Probabilidad Empírica')
        ax.set_title(f'P-P Plot: {var.replace("_", " ").title()}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].axis('off')

    plt.suptitle('Gráficos P-P (Probability-Probability)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/16_pp_plots.png')
    plt.close()
    print("✓ 16_pp_plots.png")

# =============================================================================
# FASE 4: GRÁFICOS DE CORRELACIÓN
# =============================================================================

def graficos_correlacion(df):
    """Gráficos detallados para análisis de correlación."""
    print("\n" + "=" * 70)
    print("FASE 4: GRÁFICOS DE ANÁLISIS DE CORRELACIÓN")
    print("=" * 70)

    vars_corr = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica', 'electricidad']

    # 4.1 Matriz de correlación con diferentes métodos
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Spearman
    corr_spearman = df[vars_corr].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
    sns.heatmap(corr_spearman, mask=mask, annot=True, cmap='RdBu_r', center=0,
               square=True, ax=axes[0], fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[0].set_title('Correlación de Spearman (No paramétrica)', fontweight='bold')

    # Pearson
    corr_pearson = df[vars_corr].corr(method='pearson')
    sns.heatmap(corr_pearson, mask=mask, annot=True, cmap='RdBu_r', center=0,
               square=True, ax=axes[1], fmt='.2f', cbar_kws={'shrink': 0.8})
    axes[1].set_title('Correlación de Pearson (Paramétrica)', fontweight='bold')

    plt.suptitle('Comparación de Matrices de Correlación', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/17_matrices_correlacion.png')
    plt.close()
    print("✓ 17_matrices_correlacion.png")

    # 4.2 Pairplot completo
    vars_pairplot = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios', 'pob_indigena_pct']
    g = sns.pairplot(df[vars_pairplot + ['region']], hue='region',
                     palette=COLORES_REGION, diag_kind='kde', height=2.5,
                     plot_kws={'alpha': 0.6, 's': 30})
    g.fig.suptitle('Pairplot: Variables Principales por Región', fontweight='bold', y=1.02)
    plt.savefig(f'{RUTA_GRAFICOS}/18_pairplot_completo.png')
    plt.close()
    print("✓ 18_pairplot_completo.png")

    # 4.3 Scatterplots con regresión y intervalos de confianza
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    pares = [
        ('votos_noboa_pct', 'pib_per_capita'),
        ('votos_noboa_pct', 'tasa_homicidios'),
        ('votos_noboa_pct', 'pob_indigena_pct'),
        ('agua_publica', 'pib_per_capita'),
        ('tasa_homicidios', 'poblacion'),
        ('electricidad', 'agua_publica')
    ]

    for i, (x_var, y_var) in enumerate(pares):
        ax = axes[i]
        sns.regplot(data=df, x=x_var, y=y_var, ax=ax,
                   scatter_kws={'alpha': 0.5, 'color': COLORES_PALETA[i]},
                   line_kws={'color': 'red', 'linewidth': 2},
                   ci=95)

        # Calcular correlación
        rho, p = spearmanr(df[x_var], df[y_var])
        ax.text(0.05, 0.95, f'ρ = {rho:.3f}\np = {p:.4f}',
               transform=ax.transAxes, fontsize=11, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(x_var.replace('_', ' ').title())
        ax.set_ylabel(y_var.replace('_', ' ').title())
        ax.set_title(f'{y_var.replace("_", " ").title()} vs {x_var.replace("_", " ").title()}',
                    fontweight='bold')

    plt.suptitle('Scatterplots con Regresión Lineal e Intervalo de Confianza (95%)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/19_scatterplots_regresion.png')
    plt.close()
    print("✓ 19_scatterplots_regresion.png")

    # 4.4 Gráfico de fuerza de correlaciones
    correlaciones = []
    for i in range(len(vars_corr)):
        for j in range(i+1, len(vars_corr)):
            rho, p = spearmanr(df[vars_corr[i]], df[vars_corr[j]])
            correlaciones.append({
                'par': f'{vars_corr[i][:8]}\nvs\n{vars_corr[j][:8]}',
                'rho': rho,
                'abs_rho': abs(rho),
                'p_valor': p
            })

    df_corr = pd.DataFrame(correlaciones).sort_values('abs_rho', ascending=True)

    plt.figure(figsize=(12, 10))
    colors = ['green' if abs(r) > 0.5 else 'orange' if abs(r) > 0.3 else 'gray'
              for r in df_corr['rho']]
    plt.barh(range(len(df_corr)), df_corr['rho'], color=colors, edgecolor='black')
    plt.yticks(range(len(df_corr)), df_corr['par'], fontsize=9)
    plt.axvline(0, color='black', linewidth=1)
    plt.axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Fuerte (±0.5)')
    plt.axvline(-0.5, color='green', linestyle='--', alpha=0.5)
    plt.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Moderada (±0.3)')
    plt.axvline(-0.3, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Coeficiente de Correlación de Spearman (ρ)')
    plt.title('Fuerza de las Correlaciones Entre Variables', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/20_fuerza_correlaciones.png')
    plt.close()
    print("✓ 20_fuerza_correlaciones.png")

    # 4.5 Jointplots para correlaciones principales
    var_pairs = [
        ('votos_noboa_pct', 'tasa_homicidios'),
        ('votos_noboa_pct', 'pob_indigena_pct')
    ]

    for x_var, y_var in var_pairs:
        g = sns.jointplot(data=df, x=x_var, y=y_var, kind='reg',
                         height=8, color=COLORES_PALETA[0],
                         marginal_kws=dict(bins=25, fill=True))
        rho, p = spearmanr(df[x_var], df[y_var])
        g.ax_joint.text(0.05, 0.95, f'ρ = {rho:.3f}, p = {p:.4f}',
                       transform=g.ax_joint.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat'))
        g.fig.suptitle(f'Jointplot: {y_var.replace("_", " ").title()} vs {x_var.replace("_", " ").title()}',
                      fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAFICOS}/21_jointplot_{x_var}_vs_{y_var}.png')
        plt.close()
    print("✓ 21_jointplot_votos_vs_homicidios.png")
    print("✓ 21_jointplot_votos_vs_indigena.png")

# =============================================================================
# FASE 5: GRÁFICOS POR REGIÓN
# =============================================================================

def graficos_region(df):
    """Gráficos comparativos por región geográfica."""
    print("\n" + "=" * 70)
    print("FASE 5: GRÁFICOS POR REGIÓN GEOGRÁFICA")
    print("=" * 70)

    vars_principales = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                        'pob_indigena_pct', 'agua_publica']

    # 5.1 Boxplots por región
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.boxplot(data=df, x='region', y=var, ax=axes[i],
                   palette=COLORES_REGION, order=['Costa', 'Sierra', 'Oriente', 'Insular'])
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    axes[-1].axis('off')

    plt.suptitle('Distribución de Variables por Región', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/22_boxplots_por_region.png')
    plt.close()
    print("✓ 22_boxplots_por_region.png")

    # 5.2 Violin plots por región
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.violinplot(data=df, x='region', y=var, ax=axes[i],
                      palette=COLORES_REGION, inner='box',
                      order=['Costa', 'Sierra', 'Oriente', 'Insular'])
        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    axes[-1].axis('off')

    plt.suptitle('Violin Plots por Región', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/23_violin_por_region.png')
    plt.close()
    print("✓ 23_violin_por_region.png")

    # 5.3 Gráfico de medias por región
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        means = df.groupby('region')[var].mean().reindex(['Costa', 'Sierra', 'Oriente', 'Insular'])
        stds = df.groupby('region')[var].std().reindex(['Costa', 'Sierra', 'Oriente', 'Insular'])

        ax = axes[i]
        bars = ax.bar(range(4), means.values,
                     color=[COLORES_REGION[r] for r in means.index],
                     yerr=stds.values, capsize=10, edgecolor='black')
        ax.set_xticks(range(4))
        ax.set_xticklabels(means.index)
        ax.set_title(var.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Media ± Desv. Std.')

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{bar.get_height():.1f}',
                   ha='center', va='bottom', fontsize=9)
    axes[-1].axis('off')

    plt.suptitle('Media y Desviación Estándar por Región', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/24_medias_por_region.png')
    plt.close()
    print("✓ 24_medias_por_region.png")

    # 5.4 Radar chart por región
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    # Normalizar datos para el radar
    vars_radar = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                  'pob_indigena_pct', 'agua_publica', 'electricidad']

    for idx, region in enumerate(['Costa', 'Sierra', 'Oriente', 'Insular']):
        ax = axes[idx]

        # Datos normalizados (0-1) para la región
        valores = []
        for var in vars_radar:
            val = df[df['region'] == region][var].mean()
            val_min = df[var].min()
            val_max = df[var].max()
            val_norm = (val - val_min) / (val_max - val_min) if val_max != val_min else 0
            valores.append(val_norm)

        # Cerrar el polígono
        valores.append(valores[0])

        # Ángulos
        angles = np.linspace(0, 2*np.pi, len(vars_radar), endpoint=False).tolist()
        angles.append(angles[0])

        # Plot
        ax.plot(angles, valores, 'o-', linewidth=2, color=COLORES_REGION[region])
        ax.fill(angles, valores, alpha=0.25, color=COLORES_REGION[region])

        # Etiquetas
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([v.replace('_', '\n') for v in vars_radar], fontsize=9)
        ax.set_title(f'{region}', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True)

    plt.suptitle('Radar Charts por Región (Valores Normalizados)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/25_radar_charts_region.png')
    plt.close()
    print("✓ 25_radar_charts_region.png")

    # 5.5 Facet grid de densidades
    vars_facet = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios']

    for var in vars_facet:
        g = sns.FacetGrid(df, col='region', col_wrap=2, height=4, aspect=1.5,
                         palette=COLORES_REGION, sharex=False)
        g.map(sns.histplot, var, kde=True, bins=20)
        g.fig.suptitle(f'Distribución de {var.replace("_", " ").title()} por Región',
                      fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAFICOS}/26_facet_{var}.png')
        plt.close()
    print("✓ 26_facet_votos_noboa_pct.png")
    print("✓ 26_facet_pib_per_capita.png")
    print("✓ 26_facet_tasa_homicidios.png")

    # 5.6 Strip plot por región
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_principales):
        sns.stripplot(data=df, x='region', y=var, ax=axes[i],
                     palette=COLORES_REGION, jitter=True, alpha=0.7, size=6,
                     order=['Costa', 'Sierra', 'Oriente', 'Insular'])

        # Añadir medias
        means = df.groupby('region')[var].mean()
        for j, region in enumerate(['Costa', 'Sierra', 'Oriente', 'Insular']):
            axes[i].hlines(means[region], j-0.3, j+0.3, colors='red', linewidths=3)

        axes[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    axes[-1].axis('off')

    plt.suptitle('Strip Plots por Región (Línea roja = Media)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/27_strip_por_region.png')
    plt.close()
    print("✓ 27_strip_por_region.png")

# =============================================================================
# FASE 6: GRÁFICOS DE OUTLIERS
# =============================================================================

def graficos_outliers(df):
    """Gráficos para identificación y análisis de outliers."""
    print("\n" + "=" * 70)
    print("FASE 6: GRÁFICOS DE IDENTIFICACIÓN DE OUTLIERS")
    print("=" * 70)

    vars_outliers = ['pib_per_capita', 'tasa_homicidios', 'pob_indigena_pct', 'poblacion']

    # 6.1 Boxplots con outliers identificados
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, var in enumerate(vars_outliers):
        ax = axes[i]

        # Boxplot
        bp = ax.boxplot(df[var], vert=True, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=COLORES_PALETA[i], alpha=0.7),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=8))

        # Identificar outliers
        Q1 = df[var].quantile(0.25)
        Q3 = df[var].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)]

        # Anotar algunos outliers
        for idx, row in outliers.nlargest(3, var).iterrows():
            ax.annotate(row['canton'][:15], xy=(1, row[var]), xytext=(1.2, row[var]),
                       fontsize=8, ha='left',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))

        ax.set_title(f'{var.replace("_", " ").title()}\n({len(outliers)} outliers)',
                    fontweight='bold')
        ax.set_xticklabels([var.replace('_', ' ')])
        ax.text(0.02, 0.98, f'IQR: {IQR:.1f}\nLímite sup: {Q3 + 1.5*IQR:.1f}',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.suptitle('Boxplots con Outliers Identificados', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/28_outliers_boxplots.png')
    plt.close()
    print("✓ 28_outliers_boxplots.png")

    # 6.2 Z-scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, var in enumerate(vars_outliers):
        ax = axes[i]
        z_scores = np.abs(zscore(df[var]))

        ax.scatter(range(len(z_scores)), z_scores, c=COLORES_PALETA[i], alpha=0.6, s=30)
        ax.axhline(3, color='red', linestyle='--', linewidth=2, label='Z = 3 (outlier)')
        ax.axhline(2, color='orange', linestyle='--', linewidth=1.5, label='Z = 2')

        # Identificar outliers extremos
        outliers_idx = np.where(z_scores > 3)[0]
        for idx in outliers_idx[:5]:
            ax.annotate(df.iloc[idx]['canton'][:10], xy=(idx, z_scores[idx]),
                       xytext=(idx+5, z_scores[idx]+0.3), fontsize=7,
                       arrowprops=dict(arrowstyle='->', lw=0.5))

        ax.set_xlabel('Índice del Cantón')
        ax.set_ylabel('|Z-score|')
        ax.set_title(f'{var.replace("_", " ").title()}\n({len(outliers_idx)} cantones con |Z| > 3)',
                    fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Identificación de Outliers por Z-Score', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/29_outliers_zscore.png')
    plt.close()
    print("✓ 29_outliers_zscore.png")

    # 6.3 Top outliers por variable
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, var in enumerate(vars_outliers):
        ax = axes[i]

        # Top 10 cantones
        top10 = df.nlargest(10, var)[['canton', var, 'region']]

        bars = ax.barh(range(10), top10[var].values,
                      color=[COLORES_REGION.get(r, 'gray') for r in top10['region']])
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{c[:20]} ({r})" for c, r in zip(top10['canton'], top10['region'])],
                          fontsize=9)
        ax.set_xlabel(var.replace('_', ' ').title())
        ax.set_title(f'Top 10 Cantones con Mayor {var.replace("_", " ").title()}',
                    fontweight='bold')

        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f'{bar.get_width():.0f}', va='center', ha='left', fontsize=8)

    plt.suptitle('Top 10 Cantones por Variable (Posibles Outliers)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/30_top10_outliers.png')
    plt.close()
    print("✓ 30_top10_outliers.png")

    # 6.4 Scatter plot multivariado para detectar outliers
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(df['pib_per_capita'], df['tasa_homicidios'],
                        c=df['pob_indigena_pct'], cmap='viridis',
                        s=df['poblacion']/5000, alpha=0.6, edgecolors='black')

    plt.colorbar(scatter, label='% Población Indígena')

    # Anotar outliers extremos
    outliers_pib = df.nlargest(3, 'pib_per_capita')
    outliers_hom = df.nlargest(3, 'tasa_homicidios')

    for idx, row in pd.concat([outliers_pib, outliers_hom]).drop_duplicates().iterrows():
        ax.annotate(row['canton'][:15], xy=(row['pib_per_capita'], row['tasa_homicidios']),
                   xytext=(row['pib_per_capita']+2000, row['tasa_homicidios']+5),
                   fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.set_xlabel('PIB per cápita (USD)', fontsize=12)
    ax.set_ylabel('Tasa de Homicidios', fontsize=12)
    ax.set_title('Scatter Multivariado\n(Tamaño = Población, Color = % Indígena)',
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RUTA_GRAFICOS}/31_scatter_multivariado.png')
    plt.close()
    print("✓ 31_scatter_multivariado.png")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta la generación completa de gráficos."""
    print("=" * 70)
    print("GENERACIÓN EXTENSIVA DE VISUALIZACIONES")
    print("Análisis Estadístico de Cantones de Ecuador")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Crear directorio
    os.makedirs(RUTA_GRAFICOS, exist_ok=True)
    print(f"✓ Directorio de salida: {RUTA_GRAFICOS}/\n")

    # Cargar datos
    df = cargar_y_preparar_datos()

    # Generar todos los gráficos
    graficos_exploracion(df)
    graficos_descriptivos(df)
    graficos_normalidad(df)
    graficos_correlacion(df)
    graficos_region(df)
    graficos_outliers(df)

    # Resumen
    print("\n" + "=" * 70)
    print("GENERACIÓN DE GRÁFICOS COMPLETADA")
    print("=" * 70)

    archivos = sorted([f for f in os.listdir(RUTA_GRAFICOS) if f.endswith('.png')])
    print(f"\nTotal de gráficos generados: {len(archivos)}")
    print(f"\nArchivos en '{RUTA_GRAFICOS}/':")

    for archivo in archivos:
        print(f"  - {archivo}")

    print(f"\nFin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
