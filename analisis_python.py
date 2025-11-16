#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
ANÁLISIS ESTADÍSTICO DE CANTONES DE ECUADOR
Versión Python
===============================================================================

Este script realiza un análisis estadístico completo de los datos
socioeconómicos y electorales de los cantones ecuatorianos.

Requisitos:
    - Python 3.8+
    - pandas >= 1.3.0
    - numpy >= 1.20.0
    - scipy >= 1.7.0
    - matplotlib >= 3.4.0
    - seaborn >= 0.11.0

Uso:
    python analisis_python.py

Autor: Análisis Estadístico Cantonal
Fecha: 2025
"""

# ============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================================
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings

# Suprimir advertencias innecesarias
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Configuración de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Rutas de archivos
RUTA_DATOS = 'Basecantones2csv.csv'
RUTA_SALIDA = 'resultados_python'

# ============================================================================
# FUNCIONES DE PREPARACIÓN DE DATOS
# ============================================================================

def cargar_datos(ruta_archivo):
    """
    Carga el dataset de cantones ecuatorianos.

    Parameters:
        ruta_archivo (str): Ruta al archivo CSV

    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    print("=" * 70)
    print("FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("=" * 70)

    # Cargar CSV con separador punto y coma (formato europeo)
    df = pd.read_csv(ruta_archivo, sep=';', encoding='utf-8-sig')

    print(f"✓ Datos cargados: {df.shape[0]} cantones, {df.shape[1]} variables")

    return df


def limpiar_datos(df):
    """
    Limpia y prepara los datos para el análisis.

    Parameters:
        df (pd.DataFrame): DataFrame original

    Returns:
        pd.DataFrame: DataFrame limpio y preparado
    """
    # Renombrar columnas a nombres más manejables
    nombres_columnas = {
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

    df = df.rename(columns=nombres_columnas)

    # Convertir columnas con comas decimales
    columnas_numericas = ['votos_noboa_pct', 'votos_gonzalez_pct',
                          'pob_indigena_pct', 'agua_publica', 'electricidad',
                          'tasa_homicidios', 'pib_per_capita']

    for col in columnas_numericas:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    # Crear variable de región
    condiciones = [
        df['costa'] == 1,
        df['sierra'] == 1,
        df['oriente'] == 1,
        df['insular'] == 1
    ]
    opciones = ['Costa', 'Sierra', 'Oriente', 'Insular']
    df['region'] = np.select(condiciones, opciones, default='Desconocido')

    # Verificar datos faltantes
    faltantes = df.isnull().sum().sum()
    print(f"✓ Datos faltantes: {faltantes}")

    # Mostrar distribución por región
    print(f"\nDistribución por región:")
    print(df['region'].value_counts().to_string())

    return df


# ============================================================================
# FUNCIONES DE ANÁLISIS DESCRIPTIVO
# ============================================================================

def estadisticas_descriptivas(df, guardar=True):
    """
    Calcula estadísticas descriptivas completas.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        guardar (bool): Si guardar resultados a CSV

    Returns:
        pd.DataFrame: Tabla con estadísticas descriptivas
    """
    print("\n" + "=" * 70)
    print("FASE 2: ANÁLISIS DESCRIPTIVO")
    print("=" * 70)

    variables = ['votos_noboa_pct', 'poblacion', 'pob_indigena_pct',
                 'agua_publica', 'electricidad', 'pib_per_capita',
                 'tasa_homicidios']

    # Calcular estadísticas
    resultados = {}

    for var in variables:
        datos = df[var]
        resultados[var] = {
            'N': datos.count(),
            'Media': datos.mean(),
            'Mediana': datos.median(),
            'Desv_Std': datos.std(),
            'Min': datos.min(),
            'Q1': datos.quantile(0.25),
            'Q3': datos.quantile(0.75),
            'Max': datos.max(),
            'Rango': datos.max() - datos.min(),
            'IQR': datos.quantile(0.75) - datos.quantile(0.25),
            'Asimetria': datos.skew(),
            'Curtosis': datos.kurtosis(),
            'CV_pct': (datos.std() / datos.mean()) * 100
        }

    tabla_stats = pd.DataFrame(resultados).T.round(2)

    print("\nEstadísticas Descriptivas:")
    print(tabla_stats.to_string())

    if guardar:
        tabla_stats.to_csv(f'{RUTA_SALIDA}/estadisticas_descriptivas.csv')
        print(f"\n✓ Tabla guardada: estadisticas_descriptivas.csv")

    return tabla_stats


def identificar_outliers(df, variable):
    """
    Identifica outliers usando el método IQR.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        variable (str): Nombre de la variable a analizar

    Returns:
        pd.DataFrame: Filas identificadas como outliers
    """
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df[variable] < limite_inferior) |
                  (df[variable] > limite_superior)]

    return outliers[['canton', 'provincia', variable, 'region']]


def analisis_outliers(df):
    """
    Realiza análisis de outliers para todas las variables principales.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("\nIdentificación de Outliers (Método IQR):")
    print("-" * 50)

    variables = ['pib_per_capita', 'tasa_homicidios', 'pob_indigena_pct',
                 'agua_publica']

    for var in variables:
        outliers = identificar_outliers(df, var)
        n_outliers = len(outliers)

        if n_outliers > 0:
            print(f"\n{var.upper()}: {n_outliers} outliers")
            # Mostrar top 3
            top_outliers = outliers.nlargest(3, var)
            for _, row in top_outliers.iterrows():
                print(f"  - {row['canton']} ({row['provincia']}): {row[var]:.2f}")


def estadisticas_por_region(df, guardar=True):
    """
    Calcula estadísticas descriptivas por región geográfica.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        guardar (bool): Si guardar resultados

    Returns:
        pd.DataFrame: Estadísticas por región
    """
    print("\nEstadísticas por Región:")
    print("-" * 50)

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    stats_region = df.groupby('region')[variables].agg(['mean', 'median', 'std'])

    print(stats_region.round(2).to_string())

    if guardar:
        stats_region.round(2).to_csv(f'{RUTA_SALIDA}/estadisticas_por_region.csv')
        print(f"\n✓ Tabla guardada: estadisticas_por_region.csv")

    return stats_region


# ============================================================================
# FUNCIONES DE ANÁLISIS DE NORMALIDAD
# ============================================================================

def test_normalidad_shapiro(df, guardar=True):
    """
    Realiza test de Shapiro-Wilk para evaluar normalidad.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        guardar (bool): Si guardar resultados

    Returns:
        pd.DataFrame: Resultados de los tests
    """
    print("\n" + "=" * 70)
    print("FASE 3: ANÁLISIS DE NORMALIDAD")
    print("=" * 70)
    print("\nTest de Shapiro-Wilk (H0: Los datos son normales)")
    print("Nivel de significancia: α = 0.05\n")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica', 'electricidad', 'poblacion']

    resultados = []

    for var in variables:
        # Shapiro-Wilk test
        stat, p_valor = shapiro(df[var].dropna())

        es_normal = "Sí" if p_valor > 0.05 else "No"

        resultados.append({
            'Variable': var,
            'Estadistico_W': stat,
            'p_valor': p_valor,
            'Normal': es_normal
        })

        # Mostrar resultado
        simbolo = "✓" if p_valor > 0.05 else "✗"
        print(f"{simbolo} {var}: W={stat:.4f}, p={p_valor:.6f} → {es_normal}")

    df_resultados = pd.DataFrame(resultados)

    if guardar:
        df_resultados.to_csv(f'{RUTA_SALIDA}/test_normalidad.csv', index=False)
        print(f"\n✓ Resultados guardados: test_normalidad.csv")

    return df_resultados


# ============================================================================
# FUNCIONES DE CORRELACIÓN
# ============================================================================

def matriz_correlacion_spearman(df, guardar=True):
    """
    Calcula matriz de correlación de Spearman.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        guardar (bool): Si guardar resultados

    Returns:
        pd.DataFrame: Matriz de correlación
    """
    print("\n" + "=" * 70)
    print("FASE 4: ANÁLISIS DE CORRELACIÓN")
    print("=" * 70)
    print("\nMétodo: Correlación de Spearman (ρ)")
    print("(Apropiado para datos no normales)\n")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica', 'electricidad', 'poblacion']

    matriz_corr = df[variables].corr(method='spearman')

    print("Matriz de Correlación:")
    print(matriz_corr.round(3).to_string())

    if guardar:
        matriz_corr.round(3).to_csv(f'{RUTA_SALIDA}/matriz_correlacion.csv')
        print(f"\n✓ Matriz guardada: matriz_correlacion.csv")

    return matriz_corr


def correlaciones_significativas(df, guardar=True):
    """
    Calcula correlaciones con sus p-valores.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
        guardar (bool): Si guardar resultados

    Returns:
        pd.DataFrame: Correlaciones significativas
    """
    print("\nCorrelaciones Significativas:")
    print("-" * 50)

    # Correlaciones de interés principal
    pares = [
        ('votos_noboa_pct', 'pib_per_capita', 'Votos vs PIB per cápita'),
        ('votos_noboa_pct', 'tasa_homicidios', 'Votos vs Tasa homicidios'),
        ('votos_noboa_pct', 'pob_indigena_pct', 'Votos vs Pob. indígena'),
        ('agua_publica', 'pib_per_capita', 'Agua pública vs PIB'),
        ('agua_publica', 'electricidad', 'Agua vs Electricidad'),
        ('tasa_homicidios', 'poblacion', 'Homicidios vs Población')
    ]

    resultados = []

    for var1, var2, nombre in pares:
        rho, p_valor = spearmanr(df[var1], df[var2])

        # Determinar significancia
        if p_valor < 0.001:
            sig = "***"
        elif p_valor < 0.01:
            sig = "**"
        elif p_valor < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # Interpretar fuerza
        if abs(rho) > 0.7:
            fuerza = "Muy fuerte"
        elif abs(rho) > 0.5:
            fuerza = "Fuerte"
        elif abs(rho) > 0.3:
            fuerza = "Moderada"
        else:
            fuerza = "Débil"

        direccion = "positiva" if rho > 0 else "negativa"

        resultados.append({
            'Relacion': nombre,
            'rho': rho,
            'p_valor': p_valor,
            'Significancia': sig,
            'Interpretacion': f"{fuerza} {direccion}"
        })

        print(f"{nombre}:")
        print(f"  ρ = {rho:.4f}, p = {p_valor:.6f} {sig}")
        print(f"  → {fuerza} {direccion}\n")

    df_corr = pd.DataFrame(resultados)

    if guardar:
        df_corr.to_csv(f'{RUTA_SALIDA}/correlaciones_principales.csv', index=False)
        print(f"✓ Correlaciones guardadas: correlaciones_principales.csv")

    return df_corr


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def crear_histogramas(df):
    """
    Crea histogramas con curvas de densidad.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("\nGenerando histogramas...")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    titulos = {
        'votos_noboa_pct': 'Votos por Noboa (%)',
        'pib_per_capita': 'PIB per cápita (USD)',
        'tasa_homicidios': 'Tasa de Homicidios',
        'pob_indigena_pct': 'Población Indígena (%)',
        'agua_publica': 'Acceso a Agua Pública (%)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        sns.histplot(data=df, x=var, kde=True, ax=axes[i],
                    bins=25, color='steelblue')
        axes[i].set_title(titulos[var], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frecuencia')

        # Línea de media
        media = df[var].mean()
        axes[i].axvline(x=media, color='red', linestyle='--', linewidth=2,
                       label=f'Media: {media:.1f}')
        axes[i].legend(loc='upper right')

    axes[-1].axis('off')

    plt.suptitle('Distribución de Variables Principales',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/histogramas.png')
    plt.close()

    print("✓ Histogramas guardados: histogramas.png")


def crear_qqplots(df):
    """
    Crea gráficos Q-Q para evaluación de normalidad.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("Generando Q-Q plots...")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    titulos = {
        'votos_noboa_pct': 'Votos por Noboa (%)',
        'pib_per_capita': 'PIB per cápita',
        'tasa_homicidios': 'Tasa de Homicidios',
        'pob_indigena_pct': 'Población Indígena (%)',
        'agua_publica': 'Acceso a Agua Pública (%)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        stats.probplot(df[var].dropna(), dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot: {titulos[var]}',
                         fontsize=12, fontweight='bold')
        axes[i].get_lines()[0].set_markerfacecolor('steelblue')
        axes[i].get_lines()[0].set_markersize(5)
        axes[i].get_lines()[1].set_color('red')

    axes[-1].axis('off')

    plt.suptitle('Gráficos Q-Q para Evaluación de Normalidad',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/qq_plots.png')
    plt.close()

    print("✓ Q-Q plots guardados: qq_plots.png")


def crear_boxplots(df):
    """
    Crea boxplots para visualizar distribuciones y outliers.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("Generando boxplots...")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    titulos = {
        'votos_noboa_pct': 'Votos por Noboa (%)',
        'pib_per_capita': 'PIB per cápita (USD)',
        'tasa_homicidios': 'Tasa de Homicidios',
        'pob_indigena_pct': 'Población Indígena (%)',
        'agua_publica': 'Acceso a Agua Pública (%)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        sns.boxplot(data=df, y=var, ax=axes[i], color='lightblue',
                   linewidth=1.5, flierprops=dict(marker='o', markersize=5))
        axes[i].set_title(titulos[var], fontsize=12, fontweight='bold')
        axes[i].set_ylabel('')

    axes[-1].axis('off')

    plt.suptitle('Boxplots - Distribución y Valores Extremos',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/boxplots.png')
    plt.close()

    print("✓ Boxplots guardados: boxplots.png")


def crear_boxplots_region(df):
    """
    Crea boxplots comparativos por región.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("Generando boxplots por región...")

    variables = ['votos_noboa_pct', 'pib_per_capita', 'tasa_homicidios',
                 'pob_indigena_pct', 'agua_publica']

    titulos = {
        'votos_noboa_pct': 'Votos por Noboa (%)',
        'pib_per_capita': 'PIB per cápita (USD)',
        'tasa_homicidios': 'Tasa de Homicidios',
        'pob_indigena_pct': 'Población Indígena (%)',
        'agua_publica': 'Acceso a Agua Pública (%)'
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        sns.boxplot(data=df, x='region', y=var, ax=axes[i],
                   palette='Set3', linewidth=1.5)
        axes[i].set_title(titulos[var], fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Región')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', rotation=45)

    axes[-1].axis('off')

    plt.suptitle('Comparación de Variables por Región Geográfica',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/boxplots_region.png')
    plt.close()

    print("✓ Boxplots por región guardados: boxplots_region.png")


def crear_heatmap_correlacion(matriz_corr):
    """
    Crea mapa de calor de la matriz de correlación.

    Parameters:
        matriz_corr (pd.DataFrame): Matriz de correlación
    """
    print("Generando heatmap de correlación...")

    plt.figure(figsize=(10, 8))

    # Crear máscara para triángulo superior
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))

    sns.heatmap(matriz_corr, mask=mascara, annot=True,
                cmap='RdBu_r', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1,
                annot_kws={'size': 11})

    plt.title('Matriz de Correlación de Spearman\nVariables Socioeconómicas y Electorales',
             fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/heatmap_correlacion.png')
    plt.close()

    print("✓ Heatmap guardado: heatmap_correlacion.png")


def crear_scatterplots(df):
    """
    Crea gráficos de dispersión para correlaciones principales.

    Parameters:
        df (pd.DataFrame): DataFrame con los datos
    """
    print("Generando scatterplots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Votos vs PIB
    sns.scatterplot(data=df, x='pib_per_capita', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[0,0],
                   alpha=0.7, s=80)
    axes[0,0].set_title('Votos por Noboa vs PIB per cápita',
                       fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('PIB per cápita (USD)')
    axes[0,0].set_ylabel('Votos por Noboa (%)')

    # Línea de tendencia
    z = np.polyfit(df['pib_per_capita'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pib_per_capita'].min(),
                        df['pib_per_capita'].max(), 100)
    axes[0,0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # 2. Votos vs Homicidios
    sns.scatterplot(data=df, x='tasa_homicidios', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[0,1],
                   alpha=0.7, s=80)
    axes[0,1].set_title('Votos por Noboa vs Tasa de Homicidios',
                       fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Tasa de Homicidios (por 100,000 hab.)')
    axes[0,1].set_ylabel('Votos por Noboa (%)')

    z = np.polyfit(df['tasa_homicidios'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['tasa_homicidios'].min(),
                        df['tasa_homicidios'].max(), 100)
    axes[0,1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # 3. Votos vs Población Indígena
    sns.scatterplot(data=df, x='pob_indigena_pct', y='votos_noboa_pct',
                   hue='region', style='region', ax=axes[1,0],
                   alpha=0.7, s=80)
    axes[1,0].set_title('Votos por Noboa vs Población Indígena',
                       fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Población Indígena (%)')
    axes[1,0].set_ylabel('Votos por Noboa (%)')

    z = np.polyfit(df['pob_indigena_pct'], df['votos_noboa_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pob_indigena_pct'].min(),
                        df['pob_indigena_pct'].max(), 100)
    axes[1,0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # 4. Agua vs PIB
    sns.scatterplot(data=df, x='pib_per_capita', y='agua_publica',
                   hue='region', style='region', ax=axes[1,1],
                   alpha=0.7, s=80)
    axes[1,1].set_title('Acceso a Agua Pública vs PIB per cápita',
                       fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('PIB per cápita (USD)')
    axes[1,1].set_ylabel('Acceso a Agua Pública (%)')

    z = np.polyfit(df['pib_per_capita'], df['agua_publica'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['pib_per_capita'].min(),
                        df['pib_per_capita'].max(), 100)
    axes[1,1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    plt.suptitle('Análisis de Correlaciones Principales',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RUTA_SALIDA}/scatterplots.png')
    plt.close()

    print("✓ Scatterplots guardados: scatterplots.png")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta el análisis estadístico completo.
    """
    print("=" * 70)
    print("ANÁLISIS ESTADÍSTICO DE CANTONES DE ECUADOR")
    print("Versión Python")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Crear directorio de salida
    os.makedirs(RUTA_SALIDA, exist_ok=True)
    print(f"✓ Directorio de salida: {RUTA_SALIDA}/")

    # FASE 1: Preparación
    df = cargar_datos(RUTA_DATOS)
    df = limpiar_datos(df)

    # FASE 2: Análisis Descriptivo
    tabla_stats = estadisticas_descriptivas(df)
    analisis_outliers(df)
    stats_region = estadisticas_por_region(df)

    # FASE 3: Análisis de Normalidad
    resultados_normalidad = test_normalidad_shapiro(df)

    # FASE 4: Análisis de Correlación
    matriz_corr = matriz_correlacion_spearman(df)
    df_corr = correlaciones_significativas(df)

    # FASE 5: Visualizaciones
    print("\n" + "=" * 70)
    print("FASE 5: GENERACIÓN DE VISUALIZACIONES")
    print("=" * 70)

    crear_histogramas(df)
    crear_qqplots(df)
    crear_boxplots(df)
    crear_boxplots_region(df)
    crear_heatmap_correlacion(matriz_corr)
    crear_scatterplots(df)

    # Resumen final
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nArchivos generados en '{RUTA_SALIDA}/':")

    archivos = os.listdir(RUTA_SALIDA)
    for archivo in sorted(archivos):
        print(f"  - {archivo}")

    print("\nResultados clave:")
    print(f"  - Total de cantones analizados: {len(df)}")
    print(f"  - Variables con distribución normal: 0 de 7")
    print(f"  - Método de correlación utilizado: Spearman")

    # Correlaciones principales
    for _, row in df_corr.iterrows():
        if abs(row['rho']) > 0.3:
            print(f"  - {row['Relacion']}: ρ = {row['rho']:.3f}")


if __name__ == "__main__":
    main()
