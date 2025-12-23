#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS DE VOTACIÓN INDÍGENA EN ECUADOR - ELECCIONES PRESIDENCIALES 2025
================================================================================

Investigación: Patrones de voto homogéneo en cantones con alta población indígena
Autor: Análisis Estadístico Electoral
Fecha: 2025

OBJETIVO:
---------
Examinar si existe un patrón de voto homogéneo entre cantones ecuatorianos con
alta población indígena en las elecciones presidenciales de 2025, y qué factores
socioeconómicos y territoriales explican las variaciones en sus preferencias
electorales bajo condiciones de polarización bidimensional.

HIPÓTESIS:
----------
H1: Existe un patrón de voto homogéneo pro-correísta en cantones con alta población indígena
H2: La polarización económica (PIB per cápita) modera la relación entre población indígena y voto
H3: El contexto territorial (Costa vs Sierra vs Amazonía) media la preferencia electoral indígena

METODOLOGÍA:
------------
- Variable dependiente: Proporción de votos válidos para Luisa González (correísta)
- Método: Regresión logística con familia binomial (variable acotada 0-1, no normal)
- Ponderación: Por número de votos válidos cantonales
- Control multicolinealidad: Pruebas VIF < 5
- Nivel de significancia: α = 0.05

VARIABLES:
----------
Dependiente:
  - prop_gonzalez: Proporción de votos válidos para Luisa González

Independientes principales:
  - pob_indigena_pct: Porcentaje de población indígena cantonal
  - log_pib_pc: Logaritmo del PIB per cápita
  - costa: Dummy región Costa (ref: Sierra)
  - amazonia: Dummy región Amazonía/Oriente (ref: Sierra)

Interacciones (heterogeneidad contextual):
  - indigena_x_logpib: % indígena × log PIB per cápita (H2: polarización económica)
  - indigena_x_costa: % indígena × Costa (H3: mediación territorial)
  - indigena_x_amazonia: % indígena × Amazonía (H3: mediación territorial)

Variables de control (disponibles en dataset):
  - agua_publica: Acceso a agua de red pública (%)
  - electricidad: Acceso a electricidad (%)
  - tasa_homicidios: Tasa de homicidios por 100,000 hab
  - altitud: Altitud en metros (proxy ruralidad/dispersión)
  - poblacion: Población total (control tamaño)

NOTA: Variables de control adicionales mencionadas en el diseño original
(pobreza, desempleo, densidad poblacional, % rural, educación, NBI) no están
disponibles en el dataset actual. Se recomienda incorporarlas si se obtienen.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# ============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

def cargar_datos(archivo='Basecantones2csv.csv'):
    """
    Carga el dataset de cantones ecuatorianos
    """
    print("="*80)
    print("FASE 1: CARGA Y PREPARACIÓN DE DATOS")
    print("="*80)

    # Leer CSV con formato (separador ; y decimal .)
    df = pd.read_csv(archivo, sep=';', decimal='.', encoding='utf-8-sig')

    print(f"\n✓ Dataset cargado: {len(df)} cantones")
    print(f"✓ Variables: {len(df.columns)} columnas")

    return df

def preparar_variables(df):
    """
    Prepara las variables para el análisis de regresión logística binomial
    """
    print("\n" + "-"*80)
    print("Preparación de variables")
    print("-"*80)

    # Crear copia para trabajar
    datos = df.copy()

    # Renombrar columnas para facilitar manejo
    datos.columns = ['canton', 'provincia', 'votos_noboa_abs', 'votos_gonzalez_abs',
                     'votos_noboa_pct', 'votos_gonzalez_pct', 'poblacion',
                     'pob_indigena_pct', 'agua_publica', 'electricidad',
                     'pib_per_capita', 'tasa_homicidios', 'altitud',
                     'costa', 'sierra', 'oriente', 'insular']

    # Convertir columnas numéricas a tipo numérico
    columnas_numericas = ['votos_noboa_abs', 'votos_gonzalez_abs', 'votos_noboa_pct',
                         'votos_gonzalez_pct', 'poblacion', 'pob_indigena_pct',
                         'agua_publica', 'electricidad', 'pib_per_capita',
                         'tasa_homicidios', 'altitud', 'costa', 'sierra', 'oriente', 'insular']

    for col in columnas_numericas:
        datos[col] = pd.to_numeric(datos[col], errors='coerce')

    # ========================================================================
    # VARIABLE DEPENDIENTE: Proporción de votos para González
    # ========================================================================
    # Convertir porcentaje a proporción (0-1) para regresión binomial
    datos['prop_gonzalez'] = datos['votos_gonzalez_pct'] / 100

    # Verificar que esté acotada [0,1]
    assert datos['prop_gonzalez'].min() >= 0 and datos['prop_gonzalez'].max() <= 1, \
        "ERROR: prop_gonzalez fuera del rango [0,1]"

    # ========================================================================
    # PESOS: Votos válidos totales (para ponderar observaciones)
    # ========================================================================
    datos['votos_validos'] = datos['votos_noboa_abs'] + datos['votos_gonzalez_abs']

    # ========================================================================
    # VARIABLES INDEPENDIENTES PRINCIPALES
    # ========================================================================

    # 1. Porcentaje población indígena (ya está en el dataset)
    # Verificar valores válidos
    datos['pob_indigena_pct'] = datos['pob_indigena_pct'].clip(lower=0, upper=100)

    # 2. Logaritmo del PIB per cápita (transformación para normalizar)
    # Usar log natural para interpretación como elasticidad
    datos['log_pib_pc'] = np.log(datos['pib_per_capita'])

    # Verificar valores infinitos
    if np.isinf(datos['log_pib_pc']).any():
        print("⚠ Advertencia: Valores infinitos en log_pib_pc, reemplazando con mediana")
        mediana_log_pib = datos['log_pib_pc'][~np.isinf(datos['log_pib_pc'])].median()
        datos.loc[np.isinf(datos['log_pib_pc']), 'log_pib_pc'] = mediana_log_pib

    # 3. Variables dummy regionales (Sierra es la categoría de referencia)
    # Costa ya está en el dataset
    # Renombrar Oriente a Amazonía para claridad conceptual
    datos['amazonia'] = datos['oriente']

    # Eliminar región Insular del análisis (solo 3 cantones, grupo muy pequeño)
    cantones_insular = datos['insular'].sum()
    datos = datos[datos['insular'] == 0].copy()

    print(f"\n✓ Cantones excluidos (región Insular): {cantones_insular}")
    print(f"✓ Cantones en análisis: {len(datos)}")

    # ========================================================================
    # TÉRMINOS DE INTERACCIÓN (Heterogeneidad contextual)
    # ========================================================================

    # H2: Polarización económica - % indígena × log PIB per cápita
    datos['indigena_x_logpib'] = datos['pob_indigena_pct'] * datos['log_pib_pc']

    # H3: Mediación territorial
    # % indígena × Costa
    datos['indigena_x_costa'] = datos['pob_indigena_pct'] * datos['costa']

    # % indígena × Amazonía
    datos['indigena_x_amazonia'] = datos['pob_indigena_pct'] * datos['amazonia']

    # ========================================================================
    # VARIABLES DE CONTROL
    # ========================================================================
    # Ya están en el dataset: agua_publica, electricidad, tasa_homicidios, altitud, poblacion

    # Logaritmo de población (control de escala)
    datos['log_poblacion'] = np.log(datos['poblacion'])

    # ========================================================================
    # LIMPIEZA DE DATOS FALTANTES
    # ========================================================================

    # Verificar valores faltantes
    missing = datos.isnull().sum()
    if missing.any():
        print("\n⚠ Valores faltantes detectados:")
        print(missing[missing > 0])

        # Imputar con mediana para variables numéricas
        for col in datos.select_dtypes(include=[np.number]).columns:
            if datos[col].isnull().any():
                mediana = datos[col].median()
                datos[col].fillna(mediana, inplace=True)
                print(f"  → {col}: imputado con mediana = {mediana:.2f}")

    print("\n✓ Variables preparadas exitosamente")
    print(f"✓ Observaciones finales: {len(datos)}")

    return datos

# ============================================================================
# FASE 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

def analisis_descriptivo(datos):
    """
    Análisis descriptivo de las variables clave
    """
    print("\n" + "="*80)
    print("FASE 2: ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*80)

    # Variables de interés
    vars_interes = ['prop_gonzalez', 'pob_indigena_pct', 'log_pib_pc',
                    'agua_publica', 'electricidad', 'tasa_homicidios', 'altitud']

    # Estadísticas descriptivas
    desc = datos[vars_interes].describe().T
    desc['CV'] = (desc['std'] / desc['mean']) * 100  # Coeficiente de variación
    desc['asimetria'] = datos[vars_interes].skew()
    desc['curtosis'] = datos[vars_interes].kurtosis()

    print("\nEstadísticas Descriptivas:")
    print("-"*80)
    print(desc.round(3))

    # Distribución por región
    print("\n\nDistribución de cantones por región:")
    print("-"*80)
    region_dist = pd.crosstab(
        index=[datos['costa'], datos['sierra'], datos['amazonia']],
        columns='Frecuencia'
    )
    region_names = {
        (1,0,0): 'Costa',
        (0,1,0): 'Sierra',
        (0,0,1): 'Amazonía'
    }

    for region_code, freq in region_dist.itertuples():
        region_name = region_names.get(region_code, 'Otra')
        print(f"  {region_name}: {freq} cantones ({freq/len(datos)*100:.1f}%)")

    # Estadísticas de voto por región
    print("\n\nVoto promedio por González según región:")
    print("-"*80)

    regiones = {
        'Costa': datos[datos['costa'] == 1],
        'Sierra': datos[datos['sierra'] == 1],
        'Amazonía': datos[datos['amazonia'] == 1]
    }

    for nombre, subgrupo in regiones.items():
        if len(subgrupo) > 0:
            media = subgrupo['prop_gonzalez'].mean() * 100
            std = subgrupo['prop_gonzalez'].std() * 100
            print(f"  {nombre}: {media:.2f}% (SD={std:.2f}%)")

    # Correlación bivariada clave
    print("\n\nCorrelación bivariada (Pearson):")
    print("-"*80)

    correlaciones_clave = [
        ('prop_gonzalez', 'pob_indigena_pct', 'Voto González vs % Población Indígena'),
        ('prop_gonzalez', 'log_pib_pc', 'Voto González vs Log PIB per cápita'),
        ('pob_indigena_pct', 'log_pib_pc', '% Población Indígena vs Log PIB per cápita')
    ]

    for var1, var2, descripcion in correlaciones_clave:
        r, p = pearsonr(datos[var1].dropna(), datos[var2].dropna())
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {descripcion}:")
        print(f"    r = {r:.4f}, p = {p:.4f} {sig}")

    return desc

# ============================================================================
# FASE 3: DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)
# ============================================================================

def calcular_vif(datos, variables_modelo):
    """
    Calcula el Variance Inflation Factor (VIF) para detectar multicolinealidad

    Criterio: VIF < 5 indica ausencia de multicolinealidad problemática
              VIF 5-10 indica multicolinealidad moderada
              VIF > 10 indica multicolinealidad severa
    """
    print("\n" + "="*80)
    print("FASE 3: DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)")
    print("="*80)

    print("\nCriterio conservador: VIF < 5")
    print("  VIF < 5:   ✓ Sin multicolinealidad problemática")
    print("  VIF 5-10:  ⚠ Multicolinealidad moderada")
    print("  VIF > 10:  ✗ Multicolinealidad severa")

    # Preparar datos para VIF (solo variables numéricas)
    X = datos[variables_modelo].copy()

    # Agregar constante (intercepto)
    X_const = sm.add_constant(X)

    # Calcular VIF para cada variable
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X_const.values, i+1)
                       for i in range(len(X.columns))]

    # Ordenar por VIF descendente
    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("\n\nResultados VIF:")
    print("-"*80)
    for idx, row in vif_data.iterrows():
        var = row['Variable']
        vif = row['VIF']

        if vif < 5:
            status = "✓ OK"
        elif vif < 10:
            status = "⚠ MODERADO"
        else:
            status = "✗ SEVERO"

        print(f"  {var:30s} VIF = {vif:7.3f}  {status}")

    # Verificar umbral conservador
    max_vif = vif_data['VIF'].max()
    problematicos = vif_data[vif_data['VIF'] >= 5]

    print("\n" + "-"*80)
    if max_vif < 5:
        print("✓ DIAGNÓSTICO: Todos los VIF < 5. No hay multicolinealidad problemática.")
        print("  El modelo cumple con el criterio conservador.")
    elif max_vif < 10:
        print("⚠ ADVERTENCIA: Algunos VIF entre 5-10 (multicolinealidad moderada):")
        for _, row in problematicos.iterrows():
            print(f"    - {row['Variable']}: VIF = {row['VIF']:.3f}")
        print("  Considerar centrar variables antes de crear interacciones.")
    else:
        print("✗ ALERTA: Multicolinealidad severa detectada (VIF > 10):")
        for _, row in problematicos.iterrows():
            print(f"    - {row['Variable']}: VIF = {row['VIF']:.3f}")
        print("  RECOMENDACIÓN: Eliminar términos de interacción o centrar variables.")

    return vif_data

# ============================================================================
# FASE 4: REGRESIÓN LOGÍSTICA BINOMIAL
# ============================================================================

def regresion_logistica_binomial(datos):
    """
    Estima modelo de regresión logística con familia binomial

    La variable dependiente es una proporción (0-1), por lo que se usa:
    - Familia: Binomial
    - Link: Logit (transformación log-odds)
    - Ponderación: Por votos válidos (frecuency weights)
    """
    print("\n" + "="*80)
    print("FASE 4: REGRESIÓN LOGÍSTICA BINOMIAL")
    print("="*80)

    print("\nEspecificación del modelo:")
    print("-"*80)
    print("Familia: Binomial (apropiada para proporciones [0,1])")
    print("Link: Logit")
    print("Ponderación: Votos válidos cantonales")
    print("Estimación: Máxima verosimilitud (MLE)")

    # ========================================================================
    # MODELO 1: Solo efectos principales (sin interacciones)
    # ========================================================================

    print("\n\n" + "="*80)
    print("MODELO 1: EFECTOS PRINCIPALES (Sin interacciones)")
    print("="*80)

    formula_m1 = """prop_gonzalez ~ pob_indigena_pct + log_pib_pc + costa + amazonia +
                    agua_publica + electricidad + tasa_homicidios + altitud + log_poblacion"""

    modelo1 = smf.glm(
        formula=formula_m1,
        data=datos,
        family=sm.families.Binomial(),
        freq_weights=datos['votos_validos']
    ).fit()

    print("\n" + modelo1.summary().as_text())

    # ========================================================================
    # MODELO 2: Con interacciones (H2 y H3)
    # ========================================================================

    print("\n\n" + "="*80)
    print("MODELO 2: CON INTERACCIONES (H2: Polarización económica, H3: Mediación territorial)")
    print("="*80)

    formula_m2 = """prop_gonzalez ~ pob_indigena_pct + log_pib_pc + costa + amazonia +
                    indigena_x_logpib + indigena_x_costa + indigena_x_amazonia +
                    agua_publica + electricidad + tasa_homicidios + altitud + log_poblacion"""

    modelo2 = smf.glm(
        formula=formula_m2,
        data=datos,
        family=sm.families.Binomial(),
        freq_weights=datos['votos_validos']
    ).fit()

    print("\n" + modelo2.summary().as_text())

    # ========================================================================
    # COMPARACIÓN DE MODELOS
    # ========================================================================

    print("\n\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)

    # Test de razón de verosimilitud (LR test)
    lr_stat = -2 * (modelo1.llf - modelo2.llf)
    df_diff = modelo2.df_model - modelo1.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)

    print(f"\nTest de Razón de Verosimilitud (Likelihood Ratio Test):")
    print(f"  H0: Modelo 1 (sin interacciones) es suficiente")
    print(f"  Ha: Modelo 2 (con interacciones) explica significativamente mejor")
    print(f"\n  LR statistic = {lr_stat:.4f}")
    print(f"  df = {df_diff}")
    print(f"  p-value = {p_value:.6f}")

    if p_value < 0.05:
        print(f"\n  ✓ RESULTADO: Rechazamos H0 (p < 0.05)")
        print(f"    Las interacciones mejoran significativamente el modelo.")
        print(f"    MODELO PREFERIDO: Modelo 2 (con interacciones)")
    else:
        print(f"\n  → RESULTADO: No rechazamos H0 (p ≥ 0.05)")
        print(f"    Las interacciones no mejoran significativamente el modelo.")
        print(f"    MODELO PREFERIDO: Modelo 1 (más parsimonioso)")

    # Criterios de información
    print(f"\n\nCriterios de Información:")
    print(f"{'':30s} {'Modelo 1':>15s} {'Modelo 2':>15s} {'Diferencia':>15s}")
    print("-"*80)
    print(f"{'AIC (menor es mejor)':30s} {modelo1.aic:15.2f} {modelo2.aic:15.2f} {modelo2.aic - modelo1.aic:15.2f}")
    print(f"{'BIC (menor es mejor)':30s} {modelo1.bic:15.2f} {modelo2.bic:15.2f} {modelo2.bic - modelo1.bic:15.2f}")
    print(f"{'Log-Likelihood':30s} {modelo1.llf:15.2f} {modelo2.llf:15.2f} {modelo2.llf - modelo1.llf:15.2f}")

    # Pseudo R²
    print(f"\n\nPseudo R² (McFadden):")
    print(f"  Modelo 1: {modelo1.pseudo_rsquared():.4f}")
    print(f"  Modelo 2: {modelo2.pseudo_rsquared():.4f}")
    print(f"  Incremento: {modelo2.pseudo_rsquared() - modelo1.pseudo_rsquared():.4f}")

    return modelo1, modelo2

# ============================================================================
# FASE 5: INTERPRETACIÓN Y EFECTOS MARGINALES
# ============================================================================

def interpretar_resultados(modelo, datos):
    """
    Interpreta los coeficientes del modelo logístico en términos de odds ratios
    y efectos marginales promedio
    """
    print("\n" + "="*80)
    print("FASE 5: INTERPRETACIÓN DE RESULTADOS")
    print("="*80)

    # ========================================================================
    # Odds Ratios (exponencial de coeficientes)
    # ========================================================================

    print("\n\nOdds Ratios (OR) e Intervalos de Confianza 95%:")
    print("-"*80)
    print("Interpretación: OR > 1 aumenta la probabilidad de votar por González")
    print("                OR < 1 disminuye la probabilidad de votar por González")
    print("                OR = 1 no tiene efecto")
    print("")

    # Obtener coeficientes e IC
    params = modelo.params
    conf_int = modelo.conf_int()

    # Calcular odds ratios
    odds_ratios = np.exp(params)
    or_ci_lower = np.exp(conf_int[0])
    or_ci_upper = np.exp(conf_int[1])

    # Crear tabla
    or_table = pd.DataFrame({
        'Coef': params,
        'OR': odds_ratios,
        'IC_95%_inferior': or_ci_lower,
        'IC_95%_superior': or_ci_upper,
        'p_value': modelo.pvalues
    })

    # Significancia
    or_table['Sig'] = or_table['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    print(or_table.round(4))

    # ========================================================================
    # Efectos Marginales Promedio (AME)
    # ========================================================================

    print("\n\nEfectos Marginales Promedio (AME):")
    print("-"*80)
    print("Cambio en probabilidad de votar por González ante cambio unitario en X")
    print("")

    # Calcular efectos marginales
    margeff = modelo.get_margeff()

    print(margeff.summary())

    # ========================================================================
    # Interpretación narrativa de variables clave
    # ========================================================================

    print("\n\n" + "="*80)
    print("INTERPRETACIÓN NARRATIVA DE HIPÓTESIS")
    print("="*80)

    var_interes = ['pob_indigena_pct', 'log_pib_pc', 'costa', 'amazonia']

    # Agregar interacciones si existen
    if 'indigena_x_logpib' in modelo.params.index:
        var_interes.extend(['indigena_x_logpib', 'indigena_x_costa', 'indigena_x_amazonia'])

    for var in var_interes:
        if var in modelo.params.index:
            coef = modelo.params[var]
            pval = modelo.pvalues[var]
            odds_ratio = np.exp(coef)

            sig_text = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'

            print(f"\n{var}:")
            print(f"  Coeficiente: {coef:.4f} {sig_text}")
            print(f"  Odds Ratio: {odds_ratio:.4f}")
            print(f"  p-value: {pval:.6f}")

            # Interpretación específica
            if var == 'pob_indigena_pct':
                if pval < 0.05:
                    if coef > 0:
                        print(f"  → Un aumento de 1% en población indígena incrementa los odds de votar")
                        print(f"    por González en {(odds_ratio - 1) * 100:.2f}%")
                        print(f"  → CONFIRMA H1: Patrón de voto pro-correísta en cantones indígenas")
                    else:
                        print(f"  → Un aumento de 1% en población indígena reduce los odds de votar")
                        print(f"    por González en {(1 - odds_ratio) * 100:.2f}%")
                        print(f"  → RECHAZA H1: No hay patrón pro-correísta")
                else:
                    print(f"  → Efecto NO significativo (p ≥ 0.05)")
                    print(f"  → NO se puede confirmar H1")

            elif var == 'log_pib_pc':
                if pval < 0.05:
                    if coef < 0:
                        print(f"  → Mayor PIB per cápita reduce voto por González")
                        print(f"  → Confirma polarización económica: sectores más ricos rechazan correísmo")
                    else:
                        print(f"  → Mayor PIB per cápita aumenta voto por González (resultado contraintuitivo)")
                else:
                    print(f"  → Efecto NO significativo")

            elif var == 'costa':
                if pval < 0.05:
                    print(f"  → Efecto regional significativo: Costa difiere de Sierra (ref)")
                else:
                    print(f"  → No hay diferencia significativa entre Costa y Sierra")

            elif var == 'amazonia':
                if pval < 0.05:
                    print(f"  → Efecto regional significativo: Amazonía difiere de Sierra (ref)")
                else:
                    print(f"  → No hay diferencia significativa entre Amazonía y Sierra")

            elif var == 'indigena_x_logpib':
                if pval < 0.05:
                    print(f"  → CONFIRMA H2: Polarización económica modera efecto indígena")
                    print(f"  → La relación entre población indígena y voto depende del nivel de PIB")
                else:
                    print(f"  → RECHAZA H2: No hay moderación por PIB per cápita")

            elif var == 'indigena_x_costa':
                if pval < 0.05:
                    print(f"  → CONFIRMA H3: Mediación territorial en Costa")
                    print(f"  → El efecto de población indígena es diferente en Costa vs Sierra")
                else:
                    print(f"  → NO hay mediación territorial en Costa")

            elif var == 'indigena_x_amazonia':
                if pval < 0.05:
                    print(f"  → CONFIRMA H3: Mediación territorial en Amazonía")
                    print(f"  → El efecto de población indígena es diferente en Amazonía vs Sierra")
                else:
                    print(f"  → NO hay mediación territorial en Amazonía")

    return or_table, margeff

# ============================================================================
# FASE 6: VISUALIZACIONES
# ============================================================================

def crear_visualizaciones(datos, modelo1, modelo2, directorio='resultados_votacion_indigena'):
    """
    Crea visualizaciones de los resultados del análisis
    """
    import os
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    print("\n" + "="*80)
    print("FASE 6: VISUALIZACIONES")
    print("="*80)

    # ========================================================================
    # 1. Distribución de la variable dependiente
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    axes[0].hist(datos['prop_gonzalez'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(datos['prop_gonzalez'].mean(), color='red', linestyle='--',
                    label=f'Media = {datos["prop_gonzalez"].mean():.3f}')
    axes[0].axvline(datos['prop_gonzalez'].median(), color='green', linestyle='--',
                    label=f'Mediana = {datos["prop_gonzalez"].median():.3f}')
    axes[0].set_xlabel('Proporción de votos por González')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de la Variable Dependiente')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Boxplot por región
    regiones_data = []
    regiones_labels = []

    for region, nombre in [('costa', 'Costa'), ('sierra', 'Sierra'), ('amazonia', 'Amazonía')]:
        regiones_data.append(datos[datos[region] == 1]['prop_gonzalez'])
        regiones_labels.append(nombre)

    bp = axes[1].boxplot(regiones_data, labels=regiones_labels, patch_artist=True)

    # Colorear boxplots
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    axes[1].set_ylabel('Proporción de votos por González')
    axes[1].set_title('Voto por González según Región')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{directorio}/01_variable_dependiente.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/01_variable_dependiente.png")
    plt.close()

    # ========================================================================
    # 2. Relación bivariada: % Indígena vs Voto González
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Global
    axes[0].scatter(datos['pob_indigena_pct'], datos['prop_gonzalez'],
                    alpha=0.6, s=datos['votos_validos']/100)

    # Línea de tendencia
    z = np.polyfit(datos['pob_indigena_pct'], datos['prop_gonzalez'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(datos['pob_indigena_pct'].min(), datos['pob_indigena_pct'].max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Correlación
    r, pval = pearsonr(datos['pob_indigena_pct'], datos['prop_gonzalez'])
    axes[0].text(0.05, 0.95, f'r = {r:.3f}, p = {pval:.4f}',
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[0].set_xlabel('% Población Indígena')
    axes[0].set_ylabel('Proporción de votos por González')
    axes[0].set_title('H1: Patrón de Voto Indígena (Global)')
    axes[0].grid(alpha=0.3)

    # Por región
    for region, nombre, color in [('costa', 'Costa', 'blue'),
                                   ('sierra', 'Sierra', 'green'),
                                   ('amazonia', 'Amazonía', 'red')]:
        subgrupo = datos[datos[region] == 1]
        axes[1].scatter(subgrupo['pob_indigena_pct'], subgrupo['prop_gonzalez'],
                       alpha=0.6, label=nombre, color=color, s=50)

        # Línea de tendencia por región
        if len(subgrupo) > 2:
            z = np.polyfit(subgrupo['pob_indigena_pct'], subgrupo['prop_gonzalez'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subgrupo['pob_indigena_pct'].min(),
                               subgrupo['pob_indigena_pct'].max(), 100)
            axes[1].plot(x_line, p(x_line), linestyle='--', color=color, alpha=0.6, linewidth=2)

    axes[1].set_xlabel('% Población Indígena')
    axes[1].set_ylabel('Proporción de votos por González')
    axes[1].set_title('H3: Mediación Territorial')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{directorio}/02_h1_h3_indigena_voto.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/02_h1_h3_indigena_voto.png")
    plt.close()

    # ========================================================================
    # 3. Coeficientes del modelo con IC 95%
    # ========================================================================

    # Modelo 2 (con interacciones)
    params = modelo2.params.drop('Intercept')  # Excluir intercepto
    conf_int = modelo2.conf_int().drop('Intercept')
    pvalues = modelo2.pvalues.drop('Intercept')

    # Ordenar por magnitud del coeficiente
    order = params.abs().sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(order))

    # Dibujar cada punto con su color correspondiente
    for i, var in enumerate(order):
        color = 'red' if pvalues[var] < 0.05 else 'gray'
        ax.errorbar(params[var], i,
                   xerr=[[params[var] - conf_int.loc[var, 0]],
                         [conf_int.loc[var, 1] - params[var]]],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color=color, ecolor=color)

    # Línea en cero
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

    # Etiquetas
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel('Coeficiente (log-odds)')
    ax.set_title('Coeficientes del Modelo 2 con IC 95%\n(Rojo: p < 0.05, Gris: no significativo)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{directorio}/03_coeficientes_modelo2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/03_coeficientes_modelo2.png")
    plt.close()

    # ========================================================================
    # 4. Efectos predichos: Interacción % Indígena × Log PIB (H2)
    # ========================================================================

    # Crear grid de predicción
    pob_indigena_range = np.linspace(0, 100, 50)

    # Cuartiles de log PIB
    log_pib_quartiles = datos['log_pib_pc'].quantile([0.25, 0.50, 0.75])

    fig, ax = plt.subplots(figsize=(10, 6))

    for q_name, q_val in zip(['Q1 (PIB bajo)', 'Q2 (PIB medio)', 'Q3 (PIB alto)'], log_pib_quartiles):
        # Crear datos para predicción
        pred_data = pd.DataFrame({
            'pob_indigena_pct': pob_indigena_range,
            'log_pib_pc': q_val,
            'indigena_x_logpib': pob_indigena_range * q_val,
            'costa': 0,  # Región referencia (Sierra)
            'amazonia': 0,
            'indigena_x_costa': 0,
            'indigena_x_amazonia': 0,
            'agua_publica': datos['agua_publica'].mean(),
            'electricidad': datos['electricidad'].mean(),
            'tasa_homicidios': datos['tasa_homicidios'].mean(),
            'altitud': datos['altitud'].mean(),
            'log_poblacion': datos['log_poblacion'].mean()
        })

        # Predicciones
        pred = modelo2.predict(pred_data)

        ax.plot(pob_indigena_range, pred, label=q_name, linewidth=2)

    ax.set_xlabel('% Población Indígena')
    ax.set_ylabel('Probabilidad predicha de votar por González')
    ax.set_title('H2: Polarización Económica\n(Efecto de % Indígena según nivel de PIB per cápita)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{directorio}/04_h2_polarizacion_economica.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/04_h2_polarizacion_economica.png")
    plt.close()

    # ========================================================================
    # 5. Efectos predichos: Interacción % Indígena × Región (H3)
    # ========================================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    for region, nombre, color in [('costa', 'Costa', 'blue'),
                                   ('sierra', 'Sierra', 'green'),
                                   ('amazonia', 'Amazonía', 'red')]:

        # Crear datos para predicción
        pred_data = pd.DataFrame({
            'pob_indigena_pct': pob_indigena_range,
            'log_pib_pc': datos['log_pib_pc'].mean(),
            'indigena_x_logpib': pob_indigena_range * datos['log_pib_pc'].mean(),
            'costa': 1 if region == 'costa' else 0,
            'amazonia': 1 if region == 'amazonia' else 0,
            'indigena_x_costa': pob_indigena_range if region == 'costa' else 0,
            'indigena_x_amazonia': pob_indigena_range if region == 'amazonia' else 0,
            'agua_publica': datos['agua_publica'].mean(),
            'electricidad': datos['electricidad'].mean(),
            'tasa_homicidios': datos['tasa_homicidios'].mean(),
            'altitud': datos['altitud'].mean(),
            'log_poblacion': datos['log_poblacion'].mean()
        })

        # Predicciones
        pred = modelo2.predict(pred_data)

        ax.plot(pob_indigena_range, pred, label=nombre, linewidth=2, color=color)

    ax.set_xlabel('% Población Indígena')
    ax.set_ylabel('Probabilidad predicha de votar por González')
    ax.set_title('H3: Mediación Territorial\n(Efecto de % Indígena según región)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{directorio}/05_h3_mediacion_territorial.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/05_h3_mediacion_territorial.png")
    plt.close()

    # ========================================================================
    # 6. Diagnóstico: Residuos del modelo
    # ========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Valores predichos
    fitted = modelo2.fittedvalues

    # Residuos deviance
    residuos = modelo2.resid_deviance

    # 6.1 Residuos vs Fitted
    axes[0, 0].scatter(fitted, residuos, alpha=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Valores Ajustados')
    axes[0, 0].set_ylabel('Residuos Deviance')
    axes[0, 0].set_title('Residuos vs Valores Ajustados')
    axes[0, 0].grid(alpha=0.3)

    # 6.2 Q-Q Plot
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot de Residuos')
    axes[0, 1].grid(alpha=0.3)

    # 6.3 Scale-Location
    residuos_std = np.sqrt(np.abs(residuos))
    axes[1, 0].scatter(fitted, residuos_std, alpha=0.5)
    axes[1, 0].set_xlabel('Valores Ajustados')
    axes[1, 0].set_ylabel('√|Residuos Estandarizados|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(alpha=0.3)

    # 6.4 Leverage (Influencia)
    from statsmodels.stats.outliers_influence import OLSInfluence
    # Nota: Para GLM no hay un método directo de influencia como en OLS
    # Usamos residuos estandarizados como proxy
    axes[1, 1].scatter(range(len(residuos)), residuos, alpha=0.5)
    axes[1, 1].axhline(2, color='red', linestyle='--', label='±2 SD')
    axes[1, 1].axhline(-2, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Índice de Observación')
    axes[1, 1].set_ylabel('Residuos Deviance')
    axes[1, 1].set_title('Residuos por Observación')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{directorio}/06_diagnostico_residuos.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {directorio}/06_diagnostico_residuos.png")
    plt.close()

    print(f"\n✓ Todas las visualizaciones guardadas en: {directorio}/")

# ============================================================================
# FASE 7: EXPORTAR RESULTADOS
# ============================================================================

def exportar_resultados(datos, vif_data, modelo1, modelo2, or_table, margeff,
                       directorio='resultados_votacion_indigena'):
    """
    Exporta tablas de resultados a archivos CSV
    """
    import os
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    print("\n" + "="*80)
    print("FASE 7: EXPORTACIÓN DE RESULTADOS")
    print("="*80)

    # 1. Estadísticas descriptivas
    vars_interes = ['prop_gonzalez', 'pob_indigena_pct', 'log_pib_pc',
                    'agua_publica', 'electricidad', 'tasa_homicidios', 'altitud']
    desc = datos[vars_interes].describe().T
    desc.to_csv(f'{directorio}/01_estadisticas_descriptivas.csv')
    print(f"✓ {directorio}/01_estadisticas_descriptivas.csv")

    # 2. VIF
    vif_data.to_csv(f'{directorio}/02_vif_multicolinealidad.csv', index=False)
    print(f"✓ {directorio}/02_vif_multicolinealidad.csv")

    # 3. Resultados Modelo 1
    resultados_m1 = pd.DataFrame({
        'Variable': modelo1.params.index,
        'Coeficiente': modelo1.params.values,
        'Std_Error': modelo1.bse.values,
        'z_value': modelo1.tvalues.values,
        'p_value': modelo1.pvalues.values,
        'IC_95%_inf': modelo1.conf_int()[0].values,
        'IC_95%_sup': modelo1.conf_int()[1].values
    })
    resultados_m1.to_csv(f'{directorio}/03_modelo1_efectos_principales.csv', index=False)
    print(f"✓ {directorio}/03_modelo1_efectos_principales.csv")

    # 4. Resultados Modelo 2
    resultados_m2 = pd.DataFrame({
        'Variable': modelo2.params.index,
        'Coeficiente': modelo2.params.values,
        'Std_Error': modelo2.bse.values,
        'z_value': modelo2.tvalues.values,
        'p_value': modelo2.pvalues.values,
        'IC_95%_inf': modelo2.conf_int()[0].values,
        'IC_95%_sup': modelo2.conf_int()[1].values
    })
    resultados_m2.to_csv(f'{directorio}/04_modelo2_con_interacciones.csv', index=False)
    print(f"✓ {directorio}/04_modelo2_con_interacciones.csv")

    # 5. Odds Ratios
    or_table.to_csv(f'{directorio}/05_odds_ratios.csv')
    print(f"✓ {directorio}/05_odds_ratios.csv")

    # 6. Efectos marginales
    margeff_df = margeff.summary_frame()
    margeff_df.to_csv(f'{directorio}/06_efectos_marginales.csv')
    print(f"✓ {directorio}/06_efectos_marginales.csv")

    # 7. Comparación de modelos
    comparacion = pd.DataFrame({
        'Criterio': ['AIC', 'BIC', 'Log-Likelihood', 'Pseudo R²'],
        'Modelo_1': [modelo1.aic, modelo1.bic, modelo1.llf, modelo1.pseudo_rsquared()],
        'Modelo_2': [modelo2.aic, modelo2.bic, modelo2.llf, modelo2.pseudo_rsquared()]
    })
    comparacion.to_csv(f'{directorio}/07_comparacion_modelos.csv', index=False)
    print(f"✓ {directorio}/07_comparacion_modelos.csv")

    print(f"\n✓ Todos los resultados exportados a: {directorio}/")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta el análisis completo de votación indígena
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE VOTACIÓN INDÍGENA EN ECUADOR - ELECCIONES 2025")
    print("="*80)
    print("\nRegresión Logística Binomial con Ponderación por Votos Válidos")
    print("Diagnóstico de Multicolinealidad: VIF < 5")
    print("\n" + "="*80)

    # FASE 1: Cargar y preparar datos
    df = cargar_datos('Basecantones2csv.csv')
    datos = preparar_variables(df)

    # FASE 2: Análisis exploratorio
    desc = analisis_descriptivo(datos)

    # FASE 3: Diagnóstico VIF
    # Variables para el modelo completo (Modelo 2)
    variables_modelo = ['pob_indigena_pct', 'log_pib_pc', 'costa', 'amazonia',
                        'indigena_x_logpib', 'indigena_x_costa', 'indigena_x_amazonia',
                        'agua_publica', 'electricidad', 'tasa_homicidios',
                        'altitud', 'log_poblacion']

    vif_data = calcular_vif(datos, variables_modelo)

    # FASE 4: Regresión logística
    modelo1, modelo2 = regresion_logistica_binomial(datos)

    # FASE 5: Interpretación
    or_table, margeff = interpretar_resultados(modelo2, datos)

    # FASE 6: Visualizaciones
    crear_visualizaciones(datos, modelo1, modelo2)

    # FASE 7: Exportar resultados
    exportar_resultados(datos, vif_data, modelo1, modelo2, or_table, margeff)

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\nConsulte los archivos generados en:")
    print("  - resultados_votacion_indigena/ (tablas CSV)")
    print("  - resultados_votacion_indigena/ (gráficos PNG)")
    print("\n" + "="*80)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main()
