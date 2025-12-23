# Análisis de Votación Indígena en Ecuador - Elecciones 2025

## Descripción del Proyecto

Este proyecto examina si existe un **patrón de voto homogéneo entre cantones ecuatorianos con alta población indígena** en las elecciones presidenciales de 2025, y qué factores socioeconómicos y territoriales explican las variaciones en sus preferencias electorales bajo condiciones de polarización bidimensional.

---

## 📊 Objetivos de Investigación

### Objetivo General
Analizar los determinantes del voto indígena en Ecuador durante las elecciones presidenciales de 2025, con énfasis en la candidata correísta Luisa González.

### Objetivos Específicos
1. **H1 (Homogeneidad)**: Determinar si existe un patrón de voto pro-correísta en cantones con alta población indígena
2. **H2 (Polarización Económica)**: Evaluar si el nivel de desarrollo económico (PIB per cápita) modera la relación entre población indígena y preferencia electoral
3. **H3 (Mediación Territorial)**: Identificar si el contexto regional (Costa, Sierra, Amazonía) media la preferencia electoral de cantones indígenas

---

## 🔬 Metodología Estadística

### Variable Dependiente
- **prop_gonzalez**: Proporción de votos válidos para Luisa González (candidata correísta)
  - Tipo: Proporción continua acotada [0, 1]
  - Distribución: No normal (naturaleza acotada)

### Variables Independientes Principales

#### Efectos Directos
1. **pob_indigena_pct**: Porcentaje de población indígena cantonal (0-100%)
2. **log_pib_pc**: Logaritmo natural del PIB per cápita (USD)
3. **costa**: Dummy región Costa (1 = Costa, 0 = otro)
4. **amazonia**: Dummy región Amazonía/Oriente (1 = Amazonía, 0 = otro)
   - *Categoría de referencia*: Sierra

#### Términos de Interacción (Heterogeneidad Contextual)
1. **indigena_x_logpib**: % indígena × log PIB per cápita
   - Evalúa H2: ¿Varía el efecto indígena según nivel económico?
2. **indigena_x_costa**: % indígena × Costa
   - Evalúa H3: ¿Es diferente el voto indígena en la Costa?
3. **indigena_x_amazonia**: % indígena × Amazonía
   - Evalúa H3: ¿Es diferente el voto indígena en la Amazonía?

### Variables de Control

Disponibles en el dataset actual:
- **agua_publica**: Acceso a agua de red pública (%)
- **electricidad**: Acceso a electricidad (%)
- **tasa_homicidios**: Tasa de homicidios por 100,000 habitantes
- **altitud**: Altitud en metros (proxy de ruralidad y dispersión)
- **log_poblacion**: Logaritmo de la población total (control de tamaño)

**Nota**: Variables adicionales recomendadas pero no disponibles actualmente:
- Tasa de pobreza
- Tasa de desempleo
- Densidad poblacional
- Porcentaje de población rural
- Nivel educativo promedio
- Índice de Necesidades Básicas Insatisfechas (NBI)

### Método de Estimación

#### Regresión Logística con Familia Binomial

**¿Por qué este método?**

La variable dependiente es una **proporción** (0-1), no una variable normal:
- ❌ **NO** usar regresión lineal OLS → produce predicciones fuera de [0,1]
- ✅ **SÍ** usar regresión logística binomial → respeta el rango [0,1]

**Especificaciones técnicas:**
- **Familia**: Binomial
- **Función de enlace**: Logit (log-odds)
- **Estimación**: Máxima verosimilitud (MLE)
- **Ponderación**: Por votos válidos cantonales (considera diferencias en tamaño electoral)

#### Control de Multicolinealidad

**Pruebas VIF (Variance Inflation Factor)**
- **Umbral conservador**: VIF < 5
- **Interpretación**:
  - VIF < 5: ✓ Sin multicolinealidad problemática
  - VIF 5-10: ⚠ Multicolinealidad moderada
  - VIF > 10: ✗ Multicolinealidad severa (requiere acción)

**Acción si VIF > 5**:
- Centrar variables antes de crear interacciones
- Eliminar términos de interacción redundantes
- Evaluar colinealidad estructural

---

## 📁 Estructura del Proyecto

```
Inv-Quant/
│
├── Basecantones2csv.csv                    # Dataset original (221 cantones)
│
├── analisis_votacion_indigena.py           # Script Python
├── analisis_votacion_indigena.R            # Script R
├── README_VOTACION_INDIGENA.md             # Este archivo
├── requirements.txt                         # Dependencias Python
│
└── resultados_votacion_indigena/           # Resultados del análisis
    ├── 01_estadisticas_descriptivas.csv
    ├── 02_vif_multicolinealidad.csv
    ├── 03_modelo1_efectos_principales.csv
    ├── 04_modelo2_con_interacciones.csv
    ├── 05_odds_ratios.csv
    ├── 06_efectos_marginales.csv
    ├── 07_comparacion_modelos.csv (o .txt para R)
    │
    ├── 01_variable_dependiente.png
    ├── 02_h1_h3_indigena_voto.png
    ├── 03_coeficientes_modelo2.png
    ├── 04_h2_polarizacion_economica.png
    ├── 05_h3_mediacion_territorial.png
    └── 06_diagnostico_residuos.png
```

---

## 🚀 Instalación y Uso

### Requisitos Previos

#### Python (versión 3.8+)
```bash
python --version  # Verificar versión
```

#### R (versión 4.0+)
```bash
R --version  # Verificar versión
```

---

### Opción 1: Ejecutar con Python

#### Instalación de dependencias

**Opción A: pip**
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
```

**Opción B: requirements.txt**
```bash
pip install -r requirements.txt
```

**Opción C: conda**
```bash
conda create -n votacion_indigena python=3.9
conda activate votacion_indigena
conda install pandas numpy scipy matplotlib seaborn statsmodels
```

#### Ejecución

```bash
# Desde la línea de comandos
python analisis_votacion_indigena.py

# Desde Jupyter Notebook
%run analisis_votacion_indigena.py
```

**Tiempo estimado de ejecución**: 30-60 segundos

---

### Opción 2: Ejecutar con R

#### Instalación de paquetes

**Opción A: Desde la consola de R**
```r
install.packages(c("tidyverse", "car", "lmtest", "broom", "margins",
                   "ggeffects", "stargazer", "patchwork", "scales"))
```

**Opción B: Instalación automática**
El script verifica e instala automáticamente los paquetes faltantes.

#### Ejecución

```bash
# Desde la línea de comandos
Rscript analisis_votacion_indigena.R

# O con mayor detalle
R CMD BATCH analisis_votacion_indigena.R
```

**Desde RStudio**:
1. Abrir `analisis_votacion_indigena.R`
2. Click en "Source" (o Ctrl+Shift+S)

**Desde la consola de R**:
```r
setwd("/ruta/a/Inv-Quant")
source("analisis_votacion_indigena.R")
```

**Tiempo estimado de ejecución**: 1-2 minutos

---

## 📈 Resultados Generados

### Tablas CSV

| Archivo | Descripción |
|---------|-------------|
| `01_estadisticas_descriptivas.csv` | Media, mediana, SD, min, max, CV, asimetría, curtosis |
| `02_vif_multicolinealidad.csv` | VIF de cada variable (diagnóstico de multicolinealidad) |
| `03_modelo1_efectos_principales.csv` | Coeficientes del Modelo 1 (sin interacciones) |
| `04_modelo2_con_interacciones.csv` | Coeficientes del Modelo 2 (con interacciones H2 y H3) |
| `05_odds_ratios.csv` | Odds Ratios e IC 95% (interpretación multiplicativa) |
| `06_efectos_marginales.csv` | Efectos marginales promedio (cambio en probabilidad) |
| `07_comparacion_modelos.csv` | AIC, BIC, Log-Likelihood, Pseudo R² |

### Gráficos PNG

| Archivo | Descripción |
|---------|-------------|
| `01_variable_dependiente.png` | Distribución de prop_gonzalez (histograma + boxplot por región) |
| `02_h1_h3_indigena_voto.png` | Relación % indígena vs voto (global y por región) |
| `03_coeficientes_modelo2.png` | Coeficientes del Modelo 2 con IC 95% |
| `04_h2_polarizacion_economica.png` | Efecto de % indígena según nivel de PIB (H2) |
| `05_h3_mediacion_territorial.png` | Efecto de % indígena según región (H3) |
| `06_diagnostico_residuos.png` | Diagnóstico de residuos (4 paneles) |

---

## 🔍 Interpretación de Resultados

### 1. Coeficientes del Modelo Logístico

Los coeficientes representan **log-odds** (logaritmo de razón de momios):
- **Coeficiente > 0**: Aumenta la probabilidad de votar por González
- **Coeficiente < 0**: Disminuye la probabilidad de votar por González
- **Coeficiente = 0**: No tiene efecto

**Significancia estadística**:
- `***` p < 0.001 (altamente significativo)
- `**` p < 0.01 (muy significativo)
- `*` p < 0.05 (significativo)
- `ns` p ≥ 0.05 (no significativo)

### 2. Odds Ratios (OR)

OR = exp(coeficiente)

**Interpretación**:
- **OR > 1**: Aumenta los odds de votar por González
  - Ejemplo: OR = 1.50 → Aumento de 50% en los odds
- **OR < 1**: Disminuye los odds de votar por González
  - Ejemplo: OR = 0.75 → Reducción de 25% en los odds
- **OR = 1**: Sin efecto

### 3. Efectos Marginales Promedio (AME)

**Cambio en probabilidad** ante cambio unitario en X:
- Más intuitivo que log-odds u odds ratios
- Ejemplo: AME = 0.02 → Un aumento de 1% en población indígena incrementa la probabilidad de votar por González en 2 puntos porcentuales

### 4. Interpretación de Interacciones

#### H2: Polarización Económica (indigena_x_logpib)

**Si significativo**:
- El efecto de % indígena **depende** del nivel de PIB per cápita
- Ejemplo: Cantones indígenas pobres votan más por González que cantones indígenas ricos

**Visualización**: Gráfico `04_h2_polarizacion_economica.png`
- Líneas divergentes → Interacción fuerte
- Líneas paralelas → Sin interacción

#### H3: Mediación Territorial (indigena_x_costa, indigena_x_amazonia)

**Si significativo**:
- El efecto de % indígena **varía según región**
- Ejemplo: Indígenas de la Costa votan diferente que indígenas de la Sierra

**Visualización**: Gráfico `05_h3_mediacion_territorial.png`
- Pendientes diferentes por región → Confirmación de H3

### 5. Comparación de Modelos

**Test de Razón de Verosimilitud (LR Test)**:
- **p < 0.05**: Modelo 2 (con interacciones) es significativamente mejor
- **p ≥ 0.05**: Modelo 1 (más parsimonioso) es preferible

**Criterios de Información**:
- **AIC** (Akaike): Menor es mejor
- **BIC** (Bayesiano): Menor es mejor, penaliza más complejidad
- **Pseudo R²** (McFadden): Mayor es mejor (0-1, pero rara vez > 0.4)

### 6. Diagnóstico de Residuos

**Gráfico `06_diagnostico_residuos.png` (4 paneles)**:

1. **Residuos vs Fitted**: Debe mostrar dispersión aleatoria sin patrón
2. **Q-Q Plot**: Puntos deben seguir la línea roja (normalidad de residuos)
3. **Scale-Location**: Varianza constante (homocedasticidad)
4. **Residuos por Observación**: Detectar outliers (|residuo| > 2)

---

## 🎯 Hipótesis y Predicciones

| Hipótesis | Predicción | Variable Clave | Resultado Esperado |
|-----------|-----------|----------------|-------------------|
| **H1: Homogeneidad** | Cantones con alta población indígena votan más por González | `pob_indigena_pct` | Coef > 0, p < 0.05 |
| **H2: Polarización Económica** | El efecto indígena es menor en cantones ricos | `indigena_x_logpib` | Coef < 0, p < 0.05 |
| **H3a: Mediación Costa** | Efecto indígena diferente en Costa vs Sierra | `indigena_x_costa` | Coef ≠ 0, p < 0.05 |
| **H3b: Mediación Amazonía** | Efecto indígena diferente en Amazonía vs Sierra | `indigena_x_amazonia` | Coef ≠ 0, p < 0.05 |

---

## 📚 Referencias Metodológicas

### Regresión Logística Binomial
- **Papke, L. E., & Wooldridge, J. M.** (1996). Econometric methods for fractional response variables with an application to 401 (k) plan participation rates. *Journal of Applied Econometrics*, 11(6), 619-632.

### Multicolinealidad (VIF)
- **O'Brien, R. M.** (2007). A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity*, 41(5), 673-690.

### Efectos Marginales
- **Bartus, T.** (2005). Estimation of marginal effects using margeff. *The Stata Journal*, 5(3), 309-329.

### Modelos de Interacción
- **Brambor, T., Clark, W. R., & Golder, M.** (2006). Understanding interaction models: Improving empirical analyses. *Political Analysis*, 14(1), 63-82.

---

## ⚠️ Limitaciones del Estudio

### Datos Faltantes

Variables de control recomendadas pero **no disponibles** en el dataset actual:
- ✗ Tasa de pobreza multidimensional
- ✗ Tasa de desempleo
- ✗ Densidad poblacional (hab/km²)
- ✗ Porcentaje de población rural
- ✗ Nivel educativo promedio (años de escolaridad)
- ✗ Índice de Necesidades Básicas Insatisfechas (NBI)

**Recomendación**: Incorporar estas variables si se obtienen de fuentes como:
- INEC (Instituto Nacional de Estadística y Censos)
- SIISE (Sistema Integrado de Indicadores Sociales del Ecuador)
- SENPLADES (Secretaría Nacional de Planificación)

### Causalidad vs Asociación

Este análisis es **observacional**, no experimental:
- ✓ Identifica **asociaciones** entre variables
- ✗ **NO** establece causalidad definitiva
- ⚠ Posible confusión por variables omitidas

### Inferencia Ecológica

Los datos son a nivel **cantonal** (agregado):
- ✗ **NO** se puede inferir comportamiento individual ("falacia ecológica")
- ✓ Solo válido para patrones territoriales

---

## 🔧 Solución de Problemas

### Python

**Error: ModuleNotFoundError**
```bash
pip install nombre_modulo
```

**Error: KeyError en columnas**
Verificar que el archivo CSV sea `Basecantones2csv.csv` con la estructura original.

**Gráficos no se muestran en entornos sin display**
```python
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
```

### R

**Error: paquete no encontrado**
```r
install.packages("nombre_paquete")
```

**Error de locale (separador decimal)**
El script ya maneja esto con `locale = locale(decimal_mark = ",")`.

**Gráficos no se guardan**
```r
dev.off()  # Cerrar dispositivo gráfico
```

---

## 📞 Contacto y Colaboración

Para preguntas, sugerencias o colaboraciones:
- Consultar código fuente (comentarios extensivos)
- Revisar documentación de resultados generados
- Reportar issues en el repositorio

---

## 📄 Licencia

Este proyecto está disponible para uso **educativo e investigativo**.

---

## ✅ Checklist de Análisis

Antes de interpretar resultados, verificar:

- [ ] VIF < 5 para todas las variables (o justificar VIF moderado)
- [ ] Residuos sin patrones sistemáticos (diagnóstico visual)
- [ ] Intervalos de confianza de coeficientes no cruzan cero (si significativos)
- [ ] Test LR confirma preferencia de modelo (p-value)
- [ ] Predicciones del modelo están en rango [0, 1]
- [ ] Interpretación de interacciones apoyada por visualizaciones

---

**Última actualización**: 2025
**Versión**: 1.0
**Software**: Python 3.8+ | R 4.0+
