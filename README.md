# Análisis Estadístico de Cantones del Ecuador

Este repositorio contiene scripts para realizar un análisis estadístico completo de datos socioeconómicos y electorales de los 221 cantones ecuatorianos. El análisis está disponible tanto en **Python** como en **R**.

## Contenido

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Datos](#datos)
- [Requisitos](#requisitos)
  - [Python](#requisitos-python)
  - [R](#requisitos-r)
- [Instalación](#instalación)
  - [Python](#instalación-python)
  - [R](#instalación-r)
- [Uso](#uso)
  - [Ejecutar con Python](#ejecutar-con-python)
  - [Ejecutar con R](#ejecutar-con-r)
- [Estructura del Análisis](#estructura-del-análisis)
- [Resultados Generados](#resultados-generados)
- [Interpretación de Resultados](#interpretación-de-resultados)
- [Metodología Estadística](#metodología-estadística)

---

## Descripción del Proyecto

Este proyecto realiza un análisis estadístico comprehensivo de los cantones ecuatorianos, examinando:

- **Variables electorales**: Votos por Daniel Noboa y Luisa González (elecciones 2023)
- **Variables socioeconómicas**: PIB per cápita, tasa de homicidios, acceso a servicios básicos
- **Variables demográficas**: Población, porcentaje de población indígena
- **Variables geográficas**: Región (Costa, Sierra, Oriente, Insular)

### Objetivos del Análisis

1. Caracterizar la distribución de variables socioeconómicas a nivel cantonal
2. Evaluar la normalidad de las distribuciones
3. Identificar correlaciones significativas entre variables
4. Detectar patrones regionales
5. Identificar valores extremos (outliers)

---

## Datos

El dataset principal es `Basecantones2csv.csv` con las siguientes características:

- **Observaciones**: 221 cantones
- **Variables**: 17 columnas
- **Formato**: CSV con separador punto y coma (;) y decimales con coma (,)

### Variables del Dataset

| Variable | Descripción | Tipo |
|----------|-------------|------|
| Cantón | Nombre del cantón | Texto |
| Provincia | Provincia a la que pertenece | Texto |
| Votos por Noboa (absoluto) | Votos absolutos por Noboa | Numérico |
| Votos por González (absoluto) | Votos absolutos por González | Numérico |
| Votos por Noboa (porcentaje) | Porcentaje de votos por Noboa | Numérico |
| Votos por González (porcentaje) | Porcentaje de votos por González | Numérico |
| Población | Población total del cantón | Numérico |
| Porcentaje población indígena | % de población indígena | Numérico |
| Agua red pública | % con acceso a agua potable | Numérico |
| Electricidad | % con acceso a electricidad | Numérico |
| PIB per cápita | PIB per cápita en USD | Numérico |
| Tasa Homicidios | Homicidios por 100,000 habitantes | Numérico |
| Altitud | Altitud en metros | Numérico |
| Costa | Indicador región Costa (0/1) | Binario |
| Sierra | Indicador región Sierra (0/1) | Binario |
| Oriente | Indicador región Oriente (0/1) | Binario |
| Insular | Indicador región Insular (0/1) | Binario |

---

## Requisitos

### Requisitos Python

**Versión de Python**: 3.8 o superior

**Librerías necesarias**:
```
pandas >= 1.3.0
numpy >= 1.20.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Requisitos R

**Versión de R**: 4.0.0 o superior

**Paquetes necesarios**:
```r
tidyverse     # Para manipulación de datos y visualización
psych         # Para estadísticas descriptivas avanzadas
corrplot      # Para visualización de matrices de correlación
ggpubr        # Para combinar gráficos
moments       # Para cálculos de asimetría y curtosis
```

---

## Instalación

### Instalación Python

#### Opción 1: Usando pip

```bash
# Crear entorno virtual (recomendado)
python -m venv env_ecuador
source env_ecuador/bin/activate  # Linux/Mac
# env_ecuador\Scripts\activate   # Windows

# Instalar dependencias
pip install pandas numpy scipy matplotlib seaborn
```

#### Opción 2: Usando requirements.txt

Crear archivo `requirements.txt`:
```txt
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Instalar:
```bash
pip install -r requirements.txt
```

#### Opción 3: Usando conda

```bash
conda create -n ecuador_analysis python=3.9
conda activate ecuador_analysis
conda install pandas numpy scipy matplotlib seaborn
```

### Instalación R

#### Opción 1: Desde la consola de R

```r
# Instalar todos los paquetes necesarios
install.packages(c("tidyverse", "psych", "corrplot", "ggpubr", "moments"))

# Verificar instalación
library(tidyverse)
library(psych)
library(corrplot)
library(ggpubr)
library(moments)
```

#### Opción 2: Instalación automática (incluida en el script)

El script `analisis_r.R` verifica e instala automáticamente los paquetes faltantes al ejecutarse.

#### Opción 3: Usando RStudio

1. Abrir RStudio
2. Ir a Tools > Install Packages
3. Escribir: `tidyverse, psych, corrplot, ggpubr, moments`
4. Hacer clic en "Install"

---

## Uso

### Ejecutar con Python

#### Desde la línea de comandos

```bash
# Navegar al directorio del proyecto
cd /ruta/a/Inv-Quant

# Ejecutar el script
python analisis_python.py
```

#### Desde un entorno virtual

```bash
# Activar entorno virtual
source env_ecuador/bin/activate  # Linux/Mac
# env_ecuador\Scripts\activate   # Windows

# Ejecutar
python analisis_python.py
```

#### Desde Jupyter Notebook

```python
# Ejecutar en una celda
%run analisis_python.py

# O importar funciones específicas
import analisis_python as ap
df = ap.cargar_datos('Basecantones2csv.csv')
df = ap.limpiar_datos(df)
```

### Ejecutar con R

#### Desde la línea de comandos

```bash
# Ejecutar directamente
Rscript analisis_r.R

# O con mayor detalle
R CMD BATCH analisis_r.R
```

#### Desde RStudio

1. Abrir el archivo `analisis_r.R` en RStudio
2. Hacer clic en "Source" (esquina superior derecha)
3. O ejecutar: `source("analisis_r.R")`

#### Desde la consola de R

```r
# Establecer directorio de trabajo
setwd("/ruta/a/Inv-Quant")

# Ejecutar script
source("analisis_r.R")
```

#### Ejecución por secciones (interactivo)

```r
# Seleccionar y ejecutar secciones específicas con Ctrl+Enter
# Por ejemplo, solo cargar datos:
source("analisis_r.R", echo = TRUE, max.deparse.length = 100)
```

---

## Estructura del Análisis

Ambos scripts siguen la misma estructura de 5 fases:

### FASE 1: Preparación de Datos
- Carga del archivo CSV
- Renombrado de columnas
- Conversión de tipos de datos
- Creación de variable "región"
- Verificación de datos faltantes

### FASE 2: Análisis Descriptivo
- Estadísticas de tendencia central (media, mediana)
- Estadísticas de dispersión (desviación estándar, varianza, CV)
- Estadísticas de forma (asimetría, curtosis)
- Identificación de outliers (método IQR)
- Análisis estratificado por región

### FASE 3: Análisis de Normalidad
- Test de Shapiro-Wilk para cada variable
- Nivel de significancia α = 0.05
- Determinación del método de correlación apropiado

### FASE 4: Análisis de Correlación
- Matriz de correlación de Spearman (por no normalidad)
- Cálculo de p-valores
- Interpretación de fuerza y dirección

### FASE 5: Visualizaciones
- Histogramas con curvas de densidad
- Gráficos Q-Q para normalidad
- Boxplots para distribución y outliers
- Boxplots comparativos por región
- Heatmap de correlaciones
- Scatterplots con líneas de tendencia

---

## Resultados Generados

### Python (carpeta `resultados_python/`)

```
resultados_python/
├── estadisticas_descriptivas.csv
├── estadisticas_por_region.csv
├── test_normalidad.csv
├── matriz_correlacion.csv
├── correlaciones_principales.csv
├── histogramas.png
├── qq_plots.png
├── boxplots.png
├── boxplots_region.png
├── heatmap_correlacion.png
└── scatterplots.png
```

### R (carpeta `resultados_r/`)

```
resultados_r/
├── estadisticas_descriptivas.csv
├── estadisticas_por_region.csv
├── test_normalidad.csv
├── matriz_correlacion.csv
├── correlaciones_principales.csv
├── histogramas.png
├── qq_plots.png
├── boxplots.png
├── boxplots_region.png
├── heatmap_correlacion.png
└── scatterplots.png
```

---

## Interpretación de Resultados

### Estadísticas Descriptivas

- **Media vs Mediana**: Si difieren mucho, indica asimetría en la distribución
- **CV (Coeficiente de Variación)**: >100% indica alta variabilidad
- **Asimetría**: >0 asimetría positiva (cola derecha larga), <0 asimetría negativa
- **Curtosis**: >0 leptocúrtica (colas pesadas), <0 platicúrtica (colas ligeras)

### Test de Normalidad (Shapiro-Wilk)

- **p-valor > 0.05**: No se rechaza H0, datos podrían ser normales
- **p-valor < 0.05**: Se rechaza H0, datos NO son normales
- Si la mayoría de variables no son normales → usar correlación de Spearman

### Correlaciones

| Valor de ρ | Interpretación |
|------------|----------------|
| 0.00 - 0.19 | Muy débil |
| 0.20 - 0.39 | Débil |
| 0.40 - 0.59 | Moderada |
| 0.60 - 0.79 | Fuerte |
| 0.80 - 1.00 | Muy fuerte |

**Significancia**:
- *** p < 0.001 (altamente significativo)
- ** p < 0.01 (muy significativo)
- * p < 0.05 (significativo)
- ns (no significativo)

### Outliers

Identificados usando el método IQR:
- **Outlier inferior**: x < Q1 - 1.5 * IQR
- **Outlier superior**: x > Q3 + 1.5 * IQR

Los outliers NO deben eliminarse automáticamente; representan realidades territoriales específicas (ej: cantones petroleros).

---

## Metodología Estadística

### Por qué Spearman en lugar de Pearson

1. **No normalidad**: Las variables no siguen distribución normal (p < 0.05 en Shapiro-Wilk)
2. **Robustez**: Spearman es menos sensible a outliers
3. **Relaciones monótonas**: Detecta asociaciones no lineales
4. **Datos ordinales**: Apropiado para rankings

### Nivel de Significancia

Se usa α = 0.05 como estándar, lo que significa:
- 5% de probabilidad de rechazar H0 cuando es verdadera
- 95% de confianza en los resultados

### Corrección por Comparaciones Múltiples

**Advertencia**: Al realizar múltiples tests de correlación, aumenta la probabilidad de error Tipo I. Para análisis rigurosos, considerar:

- Corrección de Bonferroni: α' = α/n
- Corrección de Holm-Bonferroni
- Tasa de Descubrimiento Falso (FDR)

---

## Ejemplos de Uso Avanzado

### Python - Análisis personalizado

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Cargar datos
df = pd.read_csv('Basecantones2csv.csv', sep=';', encoding='utf-8-sig')

# Renombrar columnas (simplificado)
df.columns = ['canton', 'provincia', 'votos_noboa_abs', 'votos_gonzalez_abs',
              'votos_noboa_pct', 'votos_gonzalez_pct', 'poblacion',
              'pob_indigena_pct', 'agua_publica', 'electricidad',
              'pib_per_capita', 'tasa_homicidios', 'altitud',
              'costa', 'sierra', 'oriente', 'insular']

# Análisis específico: top 10 cantones con más votos por Noboa
top_noboa = df.nlargest(10, 'votos_noboa_pct')[['canton', 'provincia', 'votos_noboa_pct']]
print(top_noboa)

# Correlación específica con intervalo de confianza
from scipy.stats import bootstrap
rho, p = spearmanr(df['votos_noboa_pct'], df['tasa_homicidios'])
print(f"Votos vs Homicidios: ρ={rho:.3f}, p={p:.6f}")
```

### R - Análisis personalizado

```r
library(tidyverse)

# Cargar datos
datos <- read_delim("Basecantones2csv.csv", delim = ";",
                    locale = locale(decimal_mark = ","))

# Renombrar (simplificado)
names(datos) <- c("canton", "provincia", "votos_noboa_abs", "votos_gonzalez_abs",
                  "votos_noboa_pct", "votos_gonzalez_pct", "poblacion",
                  "pob_indigena_pct", "agua_publica", "electricidad",
                  "pib_per_capita", "tasa_homicidios", "altitud",
                  "costa", "sierra", "oriente", "insular")

# Top 10 cantones con más votos por Noboa
datos %>%
  arrange(desc(votos_noboa_pct)) %>%
  head(10) %>%
  select(canton, provincia, votos_noboa_pct)

# Regresión lineal
modelo <- lm(votos_noboa_pct ~ pib_per_capita + tasa_homicidios +
             pob_indigena_pct + agua_publica, data = datos)
summary(modelo)
```

---

## Solución de Problemas

### Python

**Error: ModuleNotFoundError**
```bash
pip install nombre_modulo
```

**Error de codificación**
```python
# Especificar encoding
df = pd.read_csv('archivo.csv', encoding='utf-8-sig')
```

**Gráficos no se muestran**
```python
import matplotlib
matplotlib.use('Agg')  # Para entornos sin display
```

### R

**Error: paquete no encontrado**
```r
install.packages("nombre_paquete")
```

**Error de locale (decimales)**
```r
datos <- read_delim("archivo.csv", locale = locale(decimal_mark = ","))
```

**Gráficos no se guardan**
```r
# Cerrar dispositivo gráfico
dev.off()
```

---

## Licencia

Este proyecto está disponible para uso educativo e investigativo.

---

## Contacto

Para preguntas o sugerencias sobre el análisis estadístico, consulte la documentación detallada en cada script.

---

**Última actualización**: 2025
