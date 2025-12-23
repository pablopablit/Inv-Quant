#!/usr/bin/env Rscript
################################################################################
# ANÁLISIS DE VOTACIÓN INDÍGENA EN ECUADOR - ELECCIONES PRESIDENCIALES 2025
################################################################################
#
# Investigación: Patrones de voto homogéneo en cantones con alta población indígena
# Autor: Análisis Estadístico Electoral
# Fecha: 2025
#
# OBJETIVO:
# ---------
# Examinar si existe un patrón de voto homogéneo entre cantones ecuatorianos con
# alta población indígena en las elecciones presidenciales de 2025, y qué factores
# socioeconómicos y territoriales explican las variaciones en sus preferencias
# electorales bajo condiciones de polarización bidimensional.
#
# HIPÓTESIS:
# ----------
# H1: Existe un patrón de voto homogéneo pro-correísta en cantones con alta población indígena
# H2: La polarización económica (PIB per cápita) modera la relación entre población indígena y voto
# H3: El contexto territorial (Costa vs Sierra vs Amazonía) media la preferencia electoral indígena
#
# METODOLOGÍA:
# ------------
# - Variable dependiente: Proporción de votos válidos para Luisa González (correísta)
# - Método: Regresión logística con familia binomial (variable acotada 0-1, no normal)
# - Ponderación: Por número de votos válidos cantonales
# - Control multicolinealidad: Pruebas VIF < 5
# - Nivel de significancia: α = 0.05
#
# VARIABLES:
# ----------
# Dependiente:
#   - prop_gonzalez: Proporción de votos válidos para Luisa González
#
# Independientes principales:
#   - pob_indigena_pct: Porcentaje de población indígena cantonal
#   - log_pib_pc: Logaritmo del PIB per cápita
#   - costa: Dummy región Costa (ref: Sierra)
#   - amazonia: Dummy región Amazonía/Oriente (ref: Sierra)
#
# Interacciones (heterogeneidad contextual):
#   - indigena_x_logpib: % indígena × log PIB per cápita (H2: polarización económica)
#   - indigena_x_costa: % indígena × Costa (H3: mediación territorial)
#   - indigena_x_amazonia: % indígena × Amazonía (H3: mediación territorial)
#
# Variables de control:
#   - agua_publica: Acceso a agua de red pública (%)
#   - electricidad: Acceso a electricidad (%)
#   - tasa_homicidios: Tasa de homicidios por 100,000 hab
#   - altitud: Altitud en metros (proxy ruralidad/dispersión)
#   - log_poblacion: Logaritmo de población total
#
################################################################################

# ==============================================================================
# INSTALACIÓN Y CARGA DE PAQUETES
# ==============================================================================

# Lista de paquetes necesarios
paquetes <- c("tidyverse", "car", "lmtest", "broom", "margins",
              "ggeffects", "stargazer", "patchwork", "scales")

# Instalar paquetes faltantes
paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(paquetes_faltantes) > 0) {
  cat("Instalando paquetes faltantes:", paquetes_faltantes, "\n")
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

# Cargar paquetes
suppressPackageStartupMessages({
  library(tidyverse)    # Manipulación de datos y visualización
  library(car)          # VIF y diagnósticos
  library(lmtest)       # Tests de modelos
  library(broom)        # Convertir modelos a dataframes
  library(margins)      # Efectos marginales
  library(ggeffects)    # Predicciones del modelo
  library(stargazer)    # Tablas de regresión
  library(patchwork)    # Combinar gráficos
  library(scales)       # Formato de ejes
})

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

# Establecer semilla para reproducibilidad
set.seed(42)

# Crear directorio de resultados
directorio_resultados <- "resultados_votacion_indigena"
if (!dir.exists(directorio_resultados)) {
  dir.create(directorio_resultados, recursive = TRUE)
}

# Opciones de visualización
theme_set(theme_minimal(base_size = 12))
options(scipen = 999)  # Evitar notación científica

# ==============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

cat("\n", rep("=", 80), "\n", sep = "")
cat("FASE 1: CARGA Y PREPARACIÓN DE DATOS\n")
cat(rep("=", 80), "\n\n", sep = "")

# Cargar datos
datos <- read_delim(
  "Basecantones2csv.csv",
  delim = ";",
  locale = locale(decimal_mark = ",", encoding = "UTF-8"),
  show_col_types = FALSE
)

cat("✓ Dataset cargado:", nrow(datos), "cantones\n")
cat("✓ Variables:", ncol(datos), "columnas\n")

# Renombrar columnas
nombres_columnas <- c(
  "canton", "provincia", "votos_noboa_abs", "votos_gonzalez_abs",
  "votos_noboa_pct", "votos_gonzalez_pct", "poblacion",
  "pob_indigena_pct", "agua_publica", "electricidad",
  "pib_per_capita", "tasa_homicidios", "altitud",
  "costa", "sierra", "oriente", "insular"
)

colnames(datos) <- nombres_columnas

# ==============================================================================
# PREPARACIÓN DE VARIABLES
# ==============================================================================

cat("\n", rep("-", 80), "\n", sep = "")
cat("Preparación de variables\n")
cat(rep("-", 80), "\n\n", sep = "")

datos <- datos %>%
  mutate(
    # ========================================================================
    # VARIABLE DEPENDIENTE: Proporción de votos para González
    # ========================================================================
    prop_gonzalez = votos_gonzalez_pct / 100,

    # ========================================================================
    # PESOS: Votos válidos totales
    # ========================================================================
    votos_validos = votos_noboa_abs + votos_gonzalez_abs,

    # ========================================================================
    # VARIABLES INDEPENDIENTES PRINCIPALES
    # ========================================================================

    # Asegurar que % indígena esté en [0, 100]
    pob_indigena_pct = pmax(0, pmin(100, pob_indigena_pct)),

    # Logaritmo del PIB per cápita
    log_pib_pc = log(pib_per_capita),

    # Renombrar Oriente a Amazonía
    amazonia = oriente,

    # ========================================================================
    # TÉRMINOS DE INTERACCIÓN
    # ========================================================================

    # H2: Polarización económica
    indigena_x_logpib = pob_indigena_pct * log_pib_pc,

    # H3: Mediación territorial
    indigena_x_costa = pob_indigena_pct * costa,
    indigena_x_amazonia = pob_indigena_pct * amazonia,

    # ========================================================================
    # VARIABLES DE CONTROL
    # ========================================================================

    # Logaritmo de población
    log_poblacion = log(poblacion)
  ) %>%
  # Excluir región Insular (grupo muy pequeño)
  filter(insular == 0)

cat("✓ Cantones excluidos (región Insular):", sum(read_delim("Basecantones2csv.csv", delim = ";", locale = locale(decimal_mark = ","), show_col_types = FALSE)$Insular), "\n")
cat("✓ Cantones en análisis:", nrow(datos), "\n")

# Verificar valores faltantes
valores_faltantes <- colSums(is.na(datos))
if (any(valores_faltantes > 0)) {
  cat("\n⚠ Valores faltantes detectados:\n")
  print(valores_faltantes[valores_faltantes > 0])

  # Imputar con mediana
  datos <- datos %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

cat("\n✓ Variables preparadas exitosamente\n")
cat("✓ Observaciones finales:", nrow(datos), "\n")

# ==============================================================================
# FASE 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("FASE 2: ANÁLISIS EXPLORATORIO DE DATOS\n")
cat(rep("=", 80), "\n\n", sep = "")

# Variables de interés
vars_interes <- c("prop_gonzalez", "pob_indigena_pct", "log_pib_pc",
                  "agua_publica", "electricidad", "tasa_homicidios", "altitud")

# Estadísticas descriptivas
cat("Estadísticas Descriptivas:\n")
cat(rep("-", 80), "\n", sep = "")

desc_stats <- datos %>%
  select(all_of(vars_interes)) %>%
  summarise(across(everything(), list(
    Media = ~mean(., na.rm = TRUE),
    Mediana = ~median(., na.rm = TRUE),
    SD = ~sd(., na.rm = TRUE),
    Min = ~min(., na.rm = TRUE),
    Max = ~max(., na.rm = TRUE),
    CV = ~(sd(., na.rm = TRUE) / mean(., na.rm = TRUE)) * 100,
    Asimetría = ~moments::skewness(., na.rm = TRUE),
    Curtosis = ~moments::kurtosis(., na.rm = TRUE)
  ))) %>%
  pivot_longer(everything(), names_to = c("Variable", "Estadística"), names_sep = "_") %>%
  pivot_wider(names_from = Estadística, values_from = value)

print(desc_stats, n = Inf)

# Distribución por región
cat("\n\nDistribución de cantones por región:\n")
cat(rep("-", 80), "\n", sep = "")

datos %>%
  mutate(
    Region = case_when(
      costa == 1 ~ "Costa",
      sierra == 1 ~ "Sierra",
      amazonia == 1 ~ "Amazonía"
    )
  ) %>%
  count(Region) %>%
  mutate(Porcentaje = n / sum(n) * 100) %>%
  arrange(desc(n)) %>%
  print()

# Voto promedio por región
cat("\nVoto promedio por González según región:\n")
cat(rep("-", 80), "\n", sep = "")

datos %>%
  mutate(
    Region = case_when(
      costa == 1 ~ "Costa",
      sierra == 1 ~ "Sierra",
      amazonia == 1 ~ "Amazonía"
    )
  ) %>%
  group_by(Region) %>%
  summarise(
    Media = mean(prop_gonzalez) * 100,
    SD = sd(prop_gonzalez) * 100,
    .groups = "drop"
  ) %>%
  print()

# Correlaciones bivariadas clave
cat("\nCorrelación bivariada (Pearson):\n")
cat(rep("-", 80), "\n", sep = "")

correlaciones <- list(
  list("prop_gonzalez", "pob_indigena_pct", "Voto González vs % Población Indígena"),
  list("prop_gonzalez", "log_pib_pc", "Voto González vs Log PIB per cápita"),
  list("pob_indigena_pct", "log_pib_pc", "% Población Indígena vs Log PIB per cápita")
)

for (corr in correlaciones) {
  var1 <- corr[[1]]
  var2 <- corr[[2]]
  desc <- corr[[3]]

  test <- cor.test(datos[[var1]], datos[[var2]], method = "pearson")

  sig <- ifelse(test$p.value < 0.001, "***",
                ifelse(test$p.value < 0.01, "**",
                       ifelse(test$p.value < 0.05, "*", "ns")))

  cat(sprintf("  %s:\n", desc))
  cat(sprintf("    r = %.4f, p = %.4f %s\n", test$estimate, test$p.value, sig))
}

# Guardar estadísticas descriptivas
write_csv(desc_stats, file.path(directorio_resultados, "01_estadisticas_descriptivas.csv"))

# ==============================================================================
# FASE 3: DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("FASE 3: DIAGNÓSTICO DE MULTICOLINEALIDAD (VIF)\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("Criterio conservador: VIF < 5\n")
cat("  VIF < 5:   ✓ Sin multicolinealidad problemática\n")
cat("  VIF 5-10:  ⚠ Multicolinealidad moderada\n")
cat("  VIF > 10:  ✗ Multicolinealidad severa\n")

# Ajustar modelo lineal temporal para VIF
# (VIF no depende de la familia, solo de las X)
formula_vif <- prop_gonzalez ~ pob_indigena_pct + log_pib_pc + costa + amazonia +
  indigena_x_logpib + indigena_x_costa + indigena_x_amazonia +
  agua_publica + electricidad + tasa_homicidios + altitud + log_poblacion

modelo_vif <- lm(formula_vif, data = datos)

# Calcular VIF
vif_values <- vif(modelo_vif)

# Convertir a dataframe
vif_df <- data.frame(
  Variable = names(vif_values),
  VIF = as.numeric(vif_values)
) %>%
  arrange(desc(VIF))

cat("\n\nResultados VIF:\n")
cat(rep("-", 80), "\n", sep = "")

for (i in 1:nrow(vif_df)) {
  var <- vif_df$Variable[i]
  vif_val <- vif_df$VIF[i]

  status <- ifelse(vif_val < 5, "✓ OK",
                   ifelse(vif_val < 10, "⚠ MODERADO", "✗ SEVERO"))

  cat(sprintf("  %-30s VIF = %7.3f  %s\n", var, vif_val, status))
}

# Diagnóstico
max_vif <- max(vif_df$VIF)
cat("\n", rep("-", 80), "\n", sep = "")

if (max_vif < 5) {
  cat("✓ DIAGNÓSTICO: Todos los VIF < 5. No hay multicolinealidad problemática.\n")
  cat("  El modelo cumple con el criterio conservador.\n")
} else if (max_vif < 10) {
  cat("⚠ ADVERTENCIA: Algunos VIF entre 5-10 (multicolinealidad moderada):\n")
  vif_df %>%
    filter(VIF >= 5) %>%
    mutate(Mensaje = sprintf("    - %s: VIF = %.3f", Variable, VIF)) %>%
    pull(Mensaje) %>%
    cat(sep = "\n")
  cat("\n  Considerar centrar variables antes de crear interacciones.\n")
} else {
  cat("✗ ALERTA: Multicolinealidad severa detectada (VIF > 10):\n")
  vif_df %>%
    filter(VIF >= 10) %>%
    mutate(Mensaje = sprintf("    - %s: VIF = %.3f", Variable, VIF)) %>%
    pull(Mensaje) %>%
    cat(sep = "\n")
  cat("\n  RECOMENDACIÓN: Eliminar términos de interacción o centrar variables.\n")
}

# Guardar VIF
write_csv(vif_df, file.path(directorio_resultados, "02_vif_multicolinealidad.csv"))

# ==============================================================================
# FASE 4: REGRESIÓN LOGÍSTICA BINOMIAL
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("FASE 4: REGRESIÓN LOGÍSTICA BINOMIAL\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("Especificación del modelo:\n")
cat(rep("-", 80), "\n", sep = "")
cat("Familia: Binomial (apropiada para proporciones [0,1])\n")
cat("Link: Logit\n")
cat("Ponderación: Votos válidos cantonales\n")
cat("Estimación: Máxima verosimilitud (MLE)\n")

# ==============================================================================
# MODELO 1: Solo efectos principales (sin interacciones)
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("MODELO 1: EFECTOS PRINCIPALES (Sin interacciones)\n")
cat(rep("=", 80), "\n\n", sep = "")

formula_m1 <- prop_gonzalez ~ pob_indigena_pct + log_pib_pc + costa + amazonia +
  agua_publica + electricidad + tasa_homicidios + altitud + log_poblacion

modelo1 <- glm(
  formula = formula_m1,
  data = datos,
  family = binomial(link = "logit"),
  weights = votos_validos
)

print(summary(modelo1))

# ==============================================================================
# MODELO 2: Con interacciones (H2 y H3)
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("MODELO 2: CON INTERACCIONES (H2: Polarización económica, H3: Mediación territorial)\n")
cat(rep("=", 80), "\n\n", sep = "")

formula_m2 <- prop_gonzalez ~ pob_indigena_pct + log_pib_pc + costa + amazonia +
  indigena_x_logpib + indigena_x_costa + indigena_x_amazonia +
  agua_publica + electricidad + tasa_homicidios + altitud + log_poblacion

modelo2 <- glm(
  formula = formula_m2,
  data = datos,
  family = binomial(link = "logit"),
  weights = votos_validos
)

print(summary(modelo2))

# ==============================================================================
# COMPARACIÓN DE MODELOS
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("COMPARACIÓN DE MODELOS\n")
cat(rep("=", 80), "\n\n", sep = "")

# Test de razón de verosimilitud (LR test)
lr_test <- anova(modelo1, modelo2, test = "LRT")

cat("Test de Razón de Verosimilitud (Likelihood Ratio Test):\n")
cat("  H0: Modelo 1 (sin interacciones) es suficiente\n")
cat("  Ha: Modelo 2 (con interacciones) explica significativamente mejor\n\n")

print(lr_test)

p_value_lr <- lr_test$`Pr(>Chi)`[2]

if (!is.na(p_value_lr) && p_value_lr < 0.05) {
  cat("\n  ✓ RESULTADO: Rechazamos H0 (p < 0.05)\n")
  cat("    Las interacciones mejoran significativamente el modelo.\n")
  cat("    MODELO PREFERIDO: Modelo 2 (con interacciones)\n")
} else {
  cat("\n  → RESULTADO: No rechazamos H0 (p ≥ 0.05)\n")
  cat("    Las interacciones no mejoran significativamente el modelo.\n")
  cat("    MODELO PREFERIDO: Modelo 1 (más parsimonioso)\n")
}

# Criterios de información
cat("\n\nCriterios de Información:\n")
cat(sprintf("%-30s %15s %15s %15s\n", "", "Modelo 1", "Modelo 2", "Diferencia"))
cat(rep("-", 80), "\n", sep = "")
cat(sprintf("%-30s %15.2f %15.2f %15.2f\n", "AIC (menor es mejor)", AIC(modelo1), AIC(modelo2), AIC(modelo2) - AIC(modelo1)))
cat(sprintf("%-30s %15.2f %15.2f %15.2f\n", "BIC (menor es mejor)", BIC(modelo1), BIC(modelo2), BIC(modelo2) - BIC(modelo1)))
cat(sprintf("%-30s %15.2f %15.2f %15.2f\n", "Log-Likelihood", logLik(modelo1), logLik(modelo2), logLik(modelo2) - logLik(modelo1)))

# Pseudo R² (McFadden)
pseudo_r2_m1 <- 1 - (modelo1$deviance / modelo1$null.deviance)
pseudo_r2_m2 <- 1 - (modelo2$deviance / modelo2$null.deviance)

cat(sprintf("\n\nPseudo R² (McFadden):\n"))
cat(sprintf("  Modelo 1: %.4f\n", pseudo_r2_m1))
cat(sprintf("  Modelo 2: %.4f\n", pseudo_r2_m2))
cat(sprintf("  Incremento: %.4f\n", pseudo_r2_m2 - pseudo_r2_m1))

# Guardar resultados de modelos
tidy_m1 <- tidy(modelo1, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

tidy_m2 <- tidy(modelo2, conf.int = TRUE) %>%
  mutate(across(where(is.numeric), ~round(., 4)))

write_csv(tidy_m1, file.path(directorio_resultados, "03_modelo1_efectos_principales.csv"))
write_csv(tidy_m2, file.path(directorio_resultados, "04_modelo2_con_interacciones.csv"))

# Tabla comparativa con stargazer
sink(file.path(directorio_resultados, "07_comparacion_modelos.txt"))
stargazer(modelo1, modelo2,
          type = "text",
          title = "Comparación de Modelos de Regresión Logística",
          column.labels = c("Modelo 1", "Modelo 2"),
          model.names = FALSE,
          dep.var.labels = "Proporción de votos por González",
          covariate.labels = c(
            "% Población Indígena",
            "Log PIB per cápita",
            "Costa",
            "Amazonía",
            "Indígena × Log PIB",
            "Indígena × Costa",
            "Indígena × Amazonía",
            "Agua Pública (%)",
            "Electricidad (%)",
            "Tasa Homicidios",
            "Altitud",
            "Log Población"
          ),
          star.cutoffs = c(0.05, 0.01, 0.001),
          notes = c("* p<0.05; ** p<0.01; *** p<0.001"),
          notes.append = FALSE
)
sink()

cat("\n✓ Tabla comparativa guardada en: 07_comparacion_modelos.txt\n")

# ==============================================================================
# FASE 5: INTERPRETACIÓN Y EFECTOS MARGINALES
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("FASE 5: INTERPRETACIÓN DE RESULTADOS\n")
cat(rep("=", 80), "\n\n", sep = "")

# ==============================================================================
# Odds Ratios
# ==============================================================================

cat("Odds Ratios (OR) e Intervalos de Confianza 95%:\n")
cat(rep("-", 80), "\n", sep = "")
cat("Interpretación: OR > 1 aumenta la probabilidad de votar por González\n")
cat("                OR < 1 disminuye la probabilidad de votar por González\n")
cat("                OR = 1 no tiene efecto\n\n")

# Calcular odds ratios para Modelo 2
or_table <- tidy(modelo2, conf.int = TRUE, exponentiate = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    Sig = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01 ~ "**",
      p.value < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  ) %>%
  select(term, estimate, conf.low, conf.high, p.value, Sig) %>%
  rename(
    Variable = term,
    OR = estimate,
    IC_95_inf = conf.low,
    IC_95_sup = conf.high
  )

print(or_table, n = Inf)

# Guardar odds ratios
write_csv(or_table, file.path(directorio_resultados, "05_odds_ratios.csv"))

# ==============================================================================
# Efectos Marginales Promedio (AME)
# ==============================================================================

cat("\n\nEfectos Marginales Promedio (AME):\n")
cat(rep("-", 80), "\n", sep = "")
cat("Cambio en probabilidad de votar por González ante cambio unitario en X\n\n")

# Calcular efectos marginales
margeff <- margins(modelo2)

# Resumen de efectos marginales
margeff_summary <- summary(margeff) %>%
  as_tibble() %>%
  mutate(
    Sig = case_when(
      p < 0.001 ~ "***",
      p < 0.01 ~ "**",
      p < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

print(margeff_summary, n = Inf)

# Guardar efectos marginales
write_csv(margeff_summary, file.path(directorio_resultados, "06_efectos_marginales.csv"))

# ==============================================================================
# Interpretación narrativa
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("INTERPRETACIÓN NARRATIVA DE HIPÓTESIS\n")
cat(rep("=", 80), "\n\n", sep = "")

coefs <- tidy(modelo2)

variables_interes <- c("pob_indigena_pct", "log_pib_pc", "costa", "amazonia",
                       "indigena_x_logpib", "indigena_x_costa", "indigena_x_amazonia")

for (var in variables_interes) {
  row <- coefs %>% filter(term == var)

  if (nrow(row) > 0) {
    coef <- row$estimate
    pval <- row$p.value
    odds_ratio <- exp(coef)

    sig_text <- ifelse(pval < 0.001, "***",
                       ifelse(pval < 0.01, "**",
                              ifelse(pval < 0.05, "*", "ns")))

    cat(sprintf("\n%s:\n", var))
    cat(sprintf("  Coeficiente: %.4f %s\n", coef, sig_text))
    cat(sprintf("  Odds Ratio: %.4f\n", odds_ratio))
    cat(sprintf("  p-value: %.6f\n", pval))

    # Interpretaciones específicas
    if (var == "pob_indigena_pct") {
      if (pval < 0.05) {
        if (coef > 0) {
          cat(sprintf("  → Un aumento de 1%% en población indígena incrementa los odds de votar\n"))
          cat(sprintf("    por González en %.2f%%\n", (odds_ratio - 1) * 100))
          cat("  → CONFIRMA H1: Patrón de voto pro-correísta en cantones indígenas\n")
        } else {
          cat(sprintf("  → Un aumento de 1%% en población indígena reduce los odds de votar\n"))
          cat(sprintf("    por González en %.2f%%\n", (1 - odds_ratio) * 100))
          cat("  → RECHAZA H1: No hay patrón pro-correísta\n")
        }
      } else {
        cat("  → Efecto NO significativo (p ≥ 0.05)\n")
        cat("  → NO se puede confirmar H1\n")
      }
    } else if (var == "indigena_x_logpib") {
      if (pval < 0.05) {
        cat("  → CONFIRMA H2: Polarización económica modera efecto indígena\n")
        cat("  → La relación entre población indígena y voto depende del nivel de PIB\n")
      } else {
        cat("  → RECHAZA H2: No hay moderación por PIB per cápita\n")
      }
    } else if (var == "indigena_x_costa") {
      if (pval < 0.05) {
        cat("  → CONFIRMA H3: Mediación territorial en Costa\n")
        cat("  → El efecto de población indígena es diferente en Costa vs Sierra\n")
      } else {
        cat("  → NO hay mediación territorial en Costa\n")
      }
    } else if (var == "indigena_x_amazonia") {
      if (pval < 0.05) {
        cat("  → CONFIRMA H3: Mediación territorial en Amazonía\n")
        cat("  → El efecto de población indígena es diferente en Amazonía vs Sierra\n")
      } else {
        cat("  → NO hay mediación territorial en Amazonía\n")
      }
    }
  }
}

# ==============================================================================
# FASE 6: VISUALIZACIONES
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("FASE 6: VISUALIZACIONES\n")
cat(rep("=", 80), "\n\n", sep = "")

# ==============================================================================
# 1. Distribución de la variable dependiente
# ==============================================================================

p1 <- ggplot(datos, aes(x = prop_gonzalez)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(prop_gonzalez)), color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(aes(xintercept = median(prop_gonzalez)), color = "green", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Distribución de la Variable Dependiente",
    x = "Proporción de votos por González",
    y = "Frecuencia"
  ) +
  theme_minimal()

datos_region <- datos %>%
  mutate(
    Region = case_when(
      costa == 1 ~ "Costa",
      sierra == 1 ~ "Sierra",
      amazonia == 1 ~ "Amazonía"
    )
  )

p2 <- ggplot(datos_region, aes(x = Region, y = prop_gonzalez, fill = Region)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("Costa" = "lightblue", "Sierra" = "lightgreen", "Amazonía" = "lightcoral")) +
  labs(
    title = "Voto por González según Región",
    x = "",
    y = "Proporción de votos por González"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

p_combined1 <- p1 + p2

ggsave(
  file.path(directorio_resultados, "01_variable_dependiente.png"),
  plot = p_combined1,
  width = 14,
  height = 5,
  dpi = 300
)

cat("✓ Guardado: 01_variable_dependiente.png\n")

# ==============================================================================
# 2. H1 y H3: % Indígena vs Voto
# ==============================================================================

# Global
cor_test <- cor.test(datos$pob_indigena_pct, datos$prop_gonzalez)

p3 <- ggplot(datos, aes(x = pob_indigena_pct, y = prop_gonzalez)) +
  geom_point(aes(size = votos_validos), alpha = 0.6) +
  geom_smooth(method = "lm", color = "red", linetype = "dashed", linewidth = 1.5, se = FALSE) +
  annotate("text", x = min(datos$pob_indigena_pct) + 5, y = max(datos$prop_gonzalez) - 0.05,
           label = sprintf("r = %.3f, p = %.4f", cor_test$estimate, cor_test$p.value),
           hjust = 0, vjust = 1, size = 4, color = "black",
           fontface = "bold") +
  labs(
    title = "H1: Patrón de Voto Indígena (Global)",
    x = "% Población Indígena",
    y = "Proporción de votos por González"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Por región
p4 <- ggplot(datos_region, aes(x = pob_indigena_pct, y = prop_gonzalez, color = Region)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed", linewidth = 1.5) +
  scale_color_manual(values = c("Costa" = "blue", "Sierra" = "green", "Amazonía" = "red")) +
  labs(
    title = "H3: Mediación Territorial",
    x = "% Población Indígena",
    y = "Proporción de votos por González",
    color = "Región"
  ) +
  theme_minimal()

p_combined2 <- p3 + p4

ggsave(
  file.path(directorio_resultados, "02_h1_h3_indigena_voto.png"),
  plot = p_combined2,
  width = 14,
  height = 5,
  dpi = 300
)

cat("✓ Guardado: 02_h1_h3_indigena_voto.png\n")

# ==============================================================================
# 3. Coeficientes del Modelo 2 con IC 95%
# ==============================================================================

coefs_plot <- tidy(modelo2, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    Significativo = ifelse(p.value < 0.05, "Sí", "No"),
    term = factor(term, levels = term[order(abs(estimate))])
  )

p5 <- ggplot(coefs_plot, aes(x = estimate, y = term, color = Significativo)) +
  geom_point(size = 4) +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "solid", color = "black", linewidth = 0.8) +
  scale_color_manual(values = c("Sí" = "red", "No" = "gray")) +
  labs(
    title = "Coeficientes del Modelo 2 con IC 95%",
    subtitle = "(Rojo: p < 0.05, Gris: no significativo)",
    x = "Coeficiente (log-odds)",
    y = "",
    color = "Significativo"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(
  file.path(directorio_resultados, "03_coeficientes_modelo2.png"),
  plot = p5,
  width = 10,
  height = 8,
  dpi = 300
)

cat("✓ Guardado: 03_coeficientes_modelo2.png\n")

# ==============================================================================
# 4. H2: Polarización Económica
# ==============================================================================

# Crear grid de predicción
pob_indigena_range <- seq(0, 100, length.out = 50)
log_pib_quartiles <- quantile(datos$log_pib_pc, c(0.25, 0.50, 0.75))

pred_data_list <- list()

for (i in 1:3) {
  q_name <- c("Q1 (PIB bajo)", "Q2 (PIB medio)", "Q3 (PIB alto)")[i]
  q_val <- log_pib_quartiles[i]

  pred_data <- expand.grid(
    pob_indigena_pct = pob_indigena_range,
    log_pib_pc = q_val,
    costa = 0,
    amazonia = 0,
    agua_publica = mean(datos$agua_publica),
    electricidad = mean(datos$electricidad),
    tasa_homicidios = mean(datos$tasa_homicidios),
    altitud = mean(datos$altitud),
    log_poblacion = mean(datos$log_poblacion)
  ) %>%
    mutate(
      indigena_x_logpib = pob_indigena_pct * log_pib_pc,
      indigena_x_costa = 0,
      indigena_x_amazonia = 0,
      Cuartil = q_name
    )

  pred_data$pred <- predict(modelo2, newdata = pred_data, type = "response")

  pred_data_list[[i]] <- pred_data
}

pred_h2 <- bind_rows(pred_data_list)

p6 <- ggplot(pred_h2, aes(x = pob_indigena_pct, y = pred, color = Cuartil)) +
  geom_line(linewidth = 1.5) +
  labs(
    title = "H2: Polarización Económica",
    subtitle = "(Efecto de % Indígena según nivel de PIB per cápita)",
    x = "% Población Indígena",
    y = "Probabilidad predicha de votar por González",
    color = "Nivel de PIB"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(
  file.path(directorio_resultados, "04_h2_polarizacion_economica.png"),
  plot = p6,
  width = 10,
  height = 6,
  dpi = 300
)

cat("✓ Guardado: 04_h2_polarizacion_economica.png\n")

# ==============================================================================
# 5. H3: Mediación Territorial
# ==============================================================================

pred_data_regiones <- list()

regiones <- list(
  list("costa", "Costa", "blue"),
  list("sierra", "Sierra", "green"),
  list("amazonia", "Amazonía", "red")
)

for (reg_info in regiones) {
  reg <- reg_info[[1]]
  nombre <- reg_info[[2]]

  pred_data <- expand.grid(
    pob_indigena_pct = pob_indigena_range,
    log_pib_pc = mean(datos$log_pib_pc),
    costa = ifelse(reg == "costa", 1, 0),
    amazonia = ifelse(reg == "amazonia", 1, 0),
    agua_publica = mean(datos$agua_publica),
    electricidad = mean(datos$electricidad),
    tasa_homicidios = mean(datos$tasa_homicidios),
    altitud = mean(datos$altitud),
    log_poblacion = mean(datos$log_poblacion)
  ) %>%
    mutate(
      indigena_x_logpib = pob_indigena_pct * log_pib_pc,
      indigena_x_costa = pob_indigena_pct * costa,
      indigena_x_amazonia = pob_indigena_pct * amazonia,
      Region = nombre
    )

  pred_data$pred <- predict(modelo2, newdata = pred_data, type = "response")

  pred_data_regiones[[nombre]] <- pred_data
}

pred_h3 <- bind_rows(pred_data_regiones)

p7 <- ggplot(pred_h3, aes(x = pob_indigena_pct, y = pred, color = Region)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c("Costa" = "blue", "Sierra" = "green", "Amazonía" = "red")) +
  labs(
    title = "H3: Mediación Territorial",
    subtitle = "(Efecto de % Indígena según región)",
    x = "% Población Indígena",
    y = "Probabilidad predicha de votar por González",
    color = "Región"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave(
  file.path(directorio_resultados, "05_h3_mediacion_territorial.png"),
  plot = p7,
  width = 10,
  height = 6,
  dpi = 300
)

cat("✓ Guardado: 05_h3_mediacion_territorial.png\n")

# ==============================================================================
# 6. Diagnóstico de Residuos
# ==============================================================================

datos_diagnostico <- datos %>%
  mutate(
    fitted = fitted(modelo2),
    residuals = residuals(modelo2, type = "deviance"),
    residuals_std = sqrt(abs(residuals))
  )

p8 <- ggplot(datos_diagnostico, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residuos vs Valores Ajustados",
    x = "Valores Ajustados",
    y = "Residuos Deviance"
  ) +
  theme_minimal()

p9 <- ggplot(datos_diagnostico, aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  labs(
    title = "Q-Q Plot de Residuos",
    x = "Cuantiles Teóricos",
    y = "Cuantiles Observados"
  ) +
  theme_minimal()

p10 <- ggplot(datos_diagnostico, aes(x = fitted, y = residuals_std)) +
  geom_point(alpha = 0.5) +
  labs(
    title = "Scale-Location",
    x = "Valores Ajustados",
    y = "√|Residuos Estandarizados|"
  ) +
  theme_minimal()

p11 <- ggplot(datos_diagnostico, aes(x = seq_along(residuals), y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = c(-2, 2), color = "red", linetype = "dashed") +
  labs(
    title = "Residuos por Observación",
    x = "Índice de Observación",
    y = "Residuos Deviance"
  ) +
  theme_minimal()

p_diagnostico <- (p8 + p9) / (p10 + p11)

ggsave(
  file.path(directorio_resultados, "06_diagnostico_residuos.png"),
  plot = p_diagnostico,
  width = 14,
  height = 10,
  dpi = 300
)

cat("✓ Guardado: 06_diagnostico_residuos.png\n")

cat("\n✓ Todas las visualizaciones guardadas en:", directorio_resultados, "\n")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("ANÁLISIS COMPLETADO EXITOSAMENTE\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("Consulte los archivos generados en:\n")
cat("  -", directorio_resultados, "/ (tablas CSV)\n")
cat("  -", directorio_resultados, "/ (gráficos PNG)\n")
cat("\n", rep("=", 80), "\n", sep = "")
