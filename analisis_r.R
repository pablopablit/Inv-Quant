# ==============================================================================
# ANÁLISIS ESTADÍSTICO DE CANTONES DE ECUADOR
# Versión R
# ==============================================================================
#
# Este script realiza un análisis estadístico completo de los datos
# socioeconómicos y electorales de los cantones ecuatorianos.
#
# Requisitos:
#   - R >= 4.0.0
#   - tidyverse (dplyr, ggplot2, tidyr, readr)
#   - psych
#   - corrplot
#   - ggpubr
#   - moments
#
# Uso:
#   Rscript analisis_r.R
#   O desde RStudio: source("analisis_r.R")
#
# Autor: Análisis Estadístico Cantonal
# Fecha: 2025
# ==============================================================================

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================

# Limpiar entorno
rm(list = ls())

# Configurar opciones
options(scipen = 999)  # Evitar notación científica
options(digits = 4)

# Instalar paquetes si no están disponibles
paquetes_necesarios <- c("tidyverse", "psych", "corrplot", "ggpubr", "moments")

for (paquete in paquetes_necesarios) {
  if (!require(paquete, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Instalando paquete:", paquete, "\n"))
    install.packages(paquete, repos = "https://cloud.r-project.org/")
    library(paquete, character.only = TRUE)
  }
}

# Cargar librerías
library(tidyverse)
library(psych)
library(corrplot)
library(ggpubr)
library(moments)

# Configuración de rutas
RUTA_DATOS <- "Basecantones2csv.csv"
RUTA_SALIDA <- "resultados_r"

# Crear directorio de salida
if (!dir.exists(RUTA_SALIDA)) {
  dir.create(RUTA_SALIDA)
}

# Configuración global de gráficos
theme_set(theme_minimal(base_size = 12))

# ==============================================================================
# FASE 1: CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

cat("======================================================================\n")
cat("ANÁLISIS ESTADÍSTICO DE CANTONES DE ECUADOR\n")
cat("Versión R\n")
cat("======================================================================\n")
cat(paste("Inicio:", Sys.time(), "\n\n"))

cat("======================================================================\n")
cat("FASE 1: CARGA Y PREPARACIÓN DE DATOS\n")
cat("======================================================================\n")

# Cargar datos
datos <- read_delim(RUTA_DATOS, delim = ";", locale = locale(decimal_mark = ","))

cat(paste("✓ Datos cargados:", nrow(datos), "cantones,", ncol(datos), "variables\n"))

# Renombrar columnas
nombres_nuevos <- c(
  "canton", "provincia", "votos_noboa_abs", "votos_gonzalez_abs",
  "votos_noboa_pct", "votos_gonzalez_pct", "poblacion",
  "pob_indigena_pct", "agua_publica", "electricidad",
  "pib_per_capita", "tasa_homicidios", "altitud",
  "costa", "sierra", "oriente", "insular"
)

names(datos) <- nombres_nuevos

# Asegurar tipos de datos correctos
datos <- datos %>%
  mutate(across(c(votos_noboa_pct, votos_gonzalez_pct, pob_indigena_pct,
                  agua_publica, electricidad, tasa_homicidios),
                ~ as.numeric(gsub(",", ".", .))))

# Crear variable de región
datos <- datos %>%
  mutate(region = case_when(
    costa == 1 ~ "Costa",
    sierra == 1 ~ "Sierra",
    oriente == 1 ~ "Oriente",
    insular == 1 ~ "Insular",
    TRUE ~ "Desconocido"
  ))

# Convertir región a factor
datos$region <- factor(datos$region, levels = c("Costa", "Sierra", "Oriente", "Insular"))

# Verificar datos faltantes
faltantes <- sum(is.na(datos))
cat(paste("✓ Datos faltantes:", faltantes, "\n"))

# Mostrar distribución por región
cat("\nDistribución por región:\n")
print(table(datos$region))

# ==============================================================================
# FASE 2: ANÁLISIS DESCRIPTIVO
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 2: ANÁLISIS DESCRIPTIVO\n")
cat("======================================================================\n")

# Variables a analizar
variables_analisis <- c("votos_noboa_pct", "poblacion", "pob_indigena_pct",
                        "agua_publica", "electricidad", "pib_per_capita",
                        "tasa_homicidios")

# Calcular estadísticas descriptivas usando psych
cat("\nEstadísticas Descriptivas:\n")
cat("----------------------------------------------------------------------\n")

stats_descriptivas <- datos %>%
  select(all_of(variables_analisis)) %>%
  describe() %>%
  as.data.frame() %>%
  select(n, mean, sd, median, min, max, range, skew, kurtosis)

# Agregar coeficiente de variación
stats_descriptivas$cv_pct <- (stats_descriptivas$sd / stats_descriptivas$mean) * 100

# Calcular IQR manualmente
for (var in variables_analisis) {
  q1 <- quantile(datos[[var]], 0.25, na.rm = TRUE)
  q3 <- quantile(datos[[var]], 0.75, na.rm = TRUE)
  stats_descriptivas[var, "Q1"] <- q1
  stats_descriptivas[var, "Q3"] <- q3
  stats_descriptivas[var, "IQR"] <- q3 - q1
}

print(round(stats_descriptivas, 2))

# Guardar estadísticas
write_csv(stats_descriptivas, file.path(RUTA_SALIDA, "estadisticas_descriptivas.csv"))
cat("\n✓ Tabla guardada: estadisticas_descriptivas.csv\n")

# --- Identificación de Outliers ---
cat("\nIdentificación de Outliers (Método IQR):\n")
cat("----------------------------------------------------------------------\n")

identificar_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  limite_inf <- Q1 - 1.5 * IQR_val
  limite_sup <- Q3 + 1.5 * IQR_val
  return(x < limite_inf | x > limite_sup)
}

for (var in c("pib_per_capita", "tasa_homicidios", "pob_indigena_pct")) {
  outliers <- datos[identificar_outliers(datos[[var]]), ]
  n_outliers <- nrow(outliers)

  if (n_outliers > 0) {
    cat(paste("\n", toupper(var), ":", n_outliers, "outliers\n"))

    # Top 3 outliers
    top_outliers <- outliers %>%
      arrange(desc(.data[[var]])) %>%
      head(3)

    for (i in 1:nrow(top_outliers)) {
      cat(paste("  -", top_outliers$canton[i],
                "(", top_outliers$provincia[i], "):",
                round(top_outliers[[var]][i], 2), "\n"))
    }
  }
}

# --- Estadísticas por Región ---
cat("\nEstadísticas por Región:\n")
cat("----------------------------------------------------------------------\n")

stats_region <- datos %>%
  group_by(region) %>%
  summarise(
    n = n(),
    votos_noboa_mean = mean(votos_noboa_pct, na.rm = TRUE),
    votos_noboa_sd = sd(votos_noboa_pct, na.rm = TRUE),
    pib_mean = mean(pib_per_capita, na.rm = TRUE),
    pib_sd = sd(pib_per_capita, na.rm = TRUE),
    homicidios_mean = mean(tasa_homicidios, na.rm = TRUE),
    indigena_mean = mean(pob_indigena_pct, na.rm = TRUE),
    agua_mean = mean(agua_publica, na.rm = TRUE)
  )

print(stats_region)

write_csv(stats_region, file.path(RUTA_SALIDA, "estadisticas_por_region.csv"))
cat("\n✓ Tabla guardada: estadisticas_por_region.csv\n")

# ==============================================================================
# FASE 3: ANÁLISIS DE NORMALIDAD
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 3: ANÁLISIS DE NORMALIDAD\n")
cat("======================================================================\n")
cat("\nTest de Shapiro-Wilk (H0: Los datos son normales)\n")
cat("Nivel de significancia: α = 0.05\n\n")

variables_normalidad <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios",
                          "pob_indigena_pct", "agua_publica", "electricidad", "poblacion")

resultados_normalidad <- data.frame(
  Variable = character(),
  Estadistico_W = numeric(),
  p_valor = numeric(),
  Normal = character(),
  stringsAsFactors = FALSE
)

for (var in variables_normalidad) {
  test_resultado <- shapiro.test(datos[[var]])

  es_normal <- ifelse(test_resultado$p.value > 0.05, "Sí", "No")

  resultados_normalidad <- rbind(resultados_normalidad, data.frame(
    Variable = var,
    Estadistico_W = test_resultado$statistic,
    p_valor = test_resultado$p.value,
    Normal = es_normal
  ))

  simbolo <- ifelse(test_resultado$p.value > 0.05, "✓", "✗")
  cat(paste(simbolo, var, ": W =", round(test_resultado$statistic, 4),
            ", p =", format(test_resultado$p.value, scientific = TRUE),
            "→", es_normal, "\n"))
}

write_csv(resultados_normalidad, file.path(RUTA_SALIDA, "test_normalidad.csv"))
cat("\n✓ Resultados guardados: test_normalidad.csv\n")

# ==============================================================================
# FASE 4: ANÁLISIS DE CORRELACIÓN
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 4: ANÁLISIS DE CORRELACIÓN\n")
cat("======================================================================\n")
cat("\nMétodo: Correlación de Spearman (ρ)\n")
cat("(Apropiado para datos no normales)\n\n")

variables_corr <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios",
                    "pob_indigena_pct", "agua_publica", "electricidad", "poblacion")

# Matriz de correlación de Spearman
matriz_corr <- cor(datos[variables_corr], method = "spearman", use = "complete.obs")

cat("Matriz de Correlación de Spearman:\n")
print(round(matriz_corr, 3))

write.csv(round(matriz_corr, 3), file.path(RUTA_SALIDA, "matriz_correlacion.csv"))
cat("\n✓ Matriz guardada: matriz_correlacion.csv\n")

# Correlaciones principales con p-valores
cat("\nCorrelaciones Significativas:\n")
cat("----------------------------------------------------------------------\n")

pares <- list(
  c("votos_noboa_pct", "pib_per_capita", "Votos vs PIB per cápita"),
  c("votos_noboa_pct", "tasa_homicidios", "Votos vs Tasa homicidios"),
  c("votos_noboa_pct", "pob_indigena_pct", "Votos vs Pob. indígena"),
  c("agua_publica", "pib_per_capita", "Agua pública vs PIB"),
  c("agua_publica", "electricidad", "Agua vs Electricidad"),
  c("tasa_homicidios", "poblacion", "Homicidios vs Población")
)

resultados_corr <- data.frame(
  Relacion = character(),
  rho = numeric(),
  p_valor = numeric(),
  Significancia = character(),
  Interpretacion = character(),
  stringsAsFactors = FALSE
)

for (par in pares) {
  var1 <- par[1]
  var2 <- par[2]
  nombre <- par[3]

  test_corr <- cor.test(datos[[var1]], datos[[var2]], method = "spearman")
  rho <- test_corr$estimate
  p_valor <- test_corr$p.value

  # Significancia
  if (p_valor < 0.001) {
    sig <- "***"
  } else if (p_valor < 0.01) {
    sig <- "**"
  } else if (p_valor < 0.05) {
    sig <- "*"
  } else {
    sig <- "ns"
  }

  # Interpretación
  if (abs(rho) > 0.7) {
    fuerza <- "Muy fuerte"
  } else if (abs(rho) > 0.5) {
    fuerza <- "Fuerte"
  } else if (abs(rho) > 0.3) {
    fuerza <- "Moderada"
  } else {
    fuerza <- "Débil"
  }

  direccion <- ifelse(rho > 0, "positiva", "negativa")
  interpretacion <- paste(fuerza, direccion)

  resultados_corr <- rbind(resultados_corr, data.frame(
    Relacion = nombre,
    rho = rho,
    p_valor = p_valor,
    Significancia = sig,
    Interpretacion = interpretacion
  ))

  cat(paste(nombre, ":\n"))
  cat(paste("  ρ =", round(rho, 4), ", p =", format(p_valor, scientific = TRUE), sig, "\n"))
  cat(paste("  →", interpretacion, "\n\n"))
}

write_csv(resultados_corr, file.path(RUTA_SALIDA, "correlaciones_principales.csv"))
cat("✓ Correlaciones guardadas: correlaciones_principales.csv\n")

# ==============================================================================
# FASE 5: VISUALIZACIONES
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 5: GENERACIÓN DE VISUALIZACIONES\n")
cat("======================================================================\n")

# --- Histogramas ---
cat("Generando histogramas...\n")

variables_viz <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios",
                   "pob_indigena_pct", "agua_publica")

titulos_viz <- c(
  "Votos por Noboa (%)",
  "PIB per cápita (USD)",
  "Tasa de Homicidios",
  "Población Indígena (%)",
  "Acceso a Agua Pública (%)"
)

histogramas <- list()

for (i in seq_along(variables_viz)) {
  var <- variables_viz[i]
  titulo <- titulos_viz[i]
  media_val <- mean(datos[[var]], na.rm = TRUE)

  p <- ggplot(datos, aes_string(x = var)) +
    geom_histogram(aes(y = ..density..), bins = 25, fill = "steelblue",
                   color = "white", alpha = 0.7) +
    geom_density(color = "darkred", size = 1) +
    geom_vline(xintercept = media_val, color = "red", linetype = "dashed",
               size = 1) +
    labs(title = titulo,
         subtitle = paste("Media:", round(media_val, 2)),
         x = "", y = "Densidad") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))

  histogramas[[i]] <- p
}

# Combinar histogramas
histogramas_combinados <- ggarrange(plotlist = histogramas, ncol = 3, nrow = 2)
histogramas_combinados <- annotate_figure(histogramas_combinados,
                                          top = text_grob("Distribución de Variables Principales",
                                                          face = "bold", size = 16))

ggsave(file.path(RUTA_SALIDA, "histogramas.png"), histogramas_combinados,
       width = 15, height = 10, dpi = 300)
cat("✓ Histogramas guardados: histogramas.png\n")

# --- Q-Q Plots ---
cat("Generando Q-Q plots...\n")

qqplots <- list()

for (i in seq_along(variables_viz)) {
  var <- variables_viz[i]
  titulo <- titulos_viz[i]

  p <- ggplot(datos, aes_string(sample = var)) +
    stat_qq(color = "steelblue", size = 2) +
    stat_qq_line(color = "red", size = 1) +
    labs(title = paste("Q-Q Plot:", titulo),
         x = "Cuantiles Teóricos", y = "Cuantiles Muestrales") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"))

  qqplots[[i]] <- p
}

qqplots_combinados <- ggarrange(plotlist = qqplots, ncol = 3, nrow = 2)
qqplots_combinados <- annotate_figure(qqplots_combinados,
                                      top = text_grob("Gráficos Q-Q para Evaluación de Normalidad",
                                                      face = "bold", size = 16))

ggsave(file.path(RUTA_SALIDA, "qq_plots.png"), qqplots_combinados,
       width = 15, height = 10, dpi = 300)
cat("✓ Q-Q plots guardados: qq_plots.png\n")

# --- Boxplots ---
cat("Generando boxplots...\n")

boxplots <- list()

for (i in seq_along(variables_viz)) {
  var <- variables_viz[i]
  titulo <- titulos_viz[i]

  p <- ggplot(datos, aes_string(y = var)) +
    geom_boxplot(fill = "lightblue", color = "darkblue", outlier.color = "red",
                 outlier.size = 2) +
    labs(title = titulo, y = "") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"),
          axis.text.x = element_blank())

  boxplots[[i]] <- p
}

boxplots_combinados <- ggarrange(plotlist = boxplots, ncol = 3, nrow = 2)
boxplots_combinados <- annotate_figure(boxplots_combinados,
                                       top = text_grob("Boxplots - Distribución y Valores Extremos",
                                                       face = "bold", size = 16))

ggsave(file.path(RUTA_SALIDA, "boxplots.png"), boxplots_combinados,
       width = 15, height = 10, dpi = 300)
cat("✓ Boxplots guardados: boxplots.png\n")

# --- Boxplots por Región ---
cat("Generando boxplots por región...\n")

boxplots_region <- list()

for (i in seq_along(variables_viz)) {
  var <- variables_viz[i]
  titulo <- titulos_viz[i]

  p <- ggplot(datos, aes_string(x = "region", y = var, fill = "region")) +
    geom_boxplot(outlier.color = "red", outlier.size = 2) +
    scale_fill_brewer(palette = "Set3") +
    labs(title = titulo, x = "Región", y = "") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

  boxplots_region[[i]] <- p
}

boxplots_region_combinados <- ggarrange(plotlist = boxplots_region, ncol = 3, nrow = 2)
boxplots_region_combinados <- annotate_figure(boxplots_region_combinados,
                                              top = text_grob("Comparación de Variables por Región Geográfica",
                                                              face = "bold", size = 16))

ggsave(file.path(RUTA_SALIDA, "boxplots_region.png"), boxplots_region_combinados,
       width = 15, height = 10, dpi = 300)
cat("✓ Boxplots por región guardados: boxplots_region.png\n")

# --- Heatmap de Correlación ---
cat("Generando heatmap de correlación...\n")

png(file.path(RUTA_SALIDA, "heatmap_correlacion.png"),
    width = 10, height = 8, units = "in", res = 300)

corrplot(matriz_corr, method = "color", type = "lower",
         addCoef.col = "black", number.cex = 0.8,
         tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#BB4444", "white", "#4477AA"))(200),
         title = "\nMatriz de Correlación de Spearman",
         mar = c(0, 0, 2, 0))

dev.off()
cat("✓ Heatmap guardado: heatmap_correlacion.png\n")

# --- Scatterplots ---
cat("Generando scatterplots...\n")

# Scatterplot 1: Votos vs PIB
p1 <- ggplot(datos, aes(x = pib_per_capita, y = votos_noboa_pct, color = region)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(title = "Votos por Noboa vs PIB per cápita",
       x = "PIB per cápita (USD)", y = "Votos por Noboa (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Scatterplot 2: Votos vs Homicidios
p2 <- ggplot(datos, aes(x = tasa_homicidios, y = votos_noboa_pct, color = region)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(title = "Votos por Noboa vs Tasa de Homicidios",
       x = "Tasa de Homicidios", y = "Votos por Noboa (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Scatterplot 3: Votos vs Población Indígena
p3 <- ggplot(datos, aes(x = pob_indigena_pct, y = votos_noboa_pct, color = region)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(title = "Votos por Noboa vs Población Indígena",
       x = "Población Indígena (%)", y = "Votos por Noboa (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Scatterplot 4: Agua vs PIB
p4 <- ggplot(datos, aes(x = pib_per_capita, y = agua_publica, color = region)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linetype = "dashed") +
  labs(title = "Acceso a Agua Pública vs PIB per cápita",
       x = "PIB per cápita (USD)", y = "Acceso a Agua Pública (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

scatterplots_combinados <- ggarrange(p1, p2, p3, p4, ncol = 2, nrow = 2,
                                     common.legend = TRUE, legend = "bottom")
scatterplots_combinados <- annotate_figure(scatterplots_combinados,
                                           top = text_grob("Análisis de Correlaciones Principales",
                                                           face = "bold", size = 16))

ggsave(file.path(RUTA_SALIDA, "scatterplots.png"), scatterplots_combinados,
       width = 14, height = 12, dpi = 300)
cat("✓ Scatterplots guardados: scatterplots.png\n")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

cat("\n======================================================================\n")
cat("ANÁLISIS COMPLETADO EXITOSAMENTE\n")
cat("======================================================================\n")
cat(paste("Fin:", Sys.time(), "\n\n"))

cat(paste("Archivos generados en '", RUTA_SALIDA, "/':\n", sep = ""))
archivos_generados <- list.files(RUTA_SALIDA)
for (archivo in archivos_generados) {
  cat(paste("  -", archivo, "\n"))
}

cat("\nResultados clave:\n")
cat(paste("  - Total de cantones analizados:", nrow(datos), "\n"))
cat(paste("  - Variables con distribución normal:", sum(resultados_normalidad$Normal == "Sí"),
          "de", nrow(resultados_normalidad), "\n"))
cat("  - Método de correlación utilizado: Spearman\n")

# Correlaciones principales
cat("\nCorrelaciones más importantes:\n")
for (i in 1:nrow(resultados_corr)) {
  if (abs(resultados_corr$rho[i]) > 0.3) {
    cat(paste("  -", resultados_corr$Relacion[i], ": ρ =",
              round(resultados_corr$rho[i], 3), "\n"))
  }
}

cat("\n======================================================================\n")
cat("FIN DEL ANÁLISIS\n")
cat("======================================================================\n")
