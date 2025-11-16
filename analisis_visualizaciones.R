# ==============================================================================
# ANÁLISIS ESTADÍSTICO COMPLETO CON VISUALIZACIONES EXTENSIVAS
# Cantones de Ecuador - Versión R
# ==============================================================================
#
# Este script genera una amplia colección de gráficos para cada fase del análisis.
#
# Requisitos:
#   install.packages(c("tidyverse", "ggpubr", "corrplot", "psych", "moments", "gridExtra"))
#
# Uso:
#   Rscript analisis_visualizaciones.R
#   O desde RStudio: source("analisis_visualizaciones.R")
#
# Autor: Análisis Estadístico Cantonal
# Fecha: 2025
# ==============================================================================

# Limpiar entorno
rm(list = ls())
options(scipen = 999, digits = 4)

# Instalar y cargar paquetes necesarios
paquetes <- c("tidyverse", "ggpubr", "corrplot", "psych", "moments", "gridExtra", "scales", "ggridges")

for (pkg in paquetes) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Instalando", pkg, "...\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Configuración
RUTA_DATOS <- "Basecantones2csv.csv"
RUTA_GRAFICOS <- "graficos_r"

# Colores personalizados
COLORES_REGION <- c("Costa" = "#FF6B6B", "Sierra" = "#4ECDC4",
                    "Oriente" = "#45B7D1", "Insular" = "#96CEB4")
COLORES_PALETA <- c("#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD")

# Tema personalizado
tema_personalizado <- theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(face = "bold"),
    legend.position = "right"
  )

theme_set(tema_personalizado)

# ==============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================

cat("======================================================================\n")
cat("GENERACIÓN EXTENSIVA DE VISUALIZACIONES EN R\n")
cat("Análisis Estadístico de Cantones de Ecuador\n")
cat("======================================================================\n")
cat(paste("Inicio:", Sys.time(), "\n\n"))

# Crear directorio
if (!dir.exists(RUTA_GRAFICOS)) {
  dir.create(RUTA_GRAFICOS)
}
cat(paste("✓ Directorio de salida:", RUTA_GRAFICOS, "/\n\n"))

# Cargar datos
cat("======================================================================\n")
cat("CARGANDO Y PREPARANDO DATOS\n")
cat("======================================================================\n")

datos <- read_delim(RUTA_DATOS, delim = ";", locale = locale(decimal_mark = ","),
                    show_col_types = FALSE)

# Renombrar columnas
nombres_nuevos <- c("canton", "provincia", "votos_noboa_abs", "votos_gonzalez_abs",
                    "votos_noboa_pct", "votos_gonzalez_pct", "poblacion",
                    "pob_indigena_pct", "agua_publica", "electricidad",
                    "pib_per_capita", "tasa_homicidios", "altitud",
                    "costa", "sierra", "oriente", "insular")
names(datos) <- nombres_nuevos

# Crear variable región
datos <- datos %>%
  mutate(region = case_when(
    costa == 1 ~ "Costa",
    sierra == 1 ~ "Sierra",
    oriente == 1 ~ "Oriente",
    insular == 1 ~ "Insular",
    TRUE ~ "Desconocido"
  )) %>%
  mutate(region = factor(region, levels = c("Costa", "Sierra", "Oriente", "Insular")))

cat(paste("✓ Datos cargados:", nrow(datos), "cantones\n\n"))

# ==============================================================================
# FASE 1: GRÁFICOS DE EXPLORACIÓN INICIAL
# ==============================================================================

cat("======================================================================\n")
cat("FASE 1: GRÁFICOS DE EXPLORACIÓN INICIAL\n")
cat("======================================================================\n")

# 1.1 Distribución por provincia
p1 <- datos %>%
  count(provincia) %>%
  arrange(desc(n)) %>%
  head(15) %>%
  ggplot(aes(x = reorder(provincia, n), y = n)) +
  geom_col(fill = COLORES_PALETA[1], color = "black") +
  geom_text(aes(label = n), hjust = -0.2, fontface = "bold") +
  coord_flip() +
  labs(title = "Distribución de Cantones por Provincia (Top 15)",
       x = "Provincia", y = "Número de Cantones") +
  theme(axis.text.y = element_text(size = 10))

ggsave(file.path(RUTA_GRAFICOS, "01_cantones_por_provincia.png"), p1,
       width = 12, height = 8, dpi = 150)
cat("✓ 01_cantones_por_provincia.png\n")

# 1.2 Distribución por región (pie + bar)
region_counts <- datos %>% count(region)

p2a <- ggplot(region_counts, aes(x = "", y = n, fill = region)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = COLORES_REGION) +
  labs(title = "Distribución por Región", fill = "Región") +
  theme_void() +
  theme(plot.title = element_text(face = "bold", size = 14, hjust = 0.5)) +
  geom_text(aes(label = paste0(region, "\n", round(n/sum(n)*100, 1), "%")),
            position = position_stack(vjust = 0.5), size = 4)

p2b <- ggplot(region_counts, aes(x = region, y = n, fill = region)) +
  geom_col(color = "black") +
  scale_fill_manual(values = COLORES_REGION) +
  geom_text(aes(label = n), vjust = -0.3, fontface = "bold") +
  labs(title = "Cantidad de Cantones por Región", x = "", y = "Número") +
  theme(legend.position = "none")

p2 <- ggarrange(p2a, p2b, ncol = 2)
ggsave(file.path(RUTA_GRAFICOS, "02_distribucion_regiones.png"), p2,
       width = 14, height = 6, dpi = 150)
cat("✓ 02_distribucion_regiones.png\n")

# 1.3 Vista general de distribuciones
vars_numericas <- c("votos_noboa_pct", "votos_gonzalez_pct", "poblacion",
                    "pob_indigena_pct", "agua_publica", "electricidad",
                    "pib_per_capita", "tasa_homicidios", "altitud")

plots_vista <- list()
for (i in seq_along(vars_numericas)) {
  var <- vars_numericas[i]
  media_val <- mean(datos[[var]], na.rm = TRUE)
  mediana_val <- median(datos[[var]], na.rm = TRUE)

  plots_vista[[i]] <- ggplot(datos, aes_string(x = var)) +
    geom_histogram(aes(y = ..density..), bins = 25, fill = COLORES_PALETA[i %% 6 + 1],
                   color = "white", alpha = 0.8) +
    geom_density(color = "darkred", size = 1) +
    geom_vline(xintercept = media_val, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = mediana_val, color = "green", linetype = "dashed", size = 1) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(plot.title = element_text(size = 11))
}

p3 <- ggarrange(plotlist = plots_vista, ncol = 3, nrow = 3)
p3 <- annotate_figure(p3, top = text_grob("Vista General de Distribuciones", face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "03_vista_general_distribuciones.png"), p3,
       width = 16, height = 14, dpi = 150)
cat("✓ 03_vista_general_distribuciones.png\n")

# ==============================================================================
# FASE 2: GRÁFICOS DE ANÁLISIS DESCRIPTIVO
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 2: GRÁFICOS DE ANÁLISIS DESCRIPTIVO\n")
cat("======================================================================\n")

vars_principales <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios",
                      "pob_indigena_pct", "agua_publica")

# 2.1 Violin plots
plots_violin <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  plots_violin[[i]] <- ggplot(datos, aes_string(x = "1", y = var)) +
    geom_violin(fill = COLORES_PALETA[i], alpha = 0.7) +
    geom_boxplot(width = 0.1, fill = "white") +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
}

p4 <- ggarrange(plotlist = plots_violin, ncol = 3, nrow = 2)
p4 <- annotate_figure(p4, top = text_grob("Violin Plots - Distribución y Densidad", face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "04_violin_plots.png"), p4, width = 16, height = 10, dpi = 150)
cat("✓ 04_violin_plots.png\n")

# 2.2 Jitter plots
plots_jitter <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  media_val <- mean(datos[[var]], na.rm = TRUE)
  mediana_val <- median(datos[[var]], na.rm = TRUE)

  plots_jitter[[i]] <- ggplot(datos, aes_string(x = "1", y = var)) +
    geom_jitter(color = COLORES_PALETA[i], alpha = 0.6, width = 0.2, size = 2) +
    geom_hline(yintercept = media_val, color = "red", linetype = "dashed", size = 1) +
    geom_hline(yintercept = mediana_val, color = "green", linetype = "dashed", size = 1) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
}

p5 <- ggarrange(plotlist = plots_jitter, ncol = 3, nrow = 2)
p5 <- annotate_figure(p5, top = text_grob("Jitter Plots - Cada Punto es un Cantón (Rojo=Media, Verde=Mediana)",
                                          face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "05_jitter_plots.png"), p5, width = 16, height = 10, dpi = 150)
cat("✓ 05_jitter_plots.png\n")

# 2.3 ECDF plots
plots_ecdf <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  mediana_val <- median(datos[[var]], na.rm = TRUE)

  plots_ecdf[[i]] <- ggplot(datos, aes_string(x = var)) +
    stat_ecdf(geom = "step", color = COLORES_PALETA[i], size = 1.2) +
    geom_vline(xintercept = mediana_val, color = "red", linetype = "dashed") +
    labs(title = gsub("_", " ", var), x = "", y = "Proporción acumulada") +
    theme(plot.title = element_text(size = 11))
}

p6 <- ggarrange(plotlist = plots_ecdf, ncol = 3, nrow = 2)
p6 <- annotate_figure(p6, top = text_grob("Funciones de Distribución Acumulada Empírica (ECDF)",
                                          face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "06_ecdf_plots.png"), p6, width = 16, height = 10, dpi = 150)
cat("✓ 06_ecdf_plots.png\n")

# 2.4 Resumen estadístico
stats_summary <- datos %>%
  select(all_of(vars_principales)) %>%
  summarise(across(everything(), list(
    media = ~mean(., na.rm = TRUE),
    sd = ~sd(., na.rm = TRUE),
    cv = ~(sd(., na.rm = TRUE) / mean(., na.rm = TRUE) * 100),
    rango = ~(max(., na.rm = TRUE) - min(., na.rm = TRUE))
  )))

# Reformatear para gráfico
stats_long <- stats_summary %>%
  pivot_longer(everything(), names_to = "variable", values_to = "valor") %>%
  separate(variable, into = c("var", "stat"), sep = "_(?=[^_]+$)") %>%
  pivot_wider(names_from = stat, values_from = valor)

# Media
p7a <- ggplot(stats_long, aes(x = reorder(var, media), y = media, fill = var)) +
  geom_col(color = "black") +
  geom_text(aes(label = round(media, 1)), vjust = -0.3) +
  scale_fill_manual(values = COLORES_PALETA) +
  labs(title = "Media por Variable", x = "", y = "Valor") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

# CV
p7b <- ggplot(stats_long, aes(x = reorder(var, cv), y = cv, fill = var)) +
  geom_col(color = "black") +
  geom_hline(yintercept = 100, color = "red", linetype = "dashed", size = 1) +
  geom_text(aes(label = round(cv, 1)), vjust = -0.3) +
  scale_fill_manual(values = COLORES_PALETA) +
  labs(title = "Coeficiente de Variación (%)", x = "", y = "CV (%)") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

p7 <- ggarrange(p7a, p7b, ncol = 2)
p7 <- annotate_figure(p7, top = text_grob("Resumen de Estadísticas Descriptivas", face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "07_resumen_estadisticas.png"), p7, width = 14, height = 8, dpi = 150)
cat("✓ 07_resumen_estadisticas.png\n")

# 2.5 Boxplots con puntos
plots_box_points <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  plots_box_points[[i]] <- ggplot(datos, aes_string(x = "1", y = var)) +
    geom_boxplot(fill = "lightgray", width = 0.3, outlier.shape = NA) +
    geom_jitter(color = COLORES_PALETA[i], alpha = 0.5, width = 0.1, size = 1.5) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
}

p8 <- ggarrange(plotlist = plots_box_points, ncol = 3, nrow = 2)
p8 <- annotate_figure(p8, top = text_grob("Boxplots con Puntos Individuales", face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "08_boxplots_puntos.png"), p8, width = 16, height = 10, dpi = 150)
cat("✓ 08_boxplots_puntos.png\n")

# ==============================================================================
# FASE 3: GRÁFICOS DE ANÁLISIS DE NORMALIDAD
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 3: GRÁFICOS DE ANÁLISIS DE NORMALIDAD\n")
cat("======================================================================\n")

# 3.1 Histogramas vs curva normal teórica
plots_hist_normal <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  datos_var <- datos[[var]]
  mu <- mean(datos_var, na.rm = TRUE)
  sigma <- sd(datos_var, na.rm = TRUE)

  plots_hist_normal[[i]] <- ggplot(datos, aes_string(x = var)) +
    geom_histogram(aes(y = ..density..), bins = 25, fill = COLORES_PALETA[i],
                   color = "white", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mu, sd = sigma),
                  color = "red", size = 1.2) +
    labs(title = paste0(gsub("_", " ", var), "\nμ=", round(mu, 1), ", σ=", round(sigma, 1)),
         x = "", y = "Densidad")
}

p9 <- ggarrange(plotlist = plots_hist_normal, ncol = 3, nrow = 2)
p9 <- annotate_figure(p9, top = text_grob("Histogramas vs Distribución Normal Teórica (línea roja)",
                                          face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "09_histogramas_vs_normal.png"), p9, width = 16, height = 10, dpi = 150)
cat("✓ 09_histogramas_vs_normal.png\n")

# 3.2 Q-Q plots
plots_qq <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]

  plots_qq[[i]] <- ggplot(datos, aes_string(sample = var)) +
    stat_qq(color = COLORES_PALETA[i], size = 2, alpha = 0.7) +
    stat_qq_line(color = "red", size = 1) +
    labs(title = gsub("_", " ", var),
         x = "Cuantiles Teóricos", y = "Cuantiles Muestrales")
}

p10 <- ggarrange(plotlist = plots_qq, ncol = 3, nrow = 2)
p10 <- annotate_figure(p10, top = text_grob("Gráficos Q-Q para Evaluación de Normalidad",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "10_qq_plots.png"), p10, width = 16, height = 10, dpi = 150)
cat("✓ 10_qq_plots.png\n")

# 3.3 Densidad empírica vs normal
plots_densidad <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  mu <- mean(datos[[var]], na.rm = TRUE)
  sigma <- sd(datos[[var]], na.rm = TRUE)

  plots_densidad[[i]] <- ggplot(datos, aes_string(x = var)) +
    geom_density(fill = COLORES_PALETA[i], alpha = 0.3, color = COLORES_PALETA[i], size = 1) +
    stat_function(fun = dnorm, args = list(mean = mu, sd = sigma),
                  color = "red", linetype = "dashed", size = 1.2) +
    labs(title = gsub("_", " ", var), x = "", y = "Densidad")
}

p11 <- ggarrange(plotlist = plots_densidad, ncol = 3, nrow = 2)
p11 <- annotate_figure(p11, top = text_grob("Densidad Empírica vs Normal Teórica (línea roja)",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "11_densidad_vs_normal.png"), p11, width = 16, height = 10, dpi = 150)
cat("✓ 11_densidad_vs_normal.png\n")

# 3.4 Asimetría y Curtosis
asimetrias <- sapply(vars_principales, function(v) skewness(datos[[v]], na.rm = TRUE))
curtosis_vals <- sapply(vars_principales, function(v) kurtosis(datos[[v]], na.rm = TRUE) - 3)

df_forma <- data.frame(
  variable = vars_principales,
  asimetria = asimetrias,
  curtosis = curtosis_vals
)

p12a <- ggplot(df_forma, aes(x = reorder(variable, asimetria), y = asimetria, fill = variable)) +
  geom_col(color = "black") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -0.5, ymax = 0.5),
            fill = "green", alpha = 0.1) +
  geom_text(aes(label = round(asimetria, 2)), vjust = -0.3) +
  scale_fill_manual(values = COLORES_PALETA) +
  labs(title = "Asimetría (Skewness)", x = "", y = "Valor") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

p12b <- ggplot(df_forma, aes(x = reorder(variable, curtosis), y = curtosis, fill = variable)) +
  geom_col(color = "black") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = -1, ymax = 1),
            fill = "green", alpha = 0.1) +
  geom_text(aes(label = round(curtosis, 2)), vjust = -0.3) +
  scale_fill_manual(values = COLORES_PALETA) +
  labs(title = "Exceso de Curtosis", x = "", y = "Valor") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

p12 <- ggarrange(p12a, p12b, ncol = 2)
p12 <- annotate_figure(p12, top = text_grob("Indicadores de Forma de la Distribución",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "12_asimetria_curtosis.png"), p12, width = 14, height = 8, dpi = 150)
cat("✓ 12_asimetria_curtosis.png\n")

# 3.5 Tabla de Shapiro-Wilk
resultados_shapiro <- data.frame(
  Variable = vars_principales,
  W = numeric(length(vars_principales)),
  p_valor = numeric(length(vars_principales)),
  Normal = character(length(vars_principales)),
  stringsAsFactors = FALSE
)

for (i in seq_along(vars_principales)) {
  test <- shapiro.test(datos[[vars_principales[i]]])
  resultados_shapiro$W[i] <- test$statistic
  resultados_shapiro$p_valor[i] <- test$p.value
  resultados_shapiro$Normal[i] <- ifelse(test$p.value > 0.05, "Sí", "No")
}

# Visualizar tabla
p13 <- ggtexttable(resultados_shapiro, rows = NULL,
                   theme = ttheme("mBlue", base_size = 12)) %>%
  tab_add_title(text = "Test de Shapiro-Wilk (α = 0.05)", face = "bold", size = 16)

ggsave(file.path(RUTA_GRAFICOS, "13_tabla_shapiro.png"), p13, width = 10, height = 6, dpi = 150)
cat("✓ 13_tabla_shapiro.png\n")

# ==============================================================================
# FASE 4: GRÁFICOS DE CORRELACIÓN
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 4: GRÁFICOS DE ANÁLISIS DE CORRELACIÓN\n")
cat("======================================================================\n")

vars_corr <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios",
               "pob_indigena_pct", "agua_publica", "electricidad")

# 4.1 Matrices de correlación
corr_spearman <- cor(datos[vars_corr], method = "spearman", use = "complete.obs")
corr_pearson <- cor(datos[vars_corr], method = "pearson", use = "complete.obs")

png(file.path(RUTA_GRAFICOS, "14_matriz_correlacion_spearman.png"),
    width = 10, height = 8, units = "in", res = 150)
corrplot(corr_spearman, method = "color", type = "lower", addCoef.col = "black",
         number.cex = 0.8, tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#BB4444", "white", "#4477AA"))(200),
         title = "\nMatriz de Correlación de Spearman", mar = c(0, 0, 2, 0))
dev.off()
cat("✓ 14_matriz_correlacion_spearman.png\n")

png(file.path(RUTA_GRAFICOS, "15_matriz_correlacion_pearson.png"),
    width = 10, height = 8, units = "in", res = 150)
corrplot(corr_pearson, method = "color", type = "lower", addCoef.col = "black",
         number.cex = 0.8, tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#BB4444", "white", "#4477AA"))(200),
         title = "\nMatriz de Correlación de Pearson", mar = c(0, 0, 2, 0))
dev.off()
cat("✓ 15_matriz_correlacion_pearson.png\n")

# 4.2 Scatterplots con regresión
pares <- list(
  c("votos_noboa_pct", "pib_per_capita"),
  c("votos_noboa_pct", "tasa_homicidios"),
  c("votos_noboa_pct", "pob_indigena_pct"),
  c("agua_publica", "pib_per_capita"),
  c("tasa_homicidios", "poblacion"),
  c("electricidad", "agua_publica")
)

plots_scatter <- list()
for (i in seq_along(pares)) {
  x_var <- pares[[i]][1]
  y_var <- pares[[i]][2]

  test_corr <- cor.test(datos[[x_var]], datos[[y_var]], method = "spearman")
  rho <- test_corr$estimate
  p_val <- test_corr$p.value

  plots_scatter[[i]] <- ggplot(datos, aes_string(x = x_var, y = y_var)) +
    geom_point(color = COLORES_PALETA[i], alpha = 0.6, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "red", fill = "pink", alpha = 0.3) +
    annotate("text", x = Inf, y = Inf,
             label = paste0("ρ = ", round(rho, 3), "\np = ", format(p_val, scientific = TRUE)),
             hjust = 1.1, vjust = 1.5, size = 4,
             fontface = "bold", color = "black") +
    labs(title = paste(gsub("_", " ", y_var), "vs", gsub("_", " ", x_var)),
         x = gsub("_", " ", x_var), y = gsub("_", " ", y_var))
}

p16 <- ggarrange(plotlist = plots_scatter, ncol = 3, nrow = 2)
p16 <- annotate_figure(p16, top = text_grob("Scatterplots con Regresión Lineal (IC 95%)",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "16_scatterplots_regresion.png"), p16, width = 18, height = 12, dpi = 150)
cat("✓ 16_scatterplots_regresion.png\n")

# 4.3 Pairplot
vars_pair <- c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios", "pob_indigena_pct")

p17 <- ggpairs(datos[c(vars_pair, "region")],
               columns = 1:4, aes(color = region, alpha = 0.6),
               upper = list(continuous = wrap("cor", method = "spearman", size = 3)),
               lower = list(continuous = wrap("points", size = 1)),
               diag = list(continuous = wrap("densityDiag", alpha = 0.5))) +
  scale_color_manual(values = COLORES_REGION) +
  scale_fill_manual(values = COLORES_REGION) +
  theme_minimal()

ggsave(file.path(RUTA_GRAFICOS, "17_pairplot.png"), p17, width = 14, height = 14, dpi = 150)
cat("✓ 17_pairplot.png\n")

# 4.4 Fuerza de correlaciones
correlaciones <- data.frame()
for (i in 1:(length(vars_corr)-1)) {
  for (j in (i+1):length(vars_corr)) {
    test <- cor.test(datos[[vars_corr[i]]], datos[[vars_corr[j]]], method = "spearman")
    correlaciones <- rbind(correlaciones, data.frame(
      par = paste(substr(vars_corr[i], 1, 12), "\nvs\n", substr(vars_corr[j], 1, 12)),
      rho = test$estimate,
      abs_rho = abs(test$estimate)
    ))
  }
}

correlaciones <- correlaciones %>% arrange(abs_rho)
correlaciones$color <- ifelse(abs(correlaciones$rho) > 0.5, "Fuerte",
                              ifelse(abs(correlaciones$rho) > 0.3, "Moderada", "Débil"))

p18 <- ggplot(correlaciones, aes(x = reorder(par, abs_rho), y = rho, fill = color)) +
  geom_col(color = "black") +
  geom_hline(yintercept = 0, color = "black", size = 1) +
  geom_hline(yintercept = c(-0.5, 0.5), color = "green", linetype = "dashed") +
  geom_hline(yintercept = c(-0.3, 0.3), color = "orange", linetype = "dashed") +
  coord_flip() +
  scale_fill_manual(values = c("Fuerte" = "green", "Moderada" = "orange", "Débil" = "gray")) +
  labs(title = "Fuerza de las Correlaciones Entre Variables",
       x = "", y = "Coeficiente de Correlación de Spearman (ρ)", fill = "Fuerza") +
  theme(axis.text.y = element_text(size = 9))

ggsave(file.path(RUTA_GRAFICOS, "18_fuerza_correlaciones.png"), p18, width = 12, height = 10, dpi = 150)
cat("✓ 18_fuerza_correlaciones.png\n")

# ==============================================================================
# FASE 5: GRÁFICOS POR REGIÓN
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 5: GRÁFICOS POR REGIÓN GEOGRÁFICA\n")
cat("======================================================================\n")

# 5.1 Boxplots por región
plots_box_region <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  plots_box_region[[i]] <- ggplot(datos, aes_string(x = "region", y = var, fill = "region")) +
    geom_boxplot(outlier.color = "red", outlier.size = 2) +
    scale_fill_manual(values = COLORES_REGION) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
}

p19 <- ggarrange(plotlist = plots_box_region, ncol = 3, nrow = 2)
p19 <- annotate_figure(p19, top = text_grob("Distribución de Variables por Región",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "19_boxplots_region.png"), p19, width = 16, height = 10, dpi = 150)
cat("✓ 19_boxplots_region.png\n")

# 5.2 Violin plots por región
plots_violin_region <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]
  plots_violin_region[[i]] <- ggplot(datos, aes_string(x = "region", y = var, fill = "region")) +
    geom_violin(alpha = 0.7) +
    geom_boxplot(width = 0.1, fill = "white") +
    scale_fill_manual(values = COLORES_REGION) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
}

p20 <- ggarrange(plotlist = plots_violin_region, ncol = 3, nrow = 2)
p20 <- annotate_figure(p20, top = text_grob("Violin Plots por Región", face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "20_violin_region.png"), p20, width = 16, height = 10, dpi = 150)
cat("✓ 20_violin_region.png\n")

# 5.3 Medias por región con barras de error
plots_medias <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]

  stats_var <- datos %>%
    group_by(region) %>%
    summarise(media = mean(.data[[var]], na.rm = TRUE),
              sd = sd(.data[[var]], na.rm = TRUE)) %>%
    ungroup()

  plots_medias[[i]] <- ggplot(stats_var, aes(x = region, y = media, fill = region)) +
    geom_col(color = "black") +
    geom_errorbar(aes(ymin = media - sd, ymax = media + sd), width = 0.3) +
    geom_text(aes(label = round(media, 1)), vjust = -1, size = 3) +
    scale_fill_manual(values = COLORES_REGION) +
    labs(title = gsub("_", " ", var), x = "", y = "Media ± SD") +
    theme(legend.position = "none")
}

p21 <- ggarrange(plotlist = plots_medias, ncol = 3, nrow = 2)
p21 <- annotate_figure(p21, top = text_grob("Media y Desviación Estándar por Región",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "21_medias_region.png"), p21, width = 16, height = 10, dpi = 150)
cat("✓ 21_medias_region.png\n")

# 5.4 Facet grids
for (var in c("votos_noboa_pct", "pib_per_capita", "tasa_homicidios")) {
  p_facet <- ggplot(datos, aes_string(x = var, fill = "region")) +
    geom_histogram(aes(y = ..density..), bins = 20, alpha = 0.7, color = "white") +
    geom_density(alpha = 0.3) +
    facet_wrap(~region, scales = "free_y") +
    scale_fill_manual(values = COLORES_REGION) +
    labs(title = paste("Distribución de", gsub("_", " ", var), "por Región"),
         x = gsub("_", " ", var), y = "Densidad") +
    theme(legend.position = "none")

  ggsave(file.path(RUTA_GRAFICOS, paste0("22_facet_", var, ".png")), p_facet,
         width = 12, height = 10, dpi = 150)
}
cat("✓ 22_facet_votos_noboa_pct.png\n")
cat("✓ 22_facet_pib_per_capita.png\n")
cat("✓ 22_facet_tasa_homicidios.png\n")

# 5.5 Strip plots por región
plots_strip <- list()
for (i in seq_along(vars_principales)) {
  var <- vars_principales[i]

  medias <- datos %>% group_by(region) %>% summarise(media = mean(.data[[var]], na.rm = TRUE))

  plots_strip[[i]] <- ggplot(datos, aes_string(x = "region", y = var, color = "region")) +
    geom_jitter(alpha = 0.6, width = 0.2, size = 2) +
    geom_segment(data = medias, aes(x = as.numeric(region) - 0.3, xend = as.numeric(region) + 0.3,
                                    y = media, yend = media), color = "red", size = 1.5) +
    scale_color_manual(values = COLORES_REGION) +
    labs(title = gsub("_", " ", var), x = "", y = "") +
    theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
}

p23 <- ggarrange(plotlist = plots_strip, ncol = 3, nrow = 2)
p23 <- annotate_figure(p23, top = text_grob("Strip Plots por Región (Línea roja = Media)",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "23_strip_region.png"), p23, width = 16, height = 10, dpi = 150)
cat("✓ 23_strip_region.png\n")

# ==============================================================================
# FASE 6: GRÁFICOS DE OUTLIERS
# ==============================================================================

cat("\n======================================================================\n")
cat("FASE 6: GRÁFICOS DE IDENTIFICACIÓN DE OUTLIERS\n")
cat("======================================================================\n")

vars_outliers <- c("pib_per_capita", "tasa_homicidios", "pob_indigena_pct", "poblacion")

# 6.1 Top 10 por variable
plots_top10 <- list()
for (i in seq_along(vars_outliers)) {
  var <- vars_outliers[i]

  top10 <- datos %>%
    arrange(desc(.data[[var]])) %>%
    head(10) %>%
    mutate(label = paste0(canton, " (", region, ")"))

  plots_top10[[i]] <- ggplot(top10, aes_string(x = paste0("reorder(label, ", var, ")"),
                                               y = var, fill = "region")) +
    geom_col(color = "black") +
    geom_text(aes_string(label = paste0("round(", var, ", 0)")), hjust = -0.1, size = 3) +
    scale_fill_manual(values = COLORES_REGION) +
    coord_flip() +
    labs(title = paste("Top 10:", gsub("_", " ", var)), x = "", y = "") +
    theme(legend.position = "none", axis.text.y = element_text(size = 8))
}

p24 <- ggarrange(plotlist = plots_top10, ncol = 2, nrow = 2)
p24 <- annotate_figure(p24, top = text_grob("Top 10 Cantones por Variable (Posibles Outliers)",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "24_top10_outliers.png"), p24, width = 16, height = 14, dpi = 150)
cat("✓ 24_top10_outliers.png\n")

# 6.2 Boxplots con outliers anotados
plots_box_outliers <- list()
for (i in seq_along(vars_outliers)) {
  var <- vars_outliers[i]

  # Identificar outliers
  Q1 <- quantile(datos[[var]], 0.25, na.rm = TRUE)
  Q3 <- quantile(datos[[var]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  outliers <- datos %>% filter(.data[[var]] > Q3 + 1.5 * IQR_val)

  plots_box_outliers[[i]] <- ggplot(datos, aes_string(x = "1", y = var)) +
    geom_boxplot(fill = COLORES_PALETA[i], alpha = 0.7, outlier.color = "red", outlier.size = 3) +
    labs(title = paste0(gsub("_", " ", var), "\n(", nrow(outliers), " outliers)"),
         x = "", y = "") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    annotate("text", x = 0.5, y = max(datos[[var]], na.rm = TRUE),
             label = paste("Límite:", round(Q3 + 1.5 * IQR_val, 0)),
             size = 3, hjust = 0)
}

p25 <- ggarrange(plotlist = plots_box_outliers, ncol = 2, nrow = 2)
p25 <- annotate_figure(p25, top = text_grob("Boxplots con Identificación de Outliers",
                                            face = "bold", size = 16))
ggsave(file.path(RUTA_GRAFICOS, "25_boxplots_outliers.png"), p25, width = 12, height = 12, dpi = 150)
cat("✓ 25_boxplots_outliers.png\n")

# 6.3 Scatter multivariado
p26 <- ggplot(datos, aes(x = pib_per_capita, y = tasa_homicidios,
                        size = poblacion, color = pob_indigena_pct)) +
  geom_point(alpha = 0.7) +
  scale_size_continuous(range = c(2, 15), name = "Población") +
  scale_color_viridis_c(name = "% Pob. Indígena") +
  labs(title = "Scatter Multivariado",
       subtitle = "Tamaño = Población, Color = % Población Indígena",
       x = "PIB per cápita (USD)", y = "Tasa de Homicidios") +
  theme(legend.position = "right")

# Anotar algunos outliers
outliers_plot <- datos %>%
  arrange(desc(pib_per_capita)) %>%
  head(3)

p26 <- p26 + geom_text(data = outliers_plot,
                       aes(label = canton), hjust = -0.1, size = 3, color = "black")

ggsave(file.path(RUTA_GRAFICOS, "26_scatter_multivariado.png"), p26, width = 12, height = 10, dpi = 150)
cat("✓ 26_scatter_multivariado.png\n")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

cat("\n======================================================================\n")
cat("GENERACIÓN DE GRÁFICOS COMPLETADA\n")
cat("======================================================================\n")

archivos <- list.files(RUTA_GRAFICOS, pattern = "\\.png$")
cat(paste("\nTotal de gráficos generados:", length(archivos), "\n"))
cat(paste("\nArchivos en '", RUTA_GRAFICOS, "/':\n", sep = ""))

for (archivo in sort(archivos)) {
  cat(paste("  -", archivo, "\n"))
}

cat(paste("\nFin:", Sys.time(), "\n"))
cat("======================================================================\n")
