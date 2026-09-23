import json
from pathlib import Path

nb_path = Path("evaluation/model_comparation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Actualizar Cell 54
cell_54_source = [
    "---\n",
    "# 17. PERSISTENCIA, CONCLUSIONES Y TRABAJO FUTURO\n",
    "\n",
    "### ¿Qué contiene el artefacto `.pkl`?\n",
    "El archivo `cdv_randomforest_pipeline.pkl` es una estructura serializada con `joblib` que encapsula:\n",
    "1. El `ColumnTransformer` ya ajustado, incluyendo los valores de imputación (mediana), los parámetros de estandarización ($z$-score) y el diccionario de categorías One-Hot Encoding (44 características finales).\n",
    "2. Los 100 árboles de decisión del clasificador Random Forest optimizados para maximizar la sensibilidad clínica (Recall 99.2%).\n",
    "\n",
    "### Flujo de Integración en Arquitectura de Producción (Figure 3)\n",
    "A continuación se presenta el diseño arquitectónico desacoplado en cuatro capas que integra el motor predictivo con el sistema de soporte a la decisión clínica (CDSS) en entornos veterinarios reales:\n",
    "\n",
    "![Figure 3. Software architecture diagram and Clinical Decision Support System (CDSS) integration](../demo_reports/arquitectura_software_cdss.png)\n",
    "\n",
    "```\n",
    "  [ CAPA 1: CLIENTE ]        [ CAPA 2: API REST ]       [ CAPA 3: ML ENGINE ]       [ CAPA 4: BD ]\n",
    "  Médico Veterinario         FastAPI (Uvicorn)          CDVInferenceEngine          Neon DB\n",
    "  Formulario Web (11 var)    Validación Pydantic        Pipeline Scikit-Learn       PostgreSQL 16\n",
    "  Dashboard Analítico  ────> Endpoints /evaluar/  ────> ColumnTransformer (44 feats) Tabla pacientes_cdv\n",
    "  Modal de Riesgo            Servicios de Negocio       Random Forest (100 trees)   UUIDv4 / SSL\n",
    "```\n",
    "\n",
    "### Conclusiones Principales del Estudio:\n",
    "1. **Factibilidad del CDSS:** Es viable predecir con alta precisión el riesgo de Distemper Canino utilizando únicamente 11 variables clínicas no invasivas disponibles en la consulta inicial de triaje.\n",
    "2. **Superioridad de Random Forest:** El ensamble no lineal demostró ser el más equilibrado para la práctica clínica (Sensibilidad 99.2%, Especificidad 60.0%, ROC-AUC 0.9400), superando las limitaciones lineales de la Regresión Logística y la inestabilidad del Árbol de Decisión.\n",
    "3. **Relevancia Diagnóstica:** Fiebre, signos neurológicos, edad y estado vacunal completo se consolidan como los factores determinantes de mayor peso predictivo en Bogotá y Chía.\n",
    "\n",
    "### Limitaciones Metodológicas Transparentes:\n",
    "1. **Tamaño Muestral ($N=162$):** Constituye una cohorte preliminar; aunque se utilizó validación cruzada estratificada rigurosa, se requieren estudios prospectivos con muestras más amplias.\n",
    "2. **Alcance Geográfico y Sesgo de Selección:** Datos recolectados en Bogotá y Sabana Norte de Cundinamarca en centros de urgencia; no se debe asumir generalización automática a otras regiones con diferente endemicidad sin calibración previa.\n",
    "3. **Carácter de Apoyo:** El sistema actúa como copiloto diagnóstico y no reemplaza el criterio médico veterinario ni las pruebas de confirmación de laboratorio.\n"
]

nb["cells"][54]["source"] = cell_54_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Celda 54 actualizada con la Figura 3 y la arquitectura de software.")
