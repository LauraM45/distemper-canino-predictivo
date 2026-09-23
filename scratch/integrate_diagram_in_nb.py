import json
from pathlib import Path

nb_path = Path("evaluation/model_comparation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Actualizar Cell 20 (Markdown de Sección 5)
cell_20_source = [
    "---\n",
    "# 5. ARQUITECTURA DEL PREPROCESAMIENTO\n",
    "\n",
    "### Flujo Modular con ColumnTransformer\n",
    "El preprocesamiento implementado en `src/data/featureEngineer.py` organiza la preparación de las variables clínicas en tres ramas independientes, permitiendo aplicar a cada tipo de variable el tratamiento matemático correspondiente.\n",
    "\n",
    "![Arquitectura Modular del Preprocesamiento](../demo_reports/arquitectura_preprocesamiento_columntransformer.png)\n",
    "\n",
    "#### Flujo del procesamiento\n",
    "\n",
    "```text\n",
    "                 MATRIZ CLÍNICA X\n",
    "                    11 variables\n",
    "                         │\n",
    "          ┌──────────────┼──────────────┐\n",
    "          │              │              │\n",
    "          ▼              ▼              ▼\n",
    "       NUMÉRICA      CATEGÓRICA      BINARIA\n",
    "       1 variable    5 variables    5 variables\n",
    "          │              │              │\n",
    "       Mediana       Desconocido        0\n",
    "          │              │              │\n",
    "    StandardScaler   OneHotEncoder   Preserva\n",
    "          │          min_frequency=2    {0,1}\n",
    "          │              │              │\n",
    "          ▼              ▼              ▼\n",
    "       1 feature     38 features     5 features\n",
    "          │              │              │\n",
    "          └──────────────┼──────────────┘\n",
    "                         ▼\n",
    "                 ColumnTransformer\n",
    "                         │\n",
    "                         ▼\n",
    "                 44 FEATURES FINALES\n",
    "                         │\n",
    "                         ▼\n",
    "              MODELOS DE MACHINE LEARNING\n",
    "       Random Forest • Decision Tree • Logistic Regression\n",
    "```\n",
    "\n",
    "### Desglose de las Tres Ramas del Pipeline\n",
    "\n",
    "1. **Rama Numérica (`edad_meses`):** imputación por la mediana muestral y estandarización mediante $Z = (x - \\mu)/\\sigma$.\n",
    "\n",
    "2. **Rama Categórica (`sexo`, `raza`, `talla`, `ubicacion_procedencia`, `estado_vacunal`):** imputación con la categoría `'Desconocido'`, seguida de `OneHotEncoder` con `min_frequency=2` para agrupar categorías poco frecuentes y manejar categorías no observadas durante la inferencia.\n",
    "\n",
    "3. **Rama Binaria (5 signos clínicos):** imputación con valor $0$ y preservación de las variables binarias en el dominio $\\{0,1\\}$.\n"
]

nb["cells"][20]["source"] = cell_20_source

# Actualizar Cell 21 (Código) para incluir visualización
cell_21_source = [
    "from IPython.display import Image, display\n",
    "from src.data.featureEngineer import CDVFeaturePipelineBuilder\n",
    "\n",
    "# 1. Visualización embebida del diagrama de bloques arquitectónico\n",
    "diag_path = PROJECT_ROOT / \"demo_reports\" / \"arquitectura_preprocesamiento_columntransformer.png\"\n",
    "if diag_path.is_file():\n",
    "    display(Image(filename=str(diag_path), width=700))\n",
    "\n",
    "# 2. Cargar el pipeline operativo de Random Forest para inspeccionar su preprocesador real\n",
    "rf_pipeline = joblib.load(RF_PIPELINE_PATH)\n",
    "preprocessor = rf_pipeline.named_steps['preprocessor']\n",
    "\n",
    "# 3. Extraer nombres ordenados de las 44 características generadas\n",
    "feature_names = list(preprocessor.get_feature_names_out())\n",
    "\n",
    "print(f\"Total de características generadas: {len(feature_names)}\")\n",
    "print(f\" - Características numéricas    : {len([f for f in feature_names if f.startswith('num__')])}\")\n",
    "print(f\" - Características categóricas  : {len([f for f in feature_names if f.startswith('cat__')])}\")\n",
    "print(f\" - Características binarias     : {len([f for f in feature_names if f.startswith('bin__')])}\")\n"
]

nb["cells"][21]["source"] = cell_21_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Sección 5 de model_comparation.ipynb actualizada con el diagrama y desglose.")
