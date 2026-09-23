import json
from pathlib import Path

nb_path = Path("evaluation/model_comparation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Actualizar Cell 23 (Markdown de Sección 6)
cell_23_source = [
    "---\n",
    "# 6. PROTOCOLO DE VALIDACIÓN Y ENTRENAMIENTO\n",
    "\n",
    "### Diagrama Conceptual del Protocolo de Evaluación (Figure 2)\n",
    "Para garantizar la ausencia de sesgo de optimismo y asegurar que el rendimiento reportado sea clínicamente realista, se implementó una **validación cruzada estratificada en 5 pliegues fuera de pliegue (5-Fold Stratified OOF)**:\n",
    "\n",
    "![Figure 2. Out-of-fold stratified cross-validation evaluation protocol](../demo_reports/protocolo_validacion_cruzada_oof.png)\n",
    "\n",
    "### Principios Metodológicos Clave del Protocolo:\n",
    "1. **Estratificación Proporcional:** En cada uno de los 5 pliegues se preserva rigurosamente la prevalencia observada en la cohorte clínica real de Bogotá y Chía (75.3% casos confirmados con CDV y 24.7% controles sanos), evitando desbalances artificiales.\n",
    "2. **Aislamiento Estricto de Entrenamiento y Prueba:** En cada iteración $k \\in \\{1, 2, 3, 4, 5\\}$, el pipeline (imputación, escalado, codificación y clasificador) se ajusta únicamente sobre el 80% de los datos (~130 caninos) y predice sobre el 20% restante (~32 caninos) que **nunca participaron** en su ajuste.\n",
    "3. **Consolidación Vectorial Out-Of-Fold (OOF):** La unión de las 5 particiones genera un vector completo de **162 predicciones y probabilidades OOF**:\n",
    "   $$\\mathbf{\\hat{y}}_{\\text{OOF}} = \\bigcup_{k=1}^5 \\mathbf{\\hat{y}}_{\\text{OOF}}^{(k)} \\in \\{0, 1\\}^{162}, \\quad \\mathbf{\\hat{P}}_{\\text{OOF}} \\in [0, 1]^{162}$$\n",
    "4. **Cálculo de Métricas sin Fuga de Información:** La matriz de confusión general, las curvas ROC y Precision-Recall se construyen sobre este vector consolidado, asegurando cero fuga de datos (*zero data leakage*).\n"
]

nb["cells"][23]["source"] = cell_23_source

# Actualizar Cell 24 (Código)
cell_24_source = [
    "from IPython.display import Image, display\n",
    "\n",
    "# 1. Visualización del esquema conceptual del protocolo OOF (Figure 2)\n",
    "oof_diag = PROJECT_ROOT / \"demo_reports\" / \"protocolo_validacion_cruzada_oof.png\"\n",
    "if oof_diag.is_file():\n",
    "    display(Image(filename=str(oof_diag), width=900))\n",
    "\n",
    "# 2. Cargar el reporte oficial JSON para extraer los hiperparámetros óptimos\n",
    "with open(REPORT_JSON_PATH, 'r', encoding='utf-8') as f:\n",
    "    report_data = json.load(f)\n",
    "\n",
    "models_summary = report_data['models_summary']\n",
    "model_names = list(models_summary.keys())\n",
    "\n",
    "print(\"=== HIPERPARÁMETROS ÓPTIMOS DETERMINADOS MEDIANTE BÚSQUEDA EN GRILLA (GRIDSEARCHCV) ===\")\n",
    "for m_name in model_names:\n",
    "    print(f\"\\n[Algoritmo: {m_name}]\")\n",
    "    for param, val in models_summary[m_name]['best_params'].items():\n",
    "        clean_p = param.replace('classifier__', '')\n",
    "        print(f\"  • {clean_p:22s}: {val}\")\n"
]

nb["cells"][24]["source"] = cell_24_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Celda 23 y 24 actualizadas con la Figura 2.")
