"""Generador perfeccionado de la Figura 4 oficial para el artículo de investigación / tesis:

Figura 4. Matrices de confusión oficiales Out-Of-Fold para los tres clasificadores evaluados.
Figure 4. Official Out-Of-Fold confusion matrices for the three evaluated classifiers.

Genera una imagen compuesta de ultra-alta resolución (300 DPI) con espaciado perfecto,
sin solapamientos de texto y formato editorial científico internacional.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Rutas del proyecto
PROJECT_ROOT = Path("c:/Users/lssof/OneDrive/Desktop/Universidad/Investigacion2/distemper-canino-predictivo")
JSON_PATH = PROJECT_ROOT / "demo_reports" / "reporte_metricas_cdv.json"
OUT_IMG_PATH = PROJECT_ROOT / "demo_reports" / "figura4_matrices_confusion_oof.png"
OUT_IMG_ALT = PROJECT_ROOT / "demo_reports" / "matrices_confusion_oof_comparativas.png"

# Cargar métricas oficiales del JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    report_data = json.load(f)

models_summary = report_data["models_summary"]

# Configuración de los 3 modelos a graficar
models_config = [
    {
        "key": "RandomForest",
        "title_es": "(a) Random Forest Classifier",
        "title_en": "Selected Ensemble Model (100 Trees)",
        "badge": "★ MODELO ÓPTIMO SELECCIONADO (RECALL 95.08%)",
        "badge_color": "#1b7837",
        "border_color": "#16a34a",
        "is_best": True
    },
    {
        "key": "LogisticRegression",
        "title_es": "(b) Regresión Logística",
        "title_en": "Regularized Linear Baseline (L2)",
        "badge": "MODELO LINEAL L2 (RECALL 93.44%)",
        "badge_color": "#1d4ed8",
        "border_color": "#3b82f6",
        "is_best": False
    },
    {
        "key": "DecisionTree",
        "title_es": "(c) Árbol de Decisión (CART)",
        "title_en": "Pruned Decision Tree (max_depth=5)",
        "badge": "ÁRBOL BASE SIMPLE (RECALL 90.16%)",
        "badge_color": "#b45309",
        "border_color": "#f59e0b",
        "is_best": False
    }
]

# Configuración del lienzo ampliado verticalmente
fig, axes = plt.subplots(1, 3, figsize=(18.2, 8.4), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Encabezado editorial superior con amplio margen
fig.text(0.5, 0.975, "Figura 4 / Figure 4. Matrices de Confusión Oficiales Out-Of-Fold (5-Fold Stratified OOF)",
         ha='center', va='top', fontsize=15.5, fontweight='bold', color='#0f172a', fontfamily='sans-serif')
fig.text(0.5, 0.942, "Evaluación diagnóstica sin fuga de datos (Data Leakage) en la cohorte clínica de Bogotá D.C. y Chía (N = 165 caninos: 122 CDV+ / 43 Controles)",
         ha='center', va='top', fontsize=11, style='italic', color='#475569', fontfamily='sans-serif')

for ax, cfg in zip(axes, models_config):
    m_data = models_summary[cfg["key"]]
    cm = np.array(m_data["evaluation"]["confusion_matrix"])
    agg = m_data["evaluation"]["aggregate_metrics"]
    
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total_neg = tn + fp
    total_pos = fn + tp
    
    # Marco y fondo del gráfico
    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        if cfg["is_best"]:
            spine.set_color(cfg["border_color"])
            spine.set_linewidth(2.6)
        else:
            spine.set_color('#cbd5e1')
            spine.set_linewidth(1.2)
            
    # Coordenadas de celdas
    cell_info = [
        # (row, col, value, total_row, tag_es, tag_en, bg_color, text_color, is_alert, is_success)
        (0, 0, tn, total_neg, "Verdadero Negativo", "TN", "#f0f9ff", "#0284c7", False, False),
        (0, 1, fp, total_neg, "Falso Positivo", "FP (Sospecha)", "#fffbeb", "#d97706", False, False),
        (1, 0, fn, total_pos, "Falso Negativo", "FN (¡Omitido!)", "#fef2f2", "#dc2626", True, False),
        (1, 1, tp, total_pos, "Verdadero Positivo", "TP (Detectado)", "#f0fdf4", "#16a34a", False, True),
    ]
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    
    for r, c, val, row_tot, tag_es, tag_en, bg_col, txt_col, is_alert, is_success in cell_info:
        pct = (val / row_tot) * 100.0
        
        # Rectángulo de cuadrante
        edge_col = '#ef4444' if is_alert else ('#22c55e' if is_success else '#94a3b8')
        line_w = 2.2 if (is_alert or is_success) else 1.0
        
        rect = patches.FancyBboxPatch(
            (c - 0.44, r - 0.44), 0.88, 0.88,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=line_w,
            edgecolor=edge_col,
            facecolor=bg_col,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Valor numérico principal
        ax.text(c, r - 0.12, f"{val}", ha='center', va='center',
                fontsize=26, fontweight='bold', color=txt_col, zorder=3)
        
        # Porcentaje de la fila real
        ax.text(c, r + 0.13, f"{pct:.1f}% de clase", ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=txt_col, zorder=3)
        
        # Etiqueta clínica descriptiva
        ax.text(c, r + 0.29, f"[{tag_en}: {tag_es}]", ha='center', va='center',
                fontsize=8.4, style='italic', color=txt_col, alpha=0.92, zorder=3)
        
    # Ejes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sano / Control\n(Clase 0)", "Positivo CDV\n(Clase 1)"], fontsize=9.8, fontweight='bold', color='#1e293b')
    ax.set_yticklabels(["Sano / Control\n(Clase 0)", "Positivo CDV\n(Clase 1)"], fontsize=9.8, fontweight='bold', color='#1e293b', va='center')
    
    ax.set_xlabel("Diagnóstico Predicho por el Modelo (Predicted)", fontsize=10.5, fontweight='bold', labelpad=10, color='#0f172a')
    ax.set_ylabel("Diagnóstico Clínico Real (Ground Truth)", fontsize=10.5, fontweight='bold', labelpad=10, color='#0f172a')
    
    # Títulos del panel (muy separados del encabezado global)
    ax.text(0.5, 1.22, cfg["title_es"], ha='center', va='bottom', transform=ax.transAxes,
            fontsize=12.5, fontweight='bold', color='#0f172a')
    ax.text(0.5, 1.15, cfg["title_en"], ha='center', va='bottom', transform=ax.transAxes,
            fontsize=9.5, style='italic', color='#64748b')
    
    # Badge de estatus
    badge_rect = patches.FancyBboxPatch(
        (0.04, 1.05), 0.92, 0.065, transform=ax.transAxes,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=0, facecolor=cfg["badge_color"], zorder=4
    )
    ax.add_patch(badge_rect)
    ax.text(0.5, 1.082, cfg["badge"], ha='center', va='center', transform=ax.transAxes,
            fontsize=8.0, fontweight='bold', color='#ffffff', zorder=5)
    
    # Card de Métricas en el pie (con suficiente separación de las etiquetas del eje X)
    card_text = (
        f"  Sensibilidad (Recall) : {agg['recall']*100:.2f}%\n"
        f"  Especificidad (Spec)  : {agg['specificity']*100:.2f}%\n"
        f"  Exactitud (Accuracy)  : {agg['accuracy']*100:.2f}%\n"
        f"  F1-Score              : {agg['f1']*100:.2f}%\n"
        f"  Falsos Negativos (FN) : {fn} pacientes omitidos"
    )
    
    info_box = patches.FancyBboxPatch(
        (-0.44, 1.76), 1.88, 0.48,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.4, edgecolor=cfg["border_color"] if cfg['is_best'] else '#cbd5e1',
        facecolor='#f0fdf4' if cfg['is_best'] else '#f8fafc',
        zorder=2, clip_on=False
    )
    ax.add_patch(info_box)
    
    ax.text(0.5, 2.00, card_text, ha='center', va='center',
            fontsize=8.8, fontfamily='monospace', fontweight='bold',
            color='#14532d' if cfg['is_best'] else '#1e293b',
            linespacing=1.35, zorder=3)

# Ajuste de submárgenes
plt.subplots_adjust(top=0.76, bottom=0.26, left=0.065, right=0.975, wspace=0.33)

# Guardar figura en 300 DPI
plt.savefig(OUT_IMG_PATH, dpi=300, bbox_inches='tight')
plt.savefig(OUT_IMG_ALT, dpi=300, bbox_inches='tight')
plt.close()

print(f"[ÉXITO] Figura 4 regenerada con diseño limpio y sin solapamiento en:\n  -> {OUT_IMG_PATH}")
