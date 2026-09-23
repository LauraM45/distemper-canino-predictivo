"""Generador de la Figura 5 oficial para el artículo de investigación / tesis:

Figura 5. Curvas ROC y Precision-Recall reconstruidas fuera de pliegue (N=162).
Figure 5. Reconstructed Out-Of-Fold ROC and Precision-Recall curves (N=162).

Genera una imagen compuesta de ultra-alta resolución (300 DPI) con diseño editorial,
curvas continuas exactas OOF, áreas sombreadas y puntos de operación clínica.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import sys
PROJECT_ROOT = Path("c:/Users/lssof/OneDrive/Desktop/Universidad/Investigacion2/distemper-canino-predictivo")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_moquillo_real_v7.csv"
JSON_PATH = PROJECT_ROOT / "demo_reports" / "reporte_metricas_cdv.json"
OUT_IMG_PATH = PROJECT_ROOT / "demo_reports" / "figura5_curvas_roc_precision_recall_oof.png"
OUT_IMG_ALT = PROJECT_ROOT / "demo_reports" / "curvas_roc_precision_recall_oof.png"

# Cargar preprocesador y validador
from src.data.dataValidation import CDVDataValidator
from src.data.featureEngineer import CDVFeaturePipelineBuilder

df_raw = pd.read_csv(DATA_PATH)
validator = CDVDataValidator()
df, _ = validator.validate_and_clean(df_raw)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    report_data = json.load(f)
models_summary = report_data["models_summary"]

X_all = df[list(CDVDataValidator.FEATURE_COLUMNS)]
y_all = df[CDVDataValidator.TARGET_COLUMN].astype(int)

# Protocolo 5-Fold Stratified OOF idéntico al entrenamiento
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_meta = [
    {
        "key": "RandomForest",
        "name_es": "Random Forest (Modelo Óptimo)",
        "name_en": "Random Forest (Ensemble)",
        "clf_cls": RandomForestClassifier,
        "color": "#16a34a",
        "linestyle": "-",
        "lw": 2.6,
        "is_best": True
    },
    {
        "key": "LogisticRegression",
        "name_es": "Regresión Logística (L2)",
        "name_en": "Logistic Regression (Linear)",
        "clf_cls": LogisticRegression,
        "color": "#2563eb",
        "linestyle": "--",
        "lw": 2.1,
        "is_best": False
    },
    {
        "key": "DecisionTree",
        "name_es": "Árbol de Decisión (CART)",
        "name_en": "Decision Tree (CART)",
        "clf_cls": DecisionTreeClassifier,
        "color": "#d97706",
        "linestyle": "-.",
        "lw": 1.9,
        "is_best": False
    }
]

# Calcular probabilidades OOF continuas
oof_curves = {}
for m in models_meta:
    raw_p = models_summary[m["key"]]["best_params"]
    clean_p = {k.replace("classifier__", ""): v for k, v in raw_p.items()}
    clean_p["random_state"] = 42
    
    y_prob = np.zeros(len(y_all))
    for tr_idx, te_idx in skf.split(X_all, y_all):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        X_te = X_all.iloc[te_idx]
        pipe = CDVFeaturePipelineBuilder().build_pipeline(m["clf_cls"](**clean_p))
        pipe.fit(X_tr, y_tr)
        y_prob[te_idx] = pipe.predict_proba(X_te)[:, 1]
        
    fpr, tpr, roc_thresh = roc_curve(y_all, y_prob)
    roc_auc_val = auc(fpr, tpr)
    prec, rec, pr_thresh = precision_recall_curve(y_all, y_prob)
    ap_val = average_precision_score(y_all, y_prob)
    
    oof_curves[m["key"]] = {
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc_val,
        "prec": prec,
        "rec": rec,
        "ap": ap_val
    }

# ==============================================================
# CONFIGURACIÓN DEL LIENZO FIGURA 5
# ==============================================================
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16.8, 7.6), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Encabezado institucional bilingüe con amplio margen
fig.text(0.5, 0.970, "Figura 5 / Figure 5. Curvas ROC y Precision-Recall Reconstruidas Fuera de Pliegue (5-Fold Stratified OOF)",
         ha='center', va='top', fontsize=15.0, fontweight='bold', color='#0f172a', fontfamily='sans-serif')
fig.text(0.5, 0.935, "Evaluación continua de discriminación diagnóstica sin sesgo en la cohorte clínica de Bogotá D.C. y Chía (N = 165 [162 caninos efectivos]: 122 CDV+ / 43 Controles)",
         ha='center', va='top', fontsize=10.5, style='italic', color='#475569', fontfamily='sans-serif')

# --------------------------------------------------------------
# PANEL (A): CURVA ROC
# --------------------------------------------------------------
ax_roc.set_facecolor('#fafbfc')
for spine in ax_roc.spines.values():
    spine.set_color('#cbd5e1')
    spine.set_linewidth(1.2)

# Clasificador aleatorio
ax_roc.plot([0, 1], [0, 1], linestyle=':', color='#94a3b8', lw=1.5, label='Clasificador Azar / Random Chance (AUC = 0.5000)')

for m in models_meta:
    c_data = oof_curves[m["key"]]
    label_txt = f"{m['name_es']} — AUC = {c_data['roc_auc']:.4f}"
    ax_roc.plot(c_data["fpr"], c_data["tpr"], label=label_txt,
                color=m["color"], linestyle=m["linestyle"], lw=m["lw"], zorder=4)
    if m["is_best"]:
        ax_roc.fill_between(c_data["fpr"], c_data["tpr"], alpha=0.12, color=m["color"], zorder=2)

# Punto de operación clínica de Random Forest (threshold = 0.5)
# En RF: TN=27, FP=16 -> FPR = 16/43 = 0.3721; FN=6, TP=116 -> TPR = 116/122 = 0.9508
rf_fpr = 16.0 / 43.0
rf_tpr = 116.0 / 122.0
ax_roc.scatter([rf_fpr], [rf_tpr], color='#dc2626', s=70, zorder=6, edgecolor='#ffffff', lw=1.5)
ax_roc.annotate(
    f"Punto Clínico RF (p = 0.50)\nTPR (Recall) = {rf_tpr*100:.1f}%\nFPR (1-Spec) = {rf_fpr*100:.1f}%",
    xy=(rf_fpr, rf_tpr), xytext=(rf_fpr + 0.12, rf_tpr - 0.14),
    arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', shrink=0.08, width=1.2, headwidth=6),
    fontsize=8.8, fontweight='bold', color='#991b1b',
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#f87171", lw=1.0),
    zorder=7
)

ax_roc.set_title("(a) Curvas Receiver Operating Characteristic (ROC)", fontsize=12.5, fontweight='bold', color='#0f172a', pad=12)
ax_roc.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad) / False Positive Rate", fontsize=10.2, fontweight='bold', color='#1e293b', labelpad=8)
ax_roc.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontsize=10.2, fontweight='bold', color='#1e293b', labelpad=8)
ax_roc.set_xlim([-0.02, 1.02])
ax_roc.set_ylim([-0.02, 1.03])
ax_roc.grid(True, linestyle='--', alpha=0.55, color='#cbd5e1')
ax_roc.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.8)

# --------------------------------------------------------------
# PANEL (B): CURVA PRECISION-RECALL
# --------------------------------------------------------------
ax_pr.set_facecolor('#fafbfc')
for spine in ax_pr.spines.values():
    spine.set_color('#cbd5e1')
    spine.set_linewidth(1.2)

# Prevalencia basal
baseline_prev = y_all.mean()
ax_pr.axhline(baseline_prev, linestyle=':', color='#94a3b8', lw=1.5,
              label=f'Prevalencia Basal / Baseline ({baseline_prev*100:.1f}%)')

for m in models_meta:
    c_data = oof_curves[m["key"]]
    label_txt = f"{m['name_es']} — AP = {c_data['ap']:.4f}"
    ax_pr.plot(c_data["rec"], c_data["prec"], label=label_txt,
               color=m["color"], linestyle=m["linestyle"], lw=m["lw"], zorder=4)
    if m["is_best"]:
        ax_pr.fill_between(c_data["rec"], c_data["prec"], baseline_prev,
                           where=(c_data["prec"] >= baseline_prev),
                           alpha=0.12, color=m["color"], zorder=2)

# Punto de operación clínica de Random Forest en PR
# Precisión = 116 / (116 + 16) = 0.8788; Recall = 116 / 122 = 0.9508
rf_prec = 116.0 / (116.0 + 16.0)
rf_rec = 116.0 / 122.0
ax_pr.scatter([rf_rec], [rf_prec], color='#dc2626', s=70, zorder=6, edgecolor='#ffffff', lw=1.5)
ax_pr.annotate(
    f"Punto Clínico RF (p = 0.50)\nRecall = {rf_rec*100:.1f}%\nPrecisión = {rf_prec*100:.1f}%",
    xy=(rf_rec, rf_prec), xytext=(rf_rec - 0.42, rf_prec - 0.16),
    arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', shrink=0.08, width=1.2, headwidth=6),
    fontsize=8.8, fontweight='bold', color='#991b1b',
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#f87171", lw=1.0),
    zorder=7
)

ax_pr.set_title("(b) Curvas Precision-Recall (PR)", fontsize=12.5, fontweight='bold', color='#0f172a', pad=12)
ax_pr.set_xlabel("Sensibilidad Clínica (Exhaustividad / Recall)", fontsize=10.2, fontweight='bold', color='#1e293b', labelpad=8)
ax_pr.set_ylabel("Precisión (Valor Predictivo Positivo / PPV)", fontsize=10.2, fontweight='bold', color='#1e293b', labelpad=8)
ax_pr.set_xlim([-0.02, 1.02])
ax_pr.set_ylim([0.48, 1.03])
ax_pr.grid(True, linestyle='--', alpha=0.55, color='#cbd5e1')
ax_pr.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=8.8)

# --------------------------------------------------------------
# BANNER INFORMATIVO AL PIE
# --------------------------------------------------------------
banner_text = (
    "HALLAZGOS CLÍNICOS: Random Forest demuestra excelente discriminación global con un ROC-AUC de 0.9356 y un Average Precision (AP) de 0.9749.\n"
    "En el umbral operativo estándar (p = 0.50), garantiza una sensibilidad del 95.08% con un valor predictivo positivo del 87.88%, superando al modelo lineal y al árbol simple."
)

fig.text(0.5, 0.038, banner_text, ha='center', va='center',
         fontsize=9.0, fontfamily='sans-serif', fontweight='semibold', color='#1e293b',
         bbox=dict(boxstyle="round,pad=0.45", facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.3))

plt.subplots_adjust(top=0.83, bottom=0.17, left=0.065, right=0.975, wspace=0.22)

# Guardar figura en 300 DPI
plt.savefig(OUT_IMG_PATH, dpi=300, bbox_inches='tight')
plt.savefig(OUT_IMG_ALT, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Figura 5 generada exitosamente en:\n     1. {OUT_IMG_PATH}\n     2. {OUT_IMG_ALT}")
