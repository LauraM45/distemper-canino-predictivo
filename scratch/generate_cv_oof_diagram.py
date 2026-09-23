"""Generador del diagrama de protocolo de validación cruzada estratificada OOF (Figura 2).

Produce una figura de alta resolución (300 DPI) para publicación académica (IEEE/MDPI/Springer)
que ilustra la partición estratificada en 5 pliegues y la consolidación de las 162 predicciones OOF.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

PROJECT_ROOT = Path("c:/Users/lssof/OneDrive/Desktop/Universidad/Investigacion2/distemper-canino-predictivo")
OUTPUT_PATH = PROJECT_ROOT / "demo_reports" / "protocolo_validacion_cruzada_oof.png"

fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.5)
ax.axis('off')

# Paleta académica
COLOR_DATASET = "#0f172a"
COLOR_POS = "#ef4444"
COLOR_NEG = "#10b981"
COLOR_TRAIN = "#2563eb"
COLOR_VAL = "#ea580c"
COLOR_OOF = "#7c3aed"
COLOR_METRICS = "#1e293b"

def draw_box(x, y, w, h, text, color, title=None, subtitle=None, fontsize=9.5, text_color="white", alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                         facecolor=color, edgecolor="#334155", linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    
    cx = x + w / 2
    if title and subtitle:
        ax.text(cx, y + h * 0.72, title, ha='center', va='center', fontsize=fontsize + 0.5,
                fontweight='bold', color=text_color, zorder=3)
        ax.text(cx, y + h * 0.35, subtitle, ha='center', va='center', fontsize=fontsize - 1,
                color="#f1f5f9", zorder=3)
    elif title and text:
        ax.text(cx, y + h * 0.70, title, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, zorder=3)
        ax.text(cx, y + h * 0.32, text, ha='center', va='center', fontsize=fontsize - 1,
                color="#e2e8f0", zorder=3)
    else:
        ax.text(cx, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, zorder=3)

def draw_arrow(x1, y1, x2, y2, color="#475569", lw=1.8, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=13, shrinkA=2, shrinkB=2), zorder=1)

# ================= ENCABEZADO =================
ax.text(7.5, 9.15, "Figure 2. Out-of-fold stratified cross-validation evaluation protocol (5-Fold Stratified OOF)",
        ha='center', va='center', fontsize=13, fontweight='bold', color="#0f172a")
ax.text(7.5, 8.85, "Preservación rigurosa de la prevalencia clínica (75.3% CDV+ / 24.7% Sanos) y evaluación sin sesgo de optimismo (Zero Leakage)",
        ha='center', va='center', fontsize=9.5, color="#475569", fontstyle='italic')

# ================= 1. DATASET COMPLETO =================
draw_box(0.6, 7.5, 3.2, 1.0, "", COLOR_DATASET,
         title="Dataset Clínico Validado",
         subtitle="N = 162 Pacientes Reales (Bogotá y Chía)")

# Barra de estratificación de clases
ax.text(4.2, 8.15, "Estratificación de Clases:", fontsize=9, fontweight='bold', color="#1e293b")

# Barra roja de positivos (75.3%)
rect_pos = Rectangle((4.2, 7.6), 6.5, 0.45, facecolor=COLOR_POS, edgecolor="#991b1b", linewidth=1, zorder=2)
ax.add_patch(rect_pos)
ax.text(7.45, 7.825, "122 Casos Positivos CDV (75.3%)", color="white", fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=3)

# Barra verde de negativos (24.7%)
rect_neg = Rectangle((10.7, 7.6), 2.2, 0.45, facecolor=COLOR_NEG, edgecolor="#065f46", linewidth=1, zorder=2)
ax.add_patch(rect_neg)
ax.text(11.8, 7.825, "40 Sanos (24.7%)", color="white", fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=3)

draw_arrow(2.2, 7.5, 2.2, 6.9, color="#475569")
ax.text(2.6, 7.2, "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", fontsize=8.5, fontstyle='italic', color="#334155")

# ================= 2. PARTICIONAMIENTO EN 5 PLIEGUES =================
folds_info = [
    ("Pliegue 1", 0, "n=33", "(25 CDV+ / 8 Sanos)"),
    ("Pliegue 2", 1, "n=33", "(25 CDV+ / 8 Sanos)"),
    ("Pliegue 3", 2, "n=32", "(24 CDV+ / 8 Sanos)"),
    ("Pliegue 4", 3, "n=32", "(24 CDV+ / 8 Sanos)"),
    ("Pliegue 5", 4, "n=32", "(24 CDV+ / 8 Sanos)"),
]

start_y = 6.4
row_h = 0.72
gap_y = 0.16
col_w = 1.6
col_start_x = 2.4

for idx, (f_name, val_idx, n_size, ratio_text) in enumerate(folds_info):
    y = start_y - idx * (row_h + gap_y)
    
    # Etiqueta de la iteración
    ax.text(0.6, y + row_h/2, f_name, fontsize=9.5, fontweight='bold', va='center', color="#0f172a")
    ax.text(0.6, y + row_h/2 - 0.22, ratio_text, fontsize=7.5, color="#64748b", va='center')
    
    # Dibujar los 5 bloques del pliegue
    for b_idx in range(5):
        bx = col_start_x + b_idx * (col_w + 0.12)
        if b_idx == val_idx:
            # Bloque de Validación OOF
            draw_box(bx, y, col_w, row_h, "", COLOR_VAL,
                     title="Test OOF", subtitle=n_size, fontsize=8)
        else:
            # Bloque de Entrenamiento
            draw_box(bx, y, col_w, row_h, "", COLOR_TRAIN,
                     title="Train", subtitle="~26 pts", fontsize=8, alpha=0.9)
            
    # Flecha hacia vector OOF individual
    draw_arrow(col_start_x + 5 * (col_w + 0.12), y + row_h/2, 11.4, y + row_h/2, color=COLOR_VAL)
    draw_box(11.4, y, 2.8, row_h, "", "#4338ca",
             title=f"ŷ_OOF^({idx+1}) [{n_size}]",
             subtitle="Evaluado en modelo no expuesto", fontsize=8)

# Leyenda de Train / Test
draw_box(4.2, 2.15, 2.2, 0.45, "Entrenamiento (Train 80%)", COLOR_TRAIN, fontsize=8)
draw_box(6.6, 2.15, 2.2, 0.45, "Prueba OOF (Test 20%)", COLOR_VAL, fontsize=8)
draw_box(9.0, 2.15, 2.2, 0.45, "Predicciones OOF", "#4338ca", fontsize=8)

# ================= 3. CONSOLIDACIÓN OUT-OF-FOLD =================
draw_arrow(12.8, 2.7, 12.8, 2.05, color=COLOR_OOF, lw=2.5)

# Bloque final consolidado
draw_box(0.6, 0.45, 6.2, 1.4, "", COLOR_OOF,
         title="Consolidación Vectorial OOF Completa (N = 162)",
         subtitle="Ŷ_OOF = ŷ^(1) ∪ ŷ^(2) ∪ ŷ^(3) ∪ ŷ^(4) ∪ ŷ^(5)  ∈ {0, 1}^162\nP_OOF = p^(1) ∪ p^(2) ∪ p^(3) ∪ p^(4) ∪ p^(5)  ∈ [0, 1]^162",
         fontsize=9)

# Flecha a métricas finales
draw_arrow(6.8, 1.15, 7.6, 1.15, color=COLOR_DATASET, lw=2.5)

# Bloque de Métricas Oficiales
draw_box(7.6, 0.45, 6.8, 1.4, "", COLOR_METRICS,
         title="Métricas Oficiales sin Sesgo de Optimismo (Random Forest)",
         subtitle="• Recall (Sensibilidad): 99.2% (121/122 CDV+)   • Exactitud (Accuracy): 89.5%\n"
                  "• Especificidad: 60.0% (24/40 Sanos)            • ROC-AUC OOF: 0.9400\n"
                  "• Garantía Metodológica: 0% Fuga de Datos (Zero Data Leakage)",
         fontsize=8.5)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"[OK] Diagrama OOF guardado exitosamente en: {OUTPUT_PATH}")
