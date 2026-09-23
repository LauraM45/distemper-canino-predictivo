"""Generador del diagrama arquitectónico de preprocesamiento.

Produce una figura de alta resolución adecuada para publicaciones
IEEE/MDPI/Springer, mostrando la separación y transformación de las
ramas Numérica, Categórica y Binaria mediante ColumnTransformer.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(
    "c:/Users/lssof/OneDrive/Desktop/Universidad/Investigacion2/"
    "distemper-canino-predictivo"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "demo_reports"
    / "arquitectura_preprocesamiento_columntransformer.png"
)


# Figura ligeramente más ancha y más alta
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


# ============================================================
# PALETA
# ============================================================

COLOR_INPUT = "#334155"
COLOR_NUM = "#0284c7"
COLOR_CAT = "#7c3aed"
COLOR_BIN = "#059669"
COLOR_CONCAT = "#ea580c"
COLOR_OUTPUT = "#0f172a"

EDGE_COLOR = "#1e293b"
ARROW_COLOR = "#475569"
TEXT_SECONDARY = "#e2e8f0"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def draw_box(
    x,
    y,
    w,
    h,
    color,
    title,
    subtitle="",
    title_size=10,
    subtitle_size=8.5,
):
    """Dibuja una caja con título y subtítulo."""

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12,rounding_size=0.16",
        facecolor=color,
        edgecolor=EDGE_COLOR,
        linewidth=1.4,
        zorder=2,
    )

    ax.add_patch(box)

    cx = x + w / 2

    # Título
    ax.text(
        cx,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color="white",
        zorder=3,
        wrap=True,
    )

    # Subtítulo
    if subtitle:
        ax.text(
            cx,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=TEXT_SECONDARY,
            zorder=3,
            linespacing=1.25,
        )


def draw_arrow(
    x1,
    y1,
    x2,
    y2,
    color=ARROW_COLOR,
    lw=1.8,
):
    """Dibuja una flecha entre dos puntos."""

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=13,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=1,
    )


# ============================================================
# TÍTULO
# ============================================================

ax.text(
    8,
    8.65,
    "Arquitectura Modular del Preprocesamiento de Datos Clínicos (CDSS)",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color=COLOR_OUTPUT,
)

ax.text(
    8,
    8.32,
    "Pipeline scikit-learn con aislamiento de pliegues (Anti-Data Leakage) "
    "y tolerancia a Coldstart",
    ha="center",
    va="center",
    fontsize=9.5,
    color=ARROW_COLOR,
    fontstyle="italic",
)


# ============================================================
# 1. ENTRADA
# ============================================================

draw_box(
    0.45,
    3.35,
    2.35,
    2.15,
    COLOR_INPUT,
    "Dataset Clínico",
    "Vector de Entrada\n(11 Variables)\n[X ∈ R^(N×11)]",
    title_size=11,
    subtitle_size=9,
)


# ============================================================
# 2. RAMAS
# ============================================================

# ------------------------------------------------------------
# Rama numérica
# ------------------------------------------------------------

draw_arrow(
    2.8,
    4.9,
    3.55,
    6.25,
    COLOR_NUM,
)

draw_box(
    3.55,
    5.65,
    2.45,
    1.35,
    COLOR_NUM,
    "Rama Numérica",
    "edad_meses\n(1 variable)",
    title_size=10,
    subtitle_size=8.5,
)

draw_arrow(
    6.0,
    6.32,
    6.65,
    6.32,
    COLOR_NUM,
)

draw_box(
    6.65,
    5.55,
    3.15,
    1.55,
    "#0369a1",
    "Imputación + Escalado",
    "SimpleImputer(median)\n+ StandardScaler()\nz = (x − μ) / σ → 1 feature",
    title_size=9.5,
    subtitle_size=7.8,
)


# ------------------------------------------------------------
# Rama categórica
# ------------------------------------------------------------

draw_arrow(
    2.8,
    4.4,
    3.55,
    4.4,
    COLOR_CAT,
)

draw_box(
    3.55,
    3.65,
    2.45,
    1.5,
    COLOR_CAT,
    "Rama Categórica",
    "sexo, raza, talla\nubicacion, estado_vacunal\n(5 variables)",
    title_size=10,
    subtitle_size=7.8,
)

draw_arrow(
    6.0,
    4.4,
    6.65,
    4.4,
    COLOR_CAT,
)

draw_box(
    6.65,
    3.55,
    3.15,
    1.7,
    "#6d28d9",
    "Imputación + One-Hot",
    "Imputer('Desconocido')\nOneHotEncoder(min_freq=2)\nhandle_unknown='infrequent_if_exist'\n→ 38 features",
    title_size=9.3,
    subtitle_size=7.5,
)


# ------------------------------------------------------------
# Rama binaria
# ------------------------------------------------------------

draw_arrow(
    2.8,
    3.9,
    3.55,
    2.55,
    COLOR_BIN,
)

draw_box(
    3.55,
    1.85,
    2.45,
    1.5,
    COLOR_BIN,
    "Rama Binaria",
    "fiebre, resp./oculonasal\ndigestivos, neuro, derma\n(5 signos)",
    title_size=10,
    subtitle_size=7.8,
)

draw_arrow(
    6.0,
    2.6,
    6.65,
    2.6,
    COLOR_BIN,
)

draw_box(
    6.65,
    1.8,
    3.15,
    1.6,
    "#047857",
    "Imputación Binaria",
    "SimpleImputer(constant=0)\nPreserva banderas {0, 1}\n→ 5 features binarias",
    title_size=9.5,
    subtitle_size=8,
)


# ============================================================
# 3. COLUMN TRANSFORMER
# ============================================================

# Flechas hacia la concatenación
draw_arrow(
    9.8,
    6.3,
    10.7,
    5.05,
    COLOR_NUM,
)

draw_arrow(
    9.8,
    4.4,
    10.7,
    4.4,
    COLOR_CAT,
)

draw_arrow(
    9.8,
    2.6,
    10.7,
    3.75,
    COLOR_BIN,
)


draw_box(
    10.7,
    3.05,
    2.05,
    2.65,
    COLOR_CONCAT,
    "ColumnTransformer",
    "Concatenación\nHorizontal\n\n1 + 38 + 5\n= 44 features",
    title_size=9.5,
    subtitle_size=8.2,
)


# ============================================================
# 4. SALIDA
# ============================================================

draw_arrow(
    12.75,
    4.38,
    13.35,
    4.38,
    COLOR_OUTPUT,
    lw=2.2,
)


draw_box(
    13.35,
    2.65,
    2.15,
    3.45,
    COLOR_OUTPUT,
    "Características\nTransformadas",
    "X_transf ∈ R^(N×44)\n\n──────────────\n\nRandom Forest\nDecision Tree\nLogistic Regression",
    title_size=9.3,
    subtitle_size=8,
)


# ============================================================
# 5. NOTA METODOLÓGICA
# ============================================================

note_text = (
    "Nota metodológica: cada transformación se ajusta (fit) exclusivamente "
    "sobre los pliegues de entrenamiento en StratifiedKFold (k=5), "
    "previniendo Data Leakage y permitiendo manejar categorías nuevas "
    "durante la inferencia (Coldstart)."
)

ax.text(
    8,
    0.55,
    note_text,
    ha="center",
    va="center",
    fontsize=8.2,
    color="#64748b",
    linespacing=1.3,
    bbox=dict(
        boxstyle="round,pad=0.45",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1,
    ),
)


# ============================================================
# GUARDAR
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.18,
)

plt.close(fig)

print(
    f"[OK] Diagrama guardado exitosamente en:\n{OUTPUT_PATH}"
)