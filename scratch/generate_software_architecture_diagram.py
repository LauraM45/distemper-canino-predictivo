"""Generador del diagrama de arquitectura de software y flujo CDSS (Figura 3).

Produce una figura de alta resolución (300 DPI) para publicación académica (IEEE/MDPI/Springer)
que ilustra el flujo integral: Usuario -> Frontend Web -> FastAPI -> CDVInferenceEngine -> PostgreSQL.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

PROJECT_ROOT = Path("c:/Users/lssof/OneDrive/Desktop/Universidad/Investigacion2/distemper-canino-predictivo")
OUTPUT_PATH = PROJECT_ROOT / "demo_reports" / "arquitectura_software_cdss.png"

fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.5)
ax.axis('off')

# Paleta arquitectónica limpia
COLOR_TIER_BG = "#f8fafc"
COLOR_CLIENT = "#0284c7"      # Azul cian
COLOR_BACKEND = "#059669"     # Verde esmeralda
COLOR_ML = "#7c3aed"          # Violeta
COLOR_DB = "#ea580c"          # Naranja terracota
COLOR_BORDER = "#cbd5e1"

def draw_container(x, y, w, h, title, color_header):
    # Fondo del contenedor
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15",
                         facecolor="#f1f5f9", edgecolor=COLOR_BORDER, linewidth=1.5, zorder=1)
    ax.add_patch(box)
    
    # Encabezado del contenedor
    header = FancyBboxPatch((x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0.15,rounding_size=0.15",
                            facecolor=color_header, edgecolor=color_header, linewidth=1, zorder=2)
    ax.add_patch(header)
    ax.text(x + w / 2, y + h - 0.275, title, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="white", zorder=3)

def draw_subcard(x, y, w, h, title, subtitle, color_accent="#1e293b", bg="#ffffff"):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.12",
                          facecolor=bg, edgecolor="#94a3b8", linewidth=1.1, zorder=3)
    ax.add_patch(card)
    
    # Barra lateral de acento
    accent = Rectangle((x, y + 0.05), 0.12, h - 0.1, facecolor=color_accent, edgecolor=color_accent, zorder=4)
    ax.add_patch(accent)
    
    cx = x + w / 2 + 0.06
    if subtitle:
        ax.text(cx, y + h * 0.68, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color="#0f172a", zorder=4)
        ax.text(cx, y + h * 0.32, subtitle, ha='center', va='center', fontsize=7.5,
                color="#475569", zorder=4)
    else:
        ax.text(cx, y + h / 2, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color="#0f172a", zorder=4)

def draw_arrow(x1, y1, x2, y2, label="", color="#334155", lw=1.8, style="-|>", offset_y=0.18):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                mutation_scale=13, shrinkA=3, shrinkB=3), zorder=5)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + offset_y
        ax.text(mx, my, label, ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.8),
                zorder=6)

# ================= ENCABEZADO =================
ax.text(8.0, 9.15, "Figure 3. Software architecture diagram and Clinical Decision Support System (CDSS) integration",
        ha='center', va='center', fontsize=12.5, fontweight='bold', color="#0f172a")
ax.text(8.0, 8.85, "Flujo desacoplado en cuatro capas: Presentación Cliente → API REST FastAPI → Motor Predictivo ML → Persistencia Relacional",
        ha='center', va='center', fontsize=9.2, color="#475569", fontstyle='italic')

# ================= 1. CAPA DE PRESENTACIÓN / FRONTEND =================
draw_container(0.5, 1.2, 3.4, 7.3, "1. CAPA CLIENTE / PRESENTACIÓN", COLOR_CLIENT)

draw_subcard(0.7, 6.8, 3.0, 1.1, "Médico Veterinario / Triaje",
             "Usuario clínico en consulta médica\n(Bogotá D.C. y Chía)", COLOR_CLIENT)

draw_arrow(2.2, 6.8, 2.2, 5.9, label="Interacción UI", color=COLOR_CLIENT)

draw_subcard(0.7, 4.5, 3.0, 1.4, "Formulario Clínico Web",
             "• Captura de 11 variables clínicas\n• Interruptores de signos cardinales\n• Autocompletado de razas/zonas", COLOR_CLIENT)

draw_subcard(0.7, 2.8, 3.0, 1.4, "Modal Diagnóstico Inmediato",
             "• Barra interactiva de probabilidad (%)\n• Semáforo de Riesgo (Alto/Mod/Bajo)\n• Pauta de aislamiento sugerida", "#0284c7")

draw_subcard(0.7, 1.4, 3.0, 1.2, "Dashboard Epidemiológico",
             "• 4 Tarjetas KPI (Total, Positivos)\n• Gráfica Dona (Riesgo) y Barras (Signos)\n• Historial clínico en tiempo real", "#0284c7")

# Flecha Comunicación Frontend -> Backend
draw_arrow(3.9, 5.2, 4.6, 5.2, label="POST /api/evaluar/\n(JSON 11 variables)", color="#0f172a", lw=2.0)
draw_arrow(4.6, 2.0, 3.9, 2.0, label="GET /api/dashboard/\n(Métricas agregadas)", color="#0284c7", lw=1.6)

# ================= 2. CAPA DE TRANSPORTE Y API / BACKEND =================
draw_container(4.6, 1.2, 3.4, 7.3, "2. CAPA API REST / FASTAPI", COLOR_BACKEND)

draw_subcard(4.8, 6.8, 3.0, 1.1, "Servidor ASGI (Uvicorn)",
             "FastAPI Framework (Asíncrono)\nMiddleware CORS + Docs Swagger", COLOR_BACKEND)

draw_subcard(4.8, 5.1, 3.0, 1.4, "Validación Pydantic",
             "• PacienteInputSchema (Rangos [0.5, 240])\n• Categorías cerradas (sexo, talla)\n• Banderas binarias estrictas {0, 1}", COLOR_BACKEND)

draw_subcard(4.8, 3.4, 3.0, 1.4, "Servicio de Negocio",
             "InferenceService & DashboardService\n• Desacoplamiento de inferencia\n• Coordinación con Base de Datos", COLOR_BACKEND)

draw_subcard(4.8, 1.4, 3.0, 1.7, "Rutas y Controladores",
             "• api/routes/evaluacion.py\n• api/routes/dashboard.py\n• api/routes/pacientes.py\n(Respuestas enriquecidas JSON)", COLOR_BACKEND)

# Flecha Backend -> Machine Learning Engine
draw_arrow(8.0, 5.2, 8.7, 5.2, label="Invocación en Memoria\nCDVInferenceEngine", color="#0f172a", lw=2.0)
draw_arrow(8.7, 4.1, 8.0, 4.1, label="Diagnóstico + Prob (%)\n+ Nivel de Riesgo", color=COLOR_ML, lw=1.6)

# ================= 3. CAPA DE MACHINE LEARNING / CDSS =================
draw_container(8.7, 1.2, 3.5, 7.3, "3. MOTOR PREDICTIVO (ML ENGINE)", COLOR_ML)

draw_subcard(8.9, 6.8, 3.1, 1.1, "CDVInferenceEngine",
             "predict.py — Instancia única en RAM\nPrecargada en el ciclo de vida (Lifespan)", COLOR_ML)

draw_subcard(8.9, 5.0, 3.1, 1.5, "CDVDataValidator",
             "src/data/dataValidation.py\n• Validación de esquema contractual\n• Zero Data Leakage\n• Tolerancia a nulos", COLOR_ML)

draw_subcard(8.9, 3.0, 3.1, 1.7, "Pipeline Scikit-Learn (.pkl)",
             "cdv_randomforest_pipeline.pkl\n• ColumnTransformer (44 features)\n• StandardScaler (1 num)\n• OneHotEncoder anti-coldstart (38 cat)\n• Binary pass (5 bin)", COLOR_ML)

draw_subcard(8.9, 1.4, 3.1, 1.3, "Ensamble Random Forest",
             "100 Árboles de Decisión\n• Recall Clínico Garantizado: 99.2%\n• ROC-AUC: 0.9400", COLOR_ML)

# Flecha Backend -> Base de Datos
draw_arrow(8.0, 3.5, 12.8, 3.5, label="Persistencia Transaccional\nSQLAlchemy Session", color="#ea580c", lw=1.8, offset_y=0.22)
draw_arrow(12.8, 1.8, 8.0, 1.8, label="Consultas SQL Agregadas\n(count, avg, group by)", color="#ea580c", lw=1.6, offset_y=-0.22)

# ================= 4. CAPA DE PERSISTENCIA / BASE DE DATOS =================
draw_container(12.8, 1.2, 2.7, 7.3, "4. PERSISTENCIA / POSTGRESQL", COLOR_DB)

draw_subcard(13.0, 6.7, 2.3, 1.2, "SQLAlchemy ORM 2.0",
             "src/database/connection.py\nPool de conexiones seguro\n(SSL mode = require)", COLOR_DB)

draw_subcard(13.0, 4.4, 2.3, 2.0, "Tabla: pacientes_cdv",
             "• id: UUIDv4 (PK)\n• fecha_registro: DateTime\n• 11 Variables Clínicas\n• prediccion_cdv: Int (0/1)\n• probabilidad_cdv: Float\n• nivel_riesgo: String\n• accion_clinica: Text", COLOR_DB)

draw_subcard(13.0, 1.8, 2.3, 2.3, "Neon DB (Cloud)",
             "PostgreSQL Serverless\n• Índices optimizados por fecha, riesgo y raza\n• Constraints de integridad biológica\n• Trazabilidad y auditoría clínica", COLOR_DB)

# ================= NOTA AL PIE =================
ax.text(8.0, 0.45,
        "Nota Arquitectónica: El sistema desacopla el ciclo de entrenamiento offline (main.py) del ciclo de inferencia online en tiempo real (FastAPI + React/UI).\n"
        "La precarga del artefacto binario (.pkl) en memoria permite una latencia de diagnóstico < 20 ms por paciente.",
        ha='center', va='center', fontsize=8.2, color="#64748b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cbd5e1", lw=1))

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"[OK] Diagrama de Arquitectura CDSS guardado exitosamente en: {OUTPUT_PATH}")
