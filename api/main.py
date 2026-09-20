"""Aplicación principal de FastAPI para el Sistema Predictivo de Distemper Canino."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import dashboard, evaluacion, pacientes
from api.services.inference_service import InferenceService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ciclo de vida de la aplicación: inicializa recursos al arrancar."""
    print("\n[INFO] Inicializando servidor FastAPI...")
    print("[INFO] Precargando el motor de inferencia CDV en memoria...")
    try:
        InferenceService.get_engine()
        print("[OK] Pipeline de Machine Learning precargado y listo.")
    except Exception as exc:
        print(f"[ADVERTENCIA] No se pudo precargar el modelo: {exc}")
    yield
    print("\n[INFO] Cerrando servidor FastAPI.")


app = FastAPI(
    title="Sistema de Diagnóstico Predictivo - Distemper Canino (CDV)",
    description=(
        "API REST y servicio analítico veterinario para la detección temprana "
        "y estratificación de riesgo de Moquillo Canino basado en Machine Learning."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS para permitir conexiones desde React y clientes locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusión de las rutas API REST
app.include_router(evaluacion.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(pacientes.router, prefix="/api")


@app.get("/api/health", tags=["Estado del Sistema"])
def health_check() -> dict:
    """Verificación de operatividad del servicio."""
    return {
        "status": "online",
        "service": "CDV-Predictive-Detector",
        "version": "1.0.0",
    }


# Servir la interfaz web estática si existe el directorio frontend
if FRONTEND_DIR.is_dir():
    # Si existe una subcarpeta estática, se monta
    public_dir = FRONTEND_DIR / "public"
    if public_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(public_dir)), name="static")

    # Ruta raíz para desplegar la aplicación web directamente
    index_file = FRONTEND_DIR / "index.html"
    if index_file.is_file():
        @app.get("/", include_in_schema=False)
        def serve_frontend() -> FileResponse:
            return FileResponse(index_file)
