"""Ruta para métricas e indicadores analíticos del Dashboard veterinario."""

from __future__ import annotations

from typing import Generator
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.dashboard import DashboardResponseSchema
from api.services.dashboard_service import DashboardService
from src.database.connection import get_session_factory

router = APIRouter(prefix="/dashboard", tags=["Dashboard Clínico"])


def get_db() -> Generator[Session, None, None]:
    """Inyección de dependencias para sesión de SQLAlchemy en FastAPI."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


@router.get(
    "/metricas",
    response_model=DashboardResponseSchema,
    summary="Obtener métricas consolidadas para el Dashboard",
    description="Calcula totales, porcentajes de positividad, niveles de riesgo y prevalencia sintomatológica.",
)
def obtener_metricas_dashboard(
    db: Session = Depends(get_db),
) -> DashboardResponseSchema:
    """Devuelve el conjunto de datos para los componentes visuales y gráficas."""
    try:
        return DashboardService.obtener_metricas(db)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No se pudieron calcular las métricas del dashboard: {str(exc)}",
        )
