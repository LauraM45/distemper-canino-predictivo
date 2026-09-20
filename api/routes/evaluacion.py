"""Ruta para la evaluación y diagnóstico predictivo de pacientes caninos."""

from __future__ import annotations

from typing import Generator
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas.paciente import DiagnosticoResponseSchema, PacienteInputSchema
from api.services.inference_service import InferenceService
from src.data.dataValidation import CDVValidationError
from src.database.connection import get_session_factory

router = APIRouter(prefix="/evaluar", tags=["Evaluación Diagnóstica"])


def get_db() -> Generator[Session, None, None]:
    """Inyección de dependencias para sesión de SQLAlchemy en FastAPI."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


@router.post(
    "/",
    response_model=DiagnosticoResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Evaluar nuevo paciente canino",
    description="Recibe 11 variables clínicas, ejecuta el modelo de Scikit-Learn y persiste el diagnóstico en Neon DB.",
)
def evaluar_paciente(
    paciente_in: PacienteInputSchema,
    db: Session = Depends(get_db),
) -> DiagnosticoResponseSchema:
    """Ejecuta inferencia y persiste el paciente evaluado."""
    try:
        resultado = InferenceService.evaluar_y_guardar(paciente_in, db)
        return resultado
    except CDVValidationError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inconsistencia en variables clínicas: {str(val_err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el proceso de diagnóstico predictivo: {str(exc)}",
        )
