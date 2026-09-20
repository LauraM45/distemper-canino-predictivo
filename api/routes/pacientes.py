"""Rutas para la consulta e historial de pacientes caninos diagnosticados."""

from __future__ import annotations

import uuid
from typing import Generator, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from api.schemas.paciente import PacienteResumenSchema
from src.database.connection import get_session_factory
from src.database.models import PacienteCDV

router = APIRouter(prefix="/pacientes", tags=["Historial de Pacientes"])


def get_db() -> Generator[Session, None, None]:
    """Inyección de dependencias para sesión de SQLAlchemy en FastAPI."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


@router.get(
    "/",
    response_model=List[PacienteResumenSchema],
    summary="Listar historial de pacientes evaluados",
    description="Retorna la lista ordenada cronológicamente de los caninos diagnosticados en Neon DB.",
)
def listar_pacientes(
    limit: int = Query(50, ge=1, le=200, description="Cantidad máxima de registros a retornar"),
    offset: int = Query(0, ge=0, description="Desplazamiento para paginación"),
    db: Session = Depends(get_db),
) -> List[PacienteResumenSchema]:
    """Consulta paginada del historial de pacientes."""
    pacientes_orm = db.scalars(
        select(PacienteCDV).order_by(desc(PacienteCDV.fecha_registro)).offset(offset).limit(limit)
    ).all()

    return [
        PacienteResumenSchema(
            id=p.id,
            fecha_registro=p.fecha_registro,
            edad_meses=p.edad_meses,
            sexo=p.sexo,
            raza=p.raza,
            talla=p.talla,
            ubicacion_procedencia=p.ubicacion_procedencia,
            estado_vacunal=p.estado_vacunal,
            prediccion_cdv=p.prediccion_cdv,
            probabilidad_cdv=p.probabilidad_cdv,
            nivel_riesgo=p.nivel_riesgo,
            accion_clinica=p.accion_clinica,
        )
        for p in pacientes_orm
    ]


@router.get(
    "/{paciente_id}",
    response_model=PacienteResumenSchema,
    summary="Obtener detalle de un paciente por UUID",
)
def obtener_paciente_por_id(
    paciente_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> PacienteResumenSchema:
    """Busca un paciente específico en la base de datos."""
    paciente = db.scalar(select(PacienteCDV).where(PacienteCDV.id == paciente_id))
    if not paciente:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró ningún paciente con el ID: {paciente_id}",
        )

    return PacienteResumenSchema(
        id=paciente.id,
        fecha_registro=paciente.fecha_registro,
        edad_meses=paciente.edad_meses,
        sexo=paciente.sexo,
        raza=paciente.raza,
        talla=paciente.talla,
        ubicacion_procedencia=paciente.ubicacion_procedencia,
        estado_vacunal=paciente.estado_vacunal,
        prediccion_cdv=paciente.prediccion_cdv,
        probabilidad_cdv=paciente.probabilidad_cdv,
        nivel_riesgo=paciente.nivel_riesgo,
        accion_clinica=paciente.accion_clinica,
    )
