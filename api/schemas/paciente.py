"""Esquemas de validación Pydantic para Pacientes y Diagnósticos de Distemper Canino."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PacienteInputSchema(BaseModel):
    """Contrato clínico estricto para el ingreso de un nuevo paciente canino.
    
    Verifica las 11 variables predictoras fisiológicas requeridas por el modelo ML.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "edad_meses": 4.5,
                "sexo": "Macho",
                "raza": "Border Collie",
                "talla": "Mediano",
                "ubicacion_procedencia": "Bogota D.C. (UNAL)",
                "estado_vacunal": "Incompleto",
                "fiebre_hipertermia": 1,
                "signos_respiratorios_oculonasales": 1,
                "signos_digestivos": 0,
                "signos_neurologicos": 1,
                "signos_dermatologicos": 0,
                "notas_veterinarias": "Cachorro con tos leve y temblores musculares rítmicos.",
            }
        }
    )

    edad_meses: float = Field(
        ...,
        ge=0.5,
        le=240.0,
        description="Edad en meses del paciente canino (0.5 a 240 meses).",
    )
    sexo: Literal["Macho", "Hembra"] = Field(
        ...,
        description="Sexo fenotípico del paciente.",
    )
    raza: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Raza del canino (admite razas observadas y categorías no vistas/coldstart).",
    )
    talla: Literal["Pequeño", "Mediano", "Grande", "Gigante"] = Field(
        ...,
        description="Talla o porte somático del canino.",
    )
    ubicacion_procedencia: str = Field(
        ...,
        min_length=2,
        max_length=150,
        description="Centro veterinario, albergue o localidad de procedencia.",
    )
    estado_vacunal: Literal["Completo", "Incompleto", "No vacunado", "Desconocido"] = Field(
        ...,
        description="Esquema de inmunización registrado.",
    )
    fiebre_hipertermia: int = Field(
        0,
        ge=0,
        le=1,
        description="Presencia de hipertermia o fiebre (0 = No, 1 = Sí).",
    )
    signos_respiratorios_oculonasales: int = Field(
        0,
        ge=0,
        le=1,
        description="Secreción oculonasal purulenta o tos (0 = No, 1 = Sí).",
    )
    signos_digestivos: int = Field(
        0,
        ge=0,
        le=1,
        description="Vómito, anorexia o diarrea (0 = No, 1 = Sí).",
    )
    signos_neurologicos: int = Field(
        0,
        ge=0,
        le=1,
        description="Mioclonías, ataxia, paresia o convulsiones (0 = No, 1 = Sí).",
    )
    signos_dermatologicos: int = Field(
        0,
        ge=0,
        le=1,
        description="Hiperqueratosis plantar o dermatitis pustular (0 = No, 1 = Sí).",
    )
    diagnostico_laboratorio_real: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Resultado confirmatorio de PCR/Antígeno de laboratorio si está disponible (0/1).",
    )
    notas_veterinarias: Optional[str] = Field(
        None,
        max_length=1000,
        description="Observaciones clínicas adicionales realizadas por el médico veterinario.",
    )


class DiagnosticoResponseSchema(BaseModel):
    """Respuesta diagnóstica estructurada emitida por el modelo y guardada en BD."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Identificador único del registro clínico en Neon DB.")
    fecha_registro: datetime = Field(..., description="Marca temporal del diagnóstico.")
    
    # Resultados del modelo predictivo
    prediction: int = Field(..., description="Predicción binaria (0 = Negativo/Sano, 1 = Positivo CDV).")
    diagnosis: str = Field(..., description="Etiqueta diagnóstica humana.")
    probability_cdv: float = Field(..., description="Probabilidad calculada de sufrir Distemper (0.0 a 1.0).")
    probability_sano: float = Field(..., description="Probabilidad calculada de estar Sano (0.0 a 1.0).")
    risk_level: str = Field(..., description="Estratificación clínica ('Alto Riesgo', 'Riesgo Moderado', 'Bajo Riesgo').")
    clinical_action: str = Field(..., description="Recomendación o protocolo de aislamiento sugerido.")
    
    # Resumen del paciente
    paciente_evaluado: Dict[str, Any] = Field(..., description="Variables clínicas ingresadas en la evaluación.")


class PacienteResumenSchema(BaseModel):
    """Representación resumida de un paciente para listados e historiales clínicos."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    fecha_registro: datetime
    edad_meses: float
    sexo: str
    raza: str
    talla: str
    ubicacion_procedencia: str
    estado_vacunal: str
    prediccion_cdv: int
    probabilidad_cdv: float
    nivel_riesgo: str
    accion_clinica: Optional[str] = None
    accion_clinica_sugerida: Optional[str] = None
