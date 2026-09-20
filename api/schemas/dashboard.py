"""Esquemas Pydantic para los indicadores y métricas visuales del Dashboard."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, ConfigDict, Field

from api.schemas.paciente import PacienteResumenSchema


class DashboardSummarySchema(BaseModel):
    """Tarjetas KPI consolidadas para la vista analítica del veterinario."""
    model_config = ConfigDict(from_attributes=True)

    total_evaluados: int = Field(..., description="Número total de caninos procesados en el sistema.")
    total_positivos: int = Field(..., description="Total de casos diagnosticados como Positivos para CDV.")
    total_negativos: int = Field(..., description="Total de casos diagnosticados como Sanos/Negativos.")
    tasa_positividad: float = Field(..., description="Porcentaje de positividad global (0 a 100%).")
    alto_riesgo_count: int = Field(..., description="Pacientes en estrato de Alto Riesgo (>= 70% probabilidad).")
    moderado_riesgo_count: int = Field(..., description="Pacientes en estrato de Riesgo Moderado (40-69%).")
    bajo_riesgo_count: int = Field(..., description="Pacientes en estrato de Bajo Riesgo (< 40%).")


class PrevalenciaSignoSchema(BaseModel):
    """Frecuencia observada de un signo clínico comparando infectados vs sanos."""
    signo_clave: str
    etiqueta: str
    porcentaje_en_positivos: float
    porcentaje_en_negativos: float


class DashboardResponseSchema(BaseModel):
    """Payload completo entregado al Frontend para renderizar el Dashboard."""
    resumen: DashboardSummarySchema
    distribucion_riesgo: Dict[str, int]
    prevalencia_signos: List[PrevalenciaSignoSchema]
    ultimos_pacientes: List[PacienteResumenSchema]
