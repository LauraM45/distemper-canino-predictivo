"""Servicio de cálculo y consolidación estadística para el Dashboard veterinario."""

from __future__ import annotations

from typing import Dict, List
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from api.schemas.dashboard import (
    DashboardResponseSchema,
    DashboardSummarySchema,
    PrevalenciaSignoSchema,
)
from api.schemas.paciente import PacienteResumenSchema
from src.database.models import PacienteCDV


class DashboardService:
    """Calcula agregaciones e indicadores clínicos para la toma de decisiones."""

    SIGNOS_CLINICOS = [
        ("fiebre_hipertermia", "Fiebre / Hipertermia"),
        ("signos_respiratorios_oculonasales", "Secreción Oculonasal / Respiratoria"),
        ("signos_digestivos", "Signos Digestivos (Vómito / Diarrea)"),
        ("signos_neurologicos", "Signos Neurológicos (Mioclonías / Ataxia)"),
        ("signos_dermatologicos", "Signos Dermatológicos (Hiperqueratosis)"),
    ]

    @classmethod
    def obtener_metricas(cls, db: Session) -> DashboardResponseSchema:
        """Extrae de Neon DB los datos agregados para renderizar el Dashboard."""
        # 1. Total general de evaluaciones
        total_evaluados = db.scalar(select(func.count(PacienteCDV.id))) or 0

        if total_evaluados == 0:
            return DashboardResponseSchema(
                resumen=DashboardSummarySchema(
                    total_evaluados=0,
                    total_positivos=0,
                    total_negativos=0,
                    tasa_positividad=0.0,
                    alto_riesgo_count=0,
                    moderado_riesgo_count=0,
                    bajo_riesgo_count=0,
                ),
                distribucion_riesgo={"Alto Riesgo": 0, "Riesgo Moderado": 0, "Bajo Riesgo": 0},
                prevalencia_signos=[],
                ultimos_pacientes=[],
            )

        # 2. Conteo de positivos y negativos
        total_positivos = (
            db.scalar(select(func.count(PacienteCDV.id)).where(PacienteCDV.prediccion_cdv == 1)) or 0
        )
        total_negativos = total_evaluados - total_positivos
        tasa_positividad = round((total_positivos / total_evaluados) * 100.0, 1)

        # 3. Distribución por nivel de riesgo clínico
        alto_riesgo = (
            db.scalar(select(func.count(PacienteCDV.id)).where(PacienteCDV.nivel_riesgo == "Alto Riesgo"))
            or 0
        )
        moderado_riesgo = (
            db.scalar(
                select(func.count(PacienteCDV.id)).where(PacienteCDV.nivel_riesgo == "Riesgo Moderado")
            )
            or 0
        )
        bajo_riesgo = (
            db.scalar(select(func.count(PacienteCDV.id)).where(PacienteCDV.nivel_riesgo == "Bajo Riesgo"))
            or 0
        )

        distribucion_riesgo = {
            "Alto Riesgo": alto_riesgo,
            "Riesgo Moderado": moderado_riesgo,
            "Bajo Riesgo": bajo_riesgo,
        }

        # 4. Prevalencia de signos en positivos vs negativos
        prevalencias: List[PrevalenciaSignoSchema] = []
        for col_name, etiqueta in cls.SIGNOS_CLINICOS:
            col_attr = getattr(PacienteCDV, col_name)

            # En positivos
            pos_con_signo = (
                db.scalar(
                    select(func.count(PacienteCDV.id)).where(
                        PacienteCDV.prediccion_cdv == 1, col_attr == 1
                    )
                )
                or 0
            )
            pct_pos = round((pos_con_signo / total_positivos * 100.0), 1) if total_positivos > 0 else 0.0

            # En negativos
            neg_con_signo = (
                db.scalar(
                    select(func.count(PacienteCDV.id)).where(
                        PacienteCDV.prediccion_cdv == 0, col_attr == 1
                    )
                )
                or 0
            )
            pct_neg = round((neg_con_signo / total_negativos * 100.0), 1) if total_negativos > 0 else 0.0

            prevalencias.append(
                PrevalenciaSignoSchema(
                    signo_clave=col_name,
                    etiqueta=etiqueta,
                    porcentaje_en_positivos=pct_pos,
                    porcentaje_en_negativos=pct_neg,
                )
            )

        # 5. Últimos 10 pacientes evaluados
        ultimos_orm = (
            db.scalars(
                select(PacienteCDV).order_by(desc(PacienteCDV.fecha_registro)).limit(10)
            ).all()
        )

        ultimos_pacientes = [
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
            for p in ultimos_orm
        ]

        return DashboardResponseSchema(
            resumen=DashboardSummarySchema(
                total_evaluados=total_evaluados,
                total_positivos=total_positivos,
                total_negativos=total_negativos,
                tasa_positividad=tasa_positividad,
                alto_riesgo_count=alto_riesgo,
                moderado_riesgo_count=moderado_riesgo,
                bajo_riesgo_count=bajo_riesgo,
            ),
            distribucion_riesgo=distribucion_riesgo,
            prevalencia_signos=prevalencias,
            ultimos_pacientes=ultimos_pacientes,
        )
