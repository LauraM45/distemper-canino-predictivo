"""Servicio de orquestación de inferencia y persistencia para Distemper Canino."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from api.schemas.paciente import DiagnosticoResponseSchema, PacienteInputSchema
from predict import CDVInferenceEngine
from src.database.models import PacienteCDV


class InferenceService:
    """Orquesta la validación, evaluación con el modelo ML y persistencia en BD."""

    _engine_instance: Optional[CDVInferenceEngine] = None

    @classmethod
    def get_engine(cls) -> CDVInferenceEngine:
        """Inicializa o devuelve la instancia en caché del motor de inferencia."""
        if cls._engine_instance is None:
            cls._engine_instance = CDVInferenceEngine()
        return cls._engine_instance

    @classmethod
    def evaluar_y_guardar(
        cls, datos_paciente: PacienteInputSchema, db: Session
    ) -> DiagnosticoResponseSchema:
        """Ejecuta la predicción del modelo y persiste el registro en Neon DB.

        Parameters
        ----------
        datos_paciente : PacienteInputSchema
            Entrada validada con las 11 variables clínicas.
        db : Session
            Sesión activa de base de datos SQLAlchemy.

        Returns
        -------
        DiagnosticoResponseSchema
            Diagnóstico estructurado con id generado y nivel de riesgo.
        """
        engine = cls.get_engine()
        datos_dict = datos_paciente.model_dump(exclude={"diagnostico_laboratorio_real", "notas_veterinarias"})

        # 1. Inferencia mediante el pipeline de Scikit-Learn
        resultado_ml = engine.predict_patient(datos_dict)

        # 2. Construcción de la entidad ORM para PostgreSQL / Neon
        nuevo_id = uuid.uuid4()
        fecha_actual = datetime.now(timezone.utc)

        paciente_orm = PacienteCDV(
            id=nuevo_id,
            fecha_registro=fecha_actual,
            edad_meses=datos_paciente.edad_meses,
            sexo=datos_paciente.sexo,
            raza=datos_paciente.raza,
            talla=datos_paciente.talla,
            ubicacion_procedencia=datos_paciente.ubicacion_procedencia,
            estado_vacunal=datos_paciente.estado_vacunal,
            fiebre_hipertermia=datos_paciente.fiebre_hipertermia,
            signos_respiratorios_oculonasales=datos_paciente.signos_respiratorios_oculonasales,
            signos_digestivos=datos_paciente.signos_digestivos,
            signos_neurologicos=datos_paciente.signos_neurologicos,
            signos_dermatologicos=datos_paciente.signos_dermatologicos,
            prediccion_cdv=resultado_ml["prediction"],
            probabilidad_cdv=resultado_ml["probability_cdv"],
            probabilidad_sano=resultado_ml["probability_sano"],
            nivel_riesgo=resultado_ml["risk_level"],
            accion_clinica=resultado_ml["clinical_action"],
            modelo_version="1.0.0",
            diagnostico_laboratorio_confirmado=datos_paciente.diagnostico_laboratorio_real,
            observaciones_veterinario=datos_paciente.notas_veterinarias,
            origen_registro="interfaz_web",
        )

        # 3. Guardado seguro en base de datos
        db.add(paciente_orm)
        db.commit()
        db.refresh(paciente_orm)

        return DiagnosticoResponseSchema(
            id=nuevo_id,
            fecha_registro=fecha_actual,
            prediction=resultado_ml["prediction"],
            diagnosis=resultado_ml["diagnosis"],
            probability_cdv=resultado_ml["probability_cdv"],
            probability_sano=resultado_ml["probability_sano"],
            risk_level=resultado_ml["risk_level"],
            clinical_action=resultado_ml["clinical_action"],
            paciente_evaluado=datos_paciente.model_dump(),
        )
