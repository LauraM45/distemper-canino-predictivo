"""Modelos ORM de SQLAlchemy para el Sistema Predictivo de Distemper Canino.

Define la estructura de la tabla 'pacientes_cdv' con claves primarias UUID,
variables clínicas, resultados de inferencia y seguimiento de laboratorio.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Clase base declarativa para los modelos SQLAlchemy."""
    pass


class PacienteCDV(Base):
    """Modelo ORM que representa un paciente canino evaluado en el sistema.

    Attributes
    ----------
    id : uuid.UUID
        Identificador único global del paciente (UUID v4).
    fecha_registro : datetime
        Marca de tiempo con zona horaria de la evaluación o registro.
    edad_meses : float
        Edad del paciente en meses (0.5 a 240.0).
    sexo : str
        Sexo del paciente ('Macho', 'Hembra').
    raza : str
        Raza zootécnica (resiliente ante coldstart).
    talla : str
        Clasificación somática ('Pequeño', 'Mediano', 'Grande', 'Gigante').
    ubicacion_procedencia : str
        Centro de atención, albergue o procedencia geográfica.
    estado_vacunal : str
        Historial de inmunización ('Completo', 'Incompleto', 'No vacunado', 'Desconocido').
    fiebre_hipertermia : int
        Presencia de fiebre (0 = No, 1 = Sí).
    signos_respiratorios_oculonasales : int
        Presencia de secreción oculonasal o signos respiratorios (0 / 1).
    signos_digestivos : int
        Presencia de vómito o diarrea (0 / 1).
    signos_neurologicos : int
        Presencia de mioclonías, ataxia, convulsiones o paresia (0 / 1).
    signos_dermatologicos : int
        Presencia de hiperqueratosis o dermatitis (0 / 1).
    prediccion_cdv : int
        Predicción emitida por el modelo (0 = Negativo/Sano, 1 = Positivo CDV).
    probabilidad_cdv : float
        Probabilidad calculada para infección de Distemper Canino [0.0 - 1.0].
    probabilidad_sano : float
        Probabilidad calculada para paciente sano [0.0 - 1.0].
    nivel_riesgo : str
        Clasificación clínica ('Bajo Riesgo', 'Riesgo Moderado', 'Alto Riesgo').
    accion_clinica : str, optional
        Recomendación de conducta clínica generada automáticamente.
    modelo_version : str
        Identificador y versión del modelo que generó la inferencia.
    diagnostico_laboratorio_confirmado : int, optional
        Resultado confirmatorio de laboratorio posterior (0 = Negativo, 1 = Positivo).
    observaciones_veterinario : str, optional
        Notas clínicas adicionales registradas por el personal médico.
    origen_registro : str
        Origen de los datos ('dataset_historico', 'interfaz_web', 'batch_api').
    """

    __tablename__ = "pacientes_cdv"

    # Identificación única con UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Identificador único global del paciente (UUIDv4)",
    )

    fecha_registro: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # 11 Variables clínicas requeridas por el modelo
    edad_meses: Mapped[float] = mapped_column(Float, nullable=False)
    sexo: Mapped[str] = mapped_column(String(20), nullable=False)
    raza: Mapped[str] = mapped_column(String(100), nullable=False)
    talla: Mapped[str] = mapped_column(String(30), nullable=False)
    ubicacion_procedencia: Mapped[str] = mapped_column(String(150), nullable=False)
    estado_vacunal: Mapped[str] = mapped_column(String(50), nullable=False)
    fiebre_hipertermia: Mapped[int] = mapped_column(Integer, nullable=False)
    signos_respiratorios_oculonasales: Mapped[int] = mapped_column(Integer, nullable=False)
    signos_digestivos: Mapped[int] = mapped_column(Integer, nullable=False)
    signos_neurologicos: Mapped[int] = mapped_column(Integer, nullable=False)
    signos_dermatologicos: Mapped[int] = mapped_column(Integer, nullable=False)

    # Resultados predictivos generados por el pipeline
    prediccion_cdv: Mapped[int] = mapped_column(Integer, nullable=False)
    probabilidad_cdv: Mapped[float] = mapped_column(Float, nullable=False)
    probabilidad_sano: Mapped[float] = mapped_column(Float, nullable=False)
    nivel_riesgo: Mapped[str] = mapped_column(String(30), nullable=False)
    accion_clinica: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    modelo_version: Mapped[str] = mapped_column(String(50), default="1.0.0", nullable=False)

    # Validación clínica / Feedback posterior para reentrenamiento continuo
    diagnostico_laboratorio_confirmado: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    observaciones_veterinario: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    origen_registro: Mapped[str] = mapped_column(String(50), default="interfaz_web", nullable=False)

    # Restricciones de integridad clínica
    __table_args__ = (
        CheckConstraint("edad_meses >= 0.5 AND edad_meses <= 240.0", name="chk_edad_meses_rango"),
        CheckConstraint("sexo IN ('Macho', 'Hembra')", name="chk_sexo_valores"),
        CheckConstraint("talla IN ('Pequeño', 'Mediano', 'Grande', 'Gigante')", name="chk_talla_valores"),
        CheckConstraint(
            "estado_vacunal IN ('Completo', 'Incompleto', 'No vacunado', 'Desconocido')",
            name="chk_estado_vacunal_valores",
        ),
        CheckConstraint("fiebre_hipertermia IN (0, 1)", name="chk_fiebre_binario"),
        CheckConstraint("signos_respiratorios_oculonasales IN (0, 1)", name="chk_resp_binario"),
        CheckConstraint("signos_digestivos IN (0, 1)", name="chk_digest_binario"),
        CheckConstraint("signos_neurologicos IN (0, 1)", name="chk_neuro_binario"),
        CheckConstraint("signos_dermatologicos IN (0, 1)", name="chk_derma_binario"),
        CheckConstraint("prediccion_cdv IN (0, 1)", name="chk_prediccion_binario"),
        CheckConstraint(
            "diagnostico_laboratorio_confirmado IS NULL OR diagnostico_laboratorio_confirmado IN (0, 1)",
            name="chk_lab_confirmado_binario",
        ),
        Index("idx_pacientes_fecha_reg", "fecha_registro"),
        Index("idx_pacientes_nivel_riesgo", "nivel_riesgo"),
        Index("idx_pacientes_raza", "raza"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la entidad en un diccionario serializable."""
        return {
            "id": str(self.id),
            "fecha_registro": self.fecha_registro.isoformat() if self.fecha_registro else None,
            "edad_meses": self.edad_meses,
            "sexo": self.sexo,
            "raza": self.raza,
            "talla": self.talla,
            "ubicacion_procedencia": self.ubicacion_procedencia,
            "estado_vacunal": self.estado_vacunal,
            "fiebre_hipertermia": self.fiebre_hipertermia,
            "signos_respiratorios_oculonasales": self.signos_respiratorios_oculonasales,
            "signos_digestivos": self.signos_digestivos,
            "signos_neurologicos": self.signos_neurologicos,
            "signos_dermatologicos": self.signos_dermatologicos,
            "prediccion_cdv": self.prediccion_cdv,
            "probabilidad_cdv": self.probabilidad_cdv,
            "probabilidad_sano": self.probabilidad_sano,
            "nivel_riesgo": self.nivel_riesgo,
            "accion_clinica": self.accion_clinica,
            "modelo_version": self.modelo_version,
            "diagnostico_laboratorio_confirmado": self.diagnostico_laboratorio_confirmado,
            "observaciones_veterinario": self.observaciones_veterinario,
            "origen_registro": self.origen_registro,
        }

    def __repr__(self) -> str:
        return (
            f"<PacienteCDV id={self.id} raza='{self.raza}' "
            f"prediccion={self.prediccion_cdv} riesgo='{self.nivel_riesgo}'>"
        )
