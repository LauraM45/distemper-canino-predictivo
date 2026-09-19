"""Módulo de validación clínica y normalización para Distemper Canino (CDV).

Este módulo define el contrato de datos, las reglas de plausibilidad
veterinaria, los rangos clínicos y los metadatos auditables requeridos
para el diagnóstico predictivo de Moquillo Canino (Canine Distemper Virus - CDV).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


class CDVValidationError(ValueError):
    """Señala que los datos del paciente violan el contrato clínico de CDV.

    Esta excepción se genera cuando faltan columnas requeridas, existen
    valores fuera de los rangos fisiológicos plausibles o se detectan
    inconsistencias estructurales en las variables clínicas.
    """
    pass


# Alias para retrocompatibilidad
DataValidationError = CDVValidationError


@dataclass(frozen=True)
class ModelMetadata:
    """Metadatos formales de especificación, versión y gobernanza del modelo CDV.

    Attributes
    ----------
    model_name : str
        Nombre identificador del sistema predictivo.
    model_version : str
        Versión semántica del modelo.
    target_disease : str
        Nombre clínico de la afección patológica objetivo.
    target_column : str
        Nombre de la variable objetivo binaria.
    min_required_recall : float
        Umbral mínimo de sensibilidad clínica exigido para la clase positiva.
    feature_count : int
        Número total de variables predictoras clínicas.
    clinical_features : Tuple[str, ...]
        Nombres ordenados de las 11 variables predictoras.
    reference_record : Dict[str, Any]
        Registro clínico de referencia validado en campo.
    """

    model_name: str = "CDV-Predictive-Detector"
    model_version: str = "1.0.0"
    target_disease: str = "Canine Distemper Virus (CDV)"
    target_column: str = "diagnostico_cdv_confirmado"
    group_column: str = "ubicacion_procedencia"
    min_required_recall: float = 0.80
    feature_count: int = 11
    clinical_features: Tuple[str, ...] = (
        "edad_meses",
        "sexo",
        "raza",
        "talla",
        "ubicacion_procedencia",
        "estado_vacunal",
        "fiebre_hipertermia",
        "signos_respiratorios_oculonasales",
        "signos_digestivos",
        "signos_neurologicos",
        "signos_dermatologicos",
    )
    reference_record: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        """Inicializa el registro de referencia si no fue proporcionado."""
        if self.reference_record is None:
            object.__setattr__(
                self,
                "reference_record",
                {
                    "edad_meses": 4.0,
                    "sexo": "Macho",
                    "raza": "Border Collie",
                    "talla": "Mediano",
                    "ubicacion_procedencia": "Bogota D.C. (UNAL)",
                    "estado_vacunal": "Incompleto",
                    "fiebre_hipertermia": 1,
                    "signos_respiratorios_oculonasales": 0,
                    "signos_digestivos": 0,
                    "signos_neurologicos": 1,
                    "signos_dermatologicos": 1,
                    "diagnostico_cdv_confirmado": 1,
                },
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa los metadatos a un diccionario estándar.

        Returns
        -------
        Dict[str, Any]
            Diccionario serializable a formato JSON.
        """
        return asdict(self)


@dataclass(frozen=True)
class DataQualityReport:
    """Resultado auditable de una evaluación de calidad de datos clínicos.

    Attributes
    ----------
    row_count : int
        Número total de registros clínicos evaluados.
    missing_by_column : Dict[str, int]
        Conteo de valores nulos por columna requerida.
    coercions_to_missing : Dict[str, int]
        Conteo de valores no numéricos forzados a nulos durante el tipado.
    unique_locations : int
        Número de ubicaciones/centros veterinarios únicos detectados.
    is_valid : bool
        Indica si el dataset cumple estrictamente con el contrato.
    """

    row_count: int
    missing_by_column: Dict[str, int]
    coercions_to_missing: Dict[str, int]
    unique_locations: int
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el reporte de calidad a un diccionario serializable en JSON.

        Returns
        -------
        Dict[str, Any]
            Representación de diccionario estándar.
        """
        return asdict(self)


class CDVDataValidator:
    """Validador estricto de esquema, tipos y rangos clínicos para Distemper Canino.

    Aplica verificación contractual y normalización a nivel de paciente
    o de dataset tabular para asegurar la consistencia antes de la inferencia
    o el entrenamiento.
    """

    TARGET_COLUMN: str = "diagnostico_cdv_confirmado"
    GROUP_COLUMN: str = "ubicacion_procedencia"

    NUMERIC_COLUMNS: Tuple[str, ...] = ("edad_meses",)

    CATEGORICAL_COLUMNS: Tuple[str, ...] = (
        "sexo",
        "raza",
        "talla",
        "ubicacion_procedencia",
        "estado_vacunal",
    )

    BINARY_SIGNS_COLUMNS: Tuple[str, ...] = (
        "fiebre_hipertermia",
        "signos_respiratorios_oculonasales",
        "signos_digestivos",
        "signos_neurologicos",
        "signos_dermatologicos",
    )

    FEATURE_COLUMNS: Tuple[str, ...] = (
        *NUMERIC_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *BINARY_SIGNS_COLUMNS,
    )

    REQUIRED_COLUMNS: Tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

    # Rangos fisiológicos clínicamente admisibles en caninos
    CLINICAL_RANGES: Dict[str, Tuple[float, float]] = {
        "edad_meses": (0.5, 240.0),  # Desde 2 semanas hasta 20 años de edad
    }

    # Categorías caninas controladas
    ALLOWED_CATEGORIES: Dict[str, Set[str]] = {
        "sexo": {"Macho", "Hembra"},
        "talla": {"Pequeño", "Mediano", "Grande", "Gigante"},
        "estado_vacunal": {"Completo", "Incompleto", "No vacunado", "Desconocido"},
    }

    ALLOWED_BINARY_VALUES: Set[int] = {0, 1}

    def validate_and_clean(
        self, data: pd.DataFrame, require_target: bool = True
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Valida esquema, normaliza valores y comprueba plausibilidad clínica.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con las observaciones de pacientes caninos.
        require_target : bool, default=True
            Indica si la columna objetivo `diagnostico_cdv_confirmado` es obligatoria
            (True en entrenamiento, False en inferencia de nuevos pacientes).

        Returns
        -------
        Tuple[pd.DataFrame, DataQualityReport]
            Tupla conteniendo el DataFrame limpio y el reporte de calidad auditable.

        Raises
        ------
        CDVValidationError
            Si se violan las columnas obligatorias, existen rangos fuera de límite
            o inconsistencias de tipado.
        """
        self._validate_schema(data, require_target=require_target)
        cleaned = data.copy(deep=True)

        coercions_to_missing: Dict[str, int] = {}

        # 1. Normalización y tipado numérico
        for col in self.NUMERIC_COLUMNS:
            parsed = self._to_numeric(cleaned[col])
            coerced = int((cleaned[col].notna() & parsed.isna()).sum())
            coercions_to_missing[col] = coerced

            if coerced > 0:
                raise CDVValidationError(
                    f"La columna numérica '{col}' contiene {coerced} valor(es) "
                    "no numéricos inválidos."
                )
            cleaned[col] = parsed

        # 2. Normalización de signos binarios
        for col in self.BINARY_SIGNS_COLUMNS:
            parsed = self._to_numeric(cleaned[col])
            coerced = int((cleaned[col].notna() & parsed.isna()).sum())
            coercions_to_missing[col] = coerced

            if coerced > 0:
                raise CDVValidationError(
                    f"El signo clínico '{col}' contiene {coerced} valor(es) no válidos."
                )
            cleaned[col] = parsed

        # 3. Normalización de target si aplica
        if require_target and self.TARGET_COLUMN in cleaned.columns:
            parsed = self._to_numeric(cleaned[self.TARGET_COLUMN])
            coerced = int((cleaned[self.TARGET_COLUMN].notna() & parsed.isna()).sum())
            coercions_to_missing[self.TARGET_COLUMN] = coerced

            if coerced > 0:
                raise CDVValidationError(
                    f"La variable objetivo '{self.TARGET_COLUMN}' contiene valores no numéricos."
                )
            cleaned[self.TARGET_COLUMN] = parsed

        # 4. Normalización de cadenas de texto
        for col in self.CATEGORICAL_COLUMNS:
            cleaned[col] = cleaned[col].astype("string").str.strip()
            # Corregir posibles problemas de codificación comunes en 'Pequeño'
            if col == "talla":
                cleaned[col] = cleaned[col].replace(
                    {"Pequeo": "Pequeño", "Pequeno": "Pequeño"}
                )
            cleaned.loc[cleaned[col].eq(""), col] = pd.NA

        # 5. Reglas de plausibilidad clínica y no nulidad estructural
        self._validate_required_non_null(cleaned, require_target=require_target)
        self._validate_ranges_and_binary(cleaned, require_target=require_target)
        self._validate_controlled_categories(cleaned)

        columns_to_check = (
            self.REQUIRED_COLUMNS if require_target else self.FEATURE_COLUMNS
        )
        report = DataQualityReport(
            row_count=int(len(cleaned)),
            missing_by_column={
                col: int(cleaned[col].isna().sum()) for col in columns_to_check
            },
            coercions_to_missing=coercions_to_missing,
            unique_locations=int(
                cleaned[self.GROUP_COLUMN].nunique(dropna=True)
            ),
            is_valid=True,
        )

        return cleaned, report

    def _validate_schema(
        self, data: pd.DataFrame, require_target: bool = True
    ) -> None:
        """Verifica la presencia de todas las columnas obligatorias en el DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a inspeccionar.
        require_target : bool, default=True
            Si se debe exigir la presencia de la columna objetivo.

        Raises
        ------
        CDVValidationError
            Si el DataFrame está vacío o faltan columnas esperadas.
        """
        if data.empty:
            raise CDVValidationError("El dataset clínico está vacío.")

        expected = set(
            self.REQUIRED_COLUMNS if require_target else self.FEATURE_COLUMNS
        )
        missing = sorted(expected - set(data.columns))

        if missing:
            raise CDVValidationError(
                f"Faltan columnas requeridas en el dataset: {', '.join(missing)}"
            )

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        """Convierte series a formato numérico gestionando formatos internacionales.

        Parameters
        ----------
        series : pd.Series
            Serie con números o cadenas de texto numéricas.

        Returns
        -------
        pd.Series
            Serie convertida a tipo float/int con NaNs en valores ilegibles.
        """
        def normalize(val: Any) -> Any:
            if pd.isna(val) or isinstance(val, (int, float, np.number)):
                return val

            text = str(val).strip().replace(" ", "").replace("\u00a0", "")
            if not text:
                return np.nan

            if "," in text and "." in text:
                if text.rfind(",") > text.rfind("."):
                    return text.replace(".", "").replace(",", ".")
                return text.replace(",", "")

            return text.replace(",", ".")

        return pd.to_numeric(series.map(normalize), errors="coerce")

    def _validate_required_non_null(
        self, data: pd.DataFrame, require_target: bool = True
    ) -> None:
        """Verifica que no existan valores nulos en columnas estructurales.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame validado.
        require_target : bool, default=True
            Si se evalúa la no nulidad del target.

        Raises
        ------
        CDVValidationError
            Si existen valores nulos en ubicación o target.
        """
        non_nullable = [self.GROUP_COLUMN]
        if require_target:
            non_nullable.append(self.TARGET_COLUMN)

        failures = [col for col in non_nullable if data[col].isna().any()]
        if failures:
            raise CDVValidationError(
                f"No se admiten valores nulos en columnas estructurales: {', '.join(failures)}"
            )

    def _validate_ranges_and_binary(
        self, data: pd.DataFrame, require_target: bool = True
    ) -> None:
        """Verifica rangos fisiológicos y consistencia de banderas binarias {0, 1}.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame validado.
        require_target : bool, default=True
            Si se evalúa el target.

        Raises
        ------
        CDVValidationError
            Si un valor numérico o signo clínico se sale de los rangos válidos.
        """
        violations: List[str] = []

        # Rango de edad
        for col, (min_val, max_val) in self.CLINICAL_RANGES.items():
            invalid = data[col].notna() & ~data[col].between(min_val, max_val)
            if invalid.any():
                violations.append(
                    f"{col}: {int(invalid.sum())} registros fuera del rango [{min_val}, {max_val}]"
                )

        # Banderas binarias de signos
        binary_cols = list(self.BINARY_SIGNS_COLUMNS)
        if require_target:
            binary_cols.append(self.TARGET_COLUMN)

        for col in binary_cols:
            valid_vals = data[col].dropna()
            invalid = ~valid_vals.isin(self.ALLOWED_BINARY_VALUES)
            if invalid.any():
                violations.append(
                    f"{col}: contiene valores distintos a 0 y 1 ({int(invalid.sum())} violaciones)"
                )

        if violations:
            raise CDVValidationError("; ".join(violations))

    def _validate_controlled_categories(self, data: pd.DataFrame) -> None:
        """Verifica categorías admisibles en variables cerradas (sexo, talla, vacunas).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame validado.

        Raises
        ------
        CDVValidationError
            Si se detectan valores no permitidos en variables cerradas.
        """
        violations: List[str] = []

        for col, allowed in self.ALLOWED_CATEGORIES.items():
            if col in data.columns:
                invalid = data[col].notna() & ~data[col].isin(allowed)
                if invalid.any():
                    invalid_samples = data.loc[invalid, col].unique().tolist()
                    violations.append(
                        f"{col}: valores no permitidos {invalid_samples}. Permitidos: {allowed}"
                    )

        if violations:
            raise CDVValidationError("; ".join(violations))


# Alias de clase para máxima compatibilidad
DataValidator = CDVDataValidator