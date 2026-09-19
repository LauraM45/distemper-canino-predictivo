"""Construcción de pipeline de ingeniería de características para Distemper Canino (CDV).

Este módulo implementa la preparación, escalado, imputación y codificación
de variables clínicas, garantizando ausencia de fuga de datos (data leakage)
y alta resiliencia ante categorías no observadas en inferencia (arranque en frío / coldstart).
"""

from __future__ import annotations

from typing import Iterable, List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.dataValidation import CDVDataValidator


class CDVFeaturePipelineBuilder:
    """Constructor del transformador y pipeline scikit-learn para pacientes de CDV.

    Organiza el preprocesamiento modular dividido en tres ramas:
    1. Variables continuas (edad en meses): imputación por mediana y estandarización Z-score.
    2. Variables categóricas: imputación de categoría desconocida y One-Hot Encoding
       con agrupación de categorías infrecuentes (`min_frequency=2`) y derivación determinista
       de categorías nuevas hacia infrecuentes (`handle_unknown='infrequent_if_exist'`).
    3. Banderas clínicas binarias (signos): imputación con valor 0 (ausencia del signo).
    """

    numeric_features: tuple[str, ...] = CDVDataValidator.NUMERIC_COLUMNS
    categorical_features: tuple[str, ...] = CDVDataValidator.CATEGORICAL_COLUMNS
    binary_features: tuple[str, ...] = CDVDataValidator.BINARY_SIGNS_COLUMNS

    def build_preprocessor(self) -> ColumnTransformer:
        """Construye un ColumnTransformer especializado por tipología de variable clínica.

        Returns
        -------
        ColumnTransformer
            Transformador scikit-learn configurado y listo para ensamblar en pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Desconocido"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        min_frequency=2,
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, list(self.numeric_features)),
                ("cat", categorical_pipeline, list(self.categorical_features)),
                ("bin", binary_pipeline, list(self.binary_features)),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

    def build_pipeline(self, classifier: object) -> Pipeline:
        """Encapsula preprocesador y estimador en un único Pipeline scikit-learn.

        Parameters
        ----------
        classifier : object
            Estimador de clasificación compatible con la API de scikit-learn.

        Returns
        -------
        Pipeline
            Pipeline serializable e inmune a fuga de datos durante la validación cruzada.
        """
        return Pipeline(
            steps=[
                ("preprocessor", self.build_preprocessor()),
                ("classifier", classifier),
            ]
        )

    @staticmethod
    def transformed_feature_names(fitted_pipeline: Pipeline) -> List[str]:
        """Extrae los nombres ordenados de las variables transformadas tras el preprocesamiento.

        Parameters
        ----------
        fitted_pipeline : Pipeline
            Pipeline previamente ajustado mediante `fit()`.

        Returns
        -------
        List[str]
            Lista con los identificadores de columnas resultantes.
        """
        preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())

    @staticmethod
    def assert_feature_contract(columns: Iterable[str]) -> None:
        """Comprueba que la matriz de características cumpla con el contrato de entrada.

        Verifica que X incluya las 11 variables clínicas necesarias y no contenga
        la variable objetivo para prevenir fuga deliberada o accidental.

        Parameters
        ----------
        columns : Iterable[str]
            Nombres de las columnas de la matriz X.

        Raises
        ------
        ValueError
            Si falta alguna variable requerida o si se detecta la columna objetivo.
        """
        cols_set = set(columns)

        if CDVDataValidator.TARGET_COLUMN in cols_set:
            raise ValueError(
                f"La matriz de características no puede incluir la columna objetivo: "
                f"'{CDVDataValidator.TARGET_COLUMN}'"
            )

        missing = set(CDVDataValidator.FEATURE_COLUMNS) - cols_set
        if missing:
            raise ValueError(
                f"Faltan variables clínicas obligatorias en la entrada: {sorted(missing)}"
            )


# Alias para retrocompatibilidad
FeaturePipelineBuilder = CDVFeaturePipelineBuilder