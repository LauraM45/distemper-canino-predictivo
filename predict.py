"""Motor de inferencia clínica para Distemper Canino (CDV).

Este módulo permite cargar el pipeline serializado y evaluar pacientes caninos
de forma unitaria o por lotes, aplicando validación estricta de esquema y
resiliencia ante categorías no observadas (coldstart).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.dataValidation import CDVDataValidator, CDVValidationError, ModelMetadata

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "src" / "data" / "artifacts" / "cdv_randomforest_pipeline.pkl"


class CDVInferenceEngine:
    """Motor de inferencia para diagnóstico predictivo de Distemper Canino (CDV).

    Parameters
    ----------
    model_path : Union[str, Path], default=DEFAULT_MODEL_PATH
        Ruta al archivo `.pkl` con el pipeline entrenado de scikit-learn.

    Attributes
    ----------
    pipeline : Pipeline
        Pipeline cargado con preprocesador y clasificador.
    validator : CDVDataValidator
        Instancia para validación contractual de los pacientes.
    metadata : ModelMetadata
        Metadatos de especificación clínica.
    """

    def __init__(self, model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en: {self.model_path}. "
                "Ejecute 'main.py' primero para entrenar y persistir el modelo."
            )

        self.pipeline: Pipeline = joblib.load(self.model_path)
        self.validator = CDVDataValidator()
        self.metadata = ModelMetadata()

    def predict_patient(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa un único paciente canino y devuelve su diagnóstico y nivel de riesgo.

        Parameters
        ----------
        patient : Dict[str, Any]
            Diccionario con las 11 variables clínicas del paciente.

        Returns
        -------
        Dict[str, Any]
            Diagnóstico binario, etiqueta clínica, probabilidad estimada,
            categoría de riesgo e interpretación veterinaria.

        Raises
        ------
        CDVValidationError
            Si faltan variables clínicas o los valores están fuera de rango.
        """
        # Convertir a DataFrame unitario
        df_patient = pd.DataFrame([patient])

        # Validar contrato clínico (sin requerir columna de target)
        clean_patient, quality_report = self.validator.validate_and_clean(
            df_patient, require_target=False
        )

        X = clean_patient[list(CDVDataValidator.FEATURE_COLUMNS)]

        # Predicción y probabilidad
        prediction = int(self.pipeline.predict(X)[0])
        probabilities = self.pipeline.predict_proba(X)[0]

        prob_sano = float(probabilities[0])
        prob_cdv = float(probabilities[1])

        # Asignación de nivel de riesgo clínico
        if prob_cdv >= 0.70:
            risk_level = "Alto Riesgo"
            clinical_action = (
                "Aislamiento preventivo estricto, inicio inmediato de soporte "
                "neurológico/sistémico y toma urgente de prueba molecular (PCR / Antígeno)."
            )
        elif prob_cdv >= 0.40:
            risk_level = "Riesgo Moderado"
            clinical_action = (
                "Monitoreo cercano, aislamiento preventivo y correlación con biometría hemática."
            )
        else:
            risk_level = "Bajo Riesgo"
            clinical_action = "Continuar protocolo de medicina preventiva y seguimiento regular."

        return {
            "prediction": prediction,
            "diagnosis": "Positivo para Distemper Canino (CDV)" if prediction == 1 else "Sano / Negativo para CDV",
            "probability_cdv": round(prob_cdv, 4),
            "probability_sano": round(prob_sano, 4),
            "risk_level": risk_level,
            "clinical_action": clinical_action,
            "patient_evaluated": patient,
            "model_metadata": {
                "model_name": self.metadata.model_name,
                "model_version": self.metadata.model_version,
                "target_disease": self.metadata.target_disease,
                "required_recall": self.metadata.min_required_recall,
            },
        }

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evalúa un lote de pacientes caninos a partir de un DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame con las 11 variables clínicas requeridas.

        Returns
        -------
        pd.DataFrame
            DataFrame original enriquecido con columnas de predicción,
            probabilidad de CDV y nivel de riesgo.
        """
        clean_data, _ = self.validator.validate_and_clean(data, require_target=False)
        X = clean_data[list(CDVDataValidator.FEATURE_COLUMNS)]

        predictions = self.pipeline.predict(X).astype(int)
        probabilities = self.pipeline.predict_proba(X)[:, 1]

        result = data.copy()
        result["cdv_prediccion"] = predictions
        result["cdv_diagnostico"] = np.where(
            predictions == 1, "Positivo (CDV)", "Sano"
        )
        result["cdv_probabilidad"] = np.round(probabilities, 4)
        result["cdv_nivel_riesgo"] = np.where(
            probabilities >= 0.70,
            "Alto Riesgo",
            np.where(probabilities >= 0.40, "Riesgo Moderado", "Bajo Riesgo"),
        )
        return result


def parse_args() -> argparse.Namespace:
    """Configura los argumentos de línea de comandos para inferencia."""
    parser = argparse.ArgumentParser(
        description="Motor de Inferencia Clínica para Distemper Canino (CDV)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta al Pipeline serializado (.pkl).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Ruta a un archivo CSV con pacientes a diagnosticar.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Ruta de salida para los resultados de predicción por lote.",
    )
    parser.add_argument(
        "--coldstart-test",
        action="store_true",
        help="Ejecuta una prueba con categorías totalmente desconocidas (raza/ubicación nuevas).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal para inferencia clínica."""
    args = parse_args()
    engine = CDVInferenceEngine(model_path=args.model_path)

    if args.input_csv:
        if not args.input_csv.is_file():
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {args.input_csv}")

        data = pd.read_csv(args.input_csv)
        results = engine.predict_batch(data)

        if args.output_csv:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.output_csv, index=False)
            print(f"Predicciones por lote guardadas exitosamente en: {args.output_csv}")
        else:
            print(results[["edad_meses", "raza", "cdv_diagnostico", "cdv_probabilidad", "cdv_nivel_riesgo"]])
        return

    if args.coldstart_test:
        print("\n=== Prueba de Arranque en Frío (Coldstart / Categorías Nuevas) ===")
        unseen_patient = {
            "edad_meses": 5.0,
            "sexo": "Macho",
            "raza": "Shiba Inu",  # Categoría nunca vista en entrenamiento
            "talla": "Gigante",    # Categoría nunca vista en entrenamiento
            "ubicacion_procedencia": "Hospital Veterinario Medellin",  # Nueva ubicación
            "estado_vacunal": "No vacunado",
            "fiebre_hipertermia": 1,
            "signos_respiratorios_oculonasales": 1,
            "signos_digestivos": 0,
            "signos_neurologicos": 1,
            "signos_dermatologicos": 0,
        }
        result = engine.predict_patient(unseen_patient)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Por defecto: Evaluar el registro de referencia oficial
    print("\n=== Evaluación del Registro Clínico de Referencia ===")
    ref_patient = {
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
    }
    result = engine.predict_patient(ref_patient)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
