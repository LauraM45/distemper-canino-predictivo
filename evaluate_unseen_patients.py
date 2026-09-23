"""Evaluación de generalización clínica en pacientes y ubicaciones no vistas (CDV).

Este script evalúa la capacidad del modelo para diagnosticar pacientes
provenientes de centros veterinarios no observados durante el entrenamiento
o con perfiles clínicos atípicos (prueba de arranque en frío / coldstart).
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data.dataValidation import CDVDataValidator
from src.models.DecisionTreeTraining import DecisionTreeModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset_moquillo_real_v7.csv"


def main() -> None:
    """Ejecuta la prueba de generalización y coldstart."""
    print("=== EVALUACIÓN DE GENERALIZACIÓN Y RESILIENCIA ANTI-COLDSTART (CDV) ===")

    # 1. Cargar y validar datos
    raw_data = pd.read_csv(DATASET_PATH)
    validator = CDVDataValidator()
    clean_data, _ = validator.validate_and_clean(raw_data)

    X = clean_data.loc[:, list(CDVDataValidator.FEATURE_COLUMNS)]
    y = clean_data[CDVDataValidator.TARGET_COLUMN].astype(int)

    # 2. Ajustar el modelo con el dataset disponible
    trainer = DecisionTreeModelTrainer(n_splits=5, random_state=42)
    trainer.fit(X=X, y=y)
    best_model = trainer.best_estimator_

    print(f"Muestras de entrenamiento: {len(clean_data)} pacientes")
    print(f"Mejores parámetros del tuning: {trainer.best_params_}\n")

    # 3. Pacientes de prueba con variables nunca antes vistas (Coldstart)
    unseen_cases = pd.DataFrame([
        {
            "edad_meses": 3.0,
            "sexo": "Macho",
            "raza": "Shiba Inu",  # Raza desconocida en entrenamiento
            "talla": "Gigante",    # Talla no observada
            "ubicacion_procedencia": "Hospital Veterinario Medellin",  # Nueva ciudad/centro
            "estado_vacunal": "No vacunado",
            "fiebre_hipertermia": 1,
            "signos_respiratorios_oculonasales": 1,
            "signos_digestivos": 1,
            "signos_neurologicos": 1,
            "signos_dermatologicos": 0,
        },
        {
            "edad_meses": 48.0,
            "sexo": "Hembra",
            "raza": "Samoyedo",   # Raza desconocida
            "talla": "Grande",
            "ubicacion_procedencia": "Clinica Veterinaria Cali",  # Nueva ubicación
            "estado_vacunal": "Completo",
            "fiebre_hipertermia": 0,
            "signos_respiratorios_oculonasales": 0,
            "signos_digestivos": 0,
            "signos_neurologicos": 0,
            "signos_dermatologicos": 0,
        },
    ])

    clean_unseen, _ = validator.validate_and_clean(unseen_cases, require_target=False)
    X_unseen = clean_unseen[list(CDVDataValidator.FEATURE_COLUMNS)]

    predictions = best_model.predict(X_unseen)
    probabilities = best_model.predict_proba(X_unseen)

    for idx, row in unseen_cases.iterrows():
        pred = predictions[idx]
        prob = probabilities[idx]
        diag = "Positivo para CDV" if pred == 1 else "Sano"
        print(f"Paciente #{idx + 1} ({row['raza']}, {row['ubicacion_procedencia']}):")
        print(f"  - Diagnóstico Predicho: {diag}")
        print(f"  - Probabilidades [Sano, CDV]: [{prob[0]:.4f}, {prob[1]:.4f}]")
        print(f"  - Estado de Coldstart: Atendido exitosamente sin excepciones.\n")


if __name__ == "__main__":
    main()