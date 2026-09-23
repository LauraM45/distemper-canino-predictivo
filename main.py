"""Orquestador CRISP-ML(Q) para el Modelo Predictivo de Distemper Canino (CDV).

Este módulo centraliza la ingesta de datos clínicos, la validación contractual,
el entrenamiento y ajuste de hiperparámetros multi-algoritmo (Árbol de Decisión,
Random Forest y Regresión Logística), la evaluación rigurosa fuera de pliegue (OOF)
y la persistencia de artefactos de producción y reportes auditables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.dataValidation import CDVDataValidator, ModelMetadata
from src.models.DecisionTreeTraining import DecisionTreeModelTrainer
from src.models.RandomForest import RandomForestModelTrainer
from src.models.LogisticRegression import LogisticRegressionModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_moquillo_real_v7.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "src" / "data" / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "demo_reports"


def run(
    data_path: str | Path = DEFAULT_DATASET,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> Dict[str, Any]:
    """Ejecuta el ciclo de vida completo CRISP-ML(Q) para diagnóstico de CDV.

    Parameters
    ----------
    data_path : str | Path, default=DEFAULT_DATASET
        Ruta al archivo CSV con las observaciones clínicas de pacientes caninos.
    artifacts_dir : str | Path, default=DEFAULT_ARTIFACTS_DIR
        Directorio destino para guardar los pipelines binarios serializados (.pkl).
    reports_dir : str | Path, default=DEFAULT_REPORTS_DIR
        Directorio destino para almacenar matrices de confusión, gráficos y reportes JSON.

    Returns
    -------
    Dict[str, Any]
        Diccionario estructurado con el resumen de calidad, métricas comparativas
        y rutas de los artefactos generados.

    Raises
    ------
    FileNotFoundError
        Si el archivo de dataset de entrada no existe.
    """
    data_path = Path(data_path)
    artifacts_dir = Path(artifacts_dir)
    reports_dir = Path(reports_dir)

    if not data_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset en la ruta: {data_path}")

    # 1. Ingesta y Validación Contractual
    raw_data = pd.read_csv(data_path)
    validator = CDVDataValidator()
    clean_data, quality_report = validator.validate_and_clean(raw_data)

    X = clean_data.loc[:, list(CDVDataValidator.FEATURE_COLUMNS)]
    y = clean_data[CDVDataValidator.TARGET_COLUMN].astype(int)

    metadata = ModelMetadata()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(" SISTEMA PREDICTIVO DE DISTEMPER CANINO (CDV)")
    print(f" Versión del Modelo: {metadata.model_version}")
    print(f" Pacientes Cargados: {len(clean_data)} ({y.sum()} CDV positivos, {(y == 0).sum()} Sanos)")
    print("=======================================================\n")

    # 2. Instanciación de Entrenadores Multi-Modelo
    trainers = {
        "DecisionTree": DecisionTreeModelTrainer(n_splits=5, random_state=42),
        "RandomForest": RandomForestModelTrainer(n_splits=5, random_state=42),
        "LogisticRegression": LogisticRegressionModelTrainer(n_splits=5, random_state=42),
    }

    evaluations: Dict[str, Any] = {}
    model_paths: Dict[str, str] = {}
    plot_paths: Dict[str, str] = {}

    print("=== Entrenando y Optimizando Modelos (Métrica Objetivo: Recall >= 80%) ===")

    for name, trainer in trainers.items():
        print(f"\n[+] Procesando {name}...")
        trainer.fit(X, y)
        eval_result = trainer.evaluate_cv(X, y)
        evaluations[name] = eval_result

        rec = eval_result.aggregate_metrics["recall"]
        spec = eval_result.aggregate_metrics.get("specificity", 0.0)
        acc = eval_result.aggregate_metrics["accuracy"]
        f1 = eval_result.aggregate_metrics["f1"]
        print(f"    - Mejor F1 GridSearchCV      : {trainer.best_score_:.4f}")
        print(f"    - Recall OOF (Out-Of-Fold)   : {rec:.4f} (Meta >= 80%: {'CUMPLE' if rec >= 0.80 else 'NO CUMPLE'})")
        print(f"    - Especificidad OOF          : {spec:.4f}")
        print(f"    - F1-Score OOF               : {f1:.4f}")
        print(f"    - Exactitud (Accuracy)       : {acc:.4f}")
        print(f"    - Mejores Hiperparámetros    : {trainer.best_params_}")

        # Persistir pipelines individuales
        pkl_path = trainer.persist_model(
            artifacts_dir / f"cdv_{name.lower()}_pipeline.pkl"
        )
        model_paths[name] = str(pkl_path)

        # Generar matrices de confusión
        cm_path = trainer.save_confusion_matrix(
            eval_result,
            reports_dir / f"matriz_confusion_{name.lower()}.png",
        )
        plot_paths[f"cm_{name.lower()}"] = str(cm_path)

    # 3. Gráficos Específicos por Arquitectura
    # Diagrama de Árbol de Decisión
    tree_plot = trainers["DecisionTree"].save_tree_diagram(
        reports_dir / "arbol_decision_cdv.png"
    )
    plot_paths["tree_diagram"] = str(tree_plot)

    # Importancia de Variables en Random Forest
    rf_feat_plot = trainers["RandomForest"].save_feature_importance(
        reports_dir / "importancia_variables_random_forest.png"
    )
    plot_paths["rf_feature_importance"] = str(rf_feat_plot)

    # Coeficientes Clínicos en Regresión Logística
    lr_coef_plot = trainers["LogisticRegression"].save_coefficients_plot(
        reports_dir / "coeficientes_regresion_logistica.png"
    )
    plot_paths["lr_coefficients"] = str(lr_coef_plot)

    # 4. Consolidar Resumen Ejecutivo y Auditoría JSON
    summary: Dict[str, Any] = {
        "metadata": metadata.to_dict(),
        "data_quality": quality_report.to_dict(),
        "evaluation_protocol": (
            "Validación Cruzada Estratificada (StratifiedKFold con 5 pliegues) "
            "con optimización orientada a Recall (Sensibilidad) y protección anti-coldstart."
        ),
        "models_summary": {
            name: {
                "best_cv_recall": trainers[name].best_score_,
                "best_params": trainers[name].best_params_,
                "evaluation": evaluations[name].serializable_summary(),
                "meets_recall_threshold": evaluations[name].aggregate_metrics["recall"] >= metadata.min_required_recall,
            }
            for name in trainers
        },
        "artifacts": {
            "models": model_paths,
            "reports_and_plots": plot_paths,
        },
    }

    metrics_json_path = reports_dir / "reporte_metricas_cdv.json"
    metrics_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESUMEN COMPARATIVO FINAL (SENSIBILIDAD Y ESPECIFICIDAD) ===")
    for name, data in summary["models_summary"].items():
        recall_val = data["evaluation"]["aggregate_metrics"]["recall"]
        spec_val = data["evaluation"]["aggregate_metrics"].get("specificity", 0.0)
        status = "[CUMPLE >= 80%]" if data["meets_recall_threshold"] else "[NO CUMPLE]"
        print(f" - {name:20s}: Recall = {recall_val * 100:.1f}% | Especif = {spec_val * 100:.1f}% | {status}")

    print(f"\nReporte JSON auditable guardado en : {metrics_json_path}")
    print(f"Artefactos y gráficos guardados en : {reports_dir}")
    print("=======================================================\n")

    return summary


def parse_args() -> argparse.Namespace:
    """Configura la interfaz de línea de comandos para la ejecución del orquestador."""
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa los modelos de Distemper Canino (CDV)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ruta al archivo CSV con datos de entrenamiento.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directorio destino para guardar los pipelines serializados (.pkl).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directorio destino para guardar reportes y gráficos generados.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
    )