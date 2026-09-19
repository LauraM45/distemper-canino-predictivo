"""Entrenamiento, optimización y evaluación para Ensamble Random Forest en CDV.

Este módulo implementa el pipeline de Random Forest Classifier para diagnóstico
clínico de Distemper Canino (CDV), integrando selección de hiperparámetros por Recall
y análisis de importancia de características clínicas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.featureEngineer import CDVFeaturePipelineBuilder
from src.models.DecisionTreeTraining import EvaluationResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RandomForestModelTrainer:
    """Gestor de entrenamiento y evaluación para ensamble Random Forest en CDV.

    Parameters
    ----------
    n_splits : int, default=5
        Número de pliegues en la validación cruzada estratificada.
    random_state : int, default=42
        Semilla determinista para reproducibilidad.
    param_grid : Optional[Dict[str, List[Any]]], default=None
        Espacio de búsqueda de hiperparámetros.
    n_jobs : Optional[int], default=1
        Número de hilos de procesamiento paralelo.
    """

    DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 4, None],
        "classifier__min_samples_split": [2, 4],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_jobs: Optional[int] = 1,
    ) -> None:
        self.n_splits = n_splits
        self.random_state = random_state
        self.param_grid = param_grid or self.DEFAULT_PARAM_GRID
        self.n_jobs = n_jobs

        self.feature_builder = CDVFeaturePipelineBuilder()
        self.best_estimator_: Optional[Pipeline] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.tuning_results_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> "RandomForestModelTrainer":
        """Ajusta y optimiza el ensamble Random Forest mediante GridSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características predictoras.
        y : pd.Series
            Vector de etiquetas binarias.
        groups : Optional[pd.Series], default=None
            Información de grupos opcional.

        Returns
        -------
        RandomForestModelTrainer
            Instancia entrenada.
        """
        self._validate_training_inputs(X, y)

        search = self._build_search(n_splits=self.n_splits)
        search.fit(X, y)

        self.best_estimator_ = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_score_ = float(search.best_score_)
        self.tuning_results_ = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score"
        )

        return self

    def evaluate_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> EvaluationResult:
        """Evalúa el ensamble fuera de pliegue (OOF) mediante StratifiedKFold.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de variables clínicas.
        y : pd.Series
            Vector de etiquetas reales.
        groups : Optional[pd.Series], default=None
            Grupos opcionales.

        Returns
        -------
        EvaluationResult
            Métricas consolidadas y por pliegue.

        Raises
        ------
        RuntimeError
            Si no se ha invocado `fit()` con antelación.
        """
        self._validate_training_inputs(X, y)

        if self.best_params_ is None:
            raise RuntimeError("Debe ejecutar fit() antes de evaluate_cv().")

        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        y_true = y.to_numpy(dtype=int)
        y_pred = np.empty(shape=len(X), dtype=int)
        y_probability = np.full(shape=len(X), fill_value=np.nan, dtype=float)

        fold_metrics: List[Dict[str, float | int]] = []

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y), start=1
        ):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            inner_search = self._build_search(
                n_splits=min(self.n_splits, int(y_train.value_counts().min()))
            )
            inner_search.fit(X_train, y_train)

            fold_pipeline = inner_search.best_estimator_

            preds = fold_pipeline.predict(X_test).astype(int)
            probs = fold_pipeline.predict_proba(X_test)[:, 1]

            y_pred[test_idx] = preds
            y_probability[test_idx] = probs

            fold_metrics.append(
                {
                    "fold": fold,
                    "test_rows": int(len(test_idx)),
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, preds)
                    ),
                    "precision": float(
                        precision_score(y_test, preds, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_test, preds, zero_division=0)
                    ),
                    "f1": float(f1_score(y_test, preds, zero_division=0)),
                    "inner_best_score": float(inner_search.best_score_),
                }
            )

        cm_agg = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn_agg = cm_agg[0, 0]
        fp_agg = cm_agg[0, 1]
        agg_specificity = float(tn_agg / (tn_agg + fp_agg)) if (tn_agg + fp_agg) > 0 else 0.0

        aggregate = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": agg_specificity,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if len(np.unique(y_true)) == 2 and not np.isnan(y_probability).any():
            aggregate["roc_auc"] = float(roc_auc_score(y_true, y_probability))

        return EvaluationResult(
            aggregate_metrics=aggregate,
            fold_metrics=fold_metrics,
            classification_report=classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["Sano", "CDV"],
                output_dict=True,
                zero_division=0,
            ),
            confusion_matrix=confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist(),
            y_true=y_true,
            y_pred=y_pred,
            y_probability=y_probability,
        )

    def persist_model(self, output_path: str | Path) -> Path:
        """Guarda en disco el pipeline de Random Forest serializado.

        Parameters
        ----------
        output_path : str | Path
            Ruta de destino del archivo `.pkl`.

        Returns
        -------
        Path
            Ruta absoluta del archivo guardado.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No existe un estimador entrenado para persistir.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_estimator_, path)
        return path

    def save_confusion_matrix(
        self,
        evaluation: EvaluationResult,
        output_path: str | Path,
    ) -> Path:
        """Guarda la matriz de confusión del ensamble en formato PNG.

        Parameters
        ----------
        evaluation : EvaluationResult
            Resultados de evaluación.
        output_path : str | Path
            Ruta de destino de la imagen.

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(evaluation.confusion_matrix),
            display_labels=["Sano (0)", "CDV (1)"],
        )
        display.plot(ax=ax, cmap="Greens", colorbar=False)
        ax.set_title("Matriz de Confusión — Random Forest (CDV)")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_feature_importance(self, output_path: str | Path) -> Path:
        """Exporta un gráfico de barras horizontales con las variables de mayor peso.

        Parameters
        ----------
        output_path : str | Path
            Ruta de destino del gráfico PNG.

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No existe un modelo entrenado para calcular importancias.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        classifier: RandomForestClassifier = self.best_estimator_.named_steps[
            "classifier"
        ]
        feature_names = np.array(
            self.feature_builder.transformed_feature_names(self.best_estimator_)
        )
        importances = classifier.feature_importances_

        sorted_idx = np.argsort(importances)[-15:]  # Top 15 características

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            range(len(sorted_idx)),
            importances[sorted_idx],
            color="#2ca02c",
            align="center",
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(feature_names[sorted_idx])
        ax.set_xlabel("Importancia Relativa (Gini Importance)")
        ax.set_title("Top 15 Características Más Relevantes — Random Forest (CDV)")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _build_base_pipeline(self) -> Pipeline:
        classifier = RandomForestClassifier(random_state=self.random_state)
        return self.feature_builder.build_pipeline(classifier)

    def _build_search(self, n_splits: int) -> GridSearchCV:
        return GridSearchCV(
            estimator=self._build_base_pipeline(),
            param_grid=self.param_grid,
            scoring="f1",
            cv=StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=True,
            error_score="raise",
        )

    def _validate_training_inputs(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        self.feature_builder.assert_feature_contract(X.columns)
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma cantidad de filas.")
        classes = set(pd.Series(y).dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError("y debe contener exactamente las clases {0, 1}.")
