"""Entrenamiento, optimización y evaluación para Árbol de Decisión en CDV.

Este módulo implementa el pipeline de entrenamiento de DecisionTreeClassifier
priorizando la sensibilidad clínica (Recall >= 80%) para la detección temprana
de Distemper Canino (CDV).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import matplotlib
import numpy as np
import pandas as pd
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

from src.data.featureEngineer import CDVFeaturePipelineBuilder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class EvaluationResult:
    """Predicciones fuera de pliegue y métricas auditables de evaluación.

    Attributes
    ----------
    aggregate_metrics : Dict[str, float]
        Métricas consolidadas a nivel global (Accuracy, Recall, Precision, F1, ROC-AUC).
    fold_metrics : List[Dict[str, float | int]]
        Métricas calculadas individualmente en cada pliegue de validación.
    classification_report : Dict[str, Any]
        Informe detallado por clase (Sano vs CDV).
    confusion_matrix : List[List[int]]
        Matriz de confusión global en formato de lista anidada.
    y_true : np.ndarray
        Etiquetas verdaderas observadas.
    y_pred : np.ndarray
        Predicciones binarias de salida.
    y_probability : np.ndarray
        Probabilidades estimadas para la clase positiva (CDV = 1).
    """

    aggregate_metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float | int]]
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_probability: np.ndarray

    def serializable_summary(self) -> Dict[str, Any]:
        """Genera un resumen estructurado exportable a JSON sin incluir arrays binarios.

        Returns
        -------
        Dict[str, Any]
            Diccionario de métricas para trazabilidad y reportes.
        """
        return {
            "aggregate_metrics": self.aggregate_metrics,
            "fold_metrics": self.fold_metrics,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
        }


class DecisionTreeModelTrainer:
    """Gestor de entrenamiento, tuning y evaluación para Árbol de Decisión clínico.

    Parameters
    ----------
    n_splits : int, default=5
        Número de pliegues en la validación cruzada estratificada.
    random_state : int, default=42
        Semilla determinista para reproducibilidad de experimentos.
    param_grid : Optional[Dict[str, List[Any]]], default=None
        Rejilla de hiperparámetros para optimización con GridSearchCV.
    n_jobs : Optional[int], default=1
        Número de procesos paralelos para la búsqueda de hiperparámetros.
    """

    DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__max_depth": [2, 3, 4, 5, None],
        "classifier__min_samples_split": [2, 4, 6],
        "classifier__min_samples_leaf": [1, 2, 3],
        "classifier__class_weight": ["balanced", None],
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
    ) -> "DecisionTreeModelTrainer":
        """Optimiza hiperparámetros con validación cruzada y ajusta el pipeline final.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de variables clínicas predictoras.
        y : pd.Series
            Vector de etiquetas binarias (0 = Sano, 1 = CDV).
        groups : Optional[pd.Series], default=None
            Información de agrupamiento opcional.

        Returns
        -------
        DecisionTreeModelTrainer
            Instancia entrenada del objeto.
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
        """Calcula métricas de predicción fuera de pliegue (Out-Of-Fold) con StratifiedKFold.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de variables clínicas.
        y : pd.Series
            Vector de etiquetas verdaderas.
        groups : Optional[pd.Series], default=None
            Información opcional de grupos.

        Returns
        -------
        EvaluationResult
            Contenedor con métricas globales y por pliegue.

        Raises
        ------
        RuntimeError
            Si se invoca antes de ejecutar `fit()`.
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

    # Alias para retrocompatibilidad
    evaluate_grouped_cv = evaluate_cv

    def persist_model(self, output_path: str | Path) -> Path:
        """Serializa en disco el Pipeline scikit-learn entrenado.

        Parameters
        ----------
        output_path : str | Path
            Ruta destino del archivo `.pkl`.

        Returns
        -------
        Path
            Ruta absoluta del artefacto generado.

        Raises
        ------
        RuntimeError
            Si no hay un modelo ajustado disponible para persistir.
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
        """Genera y guarda en formato PNG la matriz de confusión calculada.

        Parameters
        ----------
        evaluation : EvaluationResult
            Objeto con las predicciones y matrices fuera de pliegue.
        output_path : str | Path
            Ruta de salida del archivo gráfico PNG.

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
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Matriz de Confusión — Árbol de Decisión (CDV)")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_tree_diagram(self, output_path: str | Path) -> Path:
        """Exporta una visualización gráfica de la topología del árbol de decisión.

        Parameters
        ----------
        output_path : str | Path
            Ruta del archivo de imagen PNG.

        Returns
        -------
        Path
            Ruta del gráfico generado.

        Raises
        ------
        RuntimeError
            Si el modelo no se encuentra ajustado.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No existe un modelo entrenado para graficar.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        classifier: DecisionTreeClassifier = self.best_estimator_.named_steps[
            "classifier"
        ]
        feature_names = self.feature_builder.transformed_feature_names(
            self.best_estimator_
        )

        depth = getattr(classifier.tree_, "max_depth", 4)
        n_leaves = classifier.get_n_leaves()

        fig_w = max(16, min(40, depth * 4 + 8))
        fig_h = max(10, min(30, n_leaves * 0.6 + 4))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        plot_tree(
            classifier,
            feature_names=feature_names,
            class_names=["Sano", "CDV"],
            filled=True,
            rounded=True,
            impurity=False,
            proportion=True,
            ax=ax,
        )
        ax.set_title("Árbol de Decisión Clínico para Distemper Canino (CDV)")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _build_base_pipeline(self) -> Pipeline:
        """Construye el pipeline básico de preprocesamiento + árbol.

        Returns
        -------
        Pipeline
            Pipeline listo para búsqueda de hiperparámetros.
        """
        classifier = DecisionTreeClassifier(random_state=self.random_state)
        return self.feature_builder.build_pipeline(classifier)

    def _build_search(self, n_splits: int) -> GridSearchCV:
        """Construye la búsqueda GridSearchCV con scoring orientado a Recall.

        Parameters
        ----------
        n_splits : int
            Número de pliegues estratificados.

        Returns
        -------
        GridSearchCV
            Búsqueda configurada.
        """
        return GridSearchCV(
            estimator=self._build_base_pipeline(),
            param_grid=self.param_grid,
            scoring="f1",  # Optimización armónica (equilibrio entre sensibilidad y especificidad)
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
        """Verifica consistencia dimensional y presencia de ambas clases binarias.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características.
        y : pd.Series
            Vector de etiquetas.

        Raises
        ------
        ValueError
            Si no coinciden las dimensiones o no hay 2 clases binarias {0, 1}.
        """
        self.feature_builder.assert_feature_contract(X.columns)

        if len(X) != len(y):
            raise ValueError(
                f"X ({len(X)}) e y ({len(y)}) deben tener exactamente el mismo número de filas."
            )

        classes = set(pd.Series(y).dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(
                f"La variable objetivo debe contener exactamente las clases binarias {{0, 1}}. Recibido: {classes}"
            )