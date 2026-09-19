"""Ejecución puntual para entrenar y persistir el Pipeline de Árbol de Decisión para CDV."""

from __future__ import annotations

import argparse
from pathlib import Path

from main import run

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_moquillo_real_v5.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "src" / "data" / "artifacts"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "demo_reports"


def parse_args() -> argparse.Namespace:
    """Obtiene parámetros de ejecución por línea de comandos.

    Returns
    -------
    argparse.Namespace
        Argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Entrena y persiste el Pipeline para Distemper Canino (CDV)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ruta al CSV de entrenamiento de CDV.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directorio de salida del Pipeline serializado.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directorio de matrices, diagramas y reporte JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data,
        artifacts_dir=args.artifacts_dir,
        reports_dir=args.reports_dir,
    )