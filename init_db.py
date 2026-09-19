"""Script de inicialización, verificación y migración de base de datos Neon DB.

Crea las tablas definidas en los modelos ORM y opcionalmente puebla
la base de datos con los 120 registros históricos del dataset clínico.

Uso:
----
python init_db.py --check-only    # Solo prueba la conexión con Neon DB
python init_db.py --create-tables # Crea las tablas en PostgreSQL
python init_db.py --seed          # Crea tablas e inserta los 120 registros iniciales
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
import pandas as pd
from sqlalchemy import func, select

from predict import CDVInferenceEngine
from src.database.connection import check_database_connection, get_db, get_engine
from src.database.models import Base, PacienteCDV

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CSV = PROJECT_ROOT / "data" / "raw" / "dataset_moquillo_real_v5.csv"


def check_connection() -> bool:
    """Verifica si las credenciales en .env son validas y alcanzan la base de datos."""
    print("\n[INFO] Verificando conectividad con la base de datos Neon...")
    info = check_database_connection()
    if info["status"] == "success":
        print("[OK] Conexion exitosa:")
        print(f"   - Base de Datos: {info['database']}")
        print(f"   - Servidor: {info['version']}")
        return True
    else:
        print(f"[ERROR] Error de conexion:\n{info['message']}\n")
        print("[AYUDA] Asegurese de haber configurado la variable DATABASE_URL en su archivo '.env'")
        return False


def create_schema() -> None:
    """Crea todas las tablas definidas en Base.metadata si no existen."""
    print("\n[INFO] Creando tablas en la base de datos...")
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("[OK] Tabla 'pacientes_cdv' verificada / creada exitosamente en PostgreSQL.")


def seed_database(force: bool = False) -> None:
    """Inserta los 120 registros historicos procesandolos con el motor de inferencia."""
    if not DATASET_CSV.is_file():
        print(f"[ERROR] No se encontro el dataset historico en: {DATASET_CSV}")
        return

    engine = get_engine()
    create_schema()

    with get_db() as session:
        # Verificar conteo actual
        conteo_existente = session.scalar(select(func.count(PacienteCDV.id))) or 0
        if conteo_existente > 0 and not force:
            print(
                f"[INFO] La base de datos ya contiene {conteo_existente} pacientes registrados. "
                "Use '--force' si desea duplicar o resembrar."
            )
            return

        print(f"\n[INFO] Poblando base de datos desde '{DATASET_CSV.name}'...")
        df = pd.read_csv(DATASET_CSV)

        # Cargar motor de inferencia clinica
        inference_engine = CDVInferenceEngine()

        registros_a_insertar = []
        for idx, row in df.iterrows():
            patient_dict = {
                "edad_meses": float(row["edad_meses"]),
                "sexo": str(row["sexo"]),
                "raza": str(row["raza"]),
                "talla": str(row["talla"]),
                "ubicacion_procedencia": str(row["ubicacion_procedencia"]),
                "estado_vacunal": str(row["estado_vacunal"]),
                "fiebre_hipertermia": int(row["fiebre_hipertermia"]),
                "signos_respiratorios_oculonasales": int(row["signos_respiratorios_oculonasales"]),
                "signos_digestivos": int(row["signos_digestivos"]),
                "signos_neurologicos": int(row["signos_neurologicos"]),
                "signos_dermatologicos": int(row["signos_dermatologicos"]),
            }

            # Evaluar con el modelo para generar prediccion y probabilidad
            eval_result = inference_engine.predict_patient(patient_dict)

            paciente_orm = PacienteCDV(
                id=uuid.uuid4(),
                edad_meses=patient_dict["edad_meses"],
                sexo=patient_dict["sexo"],
                raza=patient_dict["raza"],
                talla=patient_dict["talla"],
                ubicacion_procedencia=patient_dict["ubicacion_procedencia"],
                estado_vacunal=patient_dict["estado_vacunal"],
                fiebre_hipertermia=patient_dict["fiebre_hipertermia"],
                signos_respiratorios_oculonasales=patient_dict["signos_respiratorios_oculonasales"],
                signos_digestivos=patient_dict["signos_digestivos"],
                signos_neurologicos=patient_dict["signos_neurologicos"],
                signos_dermatologicos=patient_dict["signos_dermatologicos"],
                prediccion_cdv=eval_result["prediction"],
                probabilidad_cdv=eval_result["probability_cdv"],
                probabilidad_sano=eval_result["probability_sano"],
                nivel_riesgo=eval_result["risk_level"],
                accion_clinica=eval_result["clinical_action"],
                modelo_version=eval_result["model_metadata"]["model_version"],
                diagnostico_laboratorio_confirmado=int(row["diagnostico_cdv_confirmado"]),
                observaciones_veterinario="Registro historico del estudio clinico de referencia.",
                origen_registro="dataset_historico",
            )
            registros_a_insertar.append(paciente_orm)

        session.add_all(registros_a_insertar)
        print(f"[OK] Se sembraron exitosamente {len(registros_a_insertar)} pacientes con UUIDs unicos.")

    # Conteo final
    with get_db() as session:
        total = session.scalar(select(func.count(PacienteCDV.id)))
        print(f"[ESTADISTICAS] Total de registros actuales en 'pacientes_cdv': {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inicialización y gestión de base de datos Neon.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verifica la conexión con Neon DB y sale.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Crea las tablas en PostgreSQL si no existen.",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Crea tablas e inserta los 120 registros históricos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la inserción de registros aunque la tabla ya contenga datos.",
    )

    args = parser.parse_args()

    # Si no se pasa ningún argumento, el comportamiento por defecto es verificar y sembrar
    if not (args.check_only or args.create_tables or args.seed):
        args.seed = True

    if not check_connection():
        sys.exit(1)

    if args.check_only:
        sys.exit(0)

    if args.create_tables and not args.seed:
        create_schema()
    elif args.seed:
        seed_database(force=args.force)


if __name__ == "__main__":
    main()
