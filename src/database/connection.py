"""Gestor de conexión y sesiones para PostgreSQL (Neon DB) con SQLAlchemy.

Optimizado para bases de datos serverless: incluye pool_pre_ping,
reciclado de conexiones y reconexión automática tras hibernación.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Cargar variables de entorno desde .env en la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)


def get_database_url() -> str:
    """Obtiene y normaliza la URL de conexión a la base de datos."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise ValueError(
            "La variable de entorno 'DATABASE_URL' no está configurada o está vacía.\n"
            "Por favor, agregue su cadena de conexión de Neon en el archivo '.env':\n"
            "DATABASE_URL=postgresql://usuario:password@host/neondb?sslmode=require"
        )

    # SQLAlchemy 2.0 requiere que comience con 'postgresql://' en lugar del viejo 'postgres://'
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    return db_url


_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker[Session]] = None


def get_engine() -> Engine:
    """Retorna la instancia global del motor SQLAlchemy (Singleton con pool resiliente)."""
    global _engine, _SessionFactory
    if _engine is None:
        db_url = get_database_url()

        # Configuración adaptada a Neon Serverless:
        # pool_pre_ping: Verifica la salud del socket antes de ejecutar consultas (maneja sleep/wake de Neon)
        # pool_recycle: Recicla conexiones inactivas para evitar timeouts de red TCP
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Retorna la fábrica de sesiones de SQLAlchemy."""
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Generador de contexto para transacciones seguras en base de datos.

    Yields
    ------
    Session
        Sesión activa de SQLAlchemy. Hace commit automático en caso de éxito
        o rollback automático en caso de excepción.
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> dict:
    """Verifica la conectividad activa contra la base de datos.

    Returns
    -------
    dict
        Diccionario con el estado de la conexión, versión de PostgreSQL
        y nombre de la base de datos activa.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            version_result = conn.execute(text("SELECT version();")).scalar()
            db_name_result = conn.execute(text("SELECT current_database();")).scalar()
            return {
                "status": "success",
                "message": "Conexión establecida exitosamente con Neon DB.",
                "database": db_name_result,
                "version": version_result,
            }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Error al conectar con la base de datos: {exc}",
        }
