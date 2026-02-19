from collections.abc import Generator
import logging

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

try:
    from pgvector.psycopg import register_vector
except ModuleNotFoundError:
    register_vector = None  # type: ignore[assignment]

settings = get_settings()
logger = logging.getLogger(__name__)


def _build_engine():
    try:
        return create_engine(settings.database_url, pool_pre_ping=True)
    except ModuleNotFoundError:
        # Allows running tests without the postgres driver installed locally.
        logger.warning("Falling back to sqlite engine because psycopg is unavailable.")
        return create_engine("sqlite+pysqlite:///:memory:", pool_pre_ping=True)


engine = _build_engine()


@event.listens_for(engine, "connect")
def connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
    if register_vector is not None:
        try:
            register_vector(dbapi_connection)
        except Exception:
            logger.debug("Skipping vector registration for current DB driver.")


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def get_db_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
