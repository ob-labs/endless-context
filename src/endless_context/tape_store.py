from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from republic.tape import InMemoryQueryMixin, TapeEntry
from sqlalchemy import URL, ForeignKey, Index, Integer, String, Text, create_engine, event, func, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL as SQLAlchemyURL
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import ArgumentError, IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.sql import select, update
from sqlalchemy.types import JSON, DateTime

DEFAULT_BUB_HOME = Path.home() / ".bub"


def _default_url(bub_home: Path) -> str:
    database_path = (bub_home.expanduser() / "tapes.db").resolve()
    return str(URL.create("sqlite+pysqlite", database=str(database_path)))


class TapeStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bub_home: Path = Field(default=DEFAULT_BUB_HOME, validation_alias="BUB_HOME")
    url: str | None = Field(default=None, validation_alias="BUB_TAPESTORE_SQLALCHEMY_URL")
    echo: bool = Field(default=False, validation_alias="BUB_TAPESTORE_SQLALCHEMY_ECHO")

    @classmethod
    def from_env(cls) -> TapeStoreSettings:
        return cls()

    @field_validator("bub_home", mode="before")
    @classmethod
    def _normalize_bub_home(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()

    @property
    def resolved_url(self) -> str:
        if self.url is None or not self.url.strip():
            return _default_url(self.bub_home)
        return self.url.strip()


class Base(DeclarativeBase):
    pass


class TapeRecord(Base):
    __tablename__ = "tapes"
    __table_args__ = (Index("uq_tapes_name_key", "name_key", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    last_entry_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class TapeEntryRecord(Base):
    __tablename__ = "tape_entries"
    __table_args__ = (
        Index("idx_tape_entries_kind", "tape_id", "kind", "entry_id"),
        Index("idx_tape_entries_anchor_name_key", "tape_id", "anchor_name_key", "entry_id"),
    )

    tape_id: Mapped[int] = mapped_column(ForeignKey("tapes.id", ondelete="CASCADE"), primary_key=True)
    entry_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    anchor_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    anchor_name_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    meta: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    entry_date: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SQLAlchemyTapeStore(InMemoryQueryMixin):
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._url = self._normalize_url(url)
        self._echo = echo
        self._write_lock = threading.RLock()
        self._engine = create_engine(
            self._url,
            echo=echo,
            future=True,
            pool_pre_ping=True,
            connect_args=self._connect_args(self._url),
        )
        self._configure_engine(self._engine)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False, class_=Session)
        Base.metadata.create_all(self._engine)
        self._validate_schema()

    def list_tapes(self) -> list[str]:
        with self._session_factory() as session:
            return list(session.scalars(select(TapeRecord.name).order_by(TapeRecord.name)).all())

    def reset(self, tape: str) -> None:
        with self._write_lock:
            with self._session_factory.begin() as session:
                tape_record = self._find_tape_record(session, tape)
                if tape_record is not None:
                    session.delete(tape_record)

    def read(self, tape: str) -> list[TapeEntry] | None:
        with self._session_factory() as session:
            tape_record = self._find_tape_record(session, tape)
            if tape_record is None:
                return None
            statement = (
                select(TapeEntryRecord)
                .where(TapeEntryRecord.tape_id == tape_record.id)
                .order_by(TapeEntryRecord.entry_id)
            )
            records = session.scalars(statement).all()
        return [self._entry_from_record(record) for record in records]

    def append(self, tape: str, entry: TapeEntry) -> None:
        with self._write_lock:
            with self._session_factory.begin() as session:
                tape_record = self._load_or_create_tape(session, tape)
                next_entry_id = self._next_entry_id(session, tape_record)
                anchor_name = self._anchor_name_of(entry)
                session.add(
                    TapeEntryRecord(
                        tape_id=tape_record.id,
                        entry_id=next_entry_id,
                        kind=entry.kind,
                        anchor_name=anchor_name,
                        anchor_name_key=self._key_for(anchor_name) if anchor_name else None,
                        payload=dict(entry.payload),
                        meta=dict(entry.meta),
                        entry_date=entry.date,
                    )
                )

    @staticmethod
    def _normalize_url(url: str) -> SQLAlchemyURL:
        try:
            return make_url(url)
        except ArgumentError as exc:
            raise ValueError(f"Invalid SQLAlchemy URL: {url}") from exc

    @staticmethod
    def _connect_args(url: SQLAlchemyURL) -> dict[str, object]:
        if url.get_backend_name() == "sqlite":
            return {"check_same_thread": False, "timeout": 30}
        return {}

    @staticmethod
    def _configure_engine(engine: Engine) -> None:
        if engine.url.get_backend_name() != "sqlite":
            return

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
            del connection_record
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA busy_timeout = 30000")
            cursor.close()

    def _validate_schema(self) -> None:
        inspector = inspect(self._engine)
        table_names = set(inspector.get_table_names())
        if "tapes" not in table_names or "tape_entries" not in table_names:
            raise RuntimeError("SQLAlchemy tape store schema is incomplete.")
        tape_columns = {column["name"] for column in inspector.get_columns("tapes")}
        entry_columns = {column["name"] for column in inspector.get_columns("tape_entries")}
        required_tape_columns = {"id", "name", "name_key", "last_entry_id", "created_at"}
        required_entry_columns = {
            "tape_id",
            "entry_id",
            "kind",
            "anchor_name",
            "anchor_name_key",
            "payload",
            "meta",
            "entry_date",
            "created_at",
        }
        if not required_tape_columns.issubset(tape_columns):
            raise RuntimeError("Existing tapes table uses an incompatible schema.")
        if not required_entry_columns.issubset(entry_columns):
            raise RuntimeError("Existing tape_entries table uses an incompatible schema.")

    @staticmethod
    def _anchor_name_of(entry: TapeEntry) -> str | None:
        if entry.kind != "anchor":
            return None
        name = entry.payload.get("name")
        if isinstance(name, str) and name:
            return name
        return None

    @staticmethod
    def _entry_from_record(record: TapeEntryRecord) -> TapeEntry:
        payload = record.payload if isinstance(record.payload, dict) else {}
        meta = record.meta if isinstance(record.meta, dict) else {}
        return TapeEntry(
            id=record.entry_id,
            kind=record.kind,
            payload=dict(payload),
            meta=dict(meta),
            date=record.entry_date,
        )

    @staticmethod
    def _key_for(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _load_or_create_tape(cls, session: Session, tape: str) -> TapeRecord:
        tape_record = cls._find_tape_record(session, tape, for_update=True)
        if tape_record is not None:
            return tape_record
        try:
            with session.begin_nested():
                tape_record = TapeRecord(name=tape, name_key=cls._key_for(tape), last_entry_id=0)
                session.add(tape_record)
                session.flush()
        except IntegrityError:
            pass
        tape_record = cls._find_tape_record(session, tape, for_update=True)
        if tape_record is None:
            raise RuntimeError(f"Failed to load tape record for '{tape}'.")
        return tape_record

    @classmethod
    def _find_tape_record(cls, session: Session, tape: str, *, for_update: bool = False) -> TapeRecord | None:
        statement = select(TapeRecord).where(TapeRecord.name_key == cls._key_for(tape), TapeRecord.name == tape)
        if for_update:
            statement = statement.with_for_update()
        return session.scalar(statement)

    @classmethod
    def _next_entry_id(cls, session: Session, tape_record: TapeRecord) -> int:
        current_entry_id = tape_record.last_entry_id
        while True:
            next_entry_id = current_entry_id + 1
            result = session.execute(
                update(TapeRecord)
                .where(TapeRecord.id == tape_record.id, TapeRecord.last_entry_id == current_entry_id)
                .values(last_entry_id=next_entry_id)
            )
            if result.rowcount == 1:
                tape_record.last_entry_id = next_entry_id
                return next_entry_id
            current_entry_id = session.scalar(
                select(TapeRecord.last_entry_id).where(TapeRecord.id == tape_record.id).with_for_update()
            )
            if current_entry_id is None:
                raise RuntimeError(f"Failed to allocate entry id for tape '{tape_record.name}'.")


def _build_store() -> SQLAlchemyTapeStore:
    settings = TapeStoreSettings.from_env()
    return SQLAlchemyTapeStore(url=settings.resolved_url, echo=settings.echo)


@lru_cache(maxsize=1)
def cached_store() -> SQLAlchemyTapeStore:
    return _build_store()
