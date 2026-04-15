from __future__ import annotations

from pathlib import Path

import pymysql
from republic.tape import TapeEntry, TapeQuery

from endless_context.oceanbase import _is_savepoint_not_exist
from endless_context.tape_store import SQLAlchemyTapeStore


def test_is_savepoint_not_exist_accepts_nested_operational_error() -> None:
    error = pymysql.err.OperationalError(1305, "savepoint does not exist")

    class WrappedError(Exception):
        def __init__(self, orig: Exception) -> None:
            super().__init__("wrapped")
            self.orig = orig

    assert _is_savepoint_not_exist(error) is True
    assert _is_savepoint_not_exist(WrappedError(error)) is True


def test_sqlalchemy_tape_store_reads_current_query_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "tapes.db"
    store = SQLAlchemyTapeStore(url=f"sqlite+pysqlite:///{db_path}")
    store.append("t1", TapeEntry.anchor("session/start", state={"owner": "human"}))
    store.append("t1", TapeEntry.message({"role": "user", "content": "hello"}))
    store.append("t1", TapeEntry.message({"role": "assistant", "content": "world"}))

    entries = list(store.fetch_all(TapeQuery(tape="t1", store=store).last_anchor()))

    assert [entry.kind for entry in entries] == ["message", "message"]
    assert entries[0].payload == {"role": "user", "content": "hello"}
    assert entries[1].payload == {"role": "assistant", "content": "world"}


def test_sqlalchemy_tape_store_filters_between_dates(tmp_path: Path) -> None:
    db_path = tmp_path / "tapes.db"
    store = SQLAlchemyTapeStore(url=f"sqlite+pysqlite:///{db_path}")
    store.append("t1", TapeEntry(0, "message", {"role": "user", "content": "early"}, {}, "2026-04-14T00:00:00Z"))
    store.append("t1", TapeEntry(0, "message", {"role": "user", "content": "late"}, {}, "2026-04-15T00:00:00Z"))

    entries = list(store.fetch_all(TapeQuery(tape="t1", store=store).between_dates("2026-04-15", "2026-04-15")))

    assert len(entries) == 1
    assert entries[0].payload["content"] == "late"
