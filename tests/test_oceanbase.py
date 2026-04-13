from __future__ import annotations

import sys
from types import ModuleType

import pymysql
import pytest

from endless_context import oceanbase


def test_is_savepoint_not_exist_accepts_nested_operational_error() -> None:
    error = pymysql.err.OperationalError(1305, "savepoint does not exist")

    class WrappedError(Exception):
        def __init__(self, orig: Exception) -> None:
            super().__init__("wrapped")
            self.orig = orig

    assert oceanbase._is_savepoint_not_exist(error) is True
    assert oceanbase._is_savepoint_not_exist(WrappedError(error)) is True


def test_patch_tape_store_validate_schema_ignores_duplicate_index(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("bub_tapestore_sqlalchemy")
    store_module = ModuleType("bub_tapestore_sqlalchemy.store")

    class DuplicateIndexError(Exception):
        def __init__(self) -> None:
            super().__init__("Duplicate key name 'idx_tape_entries_anchor_name_key'")
            self.orig = type("orig", (), {"args": (1061, "Duplicate key name")})()

    class FakeTapeStore:
        def _validate_schema(self) -> None:
            raise DuplicateIndexError()

    store_module.SQLAlchemyTapeStore = FakeTapeStore
    module.store = store_module
    monkeypatch.setitem(sys.modules, "bub_tapestore_sqlalchemy", module)
    monkeypatch.setitem(sys.modules, "bub_tapestore_sqlalchemy.store", store_module)

    oceanbase._patch_tape_store_validate_schema()

    FakeTapeStore()._validate_schema()


def test_patch_tape_store_validate_schema_preserves_unrelated_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("bub_tapestore_sqlalchemy")
    store_module = ModuleType("bub_tapestore_sqlalchemy.store")

    class FakeTapeStore:
        def _validate_schema(self) -> None:
            raise RuntimeError("boom")

    store_module.SQLAlchemyTapeStore = FakeTapeStore
    module.store = store_module
    monkeypatch.setitem(sys.modules, "bub_tapestore_sqlalchemy", module)
    monkeypatch.setitem(sys.modules, "bub_tapestore_sqlalchemy.store", store_module)

    oceanbase._patch_tape_store_validate_schema()

    with pytest.raises(RuntimeError, match="boom"):
        FakeTapeStore()._validate_schema()
