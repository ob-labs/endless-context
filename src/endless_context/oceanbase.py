"""OceanBase/seekdb helpers for Bub's SQLAlchemy tape store."""

from __future__ import annotations

import pymysql
import pyobvector  # noqa: F401
from bub import hookimpl
from pyobvector.schema.dialect import OceanBaseDialect as _OceanBaseDialect
from sqlalchemy.dialects import registry


def _is_savepoint_not_exist(exc: BaseException) -> bool:
    if isinstance(exc, pymysql.err.OperationalError) and exc.args and exc.args[0] == 1305:
        return True
    orig = getattr(exc, "orig", None)
    if orig is not None and orig is not exc:
        return _is_savepoint_not_exist(orig)
    return False


class OceanBaseDialect(_OceanBaseDialect):
    supports_statement_cache = True

    def do_release_savepoint(self, connection, name: str) -> None:
        try:
            super().do_release_savepoint(connection, name)
        except Exception as exc:
            if not _is_savepoint_not_exist(exc):
                raise

    def do_rollback_to_savepoint(self, connection, name: str) -> None:
        try:
            super().do_rollback_to_savepoint(connection, name)
        except Exception as exc:
            if not _is_savepoint_not_exist(exc):
                raise


registry.register("mysql.oceanbase", "endless_context.oceanbase", "OceanBaseDialect")


def _patch_tape_store_validate_schema() -> None:
    try:
        from bub_tapestore_sqlalchemy import store as _store
    except ImportError:
        return

    store_cls = _store.SQLAlchemyTapeStore
    if getattr(store_cls, "_ec_validate_schema_patched", False):
        return

    original = store_cls._validate_schema

    def _validate_schema_tolerant(self: object) -> None:
        try:
            original(self)
        except Exception as exc:
            original_exc = getattr(exc, "orig", exc)
            args = getattr(original_exc, "args", ())
            if args and args[0] == 1061:
                return
            if "Duplicate key name" in str(exc):
                return
            raise

    store_cls._validate_schema = _validate_schema_tolerant  # type: ignore[method-assign]
    store_cls._ec_validate_schema_patched = True  # type: ignore[attr-defined]


_patch_tape_store_validate_schema()


class _OceanBasePlugin:
    @hookimpl
    def provide_tape_store(self) -> None:
        return None


def register(_framework: object) -> _OceanBasePlugin:
    return _OceanBasePlugin()
