from __future__ import annotations

import pymysql

from endless_context.oceanbase import _is_savepoint_not_exist


def test_is_savepoint_not_exist_accepts_nested_operational_error() -> None:
    error = pymysql.err.OperationalError(1305, "savepoint does not exist")

    class WrappedError(Exception):
        def __init__(self, orig: Exception) -> None:
            super().__init__("wrapped")
            self.orig = orig

    assert _is_savepoint_not_exist(error) is True
    assert _is_savepoint_not_exist(WrappedError(error)) is True
