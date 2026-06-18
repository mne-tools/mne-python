# Adapted from mne-lsl
import sys
import sysconfig

import pytest

_is_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.mark.skipif(not _is_free_threaded, reason="not a free-threaded python build")
def test_gil_disabled() -> None:
    """Assert that the GIL is disabled on free-threaded Python.

    On free-threaded builds (3.13t, 3.14t, ...), the GIL should be disabled by default.
    If a C extension that hasn't declared ``Py_MOD_GIL_NOT_USED`` gets imported, the GIL
    is silently re-enabled and a ``RuntimeWarning`` is emitted. Combined with
    ``filterwarnings = ["error"]`` in our pytest config, this test ensures that no such
    extension has been loaded during the test session.
    """
    assert not sys._is_gil_enabled(), (
        "The GIL has been re-enabled. A C extension that does not declare "
        "Py_MOD_GIL_NOT_USED was loaded during the test session."
    )
