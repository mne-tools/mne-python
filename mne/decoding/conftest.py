# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pytest


@pytest.fixture(autouse=True)
def _check_fixture():
    pytest.importorskip("sklearn")
