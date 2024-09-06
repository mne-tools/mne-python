import pytest


@pytest.fixture(autouse=True)
def _check_fixture():
    pytest.importorskip("sklearn")
