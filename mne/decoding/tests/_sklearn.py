# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial

import pytest
from sklearn.utils.estimator_checks import (
    parametrize_with_checks as _sklearn_parametrize_with_checks,
)


# TODO VERSION: Remove once scikit-learn 1.10.0 is the minimum supported version
# scikit-learn/scikit-learn#34448
def _materialize_parametrize_mark(sklearn_decorator, func):
    """Materialize generated argvalues in sklearn's pytest mark."""
    marked_func = sklearn_decorator(func)
    marks = marked_func.pytestmark
    if not isinstance(marks, list):
        marks = [marks]
    else:
        marks = list(marks)
    for index, mark in enumerate(marks):
        if mark.name == "parametrize":
            args = (*mark.args[:1], list(mark.args[1]), *mark.args[2:])
            marks[index] = pytest.mark.parametrize(*args, **mark.kwargs).mark
            break
    marked_func.pytestmark = marks
    return marked_func


def _parametrize_with_checks(estimators):
    """Parametrize estimator checks with pytest-compatible argvalues."""
    sklearn_decorator = _sklearn_parametrize_with_checks(estimators)
    return partial(_materialize_parametrize_mark, sklearn_decorator)


parametrize_with_checks = _parametrize_with_checks
