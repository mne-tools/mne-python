# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne.decoding import XdawnTransformer


@pytest.mark.filterwarnings("ignore:.*Only one sample available.*")
@parametrize_with_checks([XdawnTransformer(reg="oas")])  # oas handles few sample cases
def test_sklearn_compliance(estimator, check):
    """Test compliance with sklearn."""
    pytest.importorskip("sklearn", minversion="1.4")  # TODO VERSION remove on 1.4+
    check(estimator)
