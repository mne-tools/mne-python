# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

pytest.importorskip("sklearn")
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne.decoding import XdawnTransformer, read_xdawn_transformer


@pytest.mark.filterwarnings("ignore:.*Only one sample available.*")
@parametrize_with_checks([XdawnTransformer(reg="oas")])  # oas handles few sample cases
def test_sklearn_compliance(estimator, check):
    """Test compliance with sklearn."""
    check(estimator)


def test_xdawn_save_load(tmp_path):
    """Test that XdawnTransformer can be saved to disk and loaded correctly."""
    h5io = pytest.importorskip("h5io")
    rng = np.random.RandomState(42)
    n_epochs, n_channels, n_times = 40, 10, 50
    X = rng.randn(n_epochs, n_channels, n_times)
    y = rng.randint(0, 2, n_epochs)

    xdawn = XdawnTransformer(n_components=2)
    xdawn.fit(X, y)

    state = xdawn.__getstate__()
    assert "cov_callable" not in state
    assert "mod_ged_callable" not in state

    fname = tmp_path / "test_xdawn.h5"
    xdawn.save(fname)

    xdawn_loaded = read_xdawn_transformer(fname)

    assert hasattr(xdawn_loaded, "cov_callable")
    assert hasattr(xdawn_loaded, "mod_ged_callable")
    assert callable(xdawn_loaded.cov_callable)
    assert callable(xdawn_loaded.mod_ged_callable)

    # Check fitted array attributes are restored
    assert_array_almost_equal(xdawn.filters_, xdawn_loaded.filters_)
    assert_array_almost_equal(xdawn.patterns_, xdawn_loaded.patterns_)

    # Check scalar/param attributes
    assert xdawn.n_components == xdawn_loaded.n_components
    assert xdawn.reg == xdawn_loaded.reg
    assert xdawn.rank == xdawn_loaded.rank
    assert xdawn.restr_type == xdawn_loaded.restr_type

    # Check transform output matches
    X_orig = xdawn.transform(X)
    X_loaded = xdawn_loaded.transform(X)
    assert_array_almost_equal(X_orig, X_loaded)

    with pytest.raises(FileExistsError):
        xdawn.save(fname)
    xdawn.save(fname, overwrite=True)

    # Check that loading an HDF5 file with missing keys raises an error
    bad_fname = tmp_path / "bad_xdawn.h5"
    h5io.write_hdf5(bad_fname, dict(foo="bar"), title="mnepython", slash="replace")
    with pytest.raises(ValueError, match="missing required keys"):
        read_xdawn_transformer(bad_fname)

    with pytest.raises(OSError, match="not found"):
        read_xdawn_transformer(tmp_path / "nonexistent.h5")
