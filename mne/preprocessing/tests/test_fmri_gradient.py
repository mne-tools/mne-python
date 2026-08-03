# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import create_info
from mne.io import RawArray
from mne.preprocessing import GradientRemover, remove_fmri_gradient_artifact

N_TRS = 10
TR_CODE = 1
SAMPS_PER_TR = 100
N_CHANNELS = 8


def _sample_trs():
    return np.asarray([x * SAMPS_PER_TR for x in range(N_TRS)])


def _sample_trs_longform():
    return np.asarray([[x * SAMPS_PER_TR, 0, TR_CODE] for x in range(N_TRS)])


def _sample_data():
    return np.zeros((N_CHANNELS, N_TRS * SAMPS_PER_TR))


def test_window_validity():
    """Test validation of the template window."""
    data, trs = _sample_data(), _sample_trs()
    with pytest.raises(ValueError, match=r"Integer windows must be even"):
        GradientRemover(data, trs, 5)
    with pytest.raises(ValueError, match=r"Tuple windows must contain"):
        GradientRemover(data, trs, (2, 2, 2))
    with pytest.raises(TypeError, match=r"Window must be a positive"):
        GradientRemover(data, trs, None)
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (-1, 1))
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (1, -1))
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (0, 0))

    assert GradientRemover(data, trs, (2, 2)).window == (2, 2)
    assert GradientRemover(data, trs, 4).window == (2, 2)


def test_tr_events_validity():
    """Test validation of tr_events."""
    data = _sample_data()

    with pytest.raises(ValueError, match=r"TRs must be a 1D array or"):
        GradientRemover(data, np.asarray([[1, 2], [1, 2]]))

    with pytest.raises(ValueError, match=r"tr_tol must be a non-negative"):
        GradientRemover(data, _sample_trs(), tr_tol=-1)

    trs = _sample_trs()
    trs[1] = trs[1] + 5  # short form, spacing far outside tolerance
    with pytest.raises(ValueError, match=r"TR spacings are not"):
        GradientRemover(data, trs)
    trs = _sample_trs_longform()
    trs[1, 0] = trs[1, 0] + 5  # long form, spacing far outside tolerance
    with pytest.raises(ValueError, match=r"TR spacings are not"):
        GradientRemover(data, trs)

    # both short and long form give the same result
    gr = GradientRemover(data, _sample_trs())
    assert gr.tr_spacing == SAMPS_PER_TR
    assert gr.n_tr == N_TRS
    assert GradientRemover(data, _sample_trs_longform()).n_tr == N_TRS


def test_tr_events_jitter_tolerance():
    """Test that small TR-spacing jitter is tolerated but not accumulated."""
    data = _sample_data()
    # shift a single TR onset by 1 sample (within tr_tol=1); this perturbs
    # the two adjacent spacings by +1 and -1 respectively
    trs = _sample_trs().copy()
    trs[5] += 1
    gr = GradientRemover(data, trs, tr_tol=1)
    assert gr.tr_spacing == SAMPS_PER_TR  # median spacing unaffected
    # each TR is anchored at its own onset, so jitter does not accumulate:
    # the last TR's bounds should start near its own (jittered) sample, not
    # drift by N_TRS worth of jitter
    last_start, _ = gr._tr_bounds(N_TRS - 1)
    assert last_start == trs[-1]

    # a deviation larger than tr_tol still raises
    trs_bad = _sample_trs()
    trs_bad[1] += 2
    with pytest.raises(ValueError, match=r"TR spacings are not"):
        GradientRemover(data, trs_bad, tr_tol=1)
    # ...but is tolerated with a larger tr_tol
    GradientRemover(data, trs_bad, tr_tol=2)


def test_get_tr():
    """Test per-TR indexing."""
    gr = GradientRemover(_sample_data(), _sample_trs())
    with pytest.raises(ValueError, match=r"Index -1"):
        gr.get_tr(-1)
    with pytest.raises(ValueError, match=r"Index"):
        gr.get_tr(len(_sample_trs()) + 1)
    assert gr.get_tr(0).shape[1] == gr.tr_spacing


def test_correction_removes_artifact():
    """Test that a repeating artifact is subtracted away."""
    rng = np.random.default_rng(42)
    trs = _sample_trs()
    # identical artifact repeated every TR + small noise -> should cancel
    artifact = rng.standard_normal((N_CHANNELS, SAMPS_PER_TR))
    data = np.tile(artifact, (1, N_TRS)).astype(float)
    gr = GradientRemover(data, trs, window=(4, 4))
    corrected = gr.correct()
    # corrected property is cached and identical
    assert_allclose(gr.corrected, corrected)
    # in the interior (where a full template exists) the artifact is removed
    interior = corrected[:, 4 * SAMPS_PER_TR : 6 * SAMPS_PER_TR]
    assert_allclose(interior, 0, atol=1e-10)


def test_remove_fmri_gradient_artifact():
    """Test the Raw-level wrapper."""
    info = create_info(N_CHANNELS, sfreq=100.0, ch_types="eeg")
    artifact = np.random.default_rng(0).standard_normal((N_CHANNELS, SAMPS_PER_TR))
    data = np.tile(artifact, (1, N_TRS)).astype(float)
    raw = RawArray(data, info)

    out = remove_fmri_gradient_artifact(raw, _sample_trs(), window=(4, 4))
    assert out is not raw  # copy by default
    assert_allclose(raw.get_data(), data)  # original untouched
    interior = out.get_data()[:, 4 * SAMPS_PER_TR : 6 * SAMPS_PER_TR]
    assert_allclose(interior, 0, atol=1e-10)

    # in place
    out = remove_fmri_gradient_artifact(raw, _sample_trs_longform(), copy=False)
    assert out is raw

    with pytest.raises(ValueError, match="Invalid value for.*method"):
        remove_fmri_gradient_artifact(raw, _sample_trs(), method="bad")

    raw_nopreload = RawArray(data, info)
    raw_nopreload.preload = False
    with pytest.raises(RuntimeError, match="must be preloaded"):
        remove_fmri_gradient_artifact(raw_nopreload, _sample_trs())
