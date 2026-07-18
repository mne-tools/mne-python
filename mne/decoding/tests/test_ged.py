# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")


from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import assert_allclose
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne import Epochs, compute_rank, create_info, pick_types, read_events
from mne._fiff.proj import make_eeg_average_ref_proj
from mne.cov import Covariance, _regularized_covariance
from mne.decoding._ged import (
    _get_cov_def,
    _get_restr_mat,
    _handle_restr_mat,
    _is_cov_symm,
    _smart_ajd,
    _smart_ged,
)
from mne.decoding._mod_ged import _no_op_mod
from mne.decoding.base import _GEDTransformer
from mne.io import read_raw

data_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_dir / "test_raw.fif"
event_name = data_dir / "test-eve.fif"
tmin, tmax = -0.1, 0.2
# if stop is too small pca may fail in some cases, but we're okay on this file
start, stop = 0, 8


def _mock_info(n_channels):
    info = create_info(n_channels, 1000.0, "eeg")
    avg_eeg_projector = make_eeg_average_ref_proj(info=info, activate=False)
    info["projs"].append(avg_eeg_projector)
    return info


def _get_min_rank(covs, info):
    min_rank = dict(
        eeg=min(
            list(
                compute_rank(
                    Covariance(
                        cov,
                        info.ch_names,
                        list(),
                        list(),
                        0,
                        # verbose=_verbose_safe_false(),
                    ),
                    rank=None,
                    # _handle_default("scalings_cov_rank", None),
                    info=info,
                ).values()
            )[0]
            for cov in covs
        )
    )
    return min_rank


def _mock_cov_callable(X, y, cov_method_params=None, compute_C_ref=True):
    if cov_method_params is None:
        cov_method_params = dict()
    n_epochs, n_channels, n_times = X.shape

    # To pass sklearn check:
    if n_channels == 1:
        n_channels = 2
        X = np.tile(X, (1, n_channels, 1))

    # To make covariance estimation sensible
    if n_times == 1:
        n_times = n_channels
        X = np.tile(X, (1, 1, n_channels))

    classes = np.unique(y)
    covs, sample_weights = list(), list()
    for ci, this_class in enumerate(classes):
        class_data = X[y == this_class]
        class_data = class_data.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = _regularized_covariance(class_data, **cov_method_params)
        covs.append(cov)
        sample_weights.append(class_data.shape[0])

    ref_data = X.transpose(1, 0, 2).reshape(n_channels, -1)
    if compute_C_ref:
        C_ref = _regularized_covariance(ref_data, **cov_method_params)
    else:
        C_ref = None
    info = _mock_info(n_channels)
    rank = _get_min_rank(covs, info)
    kwargs = dict()

    # To pass sklearn check:
    if len(covs) == 1:
        covs.append(covs[0])

    elif len(covs) > 2:
        kwargs["sample_weights"] = sample_weights
    return covs, C_ref, info, rank, kwargs


def _mock_mod_ged_callable(evals, evecs, covs, **kwargs):
    sorter = None
    if evals is not None:
        ix = np.argsort(evals)[::-1]
        evals = evals[ix]
        evecs = evecs[:, ix]
        sorter = ix
    return evals, evecs, sorter


param_grid = dict(
    n_components=[4],
    cov_callable=[partial(_mock_cov_callable, cov_method_params=dict(reg="empirical"))],
    mod_ged_callable=[_mock_mod_ged_callable],
    dec_type=["single", "multi"],
    restr_type=["restricting", "whitening"],
    R_func=[None, partial(np.sum, axis=0)],
)

ged_estimators = [_GEDTransformer(**p) for p in ParameterGrid(param_grid)]


@pytest.mark.slowtest
@parametrize_with_checks(ged_estimators)
def test_sklearn_compliance(estimator, check):
    """Test GEDTransformer compliance with sklearn."""
    check(estimator)


def _get_X_y(event_id):
    raw = read_raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, stim=False, ecg=False, eog=False, exclude="bads"
    )
    picks = picks[2:12:3]  # subselect channels -> disable proj!
    raw.add_proj([], remove_existing=True)
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        proj=False,
    )
    X = epochs.get_data(copy=False, units=dict(eeg="uV", grad="fT/cm", mag="fT"))
    y = epochs.events[:, -1]
    return X, y


def test_ged_binary_cov():
    """Test GEDTransformer on audvis dataset with two covariances."""
    event_id = dict(aud_l=1, vis_l=3)
    X, y = _get_X_y(event_id)
    # Test "single" decomposition
    covs, C_ref, info, rank, kwargs = _mock_cov_callable(X, y)
    S, R = covs[0], covs[1]
    restr_mat = _get_restr_mat(C_ref, info, rank)
    evals, evecs = _smart_ged(S, R, restr_mat=restr_mat, R_func=None)
    actual_evals, actual_evecs, sorter = _mock_mod_ged_callable(
        evals, evecs, [S, R], **kwargs
    )
    actual_filters = actual_evecs.T

    ged = _GEDTransformer(
        n_components=4,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        restr_type="restricting",
    )
    ged.fit(X, y)
    desired_evals = ged.evals_
    desired_filters = ged.filters_

    assert_allclose(actual_evals, desired_evals)
    assert_allclose(actual_filters, desired_filters)

    # Test "multi" decomposition (loop), restr_mat can be reused
    all_evals, all_evecs = list(), list()
    for i in range(len(covs)):
        S = covs[i]
        evals, evecs = _smart_ged(S, R, restr_mat)
        evals, evecs, sorter = _mock_mod_ged_callable(evals, evecs, covs)
        all_evals.append(evals)
        all_evecs.append(evecs.T)
    actual_evals = np.array(all_evals)
    actual_filters = np.array(all_evecs)

    ged = _GEDTransformer(
        n_components=4,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        dec_type="multi",
        restr_type="restricting",
    )
    ged.fit(X, y)
    desired_evals = ged.evals_
    desired_filters = ged.filters_

    assert_allclose(actual_evals, desired_evals)
    assert_allclose(actual_filters, desired_filters)

    assert ged._subset_multi_components(name="foo") is None


def test_ged_multicov():
    """Test GEDTransformer on audvis dataset with multiple covariances."""
    event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
    X, y = _get_X_y(event_id)
    # Test "single" decomposition for multicov (AJD) with C_ref
    covs, C_ref, info, rank, kwargs = _mock_cov_callable(X, y)
    restr_mat = _get_restr_mat(C_ref, info, rank)
    evecs = _smart_ajd(covs, restr_mat=restr_mat)
    evals = None
    _, actual_evecs, _ = _mock_mod_ged_callable(evals, evecs, covs, **kwargs)
    actual_filters = actual_evecs.T

    ged = _GEDTransformer(
        n_components=4,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        restr_type="restricting",
    )
    ged.fit(X, y)
    desired_filters = ged.filters_

    assert_allclose(actual_filters, desired_filters)

    # Test "multi" decomposition for multicov (loop)
    R = covs[-1]
    all_evals, all_evecs = list(), list()
    for i in range(len(covs)):
        S = covs[i]
        evals, evecs = _smart_ged(S, R, restr_mat)
        evals, evecs, sorter = _mock_mod_ged_callable(evals, evecs, covs)
        all_evals.append(evals)
        all_evecs.append(evecs.T)
    actual_evals = np.array(all_evals)
    actual_filters = np.array(all_evecs)

    ged = _GEDTransformer(
        n_components=4,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        dec_type="multi",
        restr_type="restricting",
    )
    ged.fit(X, y)
    desired_evals = ged.evals_
    desired_filters = ged.filters_

    assert_allclose(actual_evals, desired_evals)
    assert_allclose(actual_filters, desired_filters)

    # Test "single" decomposition for multicov (AJD) without C_ref
    covs, C_ref, info, rank, kwargs = _mock_cov_callable(
        X, y, cov_method_params=dict(reg="oas"), compute_C_ref=False
    )
    covs = np.stack(covs)
    evecs = _smart_ajd(covs, restr_mat=None)
    evals = None
    _, actual_evecs, _ = _mock_mod_ged_callable(evals, evecs, covs, **kwargs)
    actual_filters = actual_evecs.T

    ged = _GEDTransformer(
        n_components=4,
        cov_callable=partial(
            _mock_cov_callable, cov_method_params=dict(reg="oas"), compute_C_ref=False
        ),
        mod_ged_callable=_mock_mod_ged_callable,
        restr_type="restricting",
    )
    ged.fit(X, y)
    desired_filters = ged.filters_

    assert_allclose(actual_filters, desired_filters)


def test_ged_validation_raises():
    """Test GEDTransofmer validation raises correct errors."""
    event_id = dict(aud_l=1, vis_l=3)
    X, y = _get_X_y(event_id)

    ged = _GEDTransformer(
        n_components=-1,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        restr_type="restricting",
    )
    with pytest.raises(ValueError):
        ged.fit(X, y)

    def _bad_cov_callable(X, y, foo):
        return X, y, foo

    ged = _GEDTransformer(
        n_components=1,
        cov_callable=_bad_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
        restr_type="restricting",
    )
    with pytest.raises(ValueError):
        ged.fit(X, y)


def test_ged_invalid_cov():
    """Test _validate_covariances raises proper errors."""
    ged = _GEDTransformer(
        n_components=1,
        cov_callable=_mock_cov_callable,
        mod_ged_callable=_mock_mod_ged_callable,
    )
    asymm_cov = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(ValueError, match="not symmetric"):
        ged._validate_covariances([asymm_cov, None])


def test__handle_restr_mat_invalid_restr_type():
    """Test _handle_restr_mat raises correct error when wrong restr_type."""
    C_ref = np.eye(3)
    with pytest.raises(ValueError, match="restr_type"):
        _handle_restr_mat(C_ref, restr_type="blah", info=None, rank=None)


def test_cov_validators():
    """Test that covariance validators indeed validate."""
    asymm_indef = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sing_pos_semidef = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    pos_def = np.array([[5, 1, 1], [1, 6, 2], [1, 2, 7]])

    assert not _is_cov_symm(asymm_indef)
    assert _get_cov_def(asymm_indef) == "indef"
    assert _get_cov_def(sing_pos_semidef) == "pos_semidef"
    assert _get_cov_def(pos_def) == "pos_def"


def test__smart_ajd_raises():
    """Test _smart_ajd raises proper ValueErrors."""
    asymm_indef = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sing_pos_semidef = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    pos_def1 = np.array([[5, 1, 1], [1, 6, 2], [1, 2, 7]])
    pos_def2 = np.array([[10, 1, 2], [1, 12, 3], [2, 3, 15]])

    bad_covs = np.stack([sing_pos_semidef, asymm_indef, pos_def1])
    with pytest.raises(ValueError, match="positive semi-definite"):
        _smart_ajd(bad_covs, restr_mat=pos_def2, weights=None)

    bad_covs = np.stack([sing_pos_semidef, pos_def1, pos_def2])
    with pytest.raises(ValueError, match="positive definite"):
        _smart_ajd(bad_covs, restr_mat=None, weights=None)


def test__no_op_mod():
    """Test _no_op_mod returns the same evals/evecs objects."""
    evals = np.array([[1, 2], [3, 4]])
    evecs = np.array([0, 1])
    evals_no_op, evecs_no_op, sorter_no_op = _no_op_mod(evals, evecs)
    assert evals is evals_no_op
    assert evecs is evecs_no_op
    assert sorter_no_op is None
