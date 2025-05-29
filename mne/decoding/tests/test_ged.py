# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import functools

import numpy as np
import pytest

pytest.importorskip("sklearn")


from sklearn.model_selection import ParameterGrid
from sklearn.utils.estimator_checks import parametrize_with_checks

from mne import compute_rank, create_info
from mne._fiff.proj import make_eeg_average_ref_proj
from mne.cov import Covariance, _regularized_covariance
from mne.decoding.base import GEDTransformer


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


def _mock_cov_callable(X, y, cov_method_params=None):
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
    C_ref = _regularized_covariance(ref_data, **cov_method_params)
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
    if evals is not None:
        ix = np.argsort(evals)[::-1]
        evals = evals[ix]
        evecs = evecs[:, ix]
    return evals, evecs


param_grid = dict(
    n_filters=[4],
    cov_callable=[_mock_cov_callable],
    cov_params=[
        dict(cov_method_params=dict(reg="empirical")),
    ],
    mod_ged_callable=[_mock_mod_ged_callable],
    mod_params=[dict()],
    dec_type=["single", "multi"],
    restr_type=[
        "restricting",
        "whitening",
    ],  # Not covering "ssd" here because its tests work with 2D data.
    R_func=[functools.partial(np.sum, axis=0)],
)

ged_estimators = [GEDTransformer(**p) for p in ParameterGrid(param_grid)]


@pytest.mark.slowtest
@parametrize_with_checks(ged_estimators)
def test_sklearn_compliance(estimator, check):
    """Test GEDTransformer compliance with sklearn."""
    check(estimator)
