# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD-3-Clause

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.io import savemat

import mne
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_cov
from mne.beamformer.tests.test_lcmv import _get_data
from mne.datasets import testing

data_path = testing.data_path(download=False)
ft_data_path = data_path / "fieldtrip" / "beamformer"
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_cov = data_path / "MEG" / "sample" / "sample_audvis_trunc-cov.fif"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
fname_fwd_vol = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-vol-7-fwd.fif"
fname_event = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw-eve.fif"
fname_label = data_path / "MEG" / "sample" / "labels" / "Aud-lh.label"

reject = dict(grad=4000e-13, mag=4e-12)


@pytest.fixture(scope="function", params=[testing._pytest_param()])
def _get_bf_data(save_fieldtrip=False):
    raw, epochs, evoked, data_cov, _, _, _, _, _, fwd = _get_data(proj=False)

    if save_fieldtrip is True:
        # raw needs to be saved with all channels and picked in FieldTrip
        raw.save(ft_data_path / "raw.fif", overwrite=True)

        # src (tris are not available in fwd['src'] once imported into MATLAB)
        src = fwd["src"].copy()
        mne.write_source_spaces(
            ft_data_path / "src.fif", src, verbose="error", overwrite=True
        )

    # pick gradiometers only:
    epochs.pick_types(meg="grad")
    evoked.pick_types(meg="grad")

    # compute covariance matrix (ignore false alarm about no baseline)
    data_cov = mne.compute_covariance(
        epochs, tmin=0.04, tmax=0.145, method="empirical", verbose="error"
    )

    if save_fieldtrip is True:
        # if the covariance matrix and epochs need resaving:
        # data covariance:
        cov_savepath = ft_data_path / "sample_cov.mat"
        sample_cov = {"sample_cov": data_cov["data"]}
        savemat(cov_savepath, sample_cov)
        # evoked data:
        ev_savepath = ft_data_path / "sample_evoked.mat"
        data_ev = {"sample_evoked": evoked.data}
        savemat(ev_savepath, data_ev)

    return evoked, data_cov, fwd


# beamformer types to be tested: unit-gain (vector and scalar) and
# unit-noise-gain (time series and power output [apply_lcmv_cov])
@pytest.mark.parametrize(
    "bf_type, weight_norm, pick_ori, pwr",
    [
        ["ug_scal", None, "max-power", False],
        ["ung", "unit-noise-gain", "max-power", False],
        ["ung_pow", "unit-noise-gain", "max-power", True],
        ["ug_vec", None, "vector", False],
        ["ung_vec", "unit-noise-gain", "vector", False],
    ],
)
def test_lcmv_fieldtrip(_get_bf_data, bf_type, weight_norm, pick_ori, pwr):
    """Test LCMV vs fieldtrip output."""
    pymatreader = pytest.importorskip("pymatreader")

    evoked, data_cov, fwd = _get_bf_data

    # run the MNE-Python beamformer
    filters = make_lcmv(
        evoked.info,
        fwd,
        data_cov=data_cov,
        noise_cov=None,
        pick_ori=pick_ori,
        reg=0.05,
        weight_norm=weight_norm,
    )
    if pwr:
        stc_mne = apply_lcmv_cov(data_cov, filters)
    else:
        stc_mne = apply_lcmv(evoked, filters)

    # load the FieldTrip output
    ft_fname = ft_data_path / ("ft_source_" + bf_type + "-vol.mat")
    stc_ft_data = pymatreader.read_mat(ft_fname)["stc"]
    if stc_ft_data.ndim == 1:
        stc_ft_data.shape = (stc_ft_data.size, 1)

    if stc_mne.data.ndim == 2:
        signs = np.sign((stc_mne.data * stc_ft_data).sum(-1, keepdims=True))
        if pwr:
            assert_array_equal(signs, 1.0)
        stc_mne.data *= signs
    assert stc_ft_data.shape == stc_mne.data.shape
    if pick_ori == "vector":
        # compare norms first
        assert_allclose(
            np.linalg.norm(stc_mne.data, axis=1),
            np.linalg.norm(stc_ft_data, axis=1),
            rtol=1e-6,
        )
    assert_allclose(stc_mne.data, stc_ft_data, rtol=1e-6)
