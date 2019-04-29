# Authors: Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

# preparing everything:
import mne
import numpy as np
import os.path as op
from scipy.io import savemat

from mne.datasets import testing
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_cov
from mne.beamformer.tests.test_lcmv import _get_data
from mne.utils import run_tests_if_main


data_path = testing.data_path(download=False)
ft_data_path = op.join(data_path, 'fieldtrip', 'beamformer')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')

reject = dict(grad=4000e-13, mag=4e-12)


def _get_bf_data(save_fieldtrip=False):
    raw, epochs, evoked, data_cov, _, _, _, _, _, fwd = _get_data(proj=False)

    if save_fieldtrip is True:
        # raw needs to be saved with all channels and picked in FieldTrip
        raw.save(op.join(ft_data_path, 'raw.fif'), overwrite=True)

        # src (tris are not available in fwd['src'] once imported into MATLAB)
        src = fwd['src'].copy()
        mne.write_source_spaces(op.join(ft_data_path, 'src.fif'), src)

    # pick gradiometers only:
    epochs.pick_types(meg='grad')
    evoked.pick_types(meg='grad')

    # compute covariance matrix
    data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.145,
                                      method='empirical')

    if save_fieldtrip is True:
        # if the covariance matrix and epochs need resaving:
        # data covariance:
        cov_savepath = op.join(ft_data_path, 'sample_cov')
        sample_cov = {'sample_cov': data_cov['data']}
        savemat(cov_savepath, sample_cov)
        # evoked data:
        ev_savepath = op.join(ft_data_path, 'sample_evoked')
        data_ev = {'sample_evoked': evoked.data}
        savemat(ev_savepath, data_ev)

    return evoked, data_cov, fwd


@testing.requires_testing_data
def test_lcmv_fieldtrip():
    """Test LCMV vs fieldtrip output."""
    evoked, data_cov, fwd = _get_bf_data()

    # beamformer types to be tested: unit-gain (vector and scalar) and
    # unit-noise-gain (time series and power output [apply_lcmv_cov])
    bf_types = ['ug_vec', 'ug_scal', 'ung', 'ung_pow']
    weight_norms = [None, None, 'unit-noise-gain', 'unit-noise-gain']
    pick_oris = [None, 'max-power', 'max-power', 'max-power']
    power = [False, False, False, True]

    for bf_type, weight_norm, pick_ori, pwr in zip(bf_types, weight_norms,
                                                   pick_oris, power):

        # run the MNE-Python beamformer
        filters = make_lcmv(evoked.info, fwd, data_cov=data_cov,
                            noise_cov=None, pick_ori=pick_ori, reg=0.05,
                            weight_norm=weight_norm)
        if pwr is True:
            stc_mne = apply_lcmv_cov(data_cov, filters)
        else:
            stc_mne = apply_lcmv(evoked, filters)
            # take the absolute value, since orientation can be flipped
            stc_mne.data[:, :] = np.abs(stc_mne.data)

        # load the FieldTrip output
        ft_fname = op.join(ft_data_path, 'ft_source_' + bf_type + '-vol.stc')
        stc_ft = mne.read_source_estimate(ft_fname)

        # calculate the Pearson correlation between the source solutions:
        pearson = np.corrcoef(np.concatenate(stc_mne.data),
                              np.concatenate(stc_ft.data))

        assert pearson[0, 1] >= 0.99


run_tests_if_main()
