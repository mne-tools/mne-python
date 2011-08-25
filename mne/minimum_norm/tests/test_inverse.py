import os.path as op

import numpy as np
# from numpy.testing import assert_array_almost_equal, assert_equal

from ...datasets import sample
from ...label import read_label
from ...event import read_events
from ...epochs import Epochs
from ... import fiff, Covariance, read_forward_solution
from ..inverse import minimum_norm, apply_inverse, read_inverse_operator, \
                      apply_inverse_raw, apply_inverse_epochs


examples_folder = op.join(op.dirname(__file__), '..', '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_inv = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-oct-6-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-eeg-oct-6-fwd.fif')
fname_raw = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

inverse_operator = read_inverse_operator(fname_inv)
label = read_label(fname_label)
raw = fiff.Raw(fname_raw)
snr = 3.0
lambda2 = 1.0 / snr ** 2
dSPM = True


def test_apply_mne_inverse():
    """Test MNE with precomputed inverse operator on Evoked
    """
    setno = 0
    evoked = fiff.Evoked(fname_data, setno=0, baseline=(None, 0))
    stc = apply_inverse(evoked, inverse_operator, lambda2, dSPM)

    assert np.all(stc.data > 0)
    assert np.all(stc.data < 35)


def test_apply_mne_inverse_raw():
    """Test MNE with precomputed inverse operator on Raw
    """
    stc = apply_inverse_raw(raw, inverse_operator, lambda2, dSPM=True,
                            label=label, start=0, stop=10, nave=1,
                            pick_normal=False)
    assert np.all(stc.data > 0)


def test_apply_mne_inverse_epochs():
    """Test MNE with precomputed inverse operator on Epochs
    """
    event_id, tmin, tmax = 1, -0.2, 0.5

    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=True,
                            ecg=True, eog=True, include=['STI 014'])
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    flat = dict(grad=1e-15, mag=1e-15)

    events = read_events(fname_event)[:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=reject, flat=flat)
    stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, dSPM,
                                label=label)

    assert len(stcs) == 1
    assert np.all(stcs[0].data > 0)
    assert np.all(stcs[0].data < 42)


def test_compute_minimum_norm():
    """Test MNE inverse computation starting from forward operator
    """
    setno = 0
    noise_cov = Covariance(fname_cov)
    forward = read_forward_solution(fname_fwd)
    evoked = fiff.Evoked(fname_data, setno=setno, baseline=(None, 0))
    whitener = noise_cov.get_whitener(evoked.info, mag_reg=0.1,
                                      grad_reg=0.1, eeg_reg=0.1, pca=True)
    stc = minimum_norm(evoked, forward, whitener,
                       orientation='loose', method='dspm', snr=3, loose=0.2)

    assert np.all(stc.data > 0)
    # XXX : test something
