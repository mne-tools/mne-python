# -*- coding: utf-8 -*-
"""Test the compute_current_source_density function.

For each supported file format, implement a test.
"""
# Authors: Alex Rockhill <aprockhill206@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
from scipy import io as sio

import pytest
from numpy.testing import assert_allclose
import mne
from mne.channels import make_standard_montage
from mne import create_info, Annotations
from mne.io import RawArray
from mne.utils import run_tests_if_main
from mne.datasets import testing

from mne.preprocessing import compute_current_source_density

from mne.channels.interpolation import _calc_g, _calc_h

base_path = op.join(testing.data_path(download=False), 'preprocessing')


@testing.requires_testing_data
def test_csd():
    """Test replication of the CSD MATLAB toolbox."""
    mat_contents = sio.loadmat(op.join(base_path, 'test-eeg.mat'))
    data = mat_contents['data']
    n_channels, n_epochs = data.shape[0], data.shape[1] // 386
    sfreq = 250.
    ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
    ch_types = ['eeg'] * n_channels
    sphere = (0., 0., 0., 0.095)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=data, info=info)
    montage = make_standard_montage('GSN-HydroCel-257')
    raw.set_montage(montage)
    onset = raw.times[np.arange(50, n_epochs * 386, 386)]
    raw.set_annotations(Annotations(onset=onset,
                                    duration=np.repeat(0.1, 3),
                                    description=np.repeat('foo', 3)))

    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                        preload=True, reject=None, picks=None,
                        baseline=(None, 0), verbose=False)
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')

    csd_data = sio.loadmat(op.join(base_path, 'test-eeg-csd.mat'))

    """Test G, H and CSD against matlab CSD Toolbox"""
    montage = make_standard_montage('EGI_256', head_size=0.100004)
    positions = np.array([montage.dig[pick]['r'] * 10 for pick in picks])
    cosang = np.dot(positions, positions.T)
    G = _calc_g(cosang)
    assert_allclose(G, csd_data['G'], atol=1e-3)
    H = _calc_h(cosang)
    assert_allclose(H, csd_data['H'], atol=1e-3)
    for i in range(n_epochs):
        epochs_csd = compute_current_source_density(epochs, sphere=sphere)
        assert_allclose(epochs_csd.get_data(), csd_data['X'], atol=1e-3)

    # test raw
    csd_raw = compute_current_source_density(raw, sphere=sphere)

    with pytest.raises(ValueError, match=('CSD already applied, '
                                          'should not be reapplied')):
        compute_current_source_density(csd_raw, sphere=sphere)

    csd_raw_test_array = np.array([[2.29938168e-07, 1.55737642e-07],
                                   [-9.63976630e-09, 8.31646698e-09],
                                   [-2.30898926e-07, -1.56250505e-07],
                                   [-1.81081104e-07, -5.46661150e-08],
                                   [-9.08835568e-08, 1.61788422e-07],
                                   [5.38295661e-09, 3.75240220e-07]])
    assert_allclose(csd_raw._data[:, 100:102], csd_raw_test_array, atol=1e-3)

    csd_epochs = compute_current_source_density(epochs, sphere=sphere)
    assert_allclose(csd_epochs._data, csd_data['X'], atol=1e-3)

    csd_epochs = compute_current_source_density(epochs, sphere=sphere)

    warn_raw = raw.copy()
    warn_raw.info['bads'].append(warn_raw.ch_names[3])
    with pytest.warns(UserWarning, match='Deleting bad channels'):
        compute_current_source_density(warn_raw)

    with pytest.raises(TypeError):
        csd_epochs = compute_current_source_density(None)

    fail_raw = raw.copy()
    with pytest.raises(ValueError, match='Zero or infinite position'):
        for ch in fail_raw.info['chs']:
            ch['loc'][:3] = np.array([0, 0, 0])
        compute_current_source_density(fail_raw, sphere=sphere)

    with pytest.raises(ValueError, match='Zero or infinite position'):
        fail_raw.info['chs'][3]['loc'][:3] = np.array([np.inf, np.inf, np.inf])
        compute_current_source_density(fail_raw, sphere=sphere)

    with pytest.raises(ValueError, match=('No EEG channels found.')):
        fail_raw = raw.copy()
        fail_raw.set_channel_types({ch_name: 'ecog' for ch_name in
                                    fail_raw.ch_names})
        compute_current_source_density(fail_raw, sphere=sphere)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, lambda2='0', sphere=sphere)

    with pytest.raises(ValueError, match='lambda2 must be between 0 and 1'):
        compute_current_source_density(epochs, lambda2=2, sphere=sphere)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, stiffness='0', sphere=sphere)

    with pytest.raises(ValueError, match='stiffness must be non-negative'):
        compute_current_source_density(epochs, stiffness=-2, sphere=sphere)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, n_legendre_terms=0.1,
                                       sphere=sphere)

    with pytest.raises(ValueError, match=('n_legendre_terms must be '
                                          'greater than 0')):
        compute_current_source_density(epochs, n_legendre_terms=0,
                                       sphere=sphere)

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, sphere=-0.1)

    with pytest.raises(ValueError, match=('sphere radius must be '
                                          'greater than 0')):
        compute_current_source_density(epochs, sphere=(-0.1, 0., 0., -1.))

    with pytest.raises(TypeError):
        compute_current_source_density(epochs, copy=2, sphere=sphere)

    csd_evoked = compute_current_source_density(epochs.average(),
                                                sphere=sphere)
    assert_allclose(csd_evoked.data, csd_data['X'].mean(0), atol=1e-3)
    assert_allclose(csd_evoked.data, csd_epochs._data.mean(0), atol=1e-3)


run_tests_if_main()
