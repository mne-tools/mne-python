# -*- coding: utf-8 -*-
"""Test the compute_current_source_density function.

For each supported file format, implement a test.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np

import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat
from scipy import linalg

from mne.channels import make_dig_montage
from mne import create_info, EvokedArray, pick_types, Epochs
from mne.io import read_raw_fif, RawArray
from mne.io.constants import FIFF
from mne.utils import object_diff, run_tests_if_main
from mne.datasets import testing

from mne.preprocessing import compute_current_source_density

data_path = op.join(testing.data_path(download=False), 'preprocessing')
eeg_fname = op.join(data_path, 'test_eeg.mat')
coords_fname = op.join(data_path, 'test_eeg_pos.mat')
csd_fname = op.join(data_path, 'test_eeg_csd.mat')

io_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(io_path, 'test_raw.fif')


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def evoked_csd_sphere():
    """Get the MATLAB EEG data."""
    data = loadmat(eeg_fname)['data']
    coords = loadmat(coords_fname)['coords'] * 1e-3
    csd = loadmat(csd_fname)['csd']
    sphere = np.array((0, 0, 0, 0.08500060886258405))  # meters
    sfreq = 256  # sampling rate
    # swap coordinates' shape
    pos = np.rollaxis(coords, 1)
    # swap coordinates' positions
    pos[:, [0]], pos[:, [1]] = pos[:, [1]], pos[:, [0]]
    # invert first coordinate
    pos[:, [0]] *= -1
    dists = np.linalg.norm(pos, axis=-1)
    assert_allclose(dists, sphere[-1], rtol=1e-2)  # close to spherical, meters
    # assign channel names to coordinates
    ch_names = [str(ii) for ii in range(len(pos))]
    dig_ch_pos = dict(zip(ch_names, pos))
    montage = make_dig_montage(ch_pos=dig_ch_pos, coord_frame='head')
    # create info
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    # make Evoked object
    evoked = EvokedArray(data=data, info=info, tmin=-1)
    evoked.set_montage(montage)
    return evoked, csd, sphere


def test_csd_matlab(evoked_csd_sphere):
    """Test replication of the CSD MATLAB toolbox."""
    evoked, csd, sphere = evoked_csd_sphere
    evoked_csd = compute_current_source_density(evoked, sphere=sphere)
    assert_allclose(linalg.norm(csd), 0.00177, atol=1e-5)
    # If we don't project onto the sphere, we get 1e-12 accuracy here,
    # but it's a bad assumption for real data!
    # Also, we divide by (radius ** 2) to get to units of V/m², unclear
    # why this isn't done in the upstream implementation
    evoked_csd_data = evoked_csd.data * sphere[-1] ** 2
    assert_allclose(evoked_csd_data, csd, atol=2e-7)

    with pytest.raises(ValueError, match=('CSD already applied, '
                                          'should not be reapplied')):
        compute_current_source_density(evoked_csd, sphere=sphere)

    # 1e-5 here if we don't project...
    assert_allclose(evoked_csd_data.sum(), 0.02455, atol=2e-3)


def test_csd_degenerate(evoked_csd_sphere):
    """Test degenerate conditions."""
    evoked, csd, sphere = evoked_csd_sphere
    warn_evoked = evoked.copy()
    warn_evoked.info['bads'].append(warn_evoked.ch_names[3])
    with pytest.raises(ValueError, match='Either drop.*or interpolate'):
        compute_current_source_density(warn_evoked)

    with pytest.raises(TypeError, match='must be an instance of'):
        compute_current_source_density(None)

    fail_evoked = evoked.copy()
    with pytest.raises(ValueError, match='Zero or infinite position'):
        for ch in fail_evoked.info['chs']:
            ch['loc'][:3] = np.array([0, 0, 0])
        compute_current_source_density(fail_evoked, sphere=sphere)

    with pytest.raises(ValueError, match='Zero or infinite position'):
        fail_evoked.info['chs'][3]['loc'][:3] = np.inf
        compute_current_source_density(fail_evoked, sphere=sphere)

    with pytest.raises(ValueError, match='No EEG channels found.'):
        fail_evoked = evoked.copy()
        fail_evoked.set_channel_types({ch_name: 'ecog' for ch_name in
                                       fail_evoked.ch_names})
        compute_current_source_density(fail_evoked, sphere=sphere)

    with pytest.raises(TypeError, match='lambda2'):
        compute_current_source_density(evoked, lambda2='0', sphere=sphere)

    with pytest.raises(ValueError, match='lambda2 must be between 0 and 1'):
        compute_current_source_density(evoked, lambda2=2, sphere=sphere)

    with pytest.raises(TypeError, match='stiffness must be'):
        compute_current_source_density(evoked, stiffness='0', sphere=sphere)

    with pytest.raises(ValueError, match='stiffness must be non-negative'):
        compute_current_source_density(evoked, stiffness=-2, sphere=sphere)

    with pytest.raises(TypeError, match='n_legendre_terms must be'):
        compute_current_source_density(evoked, n_legendre_terms=0.1,
                                       sphere=sphere)

    with pytest.raises(ValueError, match=('n_legendre_terms must be '
                                          'greater than 0')):
        compute_current_source_density(evoked, n_legendre_terms=0,
                                       sphere=sphere)

    with pytest.raises(ValueError, match='sphere must be'):
        compute_current_source_density(evoked, sphere=-0.1)

    with pytest.raises(ValueError, match=('sphere radius must be '
                                          'greater than 0')):
        compute_current_source_density(evoked, sphere=(-0.1, 0., 0., -1.))

    with pytest.raises(TypeError):
        compute_current_source_density(evoked, copy=2, sphere=sphere)

    # gh-7859
    raw = RawArray(evoked.data, evoked.info)
    epochs = Epochs(
        raw, [[0, 0, 1]], tmin=0, tmax=evoked.times[-1] - evoked.times[0],
        baseline=None, preload=False, proj=False)
    epochs.drop_bad()
    assert len(epochs) == 1
    assert_allclose(epochs.get_data()[0], evoked.data)
    with pytest.raises(RuntimeError, match='Computing CSD requires.*preload'):
        compute_current_source_density(epochs)
    epochs.load_data()
    raw = compute_current_source_density(raw)
    assert not np.allclose(raw.get_data(), evoked.data)
    evoked = compute_current_source_density(evoked)
    assert_allclose(raw.get_data(), evoked.data)
    epochs = compute_current_source_density(epochs)
    assert_allclose(epochs.get_data()[0], evoked.data)


def test_csd_fif():
    """Test applying CSD to FIF data."""
    raw = read_raw_fif(raw_fname).load_data()
    raw.info['bads'] = []
    picks = pick_types(raw.info, meg=False, eeg=True)
    assert 'csd' not in raw
    orig_eeg = raw.get_data('eeg')
    assert len(orig_eeg) == 60
    raw_csd = compute_current_source_density(raw)
    assert 'eeg' not in raw_csd
    new_eeg = raw_csd.get_data('csd')
    assert not (orig_eeg == new_eeg).any()

    # reset the only things that should change, and assert objects are the same
    assert raw_csd.info['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_CSD
    raw_csd.info['custom_ref_applied'] = 0
    for pick in picks:
        ch = raw_csd.info['chs'][pick]
        assert ch['coil_type'] == FIFF.FIFFV_COIL_EEG_CSD
        assert ch['unit'] == FIFF.FIFF_UNIT_V_M2
        ch.update(coil_type=FIFF.FIFFV_COIL_EEG, unit=FIFF.FIFF_UNIT_V)
        raw_csd._data[pick] = raw._data[pick]
    assert object_diff(raw.info, raw_csd.info) == ''


run_tests_if_main()
