import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal
import numpy as np
from scipy import linalg

from mne.cov import regularize
from mne import read_cov, Epochs, merge_events, \
               find_events, compute_raw_data_covariance, \
               compute_covariance
from mne.fiff import Raw, pick_channels_cov, pick_channels

cov_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-cov.fif')
cov_km_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-km-cov.fif')
raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')
erm_cov_fname = op.join('mne', 'fiff', 'tests', 'data',
                     'test_erm-cov.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    cov = read_cov(cov_fname)
    cov.save('cov.fif')
    cov2 = read_cov('cov.fif')
    assert_array_almost_equal(cov.data, cov2.data)

    cov['bads'] = ['EEG 039']
    cov_sel = pick_channels_cov(cov, exclude=cov['bads'])
    assert_true(cov_sel['dim'] == (len(cov['data']) - len(cov['bads'])))
    assert_true(cov_sel['data'].shape == (cov_sel['dim'], cov_sel['dim']))
    cov_sel.save('cov.fif')


def test_cov_estimation_on_raw_segment():
    """Estimate raw on continuous recordings (typically empty room)
    """
    raw = Raw(raw_fname)
    cov = compute_raw_data_covariance(raw)
    cov_mne = read_cov(erm_cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    print (linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro'))
    assert_true(linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 1e-6

    # test IO when computation done in Python
    cov.save('test-cov.fif')  # test saving
    cov_read = read_cov('test-cov.fif')
    assert_true(cov_read.ch_names == cov.ch_names)
    assert_true(cov_read.nfree == cov.nfree)
    assert_true((linalg.norm(cov.data - cov_read.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 1e-5)

    # test with a subset of channels
    picks = pick_channels(raw.ch_names, include=raw.ch_names[:5])
    cov = compute_raw_data_covariance(raw, picks=picks)
    assert_true(cov_mne.ch_names[:5] == cov.ch_names)
    assert_true(linalg.norm(cov.data - cov_mne.data[picks][:, picks],
                ord='fro') / linalg.norm(cov.data, ord='fro')) < 1e-6


def test_cov_estimation_with_triggers():
    """Estimate raw with triggers
    """
    raw = Raw(raw_fname)
    events = find_events(raw)
    event_ids = [1, 2, 3, 4]
    reject = dict(grad=10000e-13, mag=4e-12, eeg=80e-6, eog=150e-6)

    # cov with merged events and keep_sample_mean=True
    events_merged = merge_events(events, event_ids, 1234)
    epochs = Epochs(raw, events_merged, 1234, tmin=-0.2, tmax=0,
                    baseline=(-0.2, -0.1), proj=True,
                    reject=reject, preload=True)

    cov = compute_covariance(epochs, keep_sample_mean=True)
    cov_mne = read_cov(cov_km_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 0.005)

    # Test with tmin and tmax (different but not too much)
    cov_tmin_tmax = compute_covariance(epochs, tmin=-0.19, tmax=-0.01)
    assert_true(np.all(cov.data != cov_tmin_tmax.data))
    assert_true((linalg.norm(cov.data - cov_tmin_tmax.data, ord='fro')
            / linalg.norm(cov_tmin_tmax.data, ord='fro')) < 0.05)

    # cov using a list of epochs and keep_sample_mean=True
    epochs = [Epochs(raw, events, ev_id, tmin=-0.2, tmax=0,
              baseline=(-0.2, -0.1), proj=True, reject=reject)
              for ev_id in event_ids]

    cov2 = compute_covariance(epochs, keep_sample_mean=True)
    assert_array_almost_equal(cov.data, cov2.data)
    assert_true(cov.ch_names == cov2.ch_names)

    # cov with keep_sample_mean=False using a list of epochs
    cov = compute_covariance(epochs, keep_sample_mean=False)
    cov_mne = read_cov(cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 0.005)

    # test IO when computation done in Python
    cov.save('test-cov.fif')  # test saving
    cov_read = read_cov('test-cov.fif')
    assert_true(cov_read.ch_names == cov.ch_names)
    assert_true(cov_read.nfree == cov.nfree)
    assert_true((linalg.norm(cov.data - cov_read.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 1e-5)


def test_arithmetic_cov():
    """Test arithmetic with noise covariance matrices
    """
    cov = read_cov(cov_fname)
    cov_sum = cov + cov
    assert_array_almost_equal(2 * cov.nfree, cov_sum.nfree)
    assert_array_almost_equal(2 * cov.data, cov_sum.data)
    assert_true(cov.ch_names == cov_sum.ch_names)

    cov += cov
    assert_array_almost_equal(cov_sum.nfree, cov.nfree)
    assert_array_almost_equal(cov_sum.data, cov.data)
    assert_true(cov_sum.ch_names == cov.ch_names)


def test_regularize_cov():
    """Test cov regularization
    """
    noise_cov = read_cov(cov_fname)
    raw = Raw(raw_fname)
    # Regularize noise cov
    reg_noise_cov = regularize(noise_cov, raw.info,
                               mag=0.1, grad=0.1, eeg=0.1, proj=True)
    assert_true(noise_cov['dim'] == reg_noise_cov['dim'])
    assert_true(noise_cov['data'].shape == reg_noise_cov['data'].shape)
    assert_true(np.mean(noise_cov['data'] < reg_noise_cov['data']) < 0.08)
