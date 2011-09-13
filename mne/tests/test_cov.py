import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal
from scipy import linalg

from .. import Covariance, read_cov, Epochs, merge_events, \
               find_events, write_cov_file, compute_raw_data_covariance, \
               compute_covariance
from ..fiff import fiff_open, Raw

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
    fid, tree, _ = fiff_open(cov_fname)
    cov_type = 1
    cov = read_cov(fid, tree, cov_type)
    fid.close()

    write_cov_file('cov.fif', cov)

    fid, tree, _ = fiff_open('cov.fif')
    cov2 = read_cov(fid, tree, cov_type)
    fid.close()

    assert_array_almost_equal(cov['data'], cov2['data'])


def test_cov_estimation_on_raw_segment():
    """Estimate raw on continuous recordings (typically empty room)
    """
    raw = Raw(raw_fname)
    cov = compute_raw_data_covariance(raw)
    cov_mne = Covariance(erm_cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    print (linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro'))
    assert_true(linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 1e-6


def test_cov_estimation_with_triggers():
    """Estimate raw with triggers
    """
    raw = Raw(raw_fname)
    events = find_events(raw)
    event_ids = [1, 2, 3, 4]
    reject = dict(grad=10000e-13, mag=4e-12, eeg=80e-6, eog=150e-6)

    events = merge_events(events, event_ids, 1234)
    epochs = Epochs(raw, events, 1234, tmin=-0.2, tmax=0,
                        baseline=(-0.2, -0.1), proj=True,
                        reject=reject)

    cov = compute_covariance(epochs, keep_sample_mean=True)
    cov_mne = Covariance(cov_km_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 0.005)

    cov = compute_covariance(epochs, keep_sample_mean=False)
    cov_mne = Covariance(cov_fname)
    assert_true(cov_mne.ch_names == cov.ch_names)
    assert_true((linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 0.06)
