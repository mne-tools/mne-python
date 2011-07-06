import os.path as op

from numpy.testing import assert_array_almost_equal
from scipy import linalg

import mne
from ..fiff import fiff_open, read_evoked, Raw
from ..datasets import sample

cov_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-cov.fif')
raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')
erm_cov_fname = op.join('mne', 'fiff', 'tests', 'data',
                     'test_erm-cov.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    fid, tree, _ = fiff_open(cov_fname)
    cov_type = 1
    cov = mne.read_cov(fid, tree, cov_type)
    fid.close()

    mne.write_cov_file('cov.fif', cov)

    fid, tree, _ = fiff_open('cov.fif')
    cov2 = mne.read_cov(fid, tree, cov_type)
    fid.close()

    assert_array_almost_equal(cov['data'], cov2['data'])


def test_cov_estimation_on_raw_segment():
    """Estimate raw on continuous recordings (typically empty room)
    """
    raw = Raw(raw_fname)
    cov = mne.compute_raw_data_covariance(raw)
    cov_mne = mne.Covariance(erm_cov_fname)
    assert cov_mne.ch_names == cov.ch_names
    print (linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro'))
    assert (linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 1e-6


def test_cov_estimation_with_triggers():
    """Estimate raw with triggers
    """
    raw = Raw(raw_fname)
    events = mne.find_events(raw)
    event_ids = [1, 2, 3, 4]
    cov = mne.compute_covariance(raw, events, event_ids, tmin=-0.2, tmax=0,
                               reject=dict(grad=10000e-13, mag=4e-12,
                                           eeg=80e-6, eog=150e-6),
                               keep_sample_mean=True, proj=True)
    cov_mne = mne.Covariance(cov_fname)
    assert cov_mne.ch_names == cov.ch_names
    assert (linalg.norm(cov.data - cov_mne.data, ord='fro')
            / linalg.norm(cov.data, ord='fro')) < 0.06


def test_whitening_cov():
    """Whitening of evoked data and leadfields
    """
    data_path = sample.data_path('.')
    ave_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-ave.fif')
    cov_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-cov.fif')

    # Reading
    evoked = read_evoked(ave_fname, setno=0, baseline=(None, 0))

    cov = mne.Covariance(cov_fname)
    cov.get_whitener(evoked.info)

    # XXX : test something
