import os.path as op

from numpy.testing import assert_array_almost_equal

import mne
from ..fiff import fiff_open, read_evoked, pick_types
from ..datasets import sample

fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-cov.fif')
raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test_raw.fif')


def test_io_cov():
    """Test IO for noise covariance matrices
    """
    fid, tree, _ = fiff_open(fname)
    cov_type = 1
    cov = mne.read_cov(fid, tree, cov_type)
    fid.close()

    mne.write_cov_file('cov.fif', cov)

    fid, tree, _ = fiff_open('cov.fif')
    cov2 = mne.read_cov(fid, tree, cov_type)
    fid.close()

    assert_array_almost_equal(cov['data'], cov2['data'])


def test_cov_estimation():
    """Test estimation of noise covariance from raw data
    """
    raw = mne.fiff.Raw(raw_fname)
    # Set up pick list: MEG + STI 014 - bad channels
    want_meg = True
    want_eeg = False
    want_stim = False

    picks = pick_types(raw.info, meg=want_meg, eeg=want_eeg,
                            stim=want_stim, exclude=raw.info['bads'])

    full_cov = mne.Covariance(kind='full')
    full_cov.estimate_from_raw(raw, picks=picks)

    diagonal_cov = mne.Covariance(kind='diagonal')
    diagonal_cov.estimate_from_raw(raw, picks=picks)
    # XXX : test something


def test_whitening_cov():
    """Whitening of evoked data and leadfields
    """
    data_path = sample.data_path('.')
    fwd_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-meg-eeg-oct-6-fwd.fif')
    ave_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-ave.fif')
    cov_fname = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-cov.fif')

    # Reading
    ave = read_evoked(ave_fname, setno=0, baseline=(None, 0))
    fwd = mne.read_forward_solution(fwd_fname)

    cov = mne.Covariance()
    cov.load(cov_fname)

    ave_whiten, fwd_whiten, W = cov.whiten_evoked_and_forward(ave, fwd)
    # XXX : test something
