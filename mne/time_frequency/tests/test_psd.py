import numpy as np
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from ... import fiff, Epochs, read_events
from ...time_frequency import compute_raw_psd


raw_fname = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data',
                'test_raw.fif')

def test_psd():
    """Test PSD estimation
    """
    raw = fiff.Raw(raw_fname)

    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053'] # bads + 2 more

    # picks MEG gradiometers
    picks = fiff.pick_types(raw.info, meg='mag', eeg=False,
                                    stim=False, exclude=exclude)

    picks = picks[:2]

    tmin, tmax = 0, 10  # use the first 60s of data
    fmin, fmax = 2, 70  # look at frequencies between 5 and 70Hz
    NFFT = 124 # the FFT size (NFFT). Ideally a power of 2
    psds, freqs = compute_raw_psd(raw, tmin=tmin, tmax=tmax, picks=picks,
                                  fmin=fmin, fmax=fmax, NFFT=NFFT, n_jobs=1)

    assert_true(psds.shape == (len(picks), len(freqs)))
    assert_true(np.sum(freqs < 0) == 0)
    assert_true(np.sum(psds < 0) == 0)
