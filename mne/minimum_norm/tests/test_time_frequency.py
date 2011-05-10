import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from ...datasets import sample
from ... import fiff, find_events, Epochs
from ..inverse import read_inverse_operator
from ..time_frequency import source_induced_power


examples_folder = op.join(op.dirname(__file__), '..', '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname_inv = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis-meg-oct-6-meg-inv.fif')
fname_data = op.join(data_path, 'MEG', 'sample',
                                        'sample_audvis_raw.fif')


def test_tfr_with_inverse_operator():
    """Test MNE inverse computation"""

    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = fiff.Raw(fname_data)
    events = find_events(raw)
    inverse_operator = read_inverse_operator(fname_inv)

    include = []
    exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                                    stim=False, include=include, exclude=exclude)

    # Load condition 1
    event_id = 1
    events = events[:3]  # take 3 events to keep the computation time low
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                        preload=True)

    # Compute a source estimate per frequency band
    bands = dict(alpha=[10, 10])

    stcs = source_induced_power(epochs, inverse_operator, bands, n_cycles=2,
                                use_fft=False)

    stc = stcs['alpha']
    assert len(stcs) == len(bands.keys())
    assert np.all(stc.data > 0)
    assert_array_almost_equal(stc.times, epochs.times)

