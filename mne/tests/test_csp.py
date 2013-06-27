# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal

from mne import fiff, Epochs, read_events
from mne.csp import CSP
from mne.utils import _TempDir, requires_sklearn

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file
raw = fiff.Raw(raw_fname, preload=True)

events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=False,
                        exclude='bads')
picks = picks[1:13:3]

epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)


@requires_sklearn
def test_csp():
    """Test Common Spatial Patterns algorithm on epochs
    """
    pick_components = [[0, -1], np.arange(3), -1 * np.arange(3)]
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]

    for this_picks in pick_components:
        csp = CSP(pick_components=this_picks)

        csp.fit(epochs_data, epochs.events[:, -1])
        y = epochs.events[:, -1]
        X = csp.fit_transform(epochs_data, y)
        assert_true(csp.filters_.shape == (n_channels, n_channels))
        assert_true(csp.patterns_.shape == (n_channels, n_channels))
        assert_array_equal(csp.fit(epochs_data, y).transform(epochs_data), X)

        # test init exception
        assert_raises(ValueError, csp.fit, epochs_data,
                      np.zeros_like(epochs.events))
        assert_raises(ValueError, csp.fit, epochs, y)
        assert_raises(ValueError, csp.transform, epochs, y)

        csp.pick_components = this_picks
        sources = csp.transform(epochs_data)
        assert_true(sources.shape[1] == len(this_picks))
