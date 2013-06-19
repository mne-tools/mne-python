# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal
from itertools import product

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
    n_components = [picks.size]

    for this_n_comp, this_picks in product(n_components, pick_components):
        csp = CSP(n_components=this_n_comp, pick_components=this_picks)

        print csp  # XXX : to test repr

        csp.fit(epochs, epochs.events[:, -1])
        y = epochs.events[:, -1]
        X = csp.fit_transform(epochs, y)
        assert_true(csp.filters_.shape == (this_n_comp, this_n_comp))
        assert_true(csp.patterns_.shape == (this_n_comp, this_n_comp))
        assert_array_equal(csp.fit(epochs, y).transform(epochs), X)

        # test init exception
        assert_raises(ValueError, csp.fit, epochs,
                      np.zeros_like(epochs.events))

        sources = csp.get_sources_epochs(epochs, pick_components=this_picks)
        assert_true(sources.shape[1] == len(this_picks))
