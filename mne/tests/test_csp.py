# Author: Denis Engemann <d.engemann@fz-juelich.de>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import os
import os.path as op
import warnings

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from itertools import product

from mne import fiff, Epochs, read_events, cov
from mne.csp import CSP
from mne.utils import _TempDir, requires_sklearn

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id   = dict(aud_l=1, vis_l=3)
start, stop = 0, 8  # if stop is too small pca may fail in some cases, but
                    # we're okay on this file
raw = fiff.Raw(raw_fname, preload=True)#.crop(0, stop, False)

events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=False,
                        exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)

epochs_list = [epochs[k] for k in event_id]

@requires_sklearn
def test_csp():
    """Test Common Spatial Patterns algorithm on epochs
    """
    # setup parameter
    try:
        import sklearn.covariance
        noise_cov = [None, sklearn.covariance.OAS()]
    except:
        noise_cov = [None]
    
    # removed None cases to speed up...
    pick_components = [np.array([0,-1]), np.arange(10),-1*np.arange(10)]
    n_components = [picks.shape[0]]
    picks_ = [picks]
    iter_csp_params = product(noise_cov, components, n_components, picks_)

    # test essential core functionality
    for n_cov, n_comp, max_n, pcks in iter_csp_params:
        #######################################################################
        # test CSP epochs decomposition

        csp = CSP(cov=n_cov, n_components=max_n)

        print csp  # to test repr

        # test init exception
        assert_raises(ValueError, csp.decompose_epochs, epochs, picks=pcks)

        csp.decompose_epochs(epochs_list, picks=pcks)
        print csp  # to test repr
        
        sources = csp.get_sources_epochs(epochs, pick_components=n_comp)
        assert_true(sources.shape[1] == n_comp.shape[0])


