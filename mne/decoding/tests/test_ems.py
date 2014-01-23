# Author: Denis A. Engemann <d.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal

from mne import fiff, Epochs, read_events
from mne.decoding.csp import CSP
from mne.utils import _TempDir, requires_sklearn
from mne.decoding import compute_ems
from scipy.io import loadmat

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
curdir = op.join(op.realpath(op.curdir))

raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)


@requires_sklearn
def test_ems():
    """Test event-matched spatial filters
    """
    raw = fiff.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), preload=True)
    epochs_data = epochs.get_data()
    n_channels = epochs_data.shape[1]
    conditions = [np.intp(epochs.events[:, 2] == k) for k in (1, 3)]
    """'bootrtap'
    scipy.io.savemat('test_ems.mat', {'data': np.transpose(epochs_data,
                                                           [1, 2, 0]),
                                      'conds': conditions})
    # matlab
    epochs = load('test_ems.mat')
    [trl, sfl] = ems_ncond(epochs.data, boolean(epochs.conds))
    save('sfl.mat', 'sfl')
    """
    trial_surrogates, spatial_filter = compute_ems(epochs_data, conditions)
    trial_surrogates2, spatial_filter2 = [loadmat(op.join(curdir, k))[k[:3]]
                                          for k in ['trl.mat', 'sfl.mat']]
    for a, b in [(trial_surrogates, trial_surrogates2),
                 (spatial_filter, spatial_filter2)]:
        assert_equal(a.shape, b.shape)
        assert_array_almost_equal(a, b, 15)
