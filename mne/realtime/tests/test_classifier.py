# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_array_equal

from mne import fiff, read_events, Epochs
from mne.realtime import Scaler

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

raw = fiff.Raw(raw_fname, preload=True)
events = read_events(event_name)

picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=False,
                        exclude='bads')
picks = picks[1:13:3]

epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)


def test_Scaler():
    epochs_data = epochs.get_data()
    scaler = Scaler(epochs.info)
    y = epochs.events[:, -1]
    X = scaler.fit_transform(epochs_data, y)

    assert_true(X.shape == epochs_data.shape)
    assert_array_equal(scaler.fit(epochs_data, y).transform(epochs_data), X)

    # Test init exception
    assert_raises(ValueError, scaler.fit, epochs, y)
    assert_raises(ValueError, scaler.fit, epochs, y)
