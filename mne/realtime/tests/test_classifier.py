# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true, assert_raises

from mne.realtime import Scaler

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8
raw = fiff.Raw(raw_fname, preload=True)

events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, stim=False, ecg=False, eog=False,
                        exclude='bads')
picks = picks[1:13:3]

epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                baseline=(None, 0), preload=True)


def test_Scaler():
    epochs_data = epochs.data()
    scaler = Scaler()
        