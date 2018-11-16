"""
Example on sleep data
=====================

"""

import mne

my_event_id = {'Sleep stage ?': 1,
               'Sleep stage W': 2,
               'Sleep stage 1': 3,
               'Sleep stage 2': 4,
               'Sleep stage 3': 5,
               'Sleep stage R': 6}

# names of edf and annot files
edf_f = "SC4001E0-PSG.edf"
annot_f = "SC4001EC-Hypnogram.edf"

# read raw edf
raw = mne.io.read_raw_edf(edf_f, preload=True)

# names of EEG and EOG channels to use
channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']

# set annotations
annot = mne.read_annotations(annot_f)
raw.set_annotations(annot)

# get the events and the event ids
events, event_id = mne.events_from_annotations(raw)

# perform epoching
epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=30., baseline=None)
