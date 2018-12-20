"""
Example on sleep data
=====================

XXX add formal description of what we do here

"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD Style.

import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet import fetch_data

psg_fname, hyp_fname = fetch_data(subjects=[0])[0]

raw = mne.io.read_raw_edf(psg_fname, stim_channel=False)
annotations = mne.read_annotations(hyp_fname)

# ##############################################################################

raw.set_annotations(annotations)
raw.plot(duration=60)

mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}
raw.set_channel_types(mapping)

##############################################################################
# Extract 30s events from annotations

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}
events, event_id = mne.events_from_annotations(raw,
                    event_id=annotation_desc_2_event_id, chunk_duration=30.)

del event_id['Sleep stage 4']  # remove duplicated event_id

mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'])

##############################################################################
# Epoching

tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=tmax, baseline=None)
print(epochs)

##############################################################################
# Plot the power spectrum density (PSD) in each stage

_, ax = plt.subplots()

for stage in zip(epochs.event_id):
    epochs[stage].plot_psd(area_mode=None, ax=ax, fmin=0.1, fmax=20.)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
[line.set_color(color) for line, color in zip(ax.get_lines(), colors)]
plt.legend(list(epochs.event_id.keys()))
