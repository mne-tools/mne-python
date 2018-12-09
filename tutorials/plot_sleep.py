"""
Example on sleep data
=====================

XXX add formal description of what we do here

"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD Style.

import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.datasets.sleep_physionet import fetch_data

psg_fname, hyp_fname = fetch_data(subjects=[0])[0]

raw = mne.io.read_raw_edf(psg_fname, stim_channel=False)
annotations = mne.read_annotations(hyp_fname)
raw.set_annotations(annotations)
raw.plot(duration=60)


mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}
raw.set_channel_types(mapping)

##############################################################################
# Plot hypnogram

desc2int = {'Sleep stage W': 6,
            'Sleep stage 1': 4,
            'Sleep stage 2': 3,
            'Sleep stage 3': 2,
            'Sleep stage 4': 1,
            'Sleep stage R': 5}
int2desc = {v: k for k, v in desc2int.items()}

hypnogram = pd.Series(annotations.description, index=annotations.onset)
hypnogram = hypnogram[hypnogram.isin(desc2int.keys())]  # keep only stages
hypnogram = hypnogram.iloc[1:]  # remove first annotation

plt.figure()
hypnogram.value_counts().plot(kind='barh')
plt.tight_layout()

plt.figure()
ax = hypnogram.replace(desc2int).dropna().plot()
ax.set_yticks(range(1, 7))
ax.set_yticklabels([int2desc[k] for k in range(1, 7)])
plt.tight_layout()

##############################################################################
# Epoching

events, event_id = mne.events_from_annotations(raw)
del event_id['Sleep stage ?']

tmax = 30. - 1. / raw.info['sfreq']
epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=tmax, baseline=None)

_, ax = plt.subplots()

for stage in zip(epochs.event_id):
    epochs[stage].plot_psd(area_mode=None, ax=ax, fmax=20.)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
[line.set_color(color) for line, color in zip(ax.get_lines(), colors)]
plt.legend(list(epochs.event_id.keys()))
