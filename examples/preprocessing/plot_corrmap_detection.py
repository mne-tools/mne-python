"""
==========================================================
Identify similar ICs across multiple datasets via CORRMAP
==========================================================

After fitting ICA to multiple data sets, CORRMAP [1]_
automatically identifies similar ICs in all sets based
on a manually selected template. These ICs can then be
removed, or further investigated.

References
----------
.. [1] Viola FC, et al. Semi-automatic identification of independent components
       representing EEG artifact. Clin Neurophysiol 2009, May; 120(5): 868-77.
"""

# Authors: Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne

from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing.ica import corrmap
from mne.datasets import sample

print(__doc__)

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 30, method='iir')
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, ecg=False,
                       stim=False, exclude='bads')

events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)
reject = dict(eog=250e-6)
tmin, tmax = -0.5, 0.75


###############################################################################
# 1) Fit ICA to all "subjects".
# In a real-world case, this would instead be multiple subjects/data sets,
# here we create artificial subsets

all_epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        proj=False, picks=picks, baseline=(None, 0),
                        preload=True, reject=None, verbose=False)

all_epochs = [all_epochs[start:stop] for start, stop in
              [(0, 100), (101, 200), (201, 300)]]

icas = [ICA(n_components=20, random_state=1).fit(epochs)
        for epochs in all_epochs]

# 2) Use corrmap to identify the maps best corresponding
#    to a pre-specified template across all subsets
#    (or, in the real world, multiple participant data sets)

template = (0, 0)
fig_template, fig_detected = corrmap(icas, template=template, label="blinks",
                                     show=True, threshold=.8)

# 3) Zeroing the identified blink components for all data sets
#    results in individually cleaned data sets. Specific components
#    can be accessed using the label_ attribute.

for ica in icas:
    print(ica.labels_)
