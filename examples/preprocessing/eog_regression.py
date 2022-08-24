"""
========================
EOG regression
========================

Reduce EOG artifacts by regressing the EOG channels onto the rest of the
signal.

References
----------
[1] Croft, R. J., & Barry, R. J. (2000). Removal of ocular artifact from
the EEG: a review. Clinical Neurophysiology, 30(1), 5-19.
http://doi.org/10.1016/S0987-7053(00)00055-1

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>

License: BSD (3-clause)
"""
# %%
# Do imports and load the MNE-Sample data.

import mne
from mne.datasets import sample
from mne.preprocessing import EOGRegression
from matplotlib import pyplot as plt

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.Raw(raw_fname, preload=True)
events = mne.find_events(raw, 'STI 014')

# For this example, we only operate on the EEG channels.
raw.pick(('eeg', 'eog'))

# Bandpass filter
raw.filter(0.3, 30, method='iir', picks=('eeg', 'eog'))

# %%
# Create a template of a blink
# ----------------------------
#
# Estimate blink onsets and create blink evokeds. EOG regression weights will
# be computed using this evoked data. This may yield slightly better weights
# than performing the regression using the raw data, as the averaging procedure
# yields a more "pure" recording of the EOG artifact with ongoing EEG
# suppressed.

eog_event_id = 512
eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id)
blink_epochs = mne.Epochs(raw, eog_events, eog_event_id, tmin=-0.5, tmax=0.5,
                          baseline=(-0.5, -0.3), preload=True)
blink_evoked = blink_epochs.average('all')

# %%
# Perform regression and remove EOG
# ---------------------------------
# We are now ready to perform the regression. We compute the regression weights
# on the blink template and apply them to the continuous data.

weights = EOGRegression().fit(blink_evoked)
raw_clean = weights.apply(raw, copy=True)

# Show the filter weights in a topomap
weights.plot()

# %%
# Before-after comparison
# -----------------------
# Let's compare the signal before and after cleaning with EOG regression. This
# is best visualized by cutting epochs and plotting the evoked potential.

tmin, tmax = -0.1, 0.5
event_ids = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
evoked_before = mne.Epochs(raw, events, event_ids, tmin, tmax,
                           baseline=(tmin, 0)).average()
evoked_after = mne.Epochs(raw_clean, events, event_ids, tmin, tmax,
                          baseline=(tmin, 0)).average()

# Create epochs after EOG correction
epochs_after = mne.Epochs(raw_clean, events, event_ids, tmin, tmax,
                          baseline=(tmin, 0))
evoked_after = epochs_after.average()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
evoked_before.plot(axes=ax[0])
ax[0].set_ylim(-6, 6)
ax[0].set_title('Evoked potential before EOG regression')

evoked_after.plot(axes=ax[1])
ax[1].set_ylim(-6, 6)
ax[1].set_title('Evoked potential after EOG regression')
fig.set_tight_layout(True)
