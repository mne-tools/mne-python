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

import mne
from mne.datasets import sample
from mne.preprocessing import eog_regression
from matplotlib import pyplot as plt

print(__doc__)

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'

# Read raw data
raw = mne.io.Raw(raw_fname, preload=True)
events = mne.find_events(raw, 'STI 014')

# Bandpass filter the EEG and EOG
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
raw.filter(0.3, 30, method='iir', picks=picks)

# Create evokeds before EOG correction
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
tmin, tmax = -0.2, 0.5
event_ids = {'AudL': 1, 'AudR': 2, 'VisL': 3, 'VisR': 4}
epochs_before = mne.Epochs(raw, events, event_ids, tmin, tmax, picks=picks,
                           baseline=(tmin, 0), preload=True)
evoked_before = epochs_before.average()


# Estimate blink onsets and create blink epochs
eog_event_id = 512
eog_events = mne.preprocessing.find_eog_events(raw, eog_event_id)
blink_epochs = mne.Epochs(raw, eog_events, eog_event_id, tmin=-0.5, tmax=0.5,
                          picks=picks, baseline=(-0.5, -0.3), preload=True)

# Perform regression and remove EOG
raw_clean, weights, _ = eog_regression(raw, blink_epochs.average('all'))

# Create epochs after EOG correction
picks = mne.pick_types(raw_clean.info, meg=False, eeg=True)
epochs_after = mne.Epochs(raw_clean, events, event_ids, tmin, tmax,
                          baseline=(tmin, 0), picks=picks, preload=True)
evoked_after = epochs_after.average()

# Show the filter weights in a topomap
eeg_ch_info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))
fig, ax = plt.subplots()
im, _ = mne.viz.plot_topomap(weights[0], eeg_ch_info, outlines='skirt')
fig.colorbar(im)
ax.set_title('Regression weights')

# Plot the evoked before and after EOG regression
evoked_before.plot()
plt.ylim(-6, 6)
plt.title('Before EOG regression')

evoked_after.plot()
plt.ylim(-6, 6)
plt.title('After EOG regression')
