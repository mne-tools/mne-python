import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from sklearn.lda import LDA
# Time Split sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline

# #############################################################################
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_files)
raw.rename_channels(lambda x: x.strip('.'))
events = find_events(raw, shortest_event=0, stim_channel='STI 014')
raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Assemble a classifier
clf = make_pipeline(CSP(n_components=4, reg=None, log=True), LDA())

n_splits = 5
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

# Classification & Time-frequency parameters
tmin, tmax = -.200, 2.000
n_cycles = 10.
n_windows = 10  # this needs to be inferred from freq_max and n_cycles
n_freqs = 5
freqs = np.linspace(5., 25., n_freqs)

# init scores
scores = np.zeros((n_splits, n_freqs - 1, n_windows))
w_times = np.linspace(tmin, tmax, n_windows + 2)[1:-1]

for freq, (fmin, fmax) in enumerate(zip(freqs[:-1], freqs[1:])):
    # Apply band-pass filter
    raw_filter = raw.copy().filter(fmin, fmax, n_jobs=-1)

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw_filter, events, event_id, tmin, tmax, proj=False,
                    baseline=None, add_eeg_ref=False, preload=True)
    epochs.drop_bad()
    y = epochs.events[:, 2] - 2

    # Slice time window of interest to compute covariance.
    sfreq = raw.info['sfreq']
    w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

    # Roll covariance, csp and lda over time
    for t, w_time in enumerate(w_times):
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.
        X = epochs.copy().crop(w_tmin, w_tmax).get_data()
        # mean scores over splits directly
        scores[:, freq, t] = cross_val_score(estimator=clf,
                                             X=X, y=y,
                                             cv=cv, n_jobs=-1)


# Use TimeFreqEVoked for plotting
fig, ax = plt.subplots(1)
im = ax.matshow(scores.mean(0), extent=[tmin, tmax, freqs[0], freqs[-1]],
                origin='lower', aspect='auto')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Freqs (Hz)')
plt.colorbar(im, ax=ax)
plt.show()
