"""
============================================================================
Decoding in time-frequency space data using the Common Spatial Pattern (CSP)
============================================================================

The time-frequency decomposition is estimated by iterating over raw data that
has been band-passed at different frequencies. This is used to compute a
covariance matrix over a rolling time-window and extract the CSP filtered
signals. A linear discriminant classifier is then applied to these signals.
"""
# Authors: Laura Gwilliams <laura.gwilliams@nyu.edu>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Alex Barachant <alexandre.barachant@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from mne import Epochs, find_events, create_info
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

###############################################################################
# Set parameters and read data
event_id = dict(hands=2, feet=3)  # motor imagery: hands vs feet
subject = 1
runs = [6, 10, 14]
raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_files)

# Extract information from the raw file
sfreq = raw.info['sfreq']
events = find_events(raw, shortest_event=0, stim_channel='STI 014')
raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Assemble the classifier using scikit-learn pipeline
clf = make_pipeline(CSP(n_components=4, reg=None, log=True),
                    LinearDiscriminantAnalysis())
n_splits = 5  # how many folds to use for cross-validation
cv = KFold(n_splits=n_splits, shuffle=True)

# Classification & Time-frequency parameters
tmin, tmax = -.200, 2.000
n_cycles = 10.  # how many complete cycles: used to define window size
min_freq = 5.
max_freq = 25.
n_freqs = 8  # how many frequency bins to use

# Assemble list of frequency range tuples
freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = zip(freqs[:-1], freqs[1:])  # make freqs into a list of tuples

# Infer window spacing from the max freq and number of cycles to avoid gaps
window_spacing = (n_cycles / np.max(freqs) / 2.)
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

###############################################################################
# Loop through frequencies and time, apply classifier and save scores

# init scores
scores = np.zeros((n_freqs - 1, n_windows))

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):

    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(fmin, fmax, n_jobs=1, fir_design='firwin')

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(raw_filter, events, event_id, tmin - w_size, tmax + w_size,
                    proj=False, baseline=None, preload=True)
    epochs.drop_bad()
    y = epochs.events[:, 2] - 2

    # Roll covariance, csp and lda over time
    for t, w_time in enumerate(centered_w_times):

        # Center the min and max of the window
        w_tmin = w_time - w_size / 2.
        w_tmax = w_time + w_size / 2.

        # Crop data into time-window of interest
        X = epochs.copy().crop(w_tmin, w_tmax).get_data()

        # Save mean scores over folds for each frequency and time window
        scores[freq, t] = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                  cv=cv, n_jobs=1), axis=0)

###############################################################################
# Plot results

# Set up time frequency object
av_tfr = AverageTFR(create_info(['freq'], sfreq), scores[np.newaxis, :],
                    centered_w_times, freqs[1:], 1)

chance = np.mean(y)  # set chance level to white in the plot
av_tfr.plot([0], vmin=chance, title="Time-Frequency Decoding Scores",
            cmap=plt.cm.Reds)
