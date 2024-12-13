"""
.. _ex-decoding-csp-eeg-timefreq:

====================================================================
Decoding in time-frequency space using Common Spatial Patterns (CSP)
====================================================================

The time-frequency decomposition is estimated by iterating over raw data that
has been band-passed at different frequencies. This is used to compute a
covariance matrix over each epoch or a rolling time-window and extract the CSP
filtered signals. A linear discriminant classifier is then applied to these
signals.
"""
# Authors: Laura Gwilliams <laura.gwilliams@nyu.edu>
#          Jean-Rémi King <jeanremi.king@gmail.com>
#          Alex Barachant <alexandre.barachant@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from mne import Epochs, create_info
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import AverageTFRArray

# %%
# Set parameters and read data
subject = 1
runs = [6, 10, 14]
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f) for f in raw_fnames])
raw.annotations.rename(dict(T1="hands", T2="feet"))

# Extract information from the raw file
sfreq = raw.info["sfreq"]
raw.pick(picks="eeg", exclude="bads")
raw.load_data()

# Assemble the classifier using scikit-learn pipeline
clf = make_pipeline(
    CSP(n_components=4, reg=None, log=True, norm_trace=False),
    LinearDiscriminantAnalysis(),
)
n_splits = 3  # for cross-validation, 5 is better, here we use 3 for speed
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Classification & time-frequency parameters
tmin, tmax = -0.200, 2.000
n_cycles = 10.0  # how many complete cycles: used to define window size
min_freq = 8.0
max_freq = 20.0
n_freqs = 6  # how many frequency bins to use

# Assemble list of frequency range tuples
freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples

# Infer window spacing from the max freq and number of cycles to avoid gaps
window_spacing = n_cycles / np.max(freqs) / 2.0
centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
n_windows = len(centered_w_times)

# Instantiate label encoder
le = LabelEncoder()

# %%
# Loop through frequencies, apply classifier and save scores

# init scores
freq_scores = np.zeros((n_freqs - 1,))

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):
    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.0)  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(
        fmin, fmax, fir_design="firwin", skip_by_annotation="edge"
    )

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(
        raw_filter,
        event_id=["hands", "feet"],
        tmin=tmin - w_size,
        tmax=tmax + w_size,
        proj=False,
        baseline=None,
        preload=True,
    )
    epochs.drop_bad()
    y = le.fit_transform(epochs.events[:, 2])

    X = epochs.get_data(copy=False)

    # Save mean scores over folds for each frequency and time window
    freq_scores[freq] = np.mean(
        cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc", cv=cv), axis=0
    )

# %%
# Plot frequency results

plt.bar(
    freqs[:-1], freq_scores, width=np.diff(freqs)[0], align="edge", edgecolor="black"
)
plt.xticks(freqs)
plt.ylim([0, 1])
plt.axhline(
    len(epochs["feet"]) / len(epochs), color="k", linestyle="--", label="chance level"
)
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Decoding Scores")
plt.title("Frequency Decoding Scores")

# %%
# Loop through frequencies and time, apply classifier and save scores

# init scores
tf_scores = np.zeros((n_freqs - 1, n_windows))

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):
    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.0)  # in seconds

    # Apply band-pass filter to isolate the specified frequencies
    raw_filter = raw.copy().filter(
        fmin, fmax, fir_design="firwin", skip_by_annotation="edge"
    )

    # Extract epochs from filtered data, padded by window size
    epochs = Epochs(
        raw_filter,
        event_id=["hands", "feet"],
        tmin=tmin - w_size,
        tmax=tmax + w_size,
        proj=False,
        baseline=None,
        preload=True,
    )
    epochs.drop_bad()
    y = le.fit_transform(epochs.events[:, 2])

    # Roll covariance, csp and lda over time
    for t, w_time in enumerate(centered_w_times):
        # Center the min and max of the window
        w_tmin = w_time - w_size / 2.0
        w_tmax = w_time + w_size / 2.0

        # Crop data into time-window of interest
        X = epochs.get_data(tmin=w_tmin, tmax=w_tmax, copy=False)

        # Save mean scores over folds for each frequency and time window
        tf_scores[freq, t] = np.mean(
            cross_val_score(estimator=clf, X=X, y=y, scoring="roc_auc", cv=cv), axis=0
        )

# %%
# Plot time-frequency results

# Set up time frequency object
av_tfr = AverageTFRArray(
    info=create_info(["freq"], sfreq),
    data=tf_scores[np.newaxis, :],
    times=centered_w_times,
    freqs=freqs[1:],
    nave=1,
)

chance = np.mean(y)  # set chance level to white in the plot
av_tfr.plot(
    [0], vlim=(chance, None), title="Time-Frequency Decoding Scores", cmap=plt.cm.Reds
)
