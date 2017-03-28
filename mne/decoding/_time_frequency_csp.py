# laura gwilliams leg5@nyu.edu
# run classifier at different time and frequency windows

# Steps:
# 1) preprocess raw into epochs band-passed at different frequencies and
#    cropped with X many cycles
# 2) fit classifier with cross validation
# 3) predict
# 4) score

import numpy as np

from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.filter import band_pass_filter
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
from sklearn.model_selection import ShuffleSplit


class TimeFrequencyCSP():

    def __init__(self, X, estimator, n_jobs=1, freq_bins, n_cycles, tmin=None,
                 tmax=None, sfreq=None, scorer=None):
        self.n_cycles = n_cycles
        self.freq_bins = freq_bins
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self._csp = []  # make this a numpy array not a list

    def transform(self, X, decim=None):

        # init array to hold band-passed data
        X_bandpass = np.zeros(X.shape.tolist()[0:2] + [len(freq_bins)])
        for freq_ii, freq_range in enumerate(freq_bins):
            # filter the data at the desired frequency
            X_bandpass = self.fit_bandpass(X, freq_range)
            # then get data with a sliding window
            X_window = self.fit_windows(self, X_bandpass, self.n_cycles,
                                  freq_range[0])

        # keep the time axis as the final one
        Xt = np.swapaxes(Xt, -1, -2)

        if decim:
            Xt = Xt[::decim]

        return Xt

    def fit(self, X, y):

        Xt = self.transform(X)
        self._csp.append(self.estimator.fit(Xt, y))
        return self

    def fit_bandpass(self, X, freq_range):

        X_bandpass = band_pass_filter(X, self.sfreq,
                                      freq_range[0], freq_range[1])

        return X_bandpass

    def fit_windows(self, X_bandpass, n_cycles, freq_min):
        """
        Run a sliding window of size n_cycles over the data to make a series
        of windowed epochs.
        """

        # compute the window from the number of cycles and the min freq
        window_size = float(n_cycles)/float(freq_min)
        window_max = self.tmax - window_size
        window_length = (np.abs(self.tmin) + window_max)*self.sfreq

        # init array to hold windowed data.
        # desired shape is: trial x channel x frequency x window
        X_window = np.zeros(list(X.shape)[0:-1] + [window_length])

        # slide the window over the data
        for shift_idx, shift_ii in enumerate(range(self.tmin, window_max)):
            time = (shift_ii, shift_ii+window_size)  # window dims
            # cut out data for this time window
            X_window[..., shift_idx] = X[..., time[0]:time[1]]

        return X_window

    def predict(self, X):

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimator_'):
            raise RuntimeError('Please fit models before trying to predict')

        Xt = self.transform(X)
        return self.estimator_.predict(Xt)

    def score(self, X, y):
        y_pred = self.predict(X)

        self.scores_ = self.scorer(y, y_pred)
        return self.scores_


# preprocessing here
def time_frequency_preprocess(raw_fnames, freq_bins, n_cycles, tmin, tmax,
                              event_id=None):

    epoch_list = []

    # loop through each frequency window
    for freq_n, freq in enumerate(freq_bins):

        # load sample data
        raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
        raw = concatenate_raws(raw_files)

        # Apply band-pass filter at the given frequency
        raw.filter(freq[0], freq[1])

        # make epochs
        events = find_events(raw, shortest_event=0, stim_channel='STI 014')
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                           eog=False, exclude='bads')
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)

        # compute the window from the number of cycles and the min freq
        window_size = float(n_cycles)/float(freq[0])

        # loop through each time window
        for cycle_n in range(n_cycles):

            time = (window_size*cycle_n, window_size*(cycle_n+1))

            # cut out data for this time window
            time_region_interest = epochs.copy().crop(time[0], time[1])
            epoch_list.append(time_region_interest)

    return epoch_list, epochs.events[:, 2]

tmin, tmax = -0.5, 4.
fmin, fmax, fstep = 5, 75, 10  # Hz
freq_bins = [(f, f+fstep) for f in np.arange(fmin, fmax, fstep)]
n_cycles = 4

# get data
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
raw_fnames = eegbci.load_data(subject, runs)
event_id = dict(hands=2, feet=3)
epoch_list, y = time_frequency_preprocess(raw_fnames, freq_bins, n_cycles,
                                          tmin, tmax, event_id)

# Define a monte-carlo cross-validation generator (reduce variance):
n_splits = 10
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42,
                  train_size=None)

# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg=None, log=True)
clf = Pipeline([('CSP', csp), ('LDA', lda)])
TFCSP = TimeFrequencyCSP(estimator=clf)

# collect scores for each frequency/cycle
scores = np.zeros([len(epoch_list), n_splits])
for epoch_n, epochs in enumerate(epoch_list):
    for n_fold, (train, test) in enumerate(cv.split(epochs)):
        TFCSP.fit(epochs[train], y[train])
        scores[epoch_n, n_fold] = (TFCSP.score(epochs[test], y[test]))
        print TFCSP.y_pred_
scores = np.mean(scores, axis=1)

# The goal is to have the following workflow:
X = epochs.get_data()
y = epochs.events[:, 2]

TFCSP = TimeFrequencyCSP(estimator=clf)
TFCSP.fit(X_train, y_train)  # calls transform in order to fit the classifier over freq+time
TFCSP.score(X_test, y_test)  # calls predict and then scores
TFCSP.plot()
