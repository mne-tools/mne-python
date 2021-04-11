"""
.. _ex-decoding-ssd-csp-eeg:

============================================================================
Improving motor imagery decoding from EEG using Spatio Spectra Decomposition
============================================================================

The Spatio Spectra Decomposition (SSD) is a unsupervised spatial filtering
algorithm that can be used as a pre-processing approach for data dimensionality
reduction while the 1/f noise in the neural data is reduced
:footcite:`HaufeEtAl2014b`.
It is useful to capture induced activity during motor imagery.

In this example, SSD will be applied before extracting features with
the Common Spatial Patterns (CSP) method. A classifier will then be trained
using the extracted features from the SSD + CSP-filtered signals. The impact in
performance of using SSD before CSP will be shown.

:footcite:`NikulinEtAl2011`.
"""
# Authors: Victoria Peterson <victoriapeterson09@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import (Epochs, pick_types, events_from_annotations,
                 EvokedArray)
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from mne.decoding import SSD
###############################################################################
# Set parameters and read data

# epochs from 0.5 to 2.5 s after the cue onset will be used
tmin, tmax = 0.5, 2.5
event_id = dict(left=2, right=3)  # Motor imagery: left vs right hand
runs = [4, 8, 12]
subjects = [2, 7, 31, 34, 42, 56, 60, 62, 85, 100]
n_subjects = len(subjects)
scores_csp = np.zeros((n_subjects, 1))
std_csp = np.zeros((n_subjects, 1))
# Data filtering.
# we are going to filter data in the alpha band
freq_signal = [8, 12]
# SSD enhances signal-to-noise ratio, thus what is considered 'noise' should be
# defined. Typically bands of 2 Hz surrounding the frequencies of interest are
# taken.
freq_noise = freq_signal[0] - 2, freq_signal[1] + 2
# when applying SSD, data should be filtered in a broader band than for CSP,
# thus, a broader bandwidth will be used to filtering the raw data.
freq_ssd = freq_signal[0] - 3, freq_signal[1] + 3
# Since we are working with a highly electrode counting dataset (64 channels),
# it is interesting to see the impact of the number of component in SSD, i.e.,
# the impact in data dimensionality reduction before applying CSP.
# The minimum n_components is 4, since 4 CSP components were selected
# the maximum is the number of channels, here 64.
steps = 4
n_components = np.arange(4, 65, steps)
scores_ssd_csp_e = np.zeros((n_subjects, len(n_components)))
std_ssd_csp_e = np.zeros((n_subjects, len(n_components)))

for s_idx, subject in enumerate(subjects):
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    # we save the raw object before filtering to use it when SSD in applied.
    raw_ssd = raw.copy()
    # apply band-pass filter
    raw.filter(freq_signal[0], freq_signal[1], fir_design='firwin',
               skip_by_annotation='edge')
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    # Extract epochs
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    labels = epochs.events[:, -1] - 2
    epochs_data = epochs.get_data()
    del raw

    # Traditional CSP + LDA pipeline
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # define pipelines with monte-carlo simulations
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # pipeline methods
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    pipe_csp = Pipeline([('CSP', csp), ('LDA', lda)])
    # this is the accuracy we would like to improve by adding SSD into the
    # pipeline
    scores_csp[s_idx] = cross_val_score(pipe_csp, epochs_data, labels, cv=cv,
                                        n_jobs=1).mean()
    std_csp[s_idx] = cross_val_score(pipe_csp, epochs_data, labels, cv=cv,
                                     n_jobs=1).std()
    # SSD + CSP + LDA pipeline
    # ^^^^^^^^^^^^^^^^^^^^^^^^

    # filter raw data in the broader frequency band
    raw_ssd.filter(freq_ssd[0], freq_ssd[1], fir_design='firwin',
                   skip_by_annotation='edge')
    # define SSD filter parameters
    filt_params_signal = dict(l_freq=freq_signal[0], h_freq=freq_signal[1],
                              l_trans_bandwidth=4, h_trans_bandwidth=4)
    filt_params_noise = dict(l_freq=freq_noise[0], h_freq=freq_noise[1],
                             l_trans_bandwidth=4, h_trans_bandwidth=4)
    # epoch data
    epochs_ssd = Epochs(raw_ssd, events, event_id, tmin, tmax,
                        proj=True, picks=picks,
                        baseline=None, preload=True)
    epochs_ssd_data = epochs_ssd.get_data()

    for n_idx, n_comp in enumerate(n_components):
        print('RUNNING n_components_' + str(n_comp))
        ssd = SSD(raw_ssd.info, filt_params_signal, filt_params_noise,
                  sort_by_spectral_ratio=False, return_filtered=True,
                  n_components=n_comp)
        # a new pipeline with SSD is defined
        pipe_ssd_csp = Pipeline([('SSD', ssd), ('CSP', csp), ('LDA', lda)])
        scores_ssd_csp_e[s_idx, n_idx] = cross_val_score(pipe_ssd_csp,
                                                         epochs_ssd_data,
                                                         labels, cv=cv,
                                                         n_jobs=1).mean()
        std_ssd_csp_e[s_idx, n_idx] = cross_val_score(pipe_ssd_csp,
                                                      epochs_ssd_data,
                                                      labels,
                                                      cv=cv, n_jobs=1).std()
###############################################################################
# Let's visualize the results
mean_ssd_csp = scores_ssd_csp_e.mean(axis=0)
std_ssd_csp = std_ssd_csp_e.mean(axis=0)
mean_csp = scores_csp.mean(axis=0)
fig, ax = plt.subplots(1, dpi=150, figsize=(8, 5))
x_axis = np.arange(0, len(n_components), 1)
ax.plot(x_axis, mean_ssd_csp, label='SSD+CSP')
ax.fill_between(x_axis, mean_ssd_csp - std_ssd_csp,
                mean_ssd_csp + std_ssd_csp, alpha=0.3)
ax.yaxis.grid(True)
ax.axhline(mean_csp, linestyle='-', color='k', label='CSP')
ax.set_xlabel('number of components SSD')
ax.set_ylabel('classification accuracy')
ax.set_xticks(np.arange(0, len(n_components), 2))
ax.set_xticklabels(np.arange(4, 65, steps * 2))
ax.set_ylim(0.6, 1.1)
ax.set_title('Impact of SSD as a function of n_components')
ax.legend()

# SSD acts as a regularizer. Here we used it as a dimensionality reduction tool
# for improving discriminative power in the alpha-band. On average, the best
# performance was achieved between 6 and 12 components, value that lies
# between the range found by the authors in :footcite:`HaufeEtAl2014b`.

##############################################################################
# Neurophysiological interpretation of the solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# SSD can not only improve classification performance of the decoding pipeline,
# as we showed, but it can also help towards a better interpretation of the
# solution.

# Here we are going to investigate the topographical plots of the CSP spatial
# patterns with and without applying SSD before for one subject of this cohort.

# Just for the sake of this example, we are going to train the models using all
# available data. But, remember data should always be split into separate sets
# to ensure the generalization capability of the model.

subject = n_subjects[1]
raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)
# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))
# e save the raw object before filtering
raw_ssd = raw.copy()

# apply band-pass filter
raw.filter(freq_signal[0], freq_signal[1], fir_design='firwin',
           skip_by_annotation='edge')

events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Extract epochs
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
labels = epochs.events[:, -1] - 2
epochs_data = epochs.get_data()
##############################################################################
# CSP
# ^^^
csp = CSP(n_components=4, log=True, norm_trace=False, reg='oas', rank='full')
csp.fit(epochs_data, labels)
# Plot topographies.
pattern_epochs = EvokedArray(data=csp.patterns_[:4].T,
                             info=raw_ssd.info)
pattern_epochs.plot_topomap(units=dict(mag='A.U.'), time_format='')
##############################################################################
# SSD + CSP
# ^^^^^^^^^
n_comp_max = 6
ssd = SSD(raw_ssd.info, filt_params_signal, filt_params_noise,
          sort_by_spectral_ratio=False, return_filtered=True,
          n_components=n_comp_max)

ssd.fit(epochs_data)
data_new = ssd.apply(epochs_data)
csp.fit(data_new, labels)
# Plot topographies.
pattern_epochs = EvokedArray(data=csp.patterns_[:4].T,
                             info=raw_ssd.info)
pattern_epochs.plot_topomap(units=dict(mag='A.U.'), time_format='')

# As it can be seen from the topographical plots, the CSP patterns that were
# learned after SSD was applied better enhance the occipital region from the
# left and right side. This could be expected since we based our analysis in
# the alpha band.
##############################################################################
# References
# ----------
# .. footbibliography::
