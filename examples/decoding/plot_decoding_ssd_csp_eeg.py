"""
.. _ex-decoding-ssd-csp-eeg:

============================================================================
Improving motor imagery decoding from EEG using Spatio Spectra Decomposition
============================================================================

The Spatio Spectra Decomposition (SSD) is a supervised spatial filtering
algorithm capable of extracting components primarily related to the signal of
interest. It can be used as a pre-processing step for the analysis of neuronal
activity. In this example SSD will be used in the context of Motor Imagery
decoding from EEG data. It will be applied before extracting features with
the Common Spatial Patterns (CSP) method. A classifier will then be trained
using the extracted features from the SSD + CSP-filtered signals. The impact in
performance of using SSD before CSP will be shown here.

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

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 0.5, 2.5
event_id = dict(left=2, right=3)  # Motor imagery: left vs right hand
subject = 2
runs = [4, 8, 12]

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))
# for the sake of comparison, we save the raw object before filtering
raw_ssd = raw.copy()
##############################################################################
# Data filtering.

# we are going to filter data in the alpha band
freq_signal = [8, 12]
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
###############################################################################
# Traditional CSP + LDA pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# define pipelines with monte-carlo simulations
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# pipeline methods
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
pipe_csp = Pipeline([('CSP', csp), ('LDA', lda)])
# this is the accuracy we would like to improve by adding SSD into the pipeline
scores_csp = cross_val_score(pipe_csp, epochs_data, labels, cv=cv,
                             n_jobs=1).mean()
###############################################################################
# SSD enhances signal-to-noise ratio, thus what is considered 'noise' should be
# defined. Typically bands of 2 Hz surrounding the frequencies of interest are
# taken.
freq_noise = freq_signal[0] - 2, freq_signal[1] + 2
# when applying SSD, data should be filtered in a broader band than for CSP,
# thus, a broader bandwidth is defined for filtering the new signal.
freq_ssd = freq_signal[0] - 3, freq_signal[1] + 3
raw_ssd.filter(freq_ssd[0], freq_ssd[1], fir_design='firwin',
               skip_by_annotation='edge')
###############################################################################
# SSD + CSP + LDA pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^

# SSD can be applied either BEFORE or AFTER data epoching, each approach has
# it owns advantages and disadvantages. Here both approaches are going to be
# implemented.
###############################################################################
# SSD outside the pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^

# copy raw data
raw_ssd_transformed = raw_ssd.copy()
# define SSD filter parameters
filt_params_signal = dict(l_freq=freq_signal[0], h_freq=freq_signal[1],
                          l_trans_bandwidth=4, h_trans_bandwidth=4)
filt_params_noise = dict(l_freq=freq_noise[0], h_freq=freq_noise[1],
                         l_trans_bandwidth=4, h_trans_bandwidth=4)

# Since we are working with a dataset with 64 channels, it is interesting to
# see the impact in data dimensionality reduction by means of SSD before
# applying CSP. Thus, SSD will be applied at different numbers of component
# (:attr:n_components). The minimum value is 4, since 4 CSP components were
# selected, and the maximum is the number of channels, here 64.
steps = 4
n_components = np.arange(4, 65, steps)
scores_ssd_csp = np.zeros((len(n_components,)))
std_ssd_csp = np.zeros((len(n_components,)))
for n, n_comp in enumerate(n_components):
    print('RUNNING n_components_' + str(n_comp))
    # set return_filtered to True, so the transformed signal will be filtered
    # in the desired frequency band.
    ssd = SSD(raw_ssd.info, filt_params_signal, filt_params_noise,
              sort_by_spectral_ratio=True, return_filtered=True,
              n_components=n_comp)
    # fit and transform
    raw_ssd_transformed._data = ssd.fit_transform(raw_ssd.get_data())
    # Now extract epochs from the transformed signals
    epochs_ssd = Epochs(raw_ssd_transformed, events, event_id, tmin, tmax,
                        proj=True, picks=np.arange(0, n_comp, 1),
                        baseline=None, preload=True)
    labels = epochs_ssd.events[:, -1] - 2
    epochs_ssd_data = epochs_ssd.get_data()
    scores_ssd_csp[n] = cross_val_score(pipe_csp, epochs_ssd_data, labels,
                                        cv=cv, n_jobs=1).mean()
    std_ssd_csp[n] = cross_val_score(pipe_csp, epochs_ssd_data, labels, cv=cv,
                                     n_jobs=1).std()
###############################################################################
# Let's visualize the results
fig, ax = plt.subplots(1, dpi=150)
x_axis = np.arange(0, len(n_components), 1)
ax.plot(x_axis, scores_ssd_csp, label='SSD+CSP')
ax.fill_between(x_axis, scores_ssd_csp - std_ssd_csp,
                scores_ssd_csp + std_ssd_csp, alpha=0.3)
ax.yaxis.grid(True)
ax.axhline(scores_csp, linestyle='-', color='k', label='CSP')
ax.set_xlabel('number of components SSD')
ax.set_ylabel('classification accuracy')
ax.set_xticks(np.arange(0, len(n_components), 2))
ax.set_xticklabels(np.arange(4, 65, steps * 2))
ax.set_ylim(0.6, 1.1)
ax.set_title('Impact of SSD before epoching')
ax.legend()
# The results show how SSD can be used to improve decoding performance.
# The maximum value is nearly already achieved with the initial 4 components.
# Adding more SSD components slightly increase the classification performance,
# leading to its minimum value when the number of components is equal to 64.
###############################################################################
# SSD within the pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^

# In this context, data should first be epoched before calling pipeline.
epochs_ssd = Epochs(raw_ssd, events, event_id, tmin, tmax,
                    proj=True, picks=np.arange(0, n_comp, 1),
                    baseline=None, preload=True)
labels = epochs_ssd.events[:, -1] - 2
epochs_ssd_data = epochs_ssd.get_data()

# Since SSD will be applied on epoched data, it is very likely that the
# optimal number of SSD components will be different from the value found
# before.
scores_ssd_csp_e = np.zeros((len(n_components,)))
std_ssd_csp_e = np.zeros((len(n_components,)))
for n, n_comp in enumerate(n_components):
    print('RUNNING n_components_' + str(n_comp))
    ssd = SSD(raw_ssd.info, filt_params_signal, filt_params_noise,
              sort_by_spectral_ratio=False, return_filtered=True,
              n_components=n_comp)
    # a new pipeline with SSD is defined
    pipe_ssd_csp = Pipeline([('SSD', ssd), ('CSP', csp), ('LDA', lda)])
    scores_ssd_csp_e[n] = cross_val_score(pipe_ssd_csp, epochs_ssd_data,
                                          labels, cv=cv, n_jobs=1).mean()
    std_ssd_csp_e[n] = cross_val_score(pipe_ssd_csp, epochs_ssd_data, labels,
                                       cv=cv, n_jobs=1).std()
###############################################################################
# Let's visualize these new results
fig, ax = plt.subplots(1, dpi=150, figsize=(8, 5))
x_axis = np.arange(0, len(n_components), 1)
ax.plot(x_axis, scores_ssd_csp_e, label='SSD+CSP')
ax.fill_between(x_axis, scores_ssd_csp_e - std_ssd_csp_e,
                scores_ssd_csp_e + std_ssd_csp_e, alpha=0.3)
ax.yaxis.grid(True)
ax.axhline(scores_csp, linestyle='-', color='k', label='CSP')
ax.set_xlabel('number of components SSD')
ax.set_ylabel('classification accuracy')
ax.set_xticks(np.arange(0, len(n_components), 2))
ax.set_xticklabels(np.arange(4, 65, steps * 2))
ax.set_ylim(0.6, 1.1)
ax.set_title('Impact of SSD after epoching')
ax.legend()
# As before, we can see that SSD helps towards improving classification
# accuracy. But, in this scenario more SSD components are needed to gain
# performance. Given that in this scenario SSD is applied after epoching,
# the edge effects of filtering epoched data might explain both: why accuracy
# improvement is lower than before and why more :attr:n_components are
# needed to achieve the maximum performance.
##############################################################################
# Neurophysiological interpretation of the solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# SSD can not only improve classification performance of the decoding pipeline,
# as we showed, but it can also help towards a better interpretation of the
# solution.

# Here we are going to investigate the topographical plots of the CSP spatial
# patterns with and without applying SSD before.

# Just for the sake of this example, we are going to train the models using all
# available data. But, remember data should always be split into separate sets
# to ensure the generalization capability of the model.
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

ixd_max = np.argmax(scores_ssd_csp_e)
n_comp_max = n_components[ixd_max]
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
# learned after SSD was applied better enhance the left vs. right event related
# (de-)synchronization patterns.
##############################################################################
# References
# ----------
# .. footbibliography::
