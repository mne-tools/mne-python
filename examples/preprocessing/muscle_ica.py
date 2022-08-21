# -*- coding: utf-8 -*-
"""
.. _ex-muscle-ica:

==============================
Removing muscle ICA components
==============================

Gross movements produce widespread high-frequency activity across all channels
that is usually not recoverable and so the epoch must be rejected as shown in
:ref:`ex-muscle-artifacts`. More ubiquitously than gross movements, muscle
artifact is produced during postural maintenance. This is more appropriately
removed by ICA otherwise there wouldn't be any epochs left! Note that muscle
artifacts of this kind are much more pronounced in EEG than they are in MEG.

"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%

import mne

data_path = mne.datasets.sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
raw.crop(tmin=100, tmax=130)  # take 30 seconds for speed

# pick only EEG channels, muscle artifact is basically not picked up by MEG
# if you have a simultaneous recording, you may want to do ICA on MEG and EEG
# separately
raw.pick_types(eeg=True)

# ICA works best with a highpass filter applied
raw.load_data()
raw.filter(l_freq=1., h_freq=None)

# %%
# Run ICA
ica = mne.preprocessing.ICA(
    n_components=15, method='picard', max_iter='auto', random_state=97)
ica.fit(raw)

# %%
# Remove components with postural muscle artifact using ICA
ica.plot_sources(raw)

# %%
# By inspection, let's select out the muscle-artifact components based on
# :footcite:`DharmapraniEtAl2016` manually.
#
# The criteria are:
#
# - Positive slope of log-log power spectrum between 7 and 75 Hz
#   (here just flat because it's not in log-log)
# - Peripheral focus or dipole/multi-pole foci (the blue and red
#   blobs in the topomap are far from the vertex where the most
#   muscle is)
# - Single focal point (low spatial smoothness; there is just one focus
#   of the topomap compared to components like the first ones that are
#   more likely neural which spread across the topomap)
#
# The other attribute worth noting is that the time course in
# :func:`mne.preprocessing.ICA.plot_sources` looks like EMG; you can
# see spikes when each motor unit fires so that the time course looks fuzzy
# and sometimes has large spikes that are often at regular intervals.
#
# ICA component 13 is a textbook example of what muscle artifact looks like.
# The focus of the topomap for this component is right on the temporalis
# muscle near the ears. There is also a minimum in the power spectrum at around
# 10 Hz, then a maximum at around 25 Hz, generally resulting in a positive
# slope in log-log units; this is a very typical pattern for muscle artifact.

muscle_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14]
ica.plot_properties(raw, picks=muscle_idx, log_scale=True)

# first, remove blinks and heartbeat to compare
blink_idx = [0]
heartbeat_idx = [5]
ica.apply(raw, exclude=blink_idx + heartbeat_idx)
ica.plot_overlay(raw, exclude=muscle_idx)

# %%
# Finally, let's try an automated algorithm to find muscle components
# and ensure that it gets the same components we did manually.
muscle_idx_auto, scores = ica.find_bads_muscle(raw)
ica.plot_scores(scores, exclude=muscle_idx_auto)
print(f'Manually found muscle artifact ICA components:      {muscle_idx}\n'
      f'Automatically found muscle artifact ICA components: {muscle_idx_auto}')

# %%
# Let's now replicate this on the EEGBCI dataset
# ----------------------------------------------

for sub in (1, 2):
    raw = mne.io.read_raw_edf(
        mne.datasets.eegbci.load_data(subject=sub, runs=(1,))[0], preload=True)
    mne.datasets.eegbci.standardize(raw)  # set channel names
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.filter(l_freq=1., h_freq=None)

    # Run ICA
    ica = mne.preprocessing.ICA(
        n_components=15, method='picard', max_iter='auto', random_state=97)
    ica.fit(raw)
    ica.plot_sources(raw)
    muscle_idx_auto, scores = ica.find_bads_muscle(raw)
    ica.plot_properties(raw, picks=muscle_idx_auto, log_scale=True)
    ica.plot_scores(scores, exclude=muscle_idx_auto)

    print(f'Manually found muscle artifact ICA components:      {muscle_idx}\n'
          'Automatically found muscle artifact ICA components: '
          f'{muscle_idx_auto}')

# %%
# References
# ----------
# .. footbibliography::
