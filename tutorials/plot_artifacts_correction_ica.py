"""

.. _tut_artifacts_correct_ica:

Artifact Correction with ICA
============================

ICA finds directions in the feature space
corresponding to projections with high non-Gaussianity. We thus obtain
a decomposition into independent components, and the artifact's contribution
is localized in only a small number of components.
These components have to be correctly identified and removed.

If EOG or ECG recordings are available, they can be used in ICA to
automatically select the corresponding artifact components from the
decomposition. To do so, you have to first build an Epoch object around
blink or heartbeat event.
"""

import numpy as np

import mne
from mne.datasets import sample

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs

# getting some data ready
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, 40, n_jobs=2)  # 1Hz high pass is often helpful for fitting ICA

picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

###############################################################################
# Before applying artifact correction please learn about your actual artifacts
# by reading :ref:`tut_artifacts_detect`.

###############################################################################
# Fit ICA
# -------
#
# ICA parameters:

n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> save time

###############################################################################
# Define the ICA object instance
ica = ICA(n_components=n_components, method=method)
print(ica)

###############################################################################
# we avoid fitting ICA on crazy environmental artifacts that would
# dominate the variance and decomposition
reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)
print(ica)

###############################################################################
# Plot ICA components

ica.plot_components()  # can you see some potential bad guys?


###############################################################################
# Advanced artifact detection
# ---------------------------
#
# Let's use a more efficient way to find artefacts

eog_average = create_eog_epochs(raw, reject=dict(mag=5e-12, grad=4000e-13),
                                picks=picks_meg).average()

# We simplify things by setting the maximum number of components to reject
n_max_eog = 1  # here we bet on finding the vertical EOG components
eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
# we can see that only one component is highly correlated and that this
# component got detected by our correlation analysis (red).

ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course

###############################################################################
# That component is also showing a prototypical average vertical EOG time
# course.
#
# Pay attention to the labels, a customized read-out of the ica.labels_
print(ica.labels_)

###############################################################################
# These labels were used by the plotters and are added automatically
# by artifact detection functions. You can also manually edit them to annotate
# components.
#
# Now let's see how we would modify our signals if we would remove this
# component from the data
ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
# red -> before, black -> after. Yes! We remove quite a lot!

# to definitely register this component as a bad one to be removed
# there is the ``ica.exclude`` attribute, a simple Python list

ica.exclude.extend(eog_inds)

# from now on the ICA will reject this component even if no exclude
# parameter is passed, and this information will be stored to disk
# on saving

# uncomment this for reading and writing
# ica.save('my-ica.fif')
# ica = read_ica('my-ica.fif')

###############################################################################
# Exercise: find and remove ECG artifacts using ICA!
#
# What if we don't have an EOG channel?
# -------------------------------------
#
# 1) make a bipolar reference from frontal EEG sensors and use as virtual EOG
# channel. This can be tricky though as you can only hope that the frontal
# EEG channels only reflect EOG and not brain dynamics in the prefrontal
# cortex.
# 2) Go for a semi-automated approach, using template matching.
# In MNE-Python option 2 is easily achievable and it might be better,
# so let's have a look at it.

from mne.preprocessing.ica import corrmap  # noqa

###############################################################################
# The idea behind corrmap is that artefact patterns are similar across subjects
# and can thus be identified by correlating the different patterns resulting
# from each solution with a template. The procedure is therefore
# semi-automatic. Corrmap hence takes at least a list of ICA solutions and a
# template, that can be an index or an array. As we don't have different
# subjects or runs available today, here we will fit ICA models to different
# parts of the recording and then use as a user-defined template the ICA
# that we just fitted for detecting corresponding components in the three "new"
# ICAs. The following block of code addresses this point and should not be
# copied, ok?
# We'll start by simulating a group of subjects or runs from a subject
start, stop = [0, len(raw.times) - 1]
intervals = np.linspace(start, stop, 4, dtype=int)
icas_from_other_data = list()
raw.pick_types(meg=True, eeg=False)  # take only MEG channels
for ii, start in enumerate(intervals):
    if ii + 1 < len(intervals):
        stop = intervals[ii + 1]
        print('fitting ICA from {0} to {1} seconds'.format(start, stop))
        this_ica = ICA(n_components=n_components, method=method).fit(
            raw, start=start, stop=stop, reject=reject)
        icas_from_other_data.append(this_ica)

###############################################################################
# Do not copy this at home! You start by reading in a collections of ICA
# solutions, something like
#
# ``icas = [mne.preprocessing.read_ica(fname) for fname in ica_fnames]``
print(icas_from_other_data)

###############################################################################
# use our previous ICA as reference.
reference_ica = ica

###############################################################################
# Investigate our reference ICA, here we use the previous fit from above.
reference_ica.plot_components()

###############################################################################
# Which one is the bad EOG component?
# Here we rely on our previous detection algorithm. You will need to decide
# yourself in that situation where no other detection is available.

reference_ica.plot_sources(eog_average, exclude=eog_inds)

###############################################################################
# Indeed it looks like an EOG, also in the average time course.
#
# So our template shall be a tuple like (reference_run_index, component_index):
template = (0, eog_inds[0])

###############################################################################
# Now we can do the corrmap.
fig_template, fig_detected = corrmap(
    icas_from_other_data, template=template, label="blinks", show=True,
    threshold=.8, ch_type='mag')

###############################################################################
# Nice, we have found similar ICs from the other runs!
# This is even nicer if we have 20 or 100 ICA solutions in a list.
#
# You can also use SSP for correcting for artifacts. It is a bit simpler,
# faster but is less precise than ICA. And it requires that you
# know the event timing of your artifact.
# See :ref:`tut_artifacts_correct_ssp`.
