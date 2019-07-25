# -*- coding: utf-8 -*-
"""
.. _tut-artifact-ica:

Repairing artifacts with ICA
============================

This tutorial covers the basics of independent components analysis (ICA) and
shows how ICA can be used for artifact repair; an extended example illustrates
repair of ocular and heartbeat artifacts.

.. contents:: Page contents
   :local:
   :depth: 2

We begin as always by importing the necessary Python modules and loading some
:ref:`example data <sample-dataset>`. Because ICA can be computationally
intense, we'll also crop the data to 60 seconds; and to save ourselves from
repeatedly typing ``mne.preprocessing`` we'll directly import a few functions
and classes from that submodule:
"""

import os
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)
raw.crop(tmax=60.)

###############################################################################
# .. note::
#     Before applying ICA (or any artifact repair strategy), be sure to observe
#     the artifacts in your data to make sure you choose the right repair tool.
#     Sometimes the right tool is no tool at all — if the artifacts are small
#     enough you may not even need to repair them to get good analysis results.
#     See :ref:`tut-artifact-overview` for guidance on detecting and
#     visualizing various types of artifact.
#
#
# What is ICA?
# ^^^^^^^^^^^^
#
# Independent components analysis (ICA) is a technique for estimating
# independent source signals from a set of recordings in which the source
# signals were mixed together in unknown ratios. A common example of this is
# the problem of `blind source separation`_: with 3 musical instruments playing
# in the same room, and 3 microphones recording the performance (each picking
# up all 3 instruments, but at varying levels), can you somehow "unmix" the
# signals recorded by the 3 microphones so that you end up with a separate
# "recording" isolating the sound of each instrument?
#
# It is not hard to see how this analogy applies to EEG/MEG analysis: there are
# many "microphones" (sensor channels) simultaneously recording many
# "instruments" (blinks, heartbeats, activity in different areas of the brain,
# muscular activity from jaw clenching or swallowing, etc). As long as these
# various source signals are `statistically independent`_ and non-gaussian, it
# is usually possible to separate the sources using ICA, and then re-construct
# the sensor signals after excluding the sources that are unwanted.
#
#
# ICA in MNE-Python
# ~~~~~~~~~~~~~~~~~
#
# MNE-Python implements three different ICA algorithms: ``fastica`` (the
# default), ``picard``, and ``infomax``. FastICA and Infomax are both in fairly
# widespread use; Picard is a newer (2017) algorithm that is expected to
# converge faster than FastICA and Infomax, and is more robust than other
# algorithms in cases where the sources are not completely independent, which
# typically happens with real EEG/MEG data. See [1]_ for more information.
#
# The ICA interface in MNE-Python mirrors the standard interface in
# `scikit-learn`_: some general parameters are specified when creating an
# :class:`~mne.preprocessing.ICA` object, then the
# :class:`~mne.preprocessing.ICA` object is fit to the data using its
# :meth:`~mne.preprocessing.ICA.fit` method. The results of the fitting are
# added to the :class:`~mne.preprocessing.ICA` object as attributes that end in
# an underscore (``_``), such as ``ica.mixing_matrix_`` and
# ``ica.unmixing_matrix_``. After fitting, the ICA component(s) that you want
# to remove must be chosen, and the ICA fit must then be applied to the
# :class:`~mne.io.Raw` or :class:`~mne.Epochs` object using the
# :class:`~mne.preprocessing.ICA` object's :meth:`~mne.preprocessing.ICA.apply`
# method.
#
# As is typically done with ICA, to speed up computation the data are first
# scaled to unit variance and whitened using principal components analysis
# (PCA) before performing the ICA decomposition. You can impose a
# dimensionality reduction at this step by specifying ``max_pca_components``.
# From these retained Principal Components (PCs), the first ``n_components``
# are then passed to the ICA algorithm (``n_components`` may be an integer
# number of components to use, or a fraction of explained variance that used
# components should capture). Then, when reconstructing the signal after
# excluding the Independent Components (ICs) that capture the artifacts, the
# signal will be reconstructed from the retained ICs *plus* the retained PCs
# that were not included in the ICA step. If you want to reduce the number of
# retained PCs used at the reconstruction stage, it is controlled by the
# ``n_pca_components`` parameter. The procedure and the parameters that control
# dimensionality at various stages is summarized in the diagram below:
#
# .. graphviz:: ../../_static/diagrams/ica.dot
#    :alt: Diagram of ICA procedure in MNE-Python
#    :align: left
#
#
# See the Notes section of the :class:`~mne.preprocessing.ICA` documentation
# for further details. Next we'll walk through an extended example that
# illustrates each of these steps in greater detail.
#
#
# Example: EOG and ECG artifact repair
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let's begin by visualizing the artifacts that we want to repair. In this
# dataset they are big enough to see easily in the raw data:

# plot some channels that clearly show heartbeats and blinks
regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks))

###############################################################################
# We can get a better view of the ocular artifacts using
# :func:`~mne.preprocessing.create_eog_epochs` like we did in the
# :ref:`tut-artifact-overview` tutorial:

eog_evoked = create_eog_epochs(raw).average()
eog_evoked.plot_joint()

###############################################################################
# Now we'll do the same for the heartbeat artifacts, using
# :func:`~mne.preprocessing.create_ecg_epochs`:

ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.plot_joint()

###############################################################################
# Before we run the ICA, an important step is filtering the data to remove
# low-frequency drifts, which can negatively affect the quality of the ICA fit.
# A 1 Hz cutoff frequency is recommended; we'll keep a copy of the unfiltered
# :class:`~mne.io.Raw` object around so we can apply the ICA solution to it
# later.

filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1., h_freq=None)

###############################################################################
# Now we're ready to set up and fit the ICA. Since we know (from observing our
# raw data) that the EOG and ECG artifacts are fairly strong, we would expect
# those artifacts to be captured in the first few dimensions of the PCA
# decomposition that happens before the ICA. Therefore, we probably don't need
# a huge number of components to do a good job of isolating our artifacts. As a
# first guess, we'll run ICA with only the first 15 PCA components — a fairly
# small number given that our data has over 300 channels, but it will run
# quickly and we will able to tell easily whether it worked or not (because we
# already know what the EOG / ECG artifacts should look like). When analyzing
# your own data on a machine with good computing power, 30 or more components
# is probably more appropriate.
#
# .. sidebar:: ICA ignores the time domain
#
#     ICA finds patterns across channels, but ignores the time domain. This
#     means you can compute ICA on discontinuous :class:`~mne.Epochs` or
#     :class:`~mne.Evoked` objects (not just continuous :class:`~mne.io.Raw`
#     objects), or only use every Nth sample by passing the ``decim`` parameter
#     to ``ICA.fit()``.
#
# ICA fitting involves some randomness (e.g., the components may get a sign
# flip on different runs, or may not always be returned in the same order), so
# we'll specify a `random seed`_ so that we get identical results each time
# this tutorial is built by our web servers. Some optional parameters that we
# could pass to the :meth:`~mne.preprocessing.ICA.fit` method include ``decim``
# (to use only every Nth sample in computing the ICs, which can yield a
# considerable speed-up) and ``reject`` (for providing a rejection dictionary
# for maximum acceptable peak-to-peak amplitudes for each channel type, just
# like we used when creating epoched data in the :ref:`tut-overview` tutorial).

ica = ICA(n_components=15, random_state=97)
ica.fit(filt_raw)

###############################################################################
# Now we can examine the ICs to see what they captured.
# :meth:`~mne.preprocessing.ICA.plot_sources` will show the time series of the
# ICs, and :meth:`~mne.preprocessing.ICA.plot_components` will show the scalp
# field distributions computed from the ICA unmixing matrix. Note that in our
# call to :meth:`~mne.preprocessing.ICA.plot_sources` we can use the original,
# unfiltered :class:`~mne.io.Raw` object:

raw.load_data()

# sphinx_gallery_thumbnail_number = 9
ica.plot_sources(raw)
ica.plot_components()

###############################################################################
# Here we can pretty clearly see that the first component (``ICA000``) captures
# the EOG signal quite well, and the second component (``ICA001``) looks a lot
# like a heartbeat. In the remaining sections, we'll look at different ways of
# choosing which ICs to exclude prior to reconstructing the sensor signals.
#
#
# Selecting ICA components manually
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the plots above it's fairly obvious which ICs are capturing our EOG and
# ECG artifacts, but it can be helpful to further visualize them anyway just to
# be sure. First, we can plot an overlay of the original signal against the
# reconstructed signal with the artifactual ICs excluded, using
# :meth:`~mne.preprocessing.ICA.plot_overlay`:

# blinks
ica.plot_overlay(raw, exclude=[0], picks='eeg')
# heartbeats
ica.plot_overlay(raw, exclude=[1], picks='mag')

###############################################################################
# We can also plot some diagnostics of each IC using
# :meth:`~mne.preprocessing.ICA.plot_properties`:

ica.plot_properties(raw, picks=[0, 1])

###############################################################################
# .. note::
#
#     :meth:`~mne.preprocessing.ICA.plot_components` (which yields the scalp
#     field topographies for each component) has an optional ``inst`` parameter
#     that takes an instance of :class:`~mne.io.Raw` or :class:`~mne.Epochs`.
#     Passing ``inst`` makes the scalp topographies interactive: clicking one
#     will bring up a diagnostic :meth:`~mne.preprocessing.ICA.plot_properties`
#     window for that component.
#
#
# Once we're certain which components we want to exclude, we can specify that
# manually by setting the ``ica.exclude`` attribute. Similar to marking bad
# channels, merely setting ``ica.exclude`` doesn't do anything yet (you're
# adding the excluded ICs to a list that will get used later when it's needed).
# Once the exclusions have been set, the ICA methods will exclude those
# component(s) even if no ``exclude`` parameter is passed, and the list of
# excluded components will be preserved when using
# :meth:`mne.preprocessing.ICA.save` and :func:`mne.preprocessing.read_ica`.

ica.exclude = [0, 1]  # indices chosen based on various plots above

###############################################################################
# Now that the exclusions have been set, we can reconstruct the sensor signals
# with artifacts removed using the :meth:`~mne.preprocessing.ICA.apply` method.
# Plotting the original raw data alongside the reconstructed data shows that
# the heartbeat and blink artifacts are repaired.

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
del reconst_raw

###############################################################################
# Using an EOG channel to select ICA components
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It may have seemed easy to review the plots and manually select which ICs to
# exclude, but when processing dozens or hundreds of subjects this can become
# a tedious, rate-limiting step in the analysis pipeline. One alternative is to
# use dedicated EOG or ECG sensors as a "pattern" to check the ICs against, and
# automatically mark for exclusion any ICs that match the EOG/ECG pattern. Here
# we'll use :meth:`~mne.preprocessing.ICA.find_bads_eog` to automatically find
# the ICs that best match the EOG signal, then use
# :meth:`~mne.preprocessing.ICA.plot_scores` along with our other plotting
# functions to see which ICs it picked. We'll start by resetting
# ``ica.exclude`` back to an empty list:

ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# barplot of IC component "EOG match" scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw)

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

###############################################################################
# Note that above we used :meth:`~mne.preprocessing.ICA.plot_sources` on both
# the original :class:`~mne.io.Raw` instance and also on an
# :class:`~mne.Evoked` instance of the extracted EOG artifacts. This can be
# another way to confirm that :meth:`~mne.preprocessing.ICA.find_bads_eog` has
# identified the correct components.
#
#
# Using a simulated channel to select ICA components
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If you don't have an EOG channel,
# :meth:`~mne.preprocessing.ICA.find_bads_eog` has a ``ch_name`` parameter that
# you can use as a proxy for EOG. You can use a single channel, or create a
# bipolar reference from frontal EEG sensors and use that as virtual EOG
# channel. This carries a risk however: you must hope that the frontal EEG
# channels only reflect EOG and not brain dynamics in the prefrontal cortex (or
# you must not care about those prefrontal signals).
#
# For ECG, it is easier: :meth:`~mne.preprocessing.ICA.find_bads_ecg` uses
# cross-channel averaging to estimate a virtual ECG channel, so it is usually
# not necessary to pass a specific channel name.
# :meth:`~mne.preprocessing.ICA.find_bads_ecg` also has two options for its
# ``method`` parameter: ``'ctps'`` (cross-trial phase statistics) and
# ``'correlation'`` (Pearson correlation between data and ECG channel).

ica.exclude = []
# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
ica.exclude = ecg_indices

# barplot of IC component "ECG match" scores
ica.plot_scores(ecg_scores)

# plot diagnostics
ica.plot_properties(raw, picks=ecg_indices)

# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(raw)

# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
ica.plot_sources(ecg_evoked)

###############################################################################
# The last of these plots is especially useful: it shows us that the heartbeat
# artifact is coming through on *two* ICs, and we've only caught one of them.
# In fact, if we look closely at the output of
# :meth:`~mne.preprocessing.ICA.plot_sources` (online, you can right-click →
# "view image" to zoom in), it looks like ``ICA014`` has a weak periodic
# component that is in-phase with ``ICA001``. It might be worthwhile to re-run
# the ICA with more components to see if that second heartbeat artifact
# resolves out a little better:

# refit the ICA with 30 components this time
new_ica = ICA(n_components=30, random_state=97)
new_ica.fit(filt_raw)

# find which ICs match the ECG pattern
ecg_indices, ecg_scores = new_ica.find_bads_ecg(raw, method='correlation')
new_ica.exclude = ecg_indices

# barplot of IC component "ECG match" scores
new_ica.plot_scores(ecg_scores)

# plot diagnostics
new_ica.plot_properties(raw, picks=ecg_indices)

# plot ICs applied to raw data, with ECG matches highlighted
new_ica.plot_sources(raw)

# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
new_ica.plot_sources(ecg_evoked)

###############################################################################
# Much better! Now we've captured both ICs that are reflecting the heartbeat
# artifact (and as a result, we got two diagnostic plots: one for each IC that
# reflects the heartbeat). This demonstrates the value of checking the results
# of automated approaches like :meth:`~mne.preprocessing.ICA.find_bads_ecg`
# before accepting them.

# clean up memory before moving on
del raw, filt_raw, ica, new_ica

###############################################################################
# Selecting ICA components using template matching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When dealing with multiple subjects, it is also possible to manually select
# an IC for exclusion on one subject, and then use that component as a
# *template* for selecting which ICs to exclude from other subjects' data,
# using :func:`mne.preprocessing.corrmap`. The idea behind
# :func:`~mne.preprocessing.corrmap` is that the artifact patterns are similar
# enough across subjects that corresponding ICs can be identified by
# correlating the ICs from each ICA solution with a common template, and
# picking the ICs with the highest correlation strength.
# :func:`~mne.preprocessing.corrmap` takes a list of ICA solutions, and a
# ``template`` parameter that specifies which ICA object and which component
# within it to use as a template.
#
# Since our sample dataset only contains data from one subject, we'll use a
# different dataset with multiple subjects: the EEGBCI dataset [2]_ [3]_. The
# dataset has 109 subjects, we'll just download one run (a left/right hand
# movement task) from each of the first 4 subjects:

raws = list()
icas = list()

for subj in range(4):
    # EEGBCI subjects are 1-indexed; run 3 is a left/right hand movement task
    fname = mne.datasets.eegbci.load_data(subj + 1, runs=[3])[0]
    raw = mne.io.read_raw_edf(fname)
    # remove trailing `.` from channel names so we can set montage
    raw.rename_channels(lambda x: x.rstrip('.'))
    raw.set_montage('standard_1005')
    # fit ICA
    ica = ICA(n_components=30, random_state=97)
    ica.fit(raw)
    raws.append(raw)
    icas.append(ica)

# use the first subject as template; use Fpz as proxy for EOG
raw = raws[0]
ica = icas[0]
eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name='Fpz')
corrmap(icas, template=(0, eog_inds[0]))

###############################################################################
# The first figure shows the template map, while the second figure shows all
# the maps that were considered a "match" for the template (including the
# template itself). There were only three matches from the four subjects;
# notice the output message ``No maps selected for subject(s) 1, consider a
# more liberal threshold```.  By default the threshold is set automatically by
# trying several values; here it may have chosen a threshold that is too high.
# Let's take a look at the ICA sources for each subject:

for index, (ica, raw) in enumerate(zip(icas, raws)):
    fig = ica.plot_sources(raw)
    fig.suptitle('Subject {}'.format(index))

###############################################################################
# Notice that subject 1 *does* seem to have an IC that looks like it reflects
# blink artifacts (component ``ICA000``). Notice also that subject 3 appears to
# have *two* components that are reflecting ocular artifacts (``ICA000`` and
# ``ICA002``), but only one was caught by :func:`~mne.preprocessing.corrmap`.
# Let's try setting the threshold manually:

corrmap(icas, template=(0, eog_inds[0]), threshold=0.9)

###############################################################################
# Now we get the message ``At least 1 IC detected for each subject`` (which is
# good). At this point we'll re-run :func:`~mne.preprocessing.corrmap` with
# parameters ``label=blink, show=False`` to *label* the ICs from each subject
# that capture the blink artifacts.

corrmap(icas, template=(0, eog_inds[0]), threshold=0.9, label='blink',
        plot=False)
print([ica.labels_ for ica in icas])

###############################################################################
# Notice that the first subject has 3 different labels for the IC at index 0:
# "eog/0/Fpz", "eog", and "blink". The first two were added by
# :meth:`~mne.preprocessing.ICA.find_bads_eog`; the "blink" label was added by
# the last call to :func:`~mne.preprocessing.corrmap`. Notice also that each
# subject has at least one IC index labelled "blink", and subject 3 has two
# components (0 and 2) labelled "blink" (consistent with the plot of IC sources
# above). The ``labels_`` attribute of :class:`~mne.preprocessing.ICA` objects
# can also be manually edited to annotate the ICs with custom labels. They also
# come in handy when plotting:

icas[3].plot_components(picks=icas[3].labels_['blink'])
icas[3].exclude = icas[3].labels_['blink']
icas[3].plot_sources(raws[3])

###############################################################################
# As a final note, it is possible to extract ICs numerically using the
# :meth:`~mne.preprocessing.ICA.get_components` method of
# :class:`~mne.preprocessing.ICA` objects. This will return a :class:`NumPy
# array <numpy.ndarray>` that can be passed to
# :func:`~mne.preprocessing.corrmap` instead of the :class:`tuple` of
# ``(subject_index, component_index)`` we passed before, and will yield the
# same result:

template_eog_component = icas[0].get_components()[:, eog_inds[0]]
corrmap(icas, template=template_eog_component, threshold=0.9)
print(template_eog_component)

###############################################################################
# An advantage of using this numerical representation of an IC to capture a
# particular artifact pattern is that it can be saved and used as a template
# for future template-matching tasks using :func:`~mne.preprocessing.corrmap`
# without having to load or recompute the ICA solution that yielded the
# template originally. Put another way, when the template is a NumPy array, the
# :class:`~mne.preprocessing.ICA` object containing the template does not need
# to be in the list of ICAs provided to :func:`~mne.preprocessing.corrmap`.
#
#
# References
# ^^^^^^^^^^
#
# .. [1] Ablin P, Cardoso J, Gramfort A (2018). Faster Independent Component
#        Analysis by Preconditioning With Hessian Approximations. IEEE
#        Transactions on Signal Processing 66:4040–4049
#
# .. [2] Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR (2004).
#        BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE
#        TBME 51(6):1034-1043
#
# .. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
#        Mietus JE, Moody GB, Peng C-K, Stanley HE (2000). PhysioBank,
#        PhysioToolkit, and PhysioNet: Components of a New Research Resource
#        for Complex Physiologic Signals. Circulation 101(23):e215-e220
#
#
# .. LINKS
#
# .. _`blind source separation`:
#    https://en.wikipedia.org/wiki/Signal_separation`
# .. _`statistically independent`:
#    https://en.wikipedia.org/wiki/Independence_(probability_theory)
# .. _`scikit-learn`: https://scikit-learn.org
# .. _`random seed`: https://en.wikipedia.org/wiki/Random_seed
# .. _`regular expression`: https://www.regular-expressions.info/
