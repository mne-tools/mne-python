"""
.. _tut-reject-data-spans:

Rejecting bad data spans
========================

This tutorial covers manual marking of bad spans of data, and automated
rejection of data spans based on signal amplitude.

.. contents:: Page contents
   :local:
   :depth: 2

We begin as always by importing the necessary Python modules and loading some
:ref:`example data <sample-dataset>`; to save memory we'll use a pre-filtered
and downsampled version of the example data, and we'll also load an events
array to use when converting the continuous data to epochs:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                           'sample_audvis_filt-0-40_raw-eve.fif')
events = mne.read_events(events_file)

###############################################################################
# Annotating bad spans of data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The tutorial :ref:`tut-events-vs-annotations` describes how
# :class:`~mne.Annotations` can be read from embedded events in the raw
# recording file, and :ref:`tut-annotate-raw` describes in detail how to
# interactively annotate a :class:`~mne.io.Raw` data object. Here, we focus on
# best practices for annotating *bad* data spans so that they will be excluded
# from your analysis pipeline.
#
#
# The ``reject_by_annotation`` parameter
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the interactive ``raw.plot()`` window, the annotation controls can be
# opened by pressing :kbd:`a`. Here, new annotation labels can be created or
# existing annotation labels can be selected for use.

fig = raw.plot()
fig.canvas.key_press_event('a')

###############################################################################
# .. sidebar:: Annotating good spans
#
#     The default "BAD\_" prefix for new labels can be removed simply by
#     pressing the backspace key four times before typing your custom
#     annotation label.
#
# You can see that default annotation label is "BAD\_"; this can be edited
# prior to pressing the "Add label" button to customize the label. The intent
# is that users can annotate with as many or as few labels as makes sense for
# their research needs, but that annotations marking spans that should be
# excluded from the analysis pipeline should all begin with "BAD" or "bad"
# (e.g., "bad_cough", "bad-eyes-closed", "bad door slamming", etc). When this
# practice is followed, many processing steps in MNE-Python will automatically
# exclude the "bad"-labelled spans of data; this behavior is controlled by a
# parameter ``reject_by_annotation`` that can be found in many MNE-Python
# functions or class constructors, including:
#
# - creation of epoched data from continuous data (:class:`mne.Epochs`)
# - many methods of the independent components analysis class
#   (:class:`mne.preprocessing.ICA`)
# - functions for finding heartbeat and blink artifacts
#   (:func:`~mne.preprocessing.find_ecg_events`,
#   :func:`~mne.preprocessing.find_eog_events`)
# - covariance computations (:func:`mne.compute_raw_covariance`)
# - power spectral density computation (:meth:`mne.io.Raw.plot_psd`,
#   :func:`mne.time_frequency.psd_welch`)
#
# For example, when creating epochs from continuous data, if
# ``reject_by_annotation=True`` the :class:`~mne.Epochs` constructor will drop
# any epoch that partially or fully overlaps with an annotated span that begins
# with "bad".
#
#
# Generating annotations programmatically
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The :ref:`tut-artifact-overview` tutorial introduced the artifact detection
# functions :func:`~mne.preprocessing.find_eog_events` and
# :func:`~mne.preprocessing.find_ecg_events` (although that tutorial mostly
# relied on their higher-level wrappers
# :func:`~mne.preprocessing.create_eog_epochs` and
# :func:`~mne.preprocessing.create_ecg_epochs`). Here, for demonstration
# purposes, we make use of the lower-level artifact detection function to get
# an events array telling us where the blinks are, then automatically add
# "bad_blink" annotations around them (this is not necessary when using
# :func:`~mne.preprocessing.create_eog_epochs`, it is done here just to show
# how annotations are added non-interactively). We'll start the annotations
# 250 ms before the blink and end them 250 ms after it:

# sphinx_gallery_thumbnail_number = 3
eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ['bad blink'] * len(eog_events)
blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
raw.set_annotations(blink_annot)

###############################################################################
# Now we can confirm that the annotations are centered on the EOG events. Since
# blinks are usually easiest to see in the EEG channels, we'll only plot EEG
# here:

eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.plot(events=eog_events, order=eeg_picks)

###############################################################################
# See the section :ref:`tut-section-programmatic-annotations` for more details
# on creating annotations programmatically.
#
#
# .. _`tut-reject-epochs-section`:
#
# Rejecting Epochs based on channel amplitude
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Besides "bad" annotations, the :class:`mne.Epochs` class constructor has
# another means of rejecting epochs, based on signal amplitude thresholds for
# each channel type. In the :ref:`overview tutorial
# <tut-section-overview-epoching>` we saw an example of this: setting maximum
# acceptable peak-to-peak amplitudes for each channel type in an epoch, using
# the ``reject`` parameter. There is also a related parameter, ``flat``, that
# can be used to set *minimum* acceptable peak-to-peak amplitudes for each
# channel type in an epoch:

reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 µV
                       eog=200e-6)       # 200 µV

flat_criteria = dict(mag=1e-15,          # 1 fT
                     grad=1e-13,         # 1 fT/cm
                     eeg=1e-6)           # 1 µV

###############################################################################
# The values that are appropriate are dataset- and hardware-dependent, so some
# trial-and-error may be necessary to find the correct balance between data
# quality and loss of power due to too many dropped epochs. Here, we've set the
# rejection criteria to be fairly stringent, for illustration purposes.
#
# Two additional parameters, ``reject_tmin`` and ``reject_tmax``, are used to
# set the temporal window in which to calculate peak-to-peak amplitude for the
# purposes of epoch rejection. These default to the same ``tmin`` and ``tmax``
# of the entire epoch. As one example, if you wanted to only apply the
# rejection thresholds to the portion of the epoch that occurs *before* the
# event marker around which the epoch is created, you could set
# ``reject_tmax=0``. A summary of the causes of rejected epochs can be
# generated with the :meth:`~mne.Epochs.plot_drop_log` method:

epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True)
epochs.plot_drop_log()

###############################################################################
# Notice that we've passed ``reject_by_annotation=False`` above, in order to
# isolate the effects of the rejection thresholds. If we re-run the epoching
# with ``reject_by_annotation=True`` (the default) we see that the rejections
# due to EEG and EOG channels have disappeared (suggesting that those channel
# fluctuations were probably blink-related, and were subsumed by rejections
# based on the "bad blink" label).

epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria, preload=True)
epochs.plot_drop_log()

###############################################################################
# More importantly, note that *many* more epochs are rejected (~20% instead of
# ~2.5%) when rejecting based on the blink labels, underscoring why it is
# usually desirable to repair artifacts rather than exclude them.
#
# The :meth:`~mne.Epochs.plot_drop_log` method is a visualization of an
# :class:`~mne.Epochs` attribute, namely ``epochs.drop_log``, which stores
# empty lists for retained epochs and lists of strings for dropped epochs, with
# the strings indicating the reason(s) why the epoch was dropped. For example:

print(epochs.drop_log)

###############################################################################
# Finally, it should be noted that "dropped" epochs are not necessarily deleted
# from the :class:`~mne.Epochs` object right away. Above, we forced the
# dropping to happen when we created the :class:`~mne.Epochs` object by using
# the ``preload=True`` parameter. If we had not done that, the
# :class:`~mne.Epochs` object would have been `memory-mapped`_ (not loaded into
# RAM), in which case the criteria for dropping epochs are stored, and the
# actual dropping happens when the :class:`~mne.Epochs` data are finally loaded
# and used. There are several ways this can get triggered, such as:
#
# - explicitly loading the data into RAM with the :meth:`~mne.Epochs.load_data`
#   method
# - plotting the data (:meth:`~mne.Epochs.plot`,
#   :meth:`~mne.Epochs.plot_image`, etc)
# - using the :meth:`~mne.Epochs.average` method to create an
#   :class:`~mne.Evoked` object
#
# You can also trigger dropping with the :meth:`~mne.Epochs.drop_bad` method;
# if ``reject`` and/or ``flat`` criteria have already been provided to the
# epochs constructor, :meth:`~mne.Epochs.drop_bad` can be used without
# arguments to simply delete the epochs already marked for removal (if the
# epochs have already been dropped, nothing further will happen):

epochs.drop_bad()

###############################################################################
# Alternatively, if rejection thresholds were not originally given to the
# :class:`~mne.Epochs` constructor, they can be passed to
# :meth:`~mne.Epochs.drop_bad` later instead; this can also be a way of
# imposing progressively more stringent rejection criteria:

stronger_reject_criteria = dict(mag=2000e-15,     # 2000 fT
                                grad=2000e-13,    # 2000 fT/cm
                                eeg=100e-6,       # 100 µV
                                eog=100e-6)       # 100 µV

epochs.drop_bad(reject=stronger_reject_criteria)
print(epochs.drop_log)

###############################################################################
# Note that a complementary Python module, the `autoreject package`_, uses
# machine learning to find optimal rejection criteria, and is designed to
# integrate smoothly with MNE-Python workflows. This can be a considerable
# time-saver when working with heterogeneous datasets.
#
#
# .. LINKS
#
# .. _`memory-mapped`: https://en.wikipedia.org/wiki/Memory-mapped_file
# .. _`autoreject package`: http://autoreject.github.io/
