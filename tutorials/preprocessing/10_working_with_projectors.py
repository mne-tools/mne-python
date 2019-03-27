# -*- coding: utf-8 -*-
"""
.. _projectors-basics-tutorial:

Working with projectors in MNE-Python
=====================================

.. include:: ../../tutorial_links.inc

This tutorial covers loading and saving projectors, adding and removing
projectors from Raw objects, and the difference between "applied" and
"unapplied" projectors.
"""

###############################################################################
# As usual we'll start by importing the modules we need, and loading some
# example data:

import os
import matplotlib.pyplot as plt
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

###############################################################################
# In our example data, :ref:`SSP <ssp-tutorial>` has already been performed
# using empty room recordings, but the :term:`projectors <projector>` are
# stored alongside the raw data (they have not been *applied* yet). You can see
# the projectors in the output of :func:`~mne.io.read_raw_fif` above, and also
# in the ``projs`` field of ``raw.info``:

print(raw.info['projs'])

###############################################################################
# .. note::
#
#     In MNE-Python, the environmental noise vectors are computed using
#     `principal component analysis`_, usually abbreviated "PCA", which is why
#     the SSP projectors usually have names like "PCA-v1".
#
# ``raw.info['projs']`` is an ordinary Python list of :class:`~mne.Projection`
# objects, so you can access individual projectors by indexing into it:

first_projector = raw.info['projs'][0]
print(first_projector)

###############################################################################
# You can see the effect the projectors are having on the measured signal by
# comparing plots with and without the projectors applied. Here we'll look at
# just the magnetometers, and a 2-second sample from the beginning of the file:

# get data with and without SSP projectors applied
ending_sample = int(raw.time_as_index(2))
raw_data, times = raw.get_data(picks='mag', stop=ending_sample,
                               return_times=True)
ssp_data = raw.copy().apply_proj().get_data(picks='mag', stop=ending_sample)

# convert from teslas to femtoteslas
raw_data /= 1e-15
ssp_data /= 1e-15

# plot
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
for ax, data, label in zip(axs, [raw_data, ssp_data], ['Before', 'After']):
    ax.plot(times, data.T, color='k', linewidth=0.1)
    ax.text(0.05, 0.9, f'{label} SSP', transform=ax.transAxes)
axs[1].set_xlabel('time (s)')
fig.text(0.02, 0.5, 'flux density (fT)', va='center', rotation='vertical')
fig.tight_layout(rect=(0.04, 0, 1, 1))

###############################################################################
# You could also produce similar plots using MNE-Python plotting methods. By
# default, ``raw.plot()`` will apply the projectors in the background before
# plotting (without modifying the :class:`~mne.io.Raw` object); you can control
# this with the boolean ``proj`` parameter as shown below, or you can turn them
# on and off interactively with the projectors interface, accessed via the
# ``Proj`` button in the lower right corner of the plot window. Here are the
# equivalent MNE-Python commands to create plots similar to the one shown
# above:

mags = raw.copy().pick_types(meg='mag')
mags.plot(duration=2, butterfly=True, proj=False)
mags.plot(duration=2, butterfly=True, proj=True)

###############################################################################
# Loading and saving projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SSP can be used for other types of signal cleaning besides just reduction of
# environmental noise. You probably noticed two large deflections in the
# magnetometer signals in the previous plot that were not removed by the
# empty-room projectors — those are artifacts of the subject's heartbeat. SSP
# can be used to remove those artifacts as well. The sample data includes
# projectors for heartbeat noise reduction that were saved in a separate file
# from the raw data, which can be loaded with the :func:`mne.read_proj`
# function:

ecg_proj_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                             'sample_audvis_ecg-proj.fif')
ecg_projs = mne.read_proj(ecg_proj_file)
print(ecg_projs)

###############################################################################
# There is a corresponding :func:`mne.write_proj` function that can be used to
# save projectors to disk in ``.fif`` format:
#
# .. code-block:: python3
#
#     mne.write_proj('heartbeat-proj.fif', ecg_projs)
#
# .. note::
#
#     By convention, MNE-Python expects projectors to be saved with a filename
#     ending in ``-proj.fif`` (or ``-proj.fif.gz``), and will issue a warning
#     if you forgo this recommendation.
#
#
# Adding and removing projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Above, when we printed the ``ecg_projs`` list that we loaded from a file, it
# showed two projectors for gradiometers (the first two, marked "planar"), two
# for magnetometers (the middle two, marked "axial"), and two for EEG sensors
# (the last two, marked "eeg"). We can add them to the :class:`~mne.io.Raw`
# object using the :meth:`~mne.io.Raw.add_proj` method:

raw.add_proj(ecg_projs)

###############################################################################
# There is a corresponding method :meth:`~mne.io.Raw.del_proj` that will remove
# projectors based on their index within the ``projs`` field of ``raw.info``.
#
# To see how the ECG projectors affect the measured signal, we can once again
# apply the projectors and plot the data. We'll compare it to the ``ssp_data``
# variable we created above, which had only the empty room SSP projectors
# applied:

clean_data = raw.copy().apply_proj().get_data(picks='mag', stop=ending_sample)
clean_data /= 1e-15  # convert to femtoteslas
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
for ax, data, label in zip(axs, [ssp_data, clean_data], ['Without', 'With']):
    ax.plot(times, data.T, color='k', linewidth=0.1)
    ax.text(0.05, 0.9, f'{label} ECG projectors', transform=ax.transAxes)
axs[1].set_xlabel('time (s)')
fig.text(0.02, 0.5, 'flux density (fT)', va='center', rotation='vertical')
fig.tight_layout(rect=(0.04, 0, 1, 1))

###############################################################################
# .. note::
#
#     Remember that once a projector is applied, it can't be un-applied, so
#     during interactive / exploratory analysis it's a good idea to use the
#     :meth:`~mne.io.Raw.copy` method before applying projectors to a
#     :class:`~mne.io.Raw` object.
#
#
# When to "apply" projectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# TODO: have they been applied? (:attr:`mne.io.Raw.proj`)
#
# TODO: when to apply projectors
#
# TODO: when projectors are applied automatically
