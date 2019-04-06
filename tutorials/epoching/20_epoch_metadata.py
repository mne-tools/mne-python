# -*- coding: utf-8 -*-
"""
.. _epoch-metadata-tutorial:

Adding metadata to Epochs objects
=================================

.. include:: ../../tutorial_links.inc

This tutorial covers adding rich metadata to Epochs objects using Pandas
DataFrames.
"""

###############################################################################
# As usual we'll start by importing the modules we need, and loading some
# example data:

import os
import numpy as np
import pandas as pd
import mne

# load raw
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)
# get events
events = mne.find_events(raw, stim_channel='STI 014')
# make epochs
epochs = mne.Epochs(raw, events)
# this tutorial uses randomization to make some fake metadata, so for
# reproducibility we'll set the random seed (to the one millionth prime)
np.random.seed(15485863)

###############################################################################
# In the :ref:`introductory epochs tutorial <epochs-intro-tutorial>`, we saw
# how event dictionaries with keys that contain a ``/`` character can be used
# to pool across conditions when indexing into an :class:`~mne.Epochs` object:

epochs.event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
                   'visual/right': 4, 'face': 5, 'button': 32}
print(epochs['auditory'])

###############################################################################
# In principle you could have even more trial attributes in each event
# dictionary key, separated by multiple ``/`` characters, to allow even more
# complex epoch subselection or pooling. However, this could become unweildy
# rather quickly as experimental conditions proliferate. A better alternative
# in many cases is to encode your trial metadata in a Pandas DataFrame object
# and attach it to the :class:`~mne.Epochs` object using the ``metadata``
# parameter of the :class:`~mne.Epochs` constructor (or by setting the
# :attr:`~mne.Epochs.metadata` attribute after creating the
# :class:`~mne.Epochs` object). The metadata DataFrame should have one row per
# event in the :class:`~mne.io.Raw` object, and can have as many columns as you
# need to represent your trial data. Here, we'll use the ``events`` array to
# create some fake metadata, to illustrate how it all works:

# Let's start by adding the same info that's in the event dictionary:
mode_dict = {1: 'auditory', 2: 'auditory', 3: 'visual', 4: 'visual'}
side_dict = {1: 'left', 2: 'right', 3: 'left', 4: 'right'}
mode = [mode_dict.get(ev_id, '') for ev_id in events[:, -1]]
side = [side_dict.get(ev_id, '') for ev_id in events[:, -1]]
face = (events[:, -1] == 5)
button = (events[:, -1] == 32)

# Now let's pretend that the auditory and visual trials (event IDs 1-4)
# involved spatial eccentricity, and provide an azimuth angle for the stimuli
azimuth = np.array([np.random.randint(0, 45) if ev_id in (1, 2, 3, 4) else 0
                    for ev_id in events[:, -1]])
# ...and let's say stimuli on the left have negative angles
left_indices = np.where(np.in1d(events[:, -1], [1, 3]))
azimuth[left_indices] = -1 * azimuth[left_indices]

# now let's add a variable that encodes high or low frequency for the auditory
# stimuli and high or low elevation for the visual stimuli
pitch = [np.random.choice(['high', 'low']) if ev_id in (1, 2, 3, 4) else
         'neither' for ev_id in events[:, -1]]

# finally, let's add a boolean variable "noise" representing whether the
# stimulus was presented with (visual or auditory) noise
noise = [np.random.choice([True, False]) if ev_id in (1, 2, 3, 4, 5) else
         False for ev_id in events[:, -1]]

metadataframe = pd.DataFrame(dict(face=face, button=button, mode=mode,
                                  side=side, azimuth=azimuth, pitch=pitch,
                                  noise=noise))
print(metadataframe.head())

###############################################################################
# Now we can add this metadata to the :class:`~mne.Epochs` object. Once we do,
# instead of just querying based on ``/``-separated keywords like we did with
# the event dictionary approach, we can use the full power of Pandas
# ``Dataframe.query`` method to select epochs:

epochs.metadata = metadataframe

print(epochs['-15 < azimuth < 15 and pitch == "high" and not noise'])

###############################################################################
# .. warning:: Do not set or change the DataFrame index of ``epochs.metadata``.
#              The index is controlled by MNE to mirror ``epochs.selection``
#              (i.e., which epochs are "good" vs "bad"). You should also avoid
#              manipulating the ``epochs.metadata`` DataFrame once it has been
#              added to the :class:`~mne.Epochs` object; while some in-place
#              operations are possible, others (such as adding or dropping
#              rows) will create inconsistency between the metadata and the
#              epoch data.
#
# This approach to querying :class:`~mne.Epochs` objects is especially useful
# for experiments with a large variety of stimuli that vary along several
# dimensions. For example, if each stimulus is a single spoken or written word,
# neurolinguistic researchers may be interested in how the *concreteness* of a
# word or its *frequency of use* in daily speech affect how it is processed by
# the brain; the epochs metadata + querying approach makes it quick and easy to
# filter epochs based on these and other properties of the stimuli. See
# :ref:`epochs-metadata-pandas-tutorial` for further querying examples using
# the kiloword dataset [1]_.
#
#
# References
# ^^^^^^^^^^
#
# .. [1] Dufau S, Grainger J, Midgley KJ and Holcomb PJ. (2015). A thousand
#        words are worth a picture: Snapshots of printed-word processing in an
#        event-related potential megastudy. *Psychological Science* 26(12),
#        1887-1897. doi:10.1177/0956797615603934
