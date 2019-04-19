# -*- coding: utf-8 -*-
"""
.. _combine-evoked-tutorial:

Combining Evoked estimates across conditions
============================================

.. include:: ../../tutorial_links.inc

This tutorial covers combining Evoked objects (optionally with weighting).
"""

###############################################################################
# We'll start by importing the modules we need and loading example evoked data:

import os
import numpy as np
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_ave_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')
evokeds = mne.read_evokeds(sample_data_ave_file, baseline=[None, 0])
aud_left, aud_right, vis_left, vis_right = evokeds

###############################################################################
# A common visualization technique when comparing conditions is to subtract one
# condition from another and plot the difference between the two. In other
# cases, one might want to pool across conditions after having already averaged
# within conditions. The :func:`mne.combine_evoked` function supports both of
# these cases: it averages together two or more :class:`~mne.Evoked` objects,
# and allows you to weight the average any way you like. Let's start by
# plotting each :class:`~mne.Evoked` object separately (we'll show just the
# gradiometers, to save space):

aud_left.plot(spatial_colors=True, picks='grad')
aud_right.plot(spatial_colors=True, picks='grad')

###############################################################################
# As you can see in the upper-left corner of each plot, the different
# :class:`~mne.Evoked` objects are each based on different numbers of epochs.
# You can also get this data from the objects' ``nave`` attribute:

print(f'Left Auditory evoked response is average of {aud_left.nave} epochs')
print(f'Right Auditory evoked response is average of {aud_right.nave} epochs')

###############################################################################
# We can tell (from the color of the channel traces) that different sensor
# channels are more or less active depending on which ear received the
# stimulus. We might expect the difference between left and right conditions
# will be strongest at electrodes over auditory areas of the cortex. To test
# this, we can *subtract* the evoked response to right-auditory stimuli from
# the evoked response to left-auditory stimuli, using
# :func:`~mne.combine_evoked`. There are two ways to perform that subtraction:

# use a minus sign before one of the Evoked objects
aud_lr_difference = mne.combine_evoked([aud_left, -aud_right], weights='nave')

# provide a negative weight to one of the Evoked objects
weights = (np.array([aud_left.nave, -1 * aud_right.nave]) /
           (aud_left.nave + aud_right.nave))
aud_lr_difference2 = mne.combine_evoked([aud_left, aud_right], weights=weights)

###############################################################################
# In the first case, the parameter ``weights='nave'`` will weight each
# :class:`~mne.Evoked` object by the number of epochs used to generate it
# (there is also a setting ``weights='equal'`` that ignores the number of
# epochs used to generate each :class:`~mne.Evoked` object, and performs an
# unweighted average of the evoked data). In the second case, both
# :class:`~mne.Evoked` objects are positive but the second weight is negative
# (the weights are normalized so that their absolute values sum to one,
# similar to the ``weights='nave'`` setting); the only difference is
# where the minus sign ends up in the description:

print(aud_lr_difference)
print(aud_lr_difference2)
assert np.array_equal(aud_lr_difference.data, aud_lr_difference2.data)

###############################################################################
# When we plot this difference, we indeed see in the scalp topographies that
# the main areas of difference are above the left and right auditory cortices:

aud_lr_difference.plot_joint(picks='grad')

###############################################################################
# If instead we wanted to *pool* these :class:`~mne.Evoked` responses instead
# of subtracting, the ``weights='nave'`` and ``weights='equal'`` settings work
# just as well when all of the :class:`~mne.Evoked` objects are positive:

aud_lr_pooled = mne.combine_evoked([aud_left, aud_right], weights='nave')
aud_lr_pooled.plot_joint(picks='grad')

###############################################################################
# In principle, this approach to pooling and subtracting across *conditions*
# can be used equally well for pooling or subtracting across *subjects* or
# *groups*. However, bear in mind that electrode position relative to the
# subjects' heads/brains can vary substantially, such that different sensors
# end up most strongly reflecting similar activity in different subjects. For
# that reason, it may be prudent to take additional steps before aggregating
# across subjects in sensor-level analyses.
