# -*- coding: utf-8 -*-
"""
.. _manip-raw-time-domain-tutorial:

Manipulating the time domain of :class:`~mne.io.Raw` objects
============================================================

This tutorial covers cropping :class:`~mne.io.Raw` objects to restrict the time
domain, and how to concatenate :class:`~mne.io.Raw` objects (whether selections
from the same recording or different recordings). As always we'll start by
importing the modules we need, and loading some example data:
"""

import os
import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)

###############################################################################
# We've already seen how to extract a short section of a :class:`~mne.io.Raw`
# object using indexing ``raw[channel_selection,
# starting_sample:ending_sample]``. We can also restrict the time span by
# *cropping* the  :class:`~mne.io.Raw` object using the
# :meth:`~mne.io.Raw.crop` method, which modifies the :class:`~mne.io.Raw`
# object in place. :meth:`~mne.io.Raw.crop` takes parameters ``tmin`` and
# ``tmax``, both in seconds:

raw_selection = raw.copy().crop(tmin=10, tmax=12.5)

###############################################################################
# :meth:`~mne.io.Raw.crop` also modifies the :attr:`~mne.io.Raw.first_samp` and
# :attr:`~mne.io.Raw.times` attributes, so that ``time = 0``
# of the returned :class:`~mne.io.Raw` object corresponds to the first sample
# of the cropped object. Accordingly, if you wanted to re-crop
# ``raw_selection`` from 11 to 12.5 seconds (instead of 10 to 12.5 as above)
# then the subsequent call to :meth:`~mne.io.Raw.crop` should get ``tmin=1``
# (not ``tmin=11``), and leave ``tmax`` unspecified to keep everything from
# ``tmin`` up to the end of the object:

print(raw_selection.times.min(), raw_selection.times.max())
raw_selection.crop(tmin=1)
print(raw_selection.times.min(), raw_selection.times.max())

###############################################################################
# Remember that sample times don't always align exactly with requested ``tmin``
# or ``tmax`` values (due to sampling frequency), which is why the ``max``
# values of the cropped files don't exactly match the requested ``tmax``. See
# :ref:`time-as-index` for details.
#
# If you need to concatenate selections (or entire :class:`~mne.io.Raw` files)
# you can use the :meth:`~mne.io.Raw.append` method:

raw_selection2 = raw.copy().crop(tmin=50, tmax=51.3)    # 1.3 seconds
raw_selection3 = raw.copy().crop(tmin=80, tmax=80.1)    # 0.1 seconds
raw_selection.append([raw_selection2, raw_selection3])  # 2.9 seconds total
print(raw_selection.times.min(), raw_selection.times.max())

###############################################################################
# .. note::
#
#     Be careful with concatenated :class:`~mne.io.Raw` objects, especially
#     when saving: :meth:`~mne.io.Raw.append` only preserves the ``info``
#     attribute of the initial :class:`~mne.io.Raw` file (the one outside the
#     method call).
