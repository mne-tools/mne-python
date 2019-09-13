# -*- coding: utf-8 -*-
"""
.. _plot_montage:

Setting up a standard montage template into an old file
=======================================================

This example illustrates how to update ``ch_names`` in `DigMonge`.
"""  # noqa: D205, D400
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op

import mne
from mne.io import read_raw_fif, __file__ as _MNE_IO_FILE
from mne.channels import make_standard_montage

###############################################################################
# Check all montages against fsaverage
#

# load the raw file
fname = op.join(op.dirname(_MNE_IO_FILE), 'tests', 'data', 'test_raw.fif')
raw = read_raw_fif(fname)

# load the standard montage
montage = make_standard_montage('mgh60')


###############################################################################
# `ch_names` in raw and montage do not match. Therefore setting the montage
# directly would crash.
#

print([name for name in raw.info['ch_names'] if name.startswith('EEG')])
print(montage.ch_names)

###############################################################################
#
# To solve it we will modify the names in place based on our needs. Take into
# account that order of `ch_names` in :class:`mne.channels.DigMontage` is the
# same as the digitization locations. Therefore the order of `ch_names` should
# not be altered. In this case channel names in raw have an extra space between
# `EEG` and the number of the channel.

montage.ch_names = [name.replace('EEG', 'EEG ') for name in montage.ch_names]

# now its safe to set the montage
raw.set_montage(montage)

mne.viz.plot_alignment(raw.info)
