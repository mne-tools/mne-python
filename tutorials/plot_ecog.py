"""
======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from scipy.io import loadmat
from mayavi import mlab

import mne
from mne.viz import plot_trans

print(__doc__)

###############################################################################
# Let's load some ECoG electrode locations and names, and turn them into
# a :class:`mne.DigMontage` class.

mat = loadmat(mne.datasets.misc.data_path() + '/ecog/sample_ecog.mat')
ch_names = mat['ch_names'].tolist()
elec = mat['elec']
dig_ch_pos = dict(zip(ch_names, elec))
mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
print('Created %s channel positions' % len(ch_names))

###############################################################################
# Now that we have our electrode positions in MRI coordinates, we can create
# our measurement info structure.

info = mne.create_info(ch_names, 1000., 'ecog', montage=mon)

###############################################################################
# We can then plot the locations of our electrodes on our subject's brain.
#
# .. note:: These are not real electrodes for this subject, so they
#           do not align to the cortical surface perfectly.

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
fig = plot_trans(info, trans=None, subject='sample', subjects_dir=subjects_dir)
mlab.view(200, 70)
