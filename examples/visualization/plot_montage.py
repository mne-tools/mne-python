"""
======================================
Plotting sensor layouts of EEG Systems
======================================

Show sensor layouts of different EEG systems.

XXX: things to refer properly:
:ref:`example_eeg_sensors_on_the_scalp`
:ref:`tut_erp`
"""
# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
from mne.datasets import sample
from mne.channels.montage import _set_montage, get_builtin_montages, Digitization
from mne.viz import plot_alignment
from mayavi import mlab

# print(__doc__)

###############################################################################
# Things that need to go somewhere
#
data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
def get_foo_dig():
    raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
    return Digitization(dig_list=raw.info['dig'].copy())

# This should be a file distributed in MNE with the transformation
def get_trans():
    trans = mne.Transform(fro='head', to='mri')
    trans['trans'] = np.array([[ 0.99981296, -0.00503971,  0.01867181,  0.00255929],
                               [ 0.00692004,  0.99475515, -0.10205064, -0.02091804],
                               [-0.01805957,  0.10216076,  0.99460393, -0.04416016],
                               [ 0.,          0.,          0.,          1.        ]])
    return trans



###############################################################################
# USER CODE
#

my_info = mne.create_info(ch_names=[],
                          sfreq=1,
                          ch_types='eeg',
                          montage=get_foo_dig()
                          )

trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
fig = plot_alignment(my_info, trans=trans, subject='sample', dig=True,
                     eeg=['original'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir,
                     fig=None)

fig = plot_alignment(my_info, trans=get_trans(), subject='fsaverage', dig=True,
                     eeg=['original'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir,
                     fig=None)



###############################################################################
# Questions I've
#
# 1 - What happens with `_set_montage` and therefore `create_info` when
#     len(info.ch_names) != len(montage.ch_names)
#
# 2 - Why some montages have feducials and some does not? why they don't get
# nicely ploted with the blue red green stuff
#
# 3 - Can I read_montage(bla.fif) ?
#
# 4 -


###############################################################################
# Try to play with a dig (from sample) and fsaverage
#
#
# Things I don't understand here.
# A- `eeg=['original'], trans=trans, subject='sample'`  this should work.
# B- `eeg=['original'], trans=None, subject='sample'`  this should work and the points should be off.
# C- `eeg=['original'], trans=None, subject='fsaverage'`  this should work and the points should be off.
# D- `eeg='projected'` crasses whatever the rest
#
# Given that A works, I should be able to generate a trans file (by reading
# https://www.martinos.org/mne/stable/manual/cookbook.html#aligning-coordinate-frames and follow links there),
# add the trans into C and it should work. But I've no idea how to tackle it just from the doc
#




###############################################################################
# TODO


# raw.pick_types(meg=False, eeg=True, eog=True)
# print(raw.info['chs'][0]['loc'])

# # Nothing
# fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# fig = plot_alignment(info, trans=None, subject='fsaverage', dig=False,
#                      eeg=['projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# # With montage
# montage = mne.channels.read_montage('standard_1020')
# # raw.set_montage(montage, set_dig=True)

# _set_montage(raw.info, montage, update_ch_names=True, set_dig=True)
# fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# # with a name
# # raw.set_montage('mgh60')  # test loading with string argument
# montage = mne.channels.read_montage('standard_1020')
# _set_montage(raw.info, montage, update_ch_names=True, set_dig=True)
# fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)
