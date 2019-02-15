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
from mne.datasets import sample
from mne.channels.montage import _set_montage, get_builtin_montages
from mne.viz import plot_alignment
from mayavi import mlab

# print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'

###############################################################################
# check all montages
#

fix_units = {'EGI_256':'cm', 'GSN-HydroCel-128':'cm', 'GSN-HydroCel-129':'cm',
             'GSN-HydroCel-256':'cm', 'GSN-HydroCel-257':'cm',
             'GSN-HydroCel-32':'cm', 'GSN-HydroCel-64_1.0':'cm',
             'GSN-HydroCel-65_1.0':'cm', 'biosemi128':'mm', 'biosemi16':'mm',
             'biosemi160':'mm', 'biosemi256':'mm', 'biosemi32':'mm',
             'biosemi64':'mm', 'easycap-M1':'mm', 'easycap-M10':'mm',
             'mgh60':'m', 'mgh70':'m', 'standard_1005':'m',
             'standard_1020':'m', 'standard_alphabetic':'m',
             'standard_postfixed':'m', 'standard_prefixed':'m',
             'standard_primed':'m'}

# fig = fig if 'fig' in locals() else None
# current_montage = get_builtin_montages()[0]
for current_montage in get_builtin_montages():
    fig = None
    montage = mne.channels.read_montage(current_montage,
                                        unit=fix_units[current_montage])
    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=1,
                           ch_types='eeg',
                           montage=montage)
    #
    fig = plot_alignment(info, trans=None, subject='fsaverage', dig=False,
                         eeg=['projected'], meg=[],
                         coord_frame='head', subjects_dir=subjects_dir,
                         fig=fig)
    mlab.view(135, 80)
    mlab.title(montage.kind, figure=fig)

###############################################################################
# something weird, when the scale is smaller than it should, the points cluster
# in a really funky manner instead of getting inside the skull
#

for current_montage in (_ for _ in get_builtin_montages() if _.startswith('standard')):
    fig = None
    montage = mne.channels.read_montage(current_montage, unit='cm')
    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=1,
                           ch_types='eeg',
                           montage=montage)
    #
    fig = plot_alignment(info, trans=None, subject='fsaverage', dig=False,
                         eeg=['projected'], meg=[],
                         coord_frame='head', subjects_dir=subjects_dir,
                         fig=fig)
    mlab.title(montage.kind, figure=fig)



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
# TODO

# trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
# raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
# raw.load_data()
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
