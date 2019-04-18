# -*- coding: utf-8 -*-
"""
Plotting sensor layouts of EEG Systems
======================================

Show sensor layouts of different EEG systems.
"""  # noqa
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

from mayavi import mlab
import os.path as op

import mne
from mne.channels.montage import get_builtin_montages
from mne.viz import plot_alignment

# print(__doc__)

subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')

###############################################################################
# check all montages
#

fix_units = {'EGI_256': 'cm', 'GSN-HydroCel-128': 'cm',
             'GSN-HydroCel-129': 'cm', 'GSN-HydroCel-256': 'cm',
             'GSN-HydroCel-257': 'cm', 'GSN-HydroCel-32': 'cm',
             'GSN-HydroCel-64_1.0': 'cm', 'GSN-HydroCel-65_1.0': 'cm',
             'biosemi128': 'mm', 'biosemi16': 'mm', 'biosemi160': 'mm',
             'biosemi256': 'mm', 'biosemi32': 'mm', 'biosemi64': 'mm',
             'easycap-M1': 'mm', 'easycap-M10': 'mm',
             'mgh60': 'm', 'mgh70': 'm',
             'standard_1005': 'm', 'standard_1020': 'm',
             'standard_alphabetic': 'm',
             'standard_postfixed': 'm',
             'standard_prefixed': 'm',
             'standard_primed': 'm'}

for current_montage in get_builtin_montages():
    montage = mne.channels.read_montage(current_montage,
                                        unit=fix_units[current_montage],
                                        transform=False)

    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=1,
                           ch_types='eeg',
                           montage=montage)

    fig = plot_alignment(info, trans=None,
                         subject='fsaverage',
                         subjects_dir=subjects_dir,
                         eeg=['original', 'projected'],
                         )
    mlab.view(135, 80)
    mlab.title(montage.kind, figure=fig)

###############################################################################
# check all montages using 'auto' everywhere
#

for current_montage in get_builtin_montages():

    cant_transform = ['EGI_256', 'easycap-M1', 'easycap-M10']
    transform = False if current_montage in cant_transform else True
    montage = mne.channels.read_montage(current_montage,
                                        unit='auto',
                                        transform=transform)

    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=1,
                           ch_types='eeg',
                           montage=montage)

    fig = plot_alignment(info, trans=None,
                         subject='fsaverage',
                         subjects_dir=subjects_dir,
                         eeg=['original', 'projected'],
                         )
    mlab.view(135, 80)
    mlab.title(montage.kind, figure=fig)
