# -*- coding: utf-8 -*-
"""
Plotting sensor layouts of EEG Systems
======================================

Show sensor layouts of different EEG systems.

"""  # noqa: E501
# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

from mayavi import mlab
import os.path as op

import mne
from mne.channels.montage import get_builtin_montages
from mne.viz import plot_alignment

# print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
trans_fname = op.join(op.dirname(mne.__file__), "data", "fsaverage",
                      "fsaverage-trans.fif")

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

for current_montage in get_builtin_montages():
    good_fig = None
    montage = mne.channels.read_montage(current_montage,
                                        unit=fix_units[current_montage])
    info = mne.create_info(ch_names=montage.ch_names,
                            sfreq=1,
                            ch_types='eeg',
                            montage=montage)

    good_fig = plot_alignment(info, trans=None, subject='fsaverage', dig=False,
                            eeg=['projected'], meg=[],
                            coord_frame='head', subjects_dir=subjects_dir,
                            fig=good_fig)
    mlab.view(135, 80)
    mlab.title(montage.kind, figure=good_fig)
