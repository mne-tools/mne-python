# -*- coding: utf-8 -*-
"""
.. _plot_montage:

Plotting sensor layouts of EEG Systems
======================================

This example illustrates how to load all the EEG system montages
shipped in MNE-python, and display it on fsaverage template.
"""  # noqa: D205, D400
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op

import mne
from mne.channels.montage import get_builtin_montages
from mne.datasets import fetch_fsaverage
from mne.viz import set_3d_title, set_3d_view


###############################################################################
# Check all montages against a sphere
#

sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.085)

for current_montage in get_builtin_montages():

    montage = mne.channels.make_standard_montage(current_montage)
    fig = mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig=True, surfaces='head', bem=sphere,

        # Create dummy info
        info=mne.create_info(
            ch_names=montage.ch_names,
            sfreq=1,
            ch_types='eeg',
            montage=montage,
        ),
    )
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title=current_montage)


###############################################################################
# Check all montages against fsaverage
#

subjects_dir = op.dirname(fetch_fsaverage())

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    fig = mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig=True, surfaces='head', trans=None,
        subject='fsaverage', subjects_dir=subjects_dir,

        # Create dummy info
        info=mne.create_info(
            ch_names=montage.ch_names,
            sfreq=1,
            ch_types='eeg',
            montage=montage,
        ),
    )
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title=current_montage)
