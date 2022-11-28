# -*- coding: utf-8 -*-
"""
.. _plot_montage:

Plotting sensor layouts of EEG systems
======================================

This example illustrates how to load all the EEG system montages
shipped in MNE-python, and display it on the fsaverage template subject.
"""  # noqa: D205, D400
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD-3-Clause

# %%

import os.path as op
import numpy as np

import mne
from mne.channels.montage import get_builtin_montages
from mne.datasets import fetch_fsaverage
from mne.viz import set_3d_title, set_3d_view


# %%
# Check all montages against a sphere

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    info = mne.create_info(
        ch_names=montage.ch_names, sfreq=100., ch_types='eeg')
    info.set_montage(montage)
    sphere = mne.make_sphere_model(r0='auto', head_radius='auto', info=info)
    fig = mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig='fiducials', surfaces='head',
        trans=mne.Transform("head", "mri", trans=np.eye(4)),  # identity
        bem=sphere, info=info)
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title=current_montage)


# %%
# Check all montages against fsaverage

subjects_dir = op.dirname(fetch_fsaverage())

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    # Create dummy info
    info = mne.create_info(
        ch_names=montage.ch_names, sfreq=100., ch_types='eeg')
    info.set_montage(montage)
    fig = mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig='fiducials', surfaces='head', mri_fiducials=True,
        subject='fsaverage', subjects_dir=subjects_dir, info=info,
        coord_frame='mri',
        trans='fsaverage',  # transform from head coords to fsaverage's MRI
    )
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title=current_montage)
