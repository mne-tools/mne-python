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
from mne.channels.montage import transform_to_head
from mne.datasets import fetch_fsaverage
from mne.viz import plot_alignment, set_3d_view, set_3d_title

import warnings


warnings.simplefilter("ignore")

subjects_dir = op.dirname(fetch_fsaverage())

sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.085)

###############################################################################
# check all montages
#

for current_montage in get_builtin_montages():

    montage = mne.channels.make_standard_montage(current_montage)

    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=1,
                           ch_types='eeg',
                           montage=montage)

    fig = plot_alignment(info, trans=None,
                         subject='fsaverage',
                         subjects_dir=subjects_dir,
                         eeg=['projected'],
                         )
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title=current_montage)



###############################################################################
# Show them on the sphere with 'unk' (or whatever they have been read)
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    fig=mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig=True, surfaces='inner_skull', bem=sphere,

        # Create dummy info
        info=mne.create_info(
            ch_names=montage.ch_names,
            sfreq=1,
            ch_types='eeg',
            montage=montage,
        ),
    )
    set_3d_view(figure=fig, azimuth=135, elevation=80)
    set_3d_title(figure=fig, title='{} [unk]'.format(current_montage))

###############################################################################
# Show them on the sphere with 'head' (only those that need transform)
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    if montage._coord_frame == 'head':
        continue
    else:
        fig=mne.viz.plot_alignment(
            # Plot options
            show_axes=True, dig=True, surfaces='inner_skull', bem=sphere,

            # Create dummy info
            info=mne.create_info(
                ch_names=montage.ch_names,
                sfreq=1,
                ch_types='eeg',
                montage=transform_to_head(montage),
            ),
        )
        set_3d_view(figure=fig, azimuth=135, elevation=80)
        set_3d_title(figure=fig, title='{} [head]'.format(current_montage))

###############################################################################
# Show them on the fsaverage with 'unk' (or whatever they have been read)
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    fig=mne.viz.plot_alignment(
        # Plot options
        show_axes=True, dig=True, surfaces='inner_skull', trans=None,
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
    set_3d_title(figure=fig, title='{} [unk]'.format(current_montage))

###############################################################################
# Show them on the fsaverage with 'head' (only those that need transform)
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    if montage._coord_frame == 'head':
        continue
    else:
        fig=mne.viz.plot_alignment(
            # Plot options
            show_axes=True, dig=True, surfaces='inner_skull', trans=None,
            subject='fsaverage', subjects_dir=subjects_dir,

            # Create dummy info
            info=mne.create_info(
                ch_names=montage.ch_names,
                sfreq=1,
                ch_types='eeg',
                montage=transform_to_head(montage),
            ),
        )
        set_3d_view(figure=fig, azimuth=135, elevation=80)
        set_3d_title(figure=fig, title='{} [head]'.format(current_montage))



###############################################################################
# Plot them side by side  in sphere
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    if montage._coord_frame == 'head':
        continue
    else:
        for kk, vv in {
                'unk': montage,
                'head': transform_to_head(montage)
        }.items():
            fig=mne.viz.plot_alignment(
                # Plot options
                show_axes=True, dig=True, surfaces='inner_skull', bem=sphere,

                # Create dummy info
                info=mne.create_info(
                    ch_names=montage.ch_names,
                    sfreq=1,
                    ch_types='eeg',
                    montage=vv,
                ),
            )
            set_3d_view(figure=fig, azimuth=135, elevation=80)
            set_3d_title(
                figure=fig, title='{} [{}]'.format(current_montage, kk)
            )

###############################################################################
# Plot them side by side  in fsaverage
#

for current_montage in get_builtin_montages():
    montage = mne.channels.make_standard_montage(current_montage)
    if montage._coord_frame == 'head':
        continue
    else:
        for kk, vv in {
                'unk': montage,
                'head': transform_to_head(montage)
        }.items():
            fig=mne.viz.plot_alignment(
                # Plot options
                show_axes=True, dig=True, surfaces='inner_skull', trans=None,
                subject='fsaverage', subjects_dir=subjects_dir,

                # Create dummy info
                info=mne.create_info(
                    ch_names=montage.ch_names,
                    sfreq=1,
                    ch_types='eeg',
                    montage=vv,
                ),
            )
            set_3d_view(figure=fig, azimuth=135, elevation=80)
            set_3d_title(
                figure=fig, title='{} [{}]'.format(current_montage, kk)
            )
