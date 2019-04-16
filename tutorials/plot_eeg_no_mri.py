# -*- coding: utf-8 -*-
"""
EEG forward operator with a template MRI
========================================

This tutorial explains how to compute the forward operator from EEG data
using template MRI subject. We use here the fsaverage brain
provided by freesurfer.

.. important:: Source reconstruction without an individual T1 MRI from the
               subject will be less accurate. Do not over interpret
               activity locations which can be off by multiple centimeters.

.. contents:: This tutorial covers:
   :local:
   :depth: 2

"""  # noqa: E501

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op
import mne
from mne.datasets.sample import data_path
from mne.datasets import eegbci

subjects_dir = op.join(data_path(), 'subjects')
subject = 'fsaverage'
trans_fname = op.join(op.dirname(mne.__file__), "data", subject,
                      "fsaverage-trans.fif")

##############################################################################
# Load the data
# -------------
#
# We use here EEG data from the BCI dataset.

raw_fname, = eegbci.load_data(subject=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True, stim_channel='auto')

# Clean channel names to be able to use a standard 1005 montage
ch_names = [c.replace('.', '') for c in raw.ch_names]
raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

# Read and set the EEG electrode locations
montage = mne.channels.read_montage('standard_1005', ch_names=raw.ch_names,
                                    transform=False)
raw.set_montage(montage)

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, subject=subject, subjects_dir=subjects_dir,
    eeg=['original', 'projected'], trans=None
)

##############################################################################
# Setup source space and compute forward
# --------------------------------------

src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
print(src)

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')

conductivity = (0.3, 0.006, 0.3)  # for three layers
bem_surfaces = mne.make_bem_model(subject=subject, ico=4,
                                  conductivity=conductivity,
                                  subjects_dir=subjects_dir)
bem = mne.make_bem_solution(bem_surfaces)

fwd = mne.make_forward_solution(raw.info, trans=trans_fname, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=2)
print(fwd)

###############################################################################
# Compute sensitivity maps
# ------------------------

eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[5, 50, 100]))
