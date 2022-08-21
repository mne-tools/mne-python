# -*- coding: utf-8 -*-
"""
.. _ex-mne-helmet:

=============================
Plot the MNE brain and helmet
=============================

This tutorial shows how to make the MNE helmet + brain image.
"""

# %%

import mne

sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / 'subjects'
fname_evoked = sample_path / 'MEG' / 'sample' / 'sample_audvis-ave.fif'
fname_inv = (sample_path / 'MEG' / 'sample' /
             'sample_audvis-meg-oct-6-meg-inv.fif')
fname_trans = sample_path / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'
inv = mne.minimum_norm.read_inverse_operator(fname_inv)
evoked = mne.read_evokeds(fname_evoked, baseline=(None, 0),
                          proj=True, verbose=False, condition='Left Auditory')
maps = mne.make_field_map(evoked, trans=fname_trans, ch_type='meg',
                          subject='sample', subjects_dir=subjects_dir)
time = 0.083
fig = mne.viz.create_3d_figure((256, 256))
mne.viz.plot_alignment(
    evoked.info, subject='sample', subjects_dir=subjects_dir, fig=fig,
    trans=fname_trans, meg='sensors', eeg=False, surfaces='pial',
    coord_frame='mri')
evoked.plot_field(maps, time=time, fig=fig, time_label=None, vmax=5e-13)
mne.viz.set_3d_view(
    fig, azimuth=40, elevation=87, focalpoint=(0., -0.01, 0.04), roll=-25,
    distance=0.55)
