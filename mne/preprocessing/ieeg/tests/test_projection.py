# -*- coding: utf-8 -*-
"""Test the compute_current_source_density function.

For each supported file format, implement a test.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

import os
import os.path as op
from shutil import copyfile
import numpy as np
from numpy.testing import assert_allclose
import pytest

import mne
from mne.preprocessing.ieeg import (project_sensors_onto_brain,
                                    project_sensors_onto_inflated)
from mne.datasets import testing
from mne.transforms import _get_trans

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_project_sensors_onto_brain(tmp_path):
    """Test projecting sensors onto the brain surface."""
    tempdir = str(tmp_path)
    raw = mne.io.read_raw_fif(fname_raw)
    trans = _get_trans(fname_trans)[0]
    # test informative error for no surface first
    with pytest.raises(RuntimeError, match='requires generating a BEM'):
        project_sensors_onto_brain(raw.info, trans, 'sample',
                                   subjects_dir=tempdir)
    brain_surf_fname = op.join(tempdir, 'sample', 'bem', 'brain.surf')
    if not op.isdir(op.dirname(brain_surf_fname)):
        os.makedirs(op.dirname(brain_surf_fname))
    if not op.isfile(brain_surf_fname):
        copyfile(op.join(subjects_dir, 'sample', 'bem', 'inner_skull.surf'),
                 brain_surf_fname)
    # now make realistic ECoG grid
    raw.pick_types(meg=False, eeg=True)
    raw.load_data()
    raw.set_eeg_reference([])
    raw.set_channel_types({ch: 'ecog' for ch in raw.ch_names})
    pos = np.zeros((49, 3))
    pos[:, :2] = np.array(
        np.meshgrid(np.linspace(0, 0.02, 7),
                    np.linspace(0, 0.02, 7))).reshape(2, -1).T
    pos[:, 2] = 0.12
    raw.drop_channels(raw.ch_names[49:])
    raw.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(raw.ch_names[:49], pos)), coord_frame='head'))
    raw.info = project_sensors_onto_brain(
        raw.info, trans, 'sample', subjects_dir=tempdir)
    # plot to check, should be projected down onto inner skull
    # brain = mne.viz.Brain('sample', subjects_dir=subjects_dir, alpha=0.5,
    #                       surf='white')
    # brain.add_sensors(raw.info, trans=trans)
    test_locs = [[0.00149, -0.001588, 0.133029],
                 [0.004302, 0.001959, 0.133922],
                 [0.008602, 0.00116, 0.133723]]
    for ch, test_loc in zip(raw.info['chs'][:3], test_locs):
        assert_allclose(ch['loc'][:3], test_loc, rtol=0.01)


@testing.requires_testing_data
def test_project_sensors_onto_inflated(tmp_path):
    """Test projecting sEEG sensors onto an inflated brain surface."""
    tempdir = str(tmp_path)
    raw = mne.io.read_raw_fif(fname_raw)
    trans = _get_trans(fname_trans)[0]
    os.makedirs(op.join(tempdir, 'sample', 'surf'), exist_ok=True)
    for hemi in ('lh', 'rh'):
        # fake white surface for pial
        copyfile(op.join(subjects_dir, 'sample', 'surf', f'{hemi}.white'),
                 op.join(tempdir, 'sample', 'surf', f'{hemi}.pial'))
        copyfile(op.join(subjects_dir, 'sample', 'surf', f'{hemi}.curv'),
                 op.join(tempdir, 'sample', 'surf', f'{hemi}.curv'))
        copyfile(op.join(subjects_dir, 'sample', 'surf', f'{hemi}.inflated'),
                 op.join(tempdir, 'sample', 'surf', f'{hemi}.inflated'))
    # now make realistic sEEG locations, picked from T1
    raw.pick_types(meg=False, eeg=True)
    raw.load_data()
    raw.set_eeg_reference([])
    raw.set_channel_types({ch: 'seeg' for ch in raw.ch_names})
    pos = np.array([[25.85, 9.04, -5.38],
                    [33.56, 9.04, -5.63],
                    [40.44, 9.04, -5.06],
                    [46.75, 9.04, -6.78],
                    [-30.08, 9.04, 28.23],
                    [-32.95, 9.04, 37.99],
                    [-36.39, 9.04, 46.03]]) / 1000
    raw.drop_channels(raw.ch_names[len(pos):])
    raw.set_montage(mne.channels.make_dig_montage(
        ch_pos=dict(zip(raw.ch_names, pos)), coord_frame='head'))
    raw.info = project_sensors_onto_inflated(
        raw.info, trans, 'sample', subjects_dir=tempdir)
    # plot to check, should be projected down onto inner skull
    # brain = mne.viz.Brain('sample', subjects_dir=tempdir, alpha=0.5,
    #                       surf='inflated')
    # brain.add_sensors(raw.info, trans=trans)
    assert_allclose(raw.info['chs'][0]['loc'][:3],
                    np.array([0.0555809, 0.0034069, -0.04593032]), rtol=0.01)
    # check all on inflated surface
    x_dir = np.array([1., 0., 0.])
    head_mri_t = mne.transforms.invert_transform(trans)  # need head->mri
    for hemi in ('lh', 'rh'):
        coords, faces = mne.surface.read_surface(
            op.join(tempdir, 'sample', 'surf', f'{hemi}.inflated'))
        x_ = coords @ x_dir
        coords -= np.max(x_) * x_dir if hemi == 'lh' else \
            np.min(x_) * x_dir
        coords /= 1000  # mm -> m
        for ch in raw.info['chs']:
            loc = ch['loc'][:3]
            if not np.isnan(loc).any() and (loc[0] <= 0) == (hemi == 'lh'):
                assert np.linalg.norm(
                    coords - mne.transforms.apply_trans(head_mri_t, loc),
                    axis=1).min() < 1e-16
