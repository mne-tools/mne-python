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
from mne.preprocessing.ieeg import project_sensors_onto_brain
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
