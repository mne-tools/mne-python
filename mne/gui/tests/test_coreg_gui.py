# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import os.path as op
import re
import shutil
import sys
from unittest import SkipTest
import warnings

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import (assert_equal, assert_almost_equal, assert_false,
                        assert_raises, assert_true)

import mne
from mne.datasets import testing
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.transforms import invert_transform
from mne.utils import _TempDir, run_tests_if_main, requires_mayavi, traits_test
from mne.externals.six import string_types

# backend needs to be set early
try:
    from traits.etsconfig.api import ETSConfig
except ImportError:
    pass
else:
    ETSConfig.toolkit = 'qt4'


data_path = testing.data_path(download=False)
raw_path = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
kit_raw_path = op.join(kit_data_dir, 'test_bin_raw.fif')
subjects_dir = op.join(data_path, 'subjects')
warnings.simplefilter('always')


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_model_decimation():
    """Test CoregModel decimation of high-res to low-res head."""
    from mne.gui._coreg_gui import CoregModel
    tempdir = _TempDir()
    subject_dir = op.join(tempdir, 'sample')
    shutil.copytree(op.join(subjects_dir, 'sample'), subject_dir)
    # This makes the test much faster
    shutil.move(op.join(subject_dir, 'bem', 'outer_skin.surf'),
                op.join(subject_dir, 'surf', 'lh.seghead'))
    for fname in ('sample-head.fif', 'sample-head-dense.fif'):
        os.remove(op.join(subject_dir, 'bem', fname))

    model = CoregModel(guess_mri_subject=False)
    with warnings.catch_warnings(record=True) as w:
        model.mri.subjects_dir = tempdir
    assert model.mri.subject == 'sample'  # already set by setting subjects_dir
    assert any('No low-resolution' in str(ww.message) for ww in w)
    assert model.mri.bem_low_res.file == ''
    assert len(model.mri.bem_low_res.surf.rr) == 2562
    assert len(model.mri.bem_high_res.surf.rr) == 2562  # because we moved it


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_model():
    """Test CoregModel."""
    from mne.gui._coreg_gui import CoregModel
    tempdir = _TempDir()
    trans_dst = op.join(tempdir, 'test-trans.fif')

    model = CoregModel()
    assert_raises(RuntimeError, model.save_trans, 'blah.fif')

    model.mri.subjects_dir = subjects_dir
    model.mri.subject = 'sample'

    assert_false(model.mri.fid_ok)
    model.mri.lpa = [[-0.06, 0, 0]]
    model.mri.nasion = [[0, 0.05, 0]]
    model.mri.rpa = [[0.08, 0, 0]]
    assert_true(model.mri.fid_ok)

    model.hsp.file = raw_path
    assert_allclose(model.hsp.lpa, [[-7.137e-2, 0, 5.122e-9]], 1e-4)
    assert_allclose(model.hsp.rpa, [[+7.527e-2, 0, 5.588e-9]], 1e-4)
    assert_allclose(model.hsp.nasion, [[+3.725e-9, 1.026e-1, 4.191e-9]], 1e-4)
    assert model.has_lpa_data
    assert model.has_nasion_data
    assert model.has_rpa_data
    assert len(model.hsp.eeg_points) > 1

    assert len(model.mri.bem_low_res.surf.rr) == 2562
    assert len(model.mri.bem_high_res.surf.rr) == 267122

    lpa_distance = model.lpa_distance
    nasion_distance = model.nasion_distance
    rpa_distance = model.rpa_distance
    avg_point_distance = np.mean(model.point_distance)

    model.nasion_weight = 1.
    model.fit_fiducials(0)
    old_x = lpa_distance ** 2 + rpa_distance ** 2 + nasion_distance ** 2
    new_x = (model.lpa_distance ** 2 + model.rpa_distance ** 2 +
             model.nasion_distance ** 2)
    assert new_x < old_x

    model.fit_icp(0)
    new_dist = np.mean(model.point_distance)
    assert new_dist < avg_point_distance

    model.save_trans(trans_dst)
    trans = mne.read_trans(trans_dst)
    assert_allclose(trans['trans'], model.head_mri_t)

    # test restoring trans
    x, y, z = 100, 200, 50
    rot_x, rot_y, rot_z = np.rad2deg([1.5, 0.1, -1.2])
    model.trans_x = x
    model.trans_y = y
    model.trans_z = z
    model.rot_x = rot_x
    model.rot_y = rot_y
    model.rot_z = rot_z
    trans = model.mri_head_t
    model.reset_traits(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y",
                        "rot_z"])
    assert_equal(model.trans_x, 0)
    model.set_trans(trans)
    assert_almost_equal(model.trans_x, x)
    assert_almost_equal(model.trans_y, y)
    assert_almost_equal(model.trans_z, z)
    assert_almost_equal(model.rot_x, rot_x)
    assert_almost_equal(model.rot_y, rot_y)
    assert_almost_equal(model.rot_z, rot_z)

    # info
    assert_true(isinstance(model.fid_eval_str, string_types))
    assert_true(isinstance(model.points_eval_str, string_types))

    # scaling job
    assert_false(model.can_prepare_bem_model)
    model.n_scale_params = 1
    assert_true(model.can_prepare_bem_model)
    model.prepare_bem_model = True
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('sample2', False)
    assert_equal(sdir, subjects_dir)
    assert_equal(sfrom, 'sample')
    assert_equal(sto, 'sample2')
    assert_allclose(scale, model.parameters[6:9])
    assert_equal(skip_fiducials, False)
    # find BEM files
    bems = set()
    for fname in os.listdir(op.join(subjects_dir, 'sample', 'bem')):
        match = re.match(r'sample-(.+-bem)\.fif', fname)
        if match:
            bems.add(match.group(1))
    assert_equal(set(bemsol), bems)
    model.prepare_bem_model = False
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('sample2', True)
    assert_equal(bemsol, [])
    assert_true(skip_fiducials)

    model.load_trans(fname_trans)
    model.save_trans(trans_dst)
    trans = mne.read_trans(trans_dst)
    assert_allclose(trans['trans'], model.head_mri_t)
    assert_allclose(invert_transform(trans)['trans'][:3, 3] * 1000.,
                    [model.trans_x, model.trans_y, model.trans_z])


def _check_ci():
    if os.getenv('TRAVIS', 'false').lower() == 'true' and \
            sys.platform == 'darwin':
        raise SkipTest('Skipping GUI tests on Travis OSX')


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_gui():
    """Test CoregFrame."""
    _check_ci()
    home_dir = _TempDir()
    os.environ['_MNE_GUI_TESTING_MODE'] = 'true'
    os.environ['_MNE_FAKE_HOME_DIR'] = home_dir
    try:
        assert_raises(ValueError, mne.gui.coregistration, subject='Elvis',
                      subjects_dir=subjects_dir)

        from pyface.api import GUI
        from tvtk.api import tvtk
        gui = GUI()

        # avoid modal dialog if SUBJECTS_DIR is set to a directory that
        # does not contain valid subjects
        ui, frame = mne.gui.coregistration(subjects_dir='')

        frame.model.mri.subjects_dir = subjects_dir
        frame.model.mri.subject = 'sample'

        assert_false(frame.model.mri.fid_ok)
        frame.model.mri.lpa = [[-0.06, 0, 0]]
        frame.model.mri.nasion = [[0, 0.05, 0]]
        frame.model.mri.rpa = [[0.08, 0, 0]]
        assert_true(frame.model.mri.fid_ok)
        frame.data_panel.raw_src.file = raw_path
        assert isinstance(frame.eeg_obj.glyph.glyph.glyph_source.glyph_source,
                          tvtk.SphereSource)
        frame.data_panel.view_options_panel.eeg_obj.project_to_surface = True
        assert isinstance(frame.eeg_obj.glyph.glyph.glyph_source.glyph_source,
                          tvtk.CylinderSource)

        # grow hair (faster for low-res)
        assert frame.data_panel.view_options_panel.head_high_res
        frame.data_panel.view_options_panel.head_high_res = False
        frame.model.grow_hair = 40.

        # scale
        frame.coreg_panel.n_scale_params = 3
        frame.coreg_panel.scale_x_inc = True
        assert frame.model.scale_x == 101.
        frame.coreg_panel.scale_y_dec = True
        assert frame.model.scale_y == 99.

        # reset parameters
        frame.coreg_panel.reset_params = True
        assert_equal(frame.model.grow_hair, 0)
        assert not frame.data_panel.view_options_panel.head_high_res

        # configuration persistence
        assert_true(frame.model.prepare_bem_model)
        frame.model.prepare_bem_model = False
        frame.save_config(home_dir)
        ui.dispose()
        gui.process_events()

        ui, frame = mne.gui.coregistration(subjects_dir=subjects_dir)
        assert_false(frame.model.prepare_bem_model)
        assert not frame.data_panel.view_options_panel.head_high_res
        ui.dispose()
        gui.process_events()
    finally:
        del os.environ['_MNE_GUI_TESTING_MODE']
        del os.environ['_MNE_FAKE_HOME_DIR']


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_model_with_fsaverage():
    """Test CoregModel with the fsaverage brain data."""
    tempdir = _TempDir()
    from mne.gui._coreg_gui import CoregModel

    mne.create_default_subject(subjects_dir=tempdir,
                               fs_home=op.join(subjects_dir, '..'))

    model = CoregModel()
    model.mri.subjects_dir = tempdir
    model.mri.subject = 'fsaverage'
    assert_true(model.mri.fid_ok)

    model.hsp.file = raw_path
    lpa_distance = model.lpa_distance
    nasion_distance = model.nasion_distance
    rpa_distance = model.rpa_distance
    avg_point_distance = np.mean(model.point_distance)

    # test hsp point omission
    model.nasion_weight = 1.
    model.trans_y = -0.008
    model.fit_fiducials(0)
    model.omit_hsp_points(0.02)
    assert model.hsp.n_omitted == 1
    model.omit_hsp_points(np.inf)
    assert model.hsp.n_omitted == 0
    model.omit_hsp_points(0.02)
    assert model.hsp.n_omitted == 1
    model.omit_hsp_points(0.01)
    assert model.hsp.n_omitted == 4
    model.omit_hsp_points(0.005)
    assert model.hsp.n_omitted == 40
    model.omit_hsp_points(0.01)
    assert model.hsp.n_omitted == 4
    model.omit_hsp_points(0.02)
    assert model.hsp.n_omitted == 1

    # scale with 1 parameter
    model.n_scale_params = 1
    model.fit_fiducials(1)
    old_x = lpa_distance ** 2 + rpa_distance ** 2 + nasion_distance ** 2
    new_x = (model.lpa_distance ** 2 + model.rpa_distance ** 2 +
             model.nasion_distance ** 2)
    assert_true(new_x < old_x)

    model.fit_icp(1)
    avg_point_distance_1param = np.mean(model.point_distance)
    assert_true(avg_point_distance_1param < avg_point_distance)

    # scaling job
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('scaled', False)
    assert_equal(sdir, tempdir)
    assert_equal(sfrom, 'fsaverage')
    assert_equal(sto, 'scaled')
    assert_allclose(scale, model.parameters[6:9])
    assert_equal(set(bemsol), set(('inner_skull-bem',)))
    model.prepare_bem_model = False
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('scaled', False)
    assert_equal(bemsol, [])

    # scale with 3 parameters
    model.n_scale_params = 3
    model.fit_icp(3)
    assert_true(np.mean(model.point_distance) < avg_point_distance_1param)

    # test switching raw disables point omission
    assert_equal(model.hsp.n_omitted, 1)
    with warnings.catch_warnings(record=True):
        model.hsp.file = kit_raw_path
    assert_equal(model.hsp.n_omitted, 0)


run_tests_if_main()
