# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os
import os.path as op
import re
import shutil

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest

import mne
from mne.datasets import testing
from mne.io.kit.tests import data_dir as kit_data_dir
from mne.surface import dig_mri_distances
from mne.transforms import invert_transform
from mne.utils import (run_tests_if_main, requires_mayavi, traits_test,
                       modified_env)

data_path = testing.data_path(download=False)
raw_path = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
kit_raw_path = op.join(kit_data_dir, 'test_bin_raw.fif')
subjects_dir = op.join(data_path, 'subjects')


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_model_decimation(subjects_dir_tmp):
    """Test CoregModel decimation of high-res to low-res head."""
    from mne.gui._coreg_gui import CoregModel
    # This makes the test much faster
    subject_dir = op.join(subjects_dir_tmp, 'sample')
    shutil.move(op.join(subject_dir, 'bem', 'outer_skin.surf'),
                op.join(subject_dir, 'surf', 'lh.seghead'))
    for fname in ('sample-head.fif', 'sample-head-dense.fif'):
        os.remove(op.join(subject_dir, 'bem', fname))

    model = CoregModel(guess_mri_subject=False)
    with pytest.warns(RuntimeWarning, match='No low-resolution'):
        model.mri.subjects_dir = op.dirname(subject_dir)
    assert model.mri.subject == 'sample'  # already set by setting subjects_dir
    assert model.mri.bem_low_res.file == ''
    assert len(model.mri.bem_low_res.surf.rr) == 2562
    assert len(model.mri.bem_high_res.surf.rr) == 2562  # because we moved it


@requires_mayavi
@traits_test
def test_coreg_model(subjects_dir_tmp):
    """Test CoregModel."""
    from mne.gui._coreg_gui import CoregModel
    trans_dst = op.join(subjects_dir_tmp, 'test-trans.fif')
    # make it use MNI fiducials
    os.remove(op.join(subjects_dir_tmp, 'sample', 'bem',
                      'sample-fiducials.fif'))

    model = CoregModel()
    with pytest.raises(RuntimeError, match='Not enough information for savin'):
        model.save_trans('blah.fif')

    model.mri.subjects_dir = subjects_dir_tmp
    model.mri.subject = 'sample'

    assert model.mri.fid_ok  # automated using MNI fiducials

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
    assert model.trans_x == 0
    model.set_trans(trans)
    assert_array_almost_equal(model.trans_x, x)
    assert_array_almost_equal(model.trans_y, y)
    assert_array_almost_equal(model.trans_z, z)
    assert_array_almost_equal(model.rot_x, rot_x)
    assert_array_almost_equal(model.rot_y, rot_y)
    assert_array_almost_equal(model.rot_z, rot_z)

    # info
    assert isinstance(model.fid_eval_str, str)
    assert isinstance(model.points_eval_str, str)

    # scaling job
    assert not model.can_prepare_bem_model
    model.n_scale_params = 1
    assert model.can_prepare_bem_model
    model.prepare_bem_model = True
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('sample2', False)
    assert sdir == subjects_dir_tmp
    assert sfrom == 'sample'
    assert sto == 'sample2'
    assert_allclose(scale, model.parameters[6:9])
    assert skip_fiducials is False
    # find BEM files
    bems = set()
    for fname in os.listdir(op.join(subjects_dir, 'sample', 'bem')):
        match = re.match(r'sample-(.+-bem)\.fif', fname)
        if match:
            bems.add(match.group(1))
    assert set(bemsol) == bems
    model.prepare_bem_model = False
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('sample2', True)
    assert bemsol == []
    assert (skip_fiducials)

    model.load_trans(fname_trans)
    model.save_trans(trans_dst)
    trans = mne.read_trans(trans_dst)
    assert_allclose(trans['trans'], model.head_mri_t)
    assert_allclose(invert_transform(trans)['trans'][:3, 3] * 1000.,
                    [model.trans_x, model.trans_y, model.trans_z])


@requires_mayavi
@traits_test
def test_coreg_gui_display(subjects_dir_tmp, check_gui_ci):
    """Test CoregFrame."""
    from mayavi import mlab
    from tvtk.api import tvtk
    home_dir = subjects_dir_tmp
    # Remove the two files that will make the fiducials okay via MNI estimation
    os.remove(op.join(subjects_dir_tmp, 'sample', 'bem',
                      'sample-fiducials.fif'))
    os.remove(op.join(subjects_dir_tmp, 'sample', 'mri', 'transforms',
                      'talairach.xfm'))
    with modified_env(_MNE_GUI_TESTING_MODE='true',
                      _MNE_FAKE_HOME_DIR=home_dir):
        with pytest.raises(ValueError, match='not a valid subject'):
            mne.gui.coregistration(
                subject='Elvis', subjects_dir=subjects_dir_tmp)

        # avoid modal dialog if SUBJECTS_DIR is set to a directory that
        # does not contain valid subjects
        ui, frame = mne.gui.coregistration(subjects_dir='')
        mlab.process_ui_events()
        ui.dispose()
        mlab.process_ui_events()

        ui, frame = mne.gui.coregistration(subjects_dir=subjects_dir_tmp,
                                           subject='sample')
        mlab.process_ui_events()

        assert not frame.model.mri.fid_ok
        frame.model.mri.lpa = [[-0.06, 0, 0]]
        frame.model.mri.nasion = [[0, 0.05, 0]]
        frame.model.mri.rpa = [[0.08, 0, 0]]
        assert frame.model.mri.fid_ok
        frame.data_panel.raw_src.file = raw_path
        assert isinstance(frame.eeg_obj.glyph.glyph.glyph_source.glyph_source,
                          tvtk.SphereSource)
        frame.data_panel.view_options_panel.eeg_obj.project_to_surface = True
        assert isinstance(frame.eeg_obj.glyph.glyph.glyph_source.glyph_source,
                          tvtk.CylinderSource)
        mlab.process_ui_events()

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
        assert frame.model.grow_hair == 0
        assert not frame.data_panel.view_options_panel.head_high_res

        # configuration persistence
        assert (frame.model.prepare_bem_model)
        frame.model.prepare_bem_model = False
        frame.save_config(home_dir)
        ui.dispose()
        mlab.process_ui_events()

        ui, frame = mne.gui.coregistration(subjects_dir=subjects_dir_tmp)
        assert not frame.model.prepare_bem_model
        assert not frame.data_panel.view_options_panel.head_high_res
        ui.dispose()
        mlab.process_ui_events()


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_model_with_fsaverage(tmpdir):
    """Test CoregModel with the fsaverage brain data."""
    tempdir = str(tmpdir)
    from mne.gui._coreg_gui import CoregModel

    mne.create_default_subject(subjects_dir=tempdir,
                               fs_home=op.join(subjects_dir, '..'))

    model = CoregModel()
    model.mri.subjects_dir = tempdir
    model.mri.subject = 'fsaverage'
    assert model.mri.fid_ok

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
    assert (new_x < old_x)

    model.fit_icp(1)
    avg_point_distance_1param = np.mean(model.point_distance)
    assert (avg_point_distance_1param < avg_point_distance)

    # scaling job
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('scaled', False)
    assert sdir == tempdir
    assert sfrom == 'fsaverage'
    assert sto == 'scaled'
    assert_allclose(scale, model.parameters[6:9])
    assert set(bemsol) == {'inner_skull-bem'}
    model.prepare_bem_model = False
    sdir, sfrom, sto, scale, skip_fiducials, labels, annot, bemsol = \
        model.get_scaling_job('scaled', False)
    assert bemsol == []

    # scale with 3 parameters
    model.n_scale_params = 3
    model.fit_icp(3)
    assert (np.mean(model.point_distance) < avg_point_distance_1param)

    # test switching raw disables point omission
    assert model.hsp.n_omitted == 1
    model.hsp.file = kit_raw_path
    assert model.hsp.n_omitted == 0


@testing.requires_testing_data
@requires_mayavi
@traits_test
def test_coreg_gui_automation():
    """Test that properties get properly updated."""
    from mne.gui._file_traits import DigSource
    from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
    from mne.gui._coreg_gui import CoregModel
    subject = 'sample'
    hsp = DigSource()
    hsp.file = raw_path
    mri = MRIHeadWithFiducialsModel(subjects_dir=subjects_dir, subject=subject)
    model = CoregModel(mri=mri, hsp=hsp)
    # gh-7254
    assert not (model.nearest_transformed_high_res_mri_idx_hsp == 0).all()
    model.fit_fiducials()
    model.icp_iterations = 2
    model.nasion_weight = 2.
    model.fit_icp()
    model.omit_hsp_points(distance=5e-3)
    model.icp_iterations = 2
    model.fit_icp()
    errs_icp = np.median(
        model._get_point_distance())
    assert 2e-3 < errs_icp < 3e-3
    info = mne.io.read_info(raw_path)
    errs_nearest = np.median(
        dig_mri_distances(info, fname_trans, subject, subjects_dir))
    assert 1e-3 < errs_nearest < 2e-3


run_tests_if_main()
