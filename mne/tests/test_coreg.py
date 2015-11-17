from glob import glob
import os

from nose.tools import assert_raises, assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less

import mne
from mne.transforms import apply_trans, rotation, translation, scaling
from mne.coreg import (fit_matched_points, fit_point_cloud,
                       _point_cloud_error, _decimate_points,
                       create_default_subject, scale_mri,
                       _is_mri_subject, scale_labels, scale_source_space)
from mne.utils import (requires_mne, requires_freesurfer, _TempDir,
                       run_tests_if_main, requires_version)
from functools import reduce


@requires_mne
@requires_freesurfer
@requires_version('scipy', '0.11')
def test_scale_mri():
    """Test creating fsaverage and scaling it"""
    # create fsaverage
    tempdir = _TempDir()
    create_default_subject(subjects_dir=tempdir)
    is_mri = _is_mri_subject('fsaverage', tempdir)
    assert_true(is_mri, "Creating fsaverage failed")

    fid_path = os.path.join(tempdir, 'fsaverage', 'bem',
                            'fsaverage-fiducials.fif')
    os.remove(fid_path)
    create_default_subject(update=True, subjects_dir=tempdir)
    assert_true(os.path.exists(fid_path), "Updating fsaverage")

    # remove redundant label files
    label_temp = os.path.join(tempdir, 'fsaverage', 'label', '*.label')
    label_paths = glob(label_temp)
    for label_path in label_paths[1:]:
        os.remove(label_path)

    # create source space
    path = os.path.join(tempdir, 'fsaverage', 'bem', 'fsaverage-ico-0-src.fif')
    mne.setup_source_space('fsaverage', path, 'ico0', overwrite=True,
                           subjects_dir=tempdir, add_dist=False)

    # scale fsaverage
    os.environ['_MNE_FEW_SURFACES'] = 'true'
    scale_mri('fsaverage', 'flachkopf', [1, .2, .8], True,
              subjects_dir=tempdir)
    del os.environ['_MNE_FEW_SURFACES']
    is_mri = _is_mri_subject('flachkopf', tempdir)
    assert_true(is_mri, "Scaling fsaverage failed")
    src_path = os.path.join(tempdir, 'flachkopf', 'bem',
                            'flachkopf-ico-0-src.fif')
    assert_true(os.path.exists(src_path), "Source space was not scaled")
    scale_labels('flachkopf', subjects_dir=tempdir)

    # scale source space separately
    os.remove(src_path)
    scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)
    assert_true(os.path.exists(src_path), "Source space was not scaled")

    # add distances to source space
    src = mne.read_source_spaces(path)
    mne.add_source_space_distances(src)
    src.save(path)

    # scale with distances
    os.remove(src_path)
    scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)


def test_fit_matched_points():
    """Test fit_matched_points: fitting two matching sets of points"""
    tgt_pts = np.random.RandomState(42).uniform(size=(6, 3))

    # rotation only
    trans = rotation(2, 6, 3)
    src_pts = apply_trans(trans, tgt_pts)
    trans_est = fit_matched_points(src_pts, tgt_pts, translate=False,
                                   out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              "rotation")

    # rotation & scaling
    trans = np.dot(rotation(2, 6, 3), scaling(.5, .5, .5))
    src_pts = apply_trans(trans, tgt_pts)
    trans_est = fit_matched_points(src_pts, tgt_pts, translate=False, scale=1,
                                   out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              "rotation and scaling.")

    # rotation & translation
    trans = np.dot(translation(2, -6, 3), rotation(2, 6, 3))
    src_pts = apply_trans(trans, tgt_pts)
    trans_est = fit_matched_points(src_pts, tgt_pts, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              "rotation and translation.")

    # rotation & translation & scaling
    trans = reduce(np.dot, (translation(2, -6, 3), rotation(1.5, .3, 1.4),
                            scaling(.5, .5, .5)))
    src_pts = apply_trans(trans, tgt_pts)
    trans_est = fit_matched_points(src_pts, tgt_pts, scale=1, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    assert_array_almost_equal(tgt_pts, est_pts, 2, "fit_matched_points with "
                              "rotation, translation and scaling.")

    # test exceeding tolerance
    tgt_pts[0, :] += 20
    assert_raises(RuntimeError, fit_matched_points, tgt_pts, src_pts, tol=10)


def test_fit_point_cloud():
    """Test fit_point_cloud: fitting a set of points to a point cloud"""
    # evenly spaced target points on a sphere
    u = np.linspace(0, np.pi, 150)
    v = np.linspace(0, np.pi, 150)

    x = np.outer(np.cos(u), np.sin(v)).reshape((-1, 1))
    y = np.outer(np.sin(u), np.sin(v)).reshape((-1, 1))
    z = np.outer(np.ones(np.size(u)), np.cos(v)).reshape((-1, 1)) * 3

    tgt_pts = np.hstack((x, y, z))
    tgt_pts = _decimate_points(tgt_pts, .05)

    # pick some points to fit
    some_tgt_pts = tgt_pts[::362]

    # rotation only
    trans = rotation(1.5, .3, -0.4)
    src_pts = apply_trans(trans, some_tgt_pts)
    trans_est = fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=False,
                                scale=0, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    err = _point_cloud_error(est_pts, tgt_pts)
    assert_array_less(err, .1, "fit_point_cloud with rotation.")

    # rotation and translation
    trans = np.dot(rotation(0.5, .3, -0.4), translation(.3, .2, -.2))
    src_pts = apply_trans(trans, some_tgt_pts)
    trans_est = fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=True,
                                scale=0, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    err = _point_cloud_error(est_pts, tgt_pts)
    assert_array_less(err, .1, "fit_point_cloud with rotation and "
                      "translation.")

    # rotation and 1 scale parameter
    trans = np.dot(rotation(0.5, .3, -0.4), scaling(1.5, 1.5, 1.5))
    src_pts = apply_trans(trans, some_tgt_pts)
    trans_est = fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=False,
                                scale=1, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    err = _point_cloud_error(est_pts, tgt_pts)
    assert_array_less(err, .1, "fit_point_cloud with rotation and 1 scaling "
                      "parameter.")

    # rotation and 3 scale parameter
    trans = np.dot(rotation(0.5, .3, -0.4), scaling(1.5, 1.7, 1.1))
    src_pts = apply_trans(trans, some_tgt_pts)
    trans_est = fit_point_cloud(src_pts, tgt_pts, rotate=True, translate=False,
                                scale=3, out='trans')
    est_pts = apply_trans(trans_est, src_pts)
    err = _point_cloud_error(est_pts, tgt_pts)
    assert_array_less(err, .1, "fit_point_cloud with rotation and 3 scaling "
                      "parameters.")


run_tests_if_main()
