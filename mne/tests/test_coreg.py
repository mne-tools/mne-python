from glob import glob
import os
from shutil import copyfile

from nose.tools import assert_equal, assert_raises, assert_true, assert_is_not
import numpy as np
from numpy.testing import assert_array_almost_equal

import mne
from mne.transforms import (Transform, apply_trans, rotation, translation,
                            scaling)
from mne.coreg import (fit_matched_points, create_default_subject, scale_mri,
                       _is_mri_subject, scale_labels, scale_source_space,
                       coregister_fiducials)
from mne.io.constants import FIFF
from mne.utils import (requires_freesurfer, _TempDir, run_tests_if_main,
                       requires_version)
from mne.source_space import write_source_spaces
from functools import reduce


def test_coregister_fiducials():
    """Test coreg.coregister_fiducials()"""
    # prepare head and MRI fiducials
    trans = Transform('head', 'mri',
                      rotation(.4, .1, 0).dot(translation(.1, -.1, .1)))
    coords_orig = np.array([[-0.08061612, -0.02908875, -0.04131077],
                            [0.00146763, 0.08506715, -0.03483611],
                            [0.08436285, -0.02850276, -0.04127743]])
    coords_trans = apply_trans(trans, coords_orig)

    def make_dig(coords, cf):
        return ({'coord_frame': cf, 'ident': 1, 'kind': 1, 'r': coords[0]},
                {'coord_frame': cf, 'ident': 2, 'kind': 1, 'r': coords[1]},
                {'coord_frame': cf, 'ident': 3, 'kind': 1, 'r': coords[2]})

    mri_fiducials = make_dig(coords_trans, FIFF.FIFFV_COORD_MRI)
    info = {'dig': make_dig(coords_orig, FIFF.FIFFV_COORD_HEAD)}

    # test coregister_fiducials()
    trans_est = coregister_fiducials(info, mri_fiducials)
    assert_equal(trans_est.from_str, trans.from_str)
    assert_equal(trans_est.to_str, trans.to_str)
    assert_array_almost_equal(trans_est['trans'], trans['trans'])


@requires_freesurfer
@requires_version('scipy', '0.11')
def test_scale_mri():
    """Test creating fsaverage and scaling it"""
    # create fsaverage
    tempdir = _TempDir()
    create_default_subject(subjects_dir=tempdir)
    assert_true(_is_mri_subject('fsaverage', tempdir),
                "Creating fsaverage failed")

    fid_path = os.path.join(tempdir, 'fsaverage', 'bem',
                            'fsaverage-fiducials.fif')
    os.remove(fid_path)
    create_default_subject(update=True, subjects_dir=tempdir)
    assert_true(os.path.exists(fid_path), "Updating fsaverage")

    # copy MRI file from sample data
    path = os.path.join('%s', 'fsaverage', 'mri', 'orig.mgz')
    sample_sdir = os.path.join(mne.datasets.sample.data_path(), 'subjects')
    copyfile(path % sample_sdir, path % tempdir)

    # remove redundant label files
    label_temp = os.path.join(tempdir, 'fsaverage', 'label', '*.label')
    label_paths = glob(label_temp)
    for label_path in label_paths[1:]:
        os.remove(label_path)

    # create source space
    path = os.path.join(tempdir, 'fsaverage', 'bem', 'fsaverage-%s-src.fif')
    src = mne.setup_source_space('fsaverage', 'ico0', subjects_dir=tempdir,
                                 add_dist=False)
    write_source_spaces(path % 'ico-0', src)
    mri = os.path.join(tempdir, 'fsaverage', 'mri', 'orig.mgz')
    vsrc = mne.setup_volume_source_space('fsaverage', pos=50, mri=mri,
                                         subjects_dir=tempdir,
                                         add_interpolator=False)
    write_source_spaces(path % 'vol-50', vsrc)

    # scale fsaverage
    os.environ['_MNE_FEW_SURFACES'] = 'true'
    scale = np.array([1, .2, .8])
    scale_mri('fsaverage', 'flachkopf', scale, True, subjects_dir=tempdir)
    del os.environ['_MNE_FEW_SURFACES']
    assert_true(_is_mri_subject('flachkopf', tempdir),
                "Scaling fsaverage failed")
    spath = os.path.join(tempdir, 'flachkopf', 'bem', 'flachkopf-%s-src.fif')

    assert_true(os.path.exists(spath % 'ico-0'),
                "Source space ico-0 was not scaled")
    vsrc_s = mne.read_source_spaces(spath % 'vol-50')
    pt = np.array([0.12, 0.41, -0.22])
    assert_array_almost_equal(apply_trans(vsrc_s[0]['src_mri_t'], pt * scale),
                              apply_trans(vsrc[0]['src_mri_t'], pt))
    scale_labels('flachkopf', subjects_dir=tempdir)

    # add distances to source space
    mne.add_source_space_distances(src)
    src.save(path % 'ico-0', overwrite=True)

    # scale with distances
    os.remove(spath % 'ico-0')
    scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)
    ssrc = mne.read_source_spaces(spath % 'ico-0')
    assert_is_not(ssrc[0]['dist'], None)


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


run_tests_if_main()
