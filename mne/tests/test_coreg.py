from functools import reduce
from glob import glob
import os
import os.path as op
from shutil import copyfile, copytree

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal)

import mne
from mne.datasets import testing
from mne.transforms import (Transform, apply_trans, rotation, translation,
                            scaling)
from mne.coreg import (fit_matched_points, create_default_subject, scale_mri,
                       _is_mri_subject, scale_labels, scale_source_space,
                       coregister_fiducials, get_mni_fiducials)
from mne.io import read_fiducials
from mne.io.constants import FIFF
from mne.utils import (run_tests_if_main, requires_nibabel, modified_env,
                       check_version)
from mne.source_space import write_source_spaces

data_path = testing.data_path(download=False)


@pytest.yield_fixture
def few_surfaces():
    """Set the _MNE_FEW_SURFACES env var."""
    with modified_env(_MNE_FEW_SURFACES='true'):
        yield


def test_coregister_fiducials():
    """Test coreg.coregister_fiducials()."""
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
    assert trans_est.from_str == trans.from_str
    assert trans_est.to_str == trans.to_str
    assert_array_almost_equal(trans_est['trans'], trans['trans'])


@pytest.mark.slowtest  # can take forever on OSX Travis
@testing.requires_testing_data
@pytest.mark.parametrize('scale', (.9, [1, .2, .8]))
def test_scale_mri(tmpdir, few_surfaces, scale):
    """Test creating fsaverage and scaling it."""
    # create fsaverage using the testing "fsaverage" instead of the FreeSurfer
    # one
    tempdir = str(tmpdir)
    fake_home = testing.data_path()
    create_default_subject(subjects_dir=tempdir, fs_home=fake_home,
                           verbose=True)
    assert _is_mri_subject('fsaverage', tempdir), "Creating fsaverage failed"

    fid_path = op.join(tempdir, 'fsaverage', 'bem', 'fsaverage-fiducials.fif')
    os.remove(fid_path)
    create_default_subject(update=True, subjects_dir=tempdir,
                           fs_home=fake_home)
    assert op.exists(fid_path), "Updating fsaverage"

    # copy MRI file from sample data (shouldn't matter that it's incorrect,
    # so here choose a small one)
    path_from = op.join(testing.data_path(), 'subjects', 'sample', 'mri',
                        'T1.mgz')
    path_to = op.join(tempdir, 'fsaverage', 'mri', 'orig.mgz')
    copyfile(path_from, path_to)

    # remove redundant label files
    label_temp = op.join(tempdir, 'fsaverage', 'label', '*.label')
    label_paths = glob(label_temp)
    for label_path in label_paths[1:]:
        os.remove(label_path)

    # create source space
    print('Creating surface source space')
    path = op.join(tempdir, 'fsaverage', 'bem', 'fsaverage-%s-src.fif')
    src = mne.setup_source_space('fsaverage', 'ico0', subjects_dir=tempdir,
                                 add_dist=False)
    mri = op.join(tempdir, 'fsaverage', 'mri', 'orig.mgz')
    print('Creating volume source space')
    vsrc = mne.setup_volume_source_space(
        'fsaverage', pos=50, mri=mri, subjects_dir=tempdir,
        add_interpolator=False)
    write_source_spaces(path % 'vol-50', vsrc)

    # scale fsaverage
    write_source_spaces(path % 'ico-0', src, overwrite=True)
    with pytest.warns(None):  # sometimes missing nibabel
        scale_mri('fsaverage', 'flachkopf', scale, True,
                  subjects_dir=tempdir, verbose='debug')
    assert _is_mri_subject('flachkopf', tempdir), "Scaling failed"
    spath = op.join(tempdir, 'flachkopf', 'bem', 'flachkopf-%s-src.fif')

    assert op.exists(spath % 'ico-0'), "Source space ico-0 was not scaled"
    assert os.path.isfile(os.path.join(tempdir, 'flachkopf', 'surf',
                                       'lh.sphere.reg'))
    vsrc_s = mne.read_source_spaces(spath % 'vol-50')
    pt = np.array([0.12, 0.41, -0.22])
    assert_array_almost_equal(
        apply_trans(vsrc_s[0]['src_mri_t'], pt * np.array(scale)),
        apply_trans(vsrc[0]['src_mri_t'], pt))
    scale_labels('flachkopf', subjects_dir=tempdir)

    # add distances to source space after hacking the properties to make
    # it run *much* faster
    src_dist = src.copy()
    for s in src_dist:
        s.update(rr=s['rr'][s['vertno']], nn=s['nn'][s['vertno']],
                 tris=s['use_tris'])
        s.update(np=len(s['rr']), ntri=len(s['tris']),
                 vertno=np.arange(len(s['rr'])),
                 inuse=np.ones(len(s['rr']), int))
    mne.add_source_space_distances(src_dist)
    write_source_spaces(path % 'ico-0', src_dist, overwrite=True)

    # scale with distances
    os.remove(spath % 'ico-0')
    scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)
    ssrc = mne.read_source_spaces(spath % 'ico-0')
    assert ssrc[0]['dist'] is not None
    assert ssrc[0]['nearest'] is not None

    # check patch info computation (only if SciPy is new enough to be fast)
    if check_version('scipy', '1.3'):
        for s in src_dist:
            for key in ('dist', 'dist_limit'):
                s[key] = None
        write_source_spaces(path % 'ico-0', src_dist, overwrite=True)

        # scale with distances
        os.remove(spath % 'ico-0')
        scale_source_space('flachkopf', 'ico-0', subjects_dir=tempdir)
        ssrc = mne.read_source_spaces(spath % 'ico-0')
        assert ssrc[0]['dist'] is None
        assert ssrc[0]['nearest'] is not None


@pytest.mark.slowtest  # can take forever on OSX Travis
@testing.requires_testing_data
@requires_nibabel()
def test_scale_mri_xfm(tmpdir, few_surfaces):
    """Test scale_mri transforms and MRI scaling."""
    # scale fsaverage
    tempdir = str(tmpdir)
    fake_home = testing.data_path()
    # add fsaverage
    create_default_subject(subjects_dir=tempdir, fs_home=fake_home,
                           verbose=True)
    # add sample (with few files)
    sample_dir = op.join(tempdir, 'sample')
    os.mkdir(sample_dir)
    os.mkdir(op.join(sample_dir, 'bem'))
    for dirname in ('mri', 'surf'):
        copytree(op.join(fake_home, 'subjects', 'sample', dirname),
                 op.join(sample_dir, dirname))
    subject_to = 'flachkopf'
    spacing = 'oct2'
    for subject_from in ('fsaverage', 'sample'):
        if subject_from == 'fsaverage':
            scale = 1.  # single dim
        else:
            scale = [0.9, 2, .8]  # separate
        src_from_fname = op.join(tempdir, subject_from, 'bem',
                                 '%s-%s-src.fif' % (subject_from, spacing))
        src_from = mne.setup_source_space(
            subject_from, spacing, subjects_dir=tempdir, add_dist=False)
        write_source_spaces(src_from_fname, src_from)
        vertices_from = np.concatenate([s['vertno'] for s in src_from])
        assert len(vertices_from) == 36
        hemis = ([0] * len(src_from[0]['vertno']) +
                 [1] * len(src_from[0]['vertno']))
        mni_from = mne.vertex_to_mni(vertices_from, hemis, subject_from,
                                     subjects_dir=tempdir)
        if subject_from == 'fsaverage':  # identity transform
            source_rr = np.concatenate([s['rr'][s['vertno']]
                                        for s in src_from]) * 1e3
            assert_allclose(mni_from, source_rr)
        if subject_from == 'fsaverage':
            overwrite = skip_fiducials = False
        else:
            with pytest.raises(IOError, match='No fiducials file'):
                scale_mri(subject_from, subject_to, scale,
                          subjects_dir=tempdir)
            skip_fiducials = True
            with pytest.raises(IOError, match='already exists'):
                scale_mri(subject_from, subject_to, scale,
                          subjects_dir=tempdir, skip_fiducials=skip_fiducials)
            overwrite = True
        if subject_from == 'sample':  # support for not needing all surf files
            os.remove(op.join(sample_dir, 'surf', 'lh.curv'))
        scale_mri(subject_from, subject_to, scale, subjects_dir=tempdir,
                  verbose='debug', overwrite=overwrite,
                  skip_fiducials=skip_fiducials)
        if subject_from == 'fsaverage':
            assert _is_mri_subject(subject_to, tempdir), "Scaling failed"
        src_to_fname = op.join(tempdir, subject_to, 'bem',
                               '%s-%s-src.fif' % (subject_to, spacing))
        assert op.exists(src_to_fname), "Source space was not scaled"
        # Check MRI scaling
        fname_mri = op.join(tempdir, subject_to, 'mri', 'T1.mgz')
        assert op.exists(fname_mri), "MRI was not scaled"
        # Check MNI transform
        src = mne.read_source_spaces(src_to_fname)
        vertices = np.concatenate([s['vertno'] for s in src])
        assert_array_equal(vertices, vertices_from)
        mni = mne.vertex_to_mni(vertices, hemis, subject_to,
                                subjects_dir=tempdir)
        assert_allclose(mni, mni_from, atol=1e-3)  # 0.001 mm


def test_fit_matched_points():
    """Test fit_matched_points: fitting two matching sets of points."""
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
    pytest.raises(RuntimeError, fit_matched_points, tgt_pts, src_pts, tol=10)


@testing.requires_testing_data
@requires_nibabel()
def test_get_mni_fiducials():
    """Test get_mni_fiducials."""
    subjects_dir = op.join(data_path, 'subjects')
    fid_fname = op.join(subjects_dir, 'sample', 'bem',
                        'sample-fiducials.fif')
    fids, coord_frame = read_fiducials(fid_fname)
    assert coord_frame == FIFF.FIFFV_COORD_MRI
    assert [f['ident'] for f in fids] == list(range(1, 4))
    fids = np.array([f['r'] for f in fids])
    fids_est = get_mni_fiducials('sample', subjects_dir)
    fids_est = np.array([f['r'] for f in fids_est])
    dists = np.linalg.norm(fids - fids_est, axis=-1) * 1000
    assert (dists < 8).all(), dists


run_tests_if_main()
