from functools import reduce
from glob import glob
import os
import os.path as op
from shutil import copyfile, copytree

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal, assert_array_less)

import mne
from mne.datasets import testing
from mne.transforms import (Transform, apply_trans, rotation, translation,
                            scaling, read_trans, _angle_between_quats,
                            rot_to_quat)
from mne.coreg import (fit_matched_points, create_default_subject, scale_mri,
                       _is_mri_subject, scale_labels, scale_source_space,
                       coregister_fiducials, get_mni_fiducials, Coregistration)
from mne.io import read_fiducials, read_info
from mne.io.constants import FIFF
from mne.utils import (requires_nibabel, modified_env, check_version,
                       catch_logging)
from mne.source_space import write_source_spaces
from mne.channels import DigMontage

data_path = testing.data_path(download=False)
subjects_dir = os.path.join(data_path, 'subjects')
fid_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-fiducials.fif')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')


@pytest.fixture
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


@requires_nibabel()
@pytest.mark.slowtest  # can take forever on OSX Travis
@testing.requires_testing_data
@pytest.mark.parametrize('scale', (.9, [1, .2, .8]))
def test_scale_mri(tmp_path, few_surfaces, scale):
    """Test creating fsaverage and scaling it."""
    # create fsaverage using the testing "fsaverage" instead of the FreeSurfer
    # one
    tempdir = str(tmp_path)
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
    for vox in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]):
        idx = np.ravel_multi_index(vox, vsrc[0]['shape'], order='F')
        err_msg = f'idx={idx} @ {vox}, scale={scale}'
        assert_allclose(apply_trans(vsrc[0]['src_mri_t'], vox),
                        vsrc[0]['rr'][idx], err_msg=err_msg)
        assert_allclose(apply_trans(vsrc_s[0]['src_mri_t'], vox),
                        vsrc_s[0]['rr'][idx], err_msg=err_msg)
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
def test_scale_mri_xfm(tmp_path, few_surfaces):
    """Test scale_mri transforms and MRI scaling."""
    # scale fsaverage
    tempdir = str(tmp_path)
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
    fids, coord_frame = read_fiducials(fid_fname)
    assert coord_frame == FIFF.FIFFV_COORD_MRI
    assert [f['ident'] for f in fids] == list(range(1, 4))
    fids = np.array([f['r'] for f in fids])
    fids_est = get_mni_fiducials('sample', subjects_dir)
    fids_est = np.array([f['r'] for f in fids_est])
    dists = np.linalg.norm(fids - fids_est, axis=-1) * 1000.  # -> mm
    assert (dists < 8).all(), dists


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize(
    'scale_mode,ref_scale,grow_hair,fiducials,fid_match', [
        (None, [1., 1., 1.], 0., None, 'nearest'),
        (None, [1., 1., 1.], 0., 'estimated', 'nearest'),
        (None, [1., 1., 1.], 2., 'auto', 'nearest'),
        ('uniform', [1., 1., 1.], 0., None, 'nearest'),
        ('3-axis', [1., 1., 1.], 0., 'auto', 'nearest'),
        ('uniform', [0.8, 0.8, 0.8], 0., 'auto', 'nearest'),
        ('3-axis', [0.8, 1.2, 1.2], 0., 'auto', 'matched')])
def test_coregistration(scale_mode, ref_scale, grow_hair, fiducials,
                        fid_match):
    """Test automated coregistration."""
    subject = 'sample'
    if fiducials is None:
        fiducials, coord_frame = read_fiducials(fid_fname)
        assert coord_frame == FIFF.FIFFV_COORD_MRI
    info = read_info(raw_fname)
    for d in info['dig']:
        d['r'] = d['r'] * ref_scale
    trans = read_trans(trans_fname)
    coreg = Coregistration(info, subject=subject, subjects_dir=subjects_dir,
                           fiducials=fiducials)
    assert np.allclose(coreg._last_parameters, coreg._parameters)
    coreg.set_fid_match(fid_match)
    default_params = list(coreg._default_parameters)
    coreg.set_rotation(default_params[:3])
    coreg.set_translation(default_params[3:6])
    coreg.set_scale(default_params[6:9])
    coreg.set_grow_hair(grow_hair)
    coreg.set_scale_mode(scale_mode)
    # Identity transform
    errs_id = coreg.compute_dig_mri_distances()
    is_scaled = ref_scale != [1., 1., 1.]
    id_max = 0.03 if is_scaled and scale_mode == '3-axis' else 0.02
    assert 0.005 < np.median(errs_id) < id_max
    # Fiducial transform + scale
    coreg.fit_fiducials(verbose=True)
    assert coreg._extra_points_filter is None
    coreg.omit_head_shape_points(distance=0.02)
    assert coreg._extra_points_filter is not None
    errs_fid = coreg.compute_dig_mri_distances()
    assert_array_less(0, errs_fid)
    if is_scaled or scale_mode is not None:
        fid_max = 0.05
        fid_med = 0.02
    else:
        fid_max = 0.03
        fid_med = 0.01
    assert_array_less(errs_fid, fid_max)
    assert 0.001 < np.median(errs_fid) < fid_med
    assert not np.allclose(coreg._parameters, default_params)
    coreg.omit_head_shape_points(distance=-1)
    coreg.omit_head_shape_points(distance=5. / 1000)
    assert coreg._extra_points_filter is not None
    # ICP transform + scale
    coreg.fit_icp(verbose=True)
    assert isinstance(coreg.trans, Transform)
    errs_icp = coreg.compute_dig_mri_distances()
    assert_array_less(0, errs_icp)
    if is_scaled or scale_mode == '3-axis':
        icp_max = 0.015
    else:
        icp_max = 0.01
    assert_array_less(errs_icp, icp_max)
    assert 0.001 < np.median(errs_icp) < 0.004
    assert np.rad2deg(_angle_between_quats(
        rot_to_quat(coreg.trans['trans'][:3, :3]),
        rot_to_quat(trans['trans'][:3, :3]))) < 13
    if scale_mode is None:
        atol = 1e-7
    else:
        atol = 0.35
    assert_allclose(coreg._scale, ref_scale, atol=atol)
    coreg.reset()
    assert_allclose(coreg._parameters, default_params)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_coreg_class_gui_match():
    """Test that using Coregistration matches mne coreg."""
    fiducials, _ = read_fiducials(fid_fname)
    info = read_info(raw_fname)
    coreg = Coregistration(info, subject='sample', subjects_dir=subjects_dir,
                           fiducials=fiducials)
    assert_allclose(coreg.trans['trans'], np.eye(4), atol=1e-6)
    # mne coreg -s sample -d subjects -f MEG/sample/sample_audvis_trunc_raw.fif
    # then "Fit Fid.", Save... to get trans, read_trans:
    want_trans = [
        [9.99428809e-01, 2.94733196e-02, 1.65350307e-02, -8.76054692e-04],
        [-1.92420650e-02, 8.98512006e-01, -4.38526988e-01, 9.39774036e-04],
        [-2.77817696e-02, 4.37958330e-01, 8.98565888e-01, -8.29207990e-03],
        [0, 0, 0, 1]]
    coreg.set_fid_match('matched')
    coreg.fit_fiducials(verbose=True)
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)
    # Set ICP iterations to one, click "Fit ICP"
    want_trans = [
        [9.99512792e-01, 2.80128177e-02, 1.37659665e-02, 6.08855276e-04],
        [-1.91694051e-02, 8.98992002e-01, -4.37545270e-01, 9.66848747e-04],
        [-2.46323701e-02, 4.37068194e-01, 8.99091005e-01, -1.44129358e-02],
        [0, 0, 0, 1]]
    coreg.fit_icp(1, verbose=True)
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)
    # Set ICP iterations to 20, click "Fit ICP"
    with catch_logging() as log:
        coreg.fit_icp(20, verbose=True)
    log = log.getvalue()
    want_trans = [
        [9.97582495e-01, 2.12266613e-02, 6.61706254e-02, -5.07694029e-04],
        [1.81089472e-02, 8.39900672e-01, -5.42437911e-01, 7.81218382e-03],
        [-6.70908988e-02, 5.42324841e-01, 8.37485850e-01, -2.50057746e-02],
        [0, 0, 0, 1]]
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)
    assert 'ICP 19' in log
    assert 'ICP 20' not in log  # converged on 19
    # Change to uniform scale mode, "Fit Fiducials" in scale UI
    coreg.set_scale_mode('uniform')
    coreg.fit_fiducials()
    want_scale = [0.975] * 3
    want_trans = [
        [9.99428809e-01, 2.94733196e-02, 1.65350307e-02, -9.25998494e-04],
        [-1.92420650e-02, 8.98512006e-01, -4.38526988e-01, -1.03350170e-03],
        [-2.77817696e-02, 4.37958330e-01, 8.98565888e-01, -9.03170835e-03],
        [0, 0, 0, 1]]
    assert_allclose(coreg.scale, want_scale, atol=5e-4)
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)
    # Click "Fit ICP" in scale UI
    with catch_logging() as log:
        coreg.fit_icp(20, verbose=True)
    log = log.getvalue()
    assert 'ICP 18' in log
    assert 'ICP 19' not in log
    want_scale = [1.036] * 3
    want_trans = [
        [9.98992383e-01, 1.72388796e-02, 4.14364934e-02, 6.19427126e-04],
        [6.80460501e-03, 8.54430079e-01, -5.19521892e-01, 5.58008114e-03],
        [-4.43605632e-02, 5.19280374e-01, 8.53451848e-01, -2.03358755e-02],
        [0, 0, 0, 1]]
    assert_allclose(coreg.scale, want_scale, atol=5e-4)
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)
    # Change scale mode to 3-axis, click "Fit ICP" in scale UI
    coreg.set_scale_mode('3-axis')
    with catch_logging() as log:
        coreg.fit_icp(20, verbose=True)
    log = log.getvalue()
    assert 'ICP  7' in log
    assert 'ICP  8' not in log
    want_scale = [1.025, 1.010, 1.121]
    want_trans = [
        [9.98387098e-01, 2.04762165e-02, 5.29526398e-02, 4.97257097e-05],
        [1.13287698e-02, 8.42087150e-01, -5.39222538e-01, 7.09863892e-03],
        [-5.56319728e-02, 5.38952649e-01, 8.40496957e-01, -1.46372067e-02],
        [0, 0, 0, 1]]
    assert_allclose(coreg.scale, want_scale, atol=5e-4)
    assert_allclose(coreg.trans['trans'], want_trans, atol=1e-6)


@testing.requires_testing_data
@pytest.mark.parametrize(
    'drop_point_kind', (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_HPI,
                        FIFF.FIFFV_POINT_EXTRA, FIFF.FIFFV_POINT_EEG))
def test_coreg_class_init(drop_point_kind):
    """Test that Coregistration can be instantiated with various digs."""
    fiducials, _ = read_fiducials(fid_fname)
    info = read_info(raw_fname)

    dig_list = []
    eeg_chans = []
    for pt in info['dig']:
        if pt['kind'] != drop_point_kind:
            dig_list.append(pt)
            if pt['kind'] == FIFF.FIFFV_POINT_EEG:
                eeg_chans.append(f"EEG {pt['ident']:03d}")

    this_info = info.copy()
    this_info.set_montage(DigMontage(dig=dig_list, ch_names=eeg_chans),
                          on_missing='ignore')
    Coregistration(this_info, subject='sample',
                   subjects_dir=subjects_dir, fiducials=fiducials)
