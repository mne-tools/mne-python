# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
from shutil import copytree

import pytest
import scipy
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose, assert_equal,
                           assert_array_less)
from mne.datasets import testing
import mne
from mne import (read_source_spaces, write_source_spaces,
                 setup_source_space, setup_volume_source_space,
                 add_source_space_distances, read_bem_surfaces,
                 morph_source_spaces, SourceEstimate, make_sphere_model,
                 compute_source_morph, pick_types,
                 read_bem_solution, read_freesurfer_lut,
                 read_trans)
from mne.fixes import _get_img_fdata
from mne.utils import (requires_nibabel, run_subprocess,
                       modified_env, requires_mne, check_version)
from mne.surface import _accumulate_normals, _triangle_neighbors
from mne.source_estimate import _get_src_type
from mne.source_space import (get_volume_labels_from_src,
                              _compare_source_spaces,
                              compute_distance_to_sensors)
from mne.io.pick import _picks_to_idx
from mne.io.constants import FIFF

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
aseg_fname = op.join(data_path, 'subjects', 'sample', 'mri', 'aseg.mgz')
fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
fname_vol = op.join(subjects_dir, 'sample', 'bem',
                    'sample-volume-7mm-src.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-bem.fif')
fname_bem_sol = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-1280-bem-sol.fif')
fname_bem_3 = op.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-1280-1280-1280-bem.fif')
fname_bem_3_sol = op.join(data_path, 'subjects', 'sample', 'bem',
                          'sample-1280-1280-1280-bem-sol.fif')
fname_fs = op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')
fname_morph = op.join(subjects_dir, 'sample', 'bem',
                      'sample-fsaverage-ico-5-src.fif')
fname_src = op.join(
    data_path, 'subjects', 'sample', 'bem', 'sample-oct-4-src.fif')
fname_fwd = op.join(
    data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_small = op.join(base_dir, 'small-src.fif.gz')
fname_ave = op.join(base_dir, 'test-ave.fif')
rng = np.random.RandomState(0)


@testing.requires_testing_data
@pytest.mark.parametrize('picks, limits', [
    ('meg', (0.02, 0.250)),
    (None, (0.01, 0.250)),  # should be same as EEG
    ('eeg', (0.01, 0.250)),
])
def test_compute_distance_to_sensors(picks, limits):
    """Test computation of distances between vertices and sensors."""
    src = read_source_spaces(fname_src)
    fwd = mne.read_forward_solution(fname_fwd)
    info = fwd['info']
    trans = read_trans(trans_fname)
    # trans = fwd['info']['mri_head_t']
    if isinstance(picks, str):
        kwargs = dict()
        kwargs[picks] = True
        if picks == 'eeg':
            info['dev_head_t'] = None  # should not break anything
        use_picks = pick_types(info, **kwargs, exclude=())
    else:
        use_picks = picks
    n_picks = len(_picks_to_idx(info, use_picks, 'data', exclude=()))

    # Make sure same vertices are used in src and fwd
    src[0]['inuse'] = fwd['src'][0]['inuse']
    src[1]['inuse'] = fwd['src'][1]['inuse']
    src[0]['nuse'] = fwd['src'][0]['nuse']
    src[1]['nuse'] = fwd['src'][1]['nuse']

    n_verts = src[0]['nuse'] + src[1]['nuse']

    # minimum distances between vertices and sensors
    depths = compute_distance_to_sensors(src, info=info, picks=use_picks,
                                         trans=trans)
    assert depths.shape == (n_verts, n_picks)
    assert limits[0] * 5 > depths.min()  # meaningful choice of limits
    assert_array_less(limits[0], depths)
    assert_array_less(depths, limits[1])

    # If source space from Forward Solution and trans=None (i.e. identity) then
    # depths2 should be the same as depth.
    depths2 = compute_distance_to_sensors(src=fwd['src'], info=info,
                                          picks=use_picks, trans=None)
    assert_allclose(depths, depths2, rtol=1e-5)

    if picks != 'eeg':
        # this should break things
        info['dev_head_t'] = None
        with pytest.raises(ValueError,
                           match='Transform between meg<->head'):
            compute_distance_to_sensors(src, info, use_picks, trans)


def _read_small_src(remove=True):
    src = read_source_spaces(fname_small)
    if remove:
        for s in src:
            s['nearest'] = None
            s['nearest_dist'] = None
            s['pinfo'] = None
    return src


def test_add_patch_info(monkeypatch):
    """Test adding patch info to source space."""
    # let's setup a small source space
    src = _read_small_src(remove=False)
    src_new = _read_small_src()

    # test that no patch info is added for small dist_limit
    add_source_space_distances(src_new, dist_limit=0.00001)
    assert all(s['nearest'] is None for s in src_new)
    assert all(s['nearest_dist'] is None for s in src_new)
    assert all(s['pinfo'] is None for s in src_new)

    # now let's use one that works (and test our warning-throwing)
    with monkeypatch.context() as m:
        m.setattr(mne.source_space, '_DIST_WARN_LIMIT', 1)
        with pytest.warns(RuntimeWarning, match='Computing distances for 258'):
            add_source_space_distances(src_new)
    _compare_source_spaces(src, src_new, 'approx')

    # Old SciPy can't do patch info only
    src_new = _read_small_src()
    with monkeypatch.context() as m:
        m.setattr(scipy, '__version__', '1.0')
        with pytest.raises(RuntimeError, match='required to calculate patch '):
            add_source_space_distances(src_new, dist_limit=0)

    # New SciPy can
    if check_version('scipy', '1.3'):
        src_nodist = src.copy()
        for s in src_nodist:
            for key in ('dist', 'dist_limit'):
                s[key] = None
        add_source_space_distances(src_new, dist_limit=0)
        _compare_source_spaces(src, src_new, 'approx')


@testing.requires_testing_data
def test_add_source_space_distances_limited(tmp_path):
    """Test adding distances to source space with a dist_limit."""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 200  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = tmp_path / 'temp-src.fif'
    add_source_space_distances(src_new, dist_limit=0.007)
    write_source_spaces(out_name, src_new)
    src_new = read_source_spaces(out_name)

    for so, sn in zip(src, src_new):
        assert_array_equal(so['dist_limit'], np.array([-0.007], np.float32))
        assert_array_equal(sn['dist_limit'], np.array([0.007], np.float32))
        do = so['dist']
        dn = sn['dist']

        # clean out distances > 0.007 in C code
        do.data[do.data > 0.007] = 0
        do.eliminate_zeros()

        # make sure we have some comparable distances
        assert np.sum(do.data < 0.007) > 400

        # do comparison over the region computed
        d = (do - dn)[:sn['vertno'][n_do - 1]][:, :sn['vertno'][n_do - 1]]
        assert_allclose(np.zeros_like(d.data), d.data, rtol=0, atol=1e-6)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_add_source_space_distances(tmp_path):
    """Test adding distances to source space."""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 19  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = tmp_path / 'temp-src.fif'
    n_jobs = 2
    assert n_do % n_jobs != 0
    with pytest.raises(ValueError, match='non-negative'):
        add_source_space_distances(src_new, dist_limit=-1)
    add_source_space_distances(src_new, n_jobs=n_jobs)
    write_source_spaces(out_name, src_new)
    src_new = read_source_spaces(out_name)

    # iterate over both hemispheres
    for so, sn in zip(src, src_new):
        v = so['vertno'][:n_do]
        assert_array_equal(so['dist_limit'], np.array([-0.007], np.float32))
        assert_array_equal(sn['dist_limit'], np.array([np.inf], np.float32))
        do = so['dist']
        dn = sn['dist']

        # clean out distances > 0.007 in C code (some residual), and Python
        ds = list()
        for d in [do, dn]:
            d.data[d.data > 0.007] = 0
            d = d[v][:, v]
            d.eliminate_zeros()
            ds.append(d)

        # make sure we actually calculated some comparable distances
        assert np.sum(ds[0].data < 0.007) > 10

        # do comparison
        d = ds[0] - ds[1]
        assert_allclose(np.zeros_like(d.data), d.data, rtol=0, atol=1e-9)


@testing.requires_testing_data
@requires_mne
def test_discrete_source_space(tmp_path):
    """Test setting up (and reading/writing) discrete source spaces."""
    src = read_source_spaces(fname)
    v = src[0]['vertno']

    # let's make a discrete version with the C code, and with ours
    temp_name = tmp_path / 'temp-src.fif'
    # save
    temp_pos = tmp_path / 'temp-pos.txt'
    np.savetxt(str(temp_pos), np.c_[src[0]['rr'][v], src[0]['nn'][v]])
    # let's try the spherical one (no bem or surf supplied)
    run_subprocess(['mne_volume_source_space', '--meters',
                    '--pos', temp_pos, '--src', temp_name])
    src_c = read_source_spaces(temp_name)
    pos_dict = dict(rr=src[0]['rr'][v], nn=src[0]['nn'][v])
    src_new = setup_volume_source_space(pos=pos_dict)
    assert src_new.kind == 'discrete'
    _compare_source_spaces(src_c, src_new, mode='approx')
    assert_allclose(src[0]['rr'][v], src_new[0]['rr'],
                    rtol=1e-3, atol=1e-6)
    assert_allclose(src[0]['nn'][v], src_new[0]['nn'],
                    rtol=1e-3, atol=1e-6)

    # now do writing
    write_source_spaces(temp_name, src_c, overwrite=True)
    src_c2 = read_source_spaces(temp_name)
    _compare_source_spaces(src_c, src_c2)

    # now do MRI
    with pytest.raises(ValueError, match='Cannot create interpolation'):
        setup_volume_source_space('sample', pos=pos_dict, mri=fname_mri)
    assert repr(src_new).split('~')[0] == repr(src_c).split('~')[0]
    assert ' kB' in repr(src_new)
    assert src_new.kind == 'discrete'
    assert _get_src_type(src_new, None) == 'discrete'

    with pytest.raises(RuntimeError, match='finite'):
        setup_volume_source_space(
            pos=dict(rr=[[0, 0, float('inf')]], nn=[[0, 1, 0]]))


@requires_nibabel()
@pytest.mark.slowtest
@testing.requires_testing_data
def test_volume_source_space(tmp_path):
    """Test setting up volume source spaces."""
    src = read_source_spaces(fname_vol)
    temp_name = tmp_path / 'temp-src.fif'
    surf = read_bem_surfaces(fname_bem, s_id=FIFF.FIFFV_BEM_SURF_ID_BRAIN)
    surf['rr'] *= 1e3  # convert to mm
    bem_sol = read_bem_solution(fname_bem_3_sol)
    bem = read_bem_solution(fname_bem_sol)
    # The one in the testing dataset (uses bem as bounds)
    for this_bem, this_surf in zip(
            (bem, fname_bem, fname_bem_3, bem_sol, fname_bem_3_sol, None),
            (None, None, None, None, None, surf)):
        src_new = setup_volume_source_space(
            'sample', pos=7.0, bem=this_bem, surface=this_surf,
            subjects_dir=subjects_dir)
        write_source_spaces(temp_name, src_new, overwrite=True)
        src[0]['subject_his_id'] = 'sample'  # XXX: to make comparison pass
        _compare_source_spaces(src, src_new, mode='approx')
        del src_new
        src_new = read_source_spaces(temp_name)
        _compare_source_spaces(src, src_new, mode='approx')
    with pytest.raises(IOError, match='surface file.*not found'):
        setup_volume_source_space(
            'sample', surface='foo', mri=fname_mri, subjects_dir=subjects_dir)
    bem['surfs'][-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    with pytest.raises(ValueError, match='BEM is not in MRI coord.* got head'):
        setup_volume_source_space(
            'sample', bem=bem, mri=fname_mri, subjects_dir=subjects_dir)
    bem['surfs'] = bem['surfs'][:-1]  # no inner skull surf
    with pytest.raises(ValueError, match='Could not get inner skul.*from BEM'):
        setup_volume_source_space(
            'sample', bem=bem, mri=fname_mri, subjects_dir=subjects_dir)
    del bem
    assert repr(src) == repr(src_new)
    assert ' MB' in repr(src)
    assert src.kind == 'volume'
    # Spheres
    sphere = make_sphere_model(r0=(0., 0., 0.), head_radius=0.1,
                               relative_radii=(0.9, 1.0), sigmas=(0.33, 1.0))
    src = setup_volume_source_space(pos=10, sphere=(0., 0., 0., 0.09))
    src_new = setup_volume_source_space(pos=10, sphere=sphere)
    _compare_source_spaces(src, src_new, mode='exact')
    with pytest.raises(ValueError, match='sphere, if str'):
        setup_volume_source_space(sphere='foo')
    # Need a radius
    sphere = make_sphere_model(head_radius=None)
    with pytest.raises(ValueError, match='be spherical with multiple layers'):
        setup_volume_source_space(sphere=sphere)


@testing.requires_testing_data
@requires_mne
def test_other_volume_source_spaces(tmp_path):
    """Test setting up other volume source spaces."""
    # these are split off because they require the MNE tools, and
    # Travis doesn't seem to like them

    # let's try the spherical one (no bem or surf supplied)
    temp_name = tmp_path / 'temp-src.fif'
    run_subprocess(['mne_volume_source_space',
                    '--grid', '7.0',
                    '--src', temp_name,
                    '--mri', fname_mri])
    src = read_source_spaces(temp_name)
    sphere = (0., 0., 0., 0.09)
    src_new = setup_volume_source_space(None, pos=7.0, mri=fname_mri,
                                        subjects_dir=subjects_dir,
                                        sphere=sphere)
    # we use a more accurate elimination criteria, so let's fix the MNE-C
    # source space
    assert len(src_new[0]['vertno']) == 7497
    assert len(src) == 1
    assert len(src_new) == 1
    good_mask = np.in1d(src[0]['vertno'], src_new[0]['vertno'])
    src[0]['inuse'][src[0]['vertno'][~good_mask]] = 0
    assert src[0]['inuse'].sum() == 7497
    src[0]['vertno'] = src[0]['vertno'][good_mask]
    assert len(src[0]['vertno']) == 7497
    src[0]['nuse'] = len(src[0]['vertno'])
    assert src[0]['nuse'] == 7497
    _compare_source_spaces(src_new, src, mode='approx')
    assert 'volume, shape' in repr(src)
    del src
    del src_new
    pytest.raises(ValueError, setup_volume_source_space, 'sample', pos=7.0,
                  sphere=[1., 1.], mri=fname_mri,  # bad sphere
                  subjects_dir=subjects_dir)

    # now without MRI argument, it should give an error when we try
    # to read it
    run_subprocess(['mne_volume_source_space',
                    '--grid', '7.0',
                    '--src', temp_name])
    pytest.raises(ValueError, read_source_spaces, temp_name)


@pytest.mark.timeout(60)  # can be slow on OSX Travis
@pytest.mark.slowtest
@testing.requires_testing_data
def test_triangle_neighbors():
    """Test efficient vertex neighboring triangles for surfaces."""
    this = read_source_spaces(fname)[0]
    this['neighbor_tri'] = [list() for _ in range(this['np'])]
    for p in range(this['ntri']):
        verts = this['tris'][p]
        this['neighbor_tri'][verts[0]].append(p)
        this['neighbor_tri'][verts[1]].append(p)
        this['neighbor_tri'][verts[2]].append(p)
    this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]

    neighbor_tri = _triangle_neighbors(this['tris'], this['np'])
    assert all(np.array_equal(nt1, nt2)
               for nt1, nt2 in zip(neighbor_tri, this['neighbor_tri']))


def test_accumulate_normals():
    """Test efficient normal accumulation for surfaces."""
    # set up comparison
    n_pts = int(1.6e5)  # approx number in sample source space
    n_tris = int(3.2e5)
    # use all positive to make a worst-case for cumulative summation
    # (real "nn" vectors will have both positive and negative values)
    tris = (rng.rand(n_tris, 1) * (n_pts - 2)).astype(int)
    tris = np.c_[tris, tris + 1, tris + 2]
    tri_nn = rng.rand(n_tris, 3)
    this = dict(tris=tris, np=n_pts, ntri=n_tris, tri_nn=tri_nn)

    # cut-and-paste from original code in surface.py:
    #    Find neighboring triangles and accumulate vertex normals
    this['nn'] = np.zeros((this['np'], 3))
    for p in range(this['ntri']):
        # vertex normals
        verts = this['tris'][p]
        this['nn'][verts, :] += this['tri_nn'][p, :]
    nn = _accumulate_normals(this['tris'], this['tri_nn'], this['np'])

    # the moment of truth (or reckoning)
    assert_allclose(nn, this['nn'], rtol=1e-7, atol=1e-7)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_setup_source_space(tmp_path):
    """Test setting up ico, oct, and all source spaces."""
    fname_ico = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')
    # first lets test some input params
    for spacing in ('oct', 'oct6e'):
        with pytest.raises(ValueError, match='subdivision must be an integer'):
            setup_source_space('sample', spacing=spacing,
                               add_dist=False, subjects_dir=subjects_dir)
    for spacing in ('oct0', 'oct-4'):
        with pytest.raises(ValueError, match='oct subdivision must be >= 1'):
            setup_source_space('sample', spacing=spacing,
                               add_dist=False, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='ico subdivision must be >= 0'):
        setup_source_space('sample', spacing='ico-4',
                           add_dist=False, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='must be a string with values'):
        setup_source_space('sample', spacing='7emm',
                           add_dist=False, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='must be a string with values'):
        setup_source_space('sample', spacing='alls',
                           add_dist=False, subjects_dir=subjects_dir)

    # ico 5 (fsaverage) - write to temp file
    src = read_source_spaces(fname_ico)
    with pytest.warns(None):  # sklearn equiv neighbors
        src_new = setup_source_space('fsaverage', spacing='ico5',
                                     subjects_dir=subjects_dir, add_dist=False)
    _compare_source_spaces(src, src_new, mode='approx')
    assert repr(src).split('~')[0] == repr(src_new).split('~')[0]
    assert repr(src).count('surface (') == 2
    assert_array_equal(src[0]['vertno'], np.arange(10242))
    assert_array_equal(src[1]['vertno'], np.arange(10242))

    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = tmp_path / 'temp-src.fif'
    with pytest.warns(None):  # sklearn equiv neighbors
        src_new = setup_source_space('sample', spacing='oct6',
                                     subjects_dir=subjects_dir, add_dist=False)
        write_source_spaces(temp_name, src_new, overwrite=True)
    assert_equal(src_new[0]['nuse'], 4098)
    _compare_source_spaces(src, src_new, mode='approx', nearest=False)
    src_new = read_source_spaces(temp_name)
    _compare_source_spaces(src, src_new, mode='approx', nearest=False)

    # all source points - no file writing
    src_new = setup_source_space('sample', spacing='all',
                                 subjects_dir=subjects_dir, add_dist=False)
    assert src_new[0]['nuse'] == len(src_new[0]['rr'])
    assert src_new[1]['nuse'] == len(src_new[1]['rr'])

    # dense source space to hit surf['inuse'] lines of _create_surf_spacing
    pytest.raises(RuntimeError, setup_source_space, 'sample',
                  spacing='ico6', subjects_dir=subjects_dir, add_dist=False)


@testing.requires_testing_data
@requires_mne
@pytest.mark.slowtest
@pytest.mark.timeout(60)
@pytest.mark.parametrize('spacing', [2, 7])
def test_setup_source_space_spacing(tmp_path, spacing):
    """Test setting up surface source spaces using a given spacing."""
    copytree(op.join(subjects_dir, 'sample'), tmp_path / 'sample')
    args = [] if spacing == 7 else ['--spacing', str(spacing)]
    with modified_env(SUBJECTS_DIR=str(tmp_path), SUBJECT='sample'):
        run_subprocess(['mne_setup_source_space'] + args)
    src = read_source_spaces(
        tmp_path / 'sample' / 'bem' / ('sample-%d-src.fif' % spacing)
    )
    src_new = setup_source_space('sample', spacing=spacing, add_dist=False,
                                 subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx', nearest=True)
    # Degenerate conditions
    with pytest.raises(TypeError, match='spacing must be.*got.*float.*'):
        setup_source_space('sample', 7., subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='spacing must be >= 2, got 1'):
        setup_source_space('sample', 1, subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_read_source_spaces():
    """Test reading of source space meshes."""
    src = read_source_spaces(fname, patch_stats=True)

    # 3D source space
    lh_points = src[0]['rr']
    lh_faces = src[0]['tris']
    lh_use_faces = src[0]['use_tris']
    rh_points = src[1]['rr']
    rh_faces = src[1]['tris']
    rh_use_faces = src[1]['use_tris']
    assert lh_faces.min() == 0
    assert lh_faces.max() == lh_points.shape[0] - 1
    assert lh_use_faces.min() >= 0
    assert lh_use_faces.max() <= lh_points.shape[0] - 1
    assert rh_faces.min() == 0
    assert rh_faces.max() == rh_points.shape[0] - 1
    assert rh_use_faces.min() >= 0
    assert rh_use_faces.max() <= rh_points.shape[0] - 1


@pytest.mark.slowtest
@testing.requires_testing_data
def test_write_source_space(tmp_path):
    """Test reading and writing of source spaces."""
    src0 = read_source_spaces(fname, patch_stats=False)
    temp_fname = tmp_path / 'tmp-src.fif'
    write_source_spaces(temp_fname, src0)
    src1 = read_source_spaces(temp_fname, patch_stats=False)
    _compare_source_spaces(src0, src1)

    # test warnings on bad filenames
    src_badname = tmp_path / 'test-bad-name.fif.gz'
    with pytest.warns(RuntimeWarning, match='-src.fif'):
        write_source_spaces(src_badname, src0)
    with pytest.warns(RuntimeWarning, match='-src.fif'):
        read_source_spaces(src_badname)


@testing.requires_testing_data
@requires_nibabel()
@pytest.mark.parametrize('pass_ids', (True, False))
def test_source_space_from_label(tmp_path, pass_ids):
    """Test generating a source space from volume label."""
    aseg_short = 'aseg.mgz'
    atlas_ids, _ = read_freesurfer_lut()
    volume_label = 'Left-Cerebellum-Cortex'

    # Test pos as dict
    pos = dict()
    with pytest.raises(ValueError, match='mri must be None if pos is a dict'):
        setup_volume_source_space(
            'sample', pos=pos, volume_label=volume_label, mri=aseg_short,
            subjects_dir=subjects_dir)

    # Test T1.mgz provided
    with pytest.raises(RuntimeError, match=r'Must use a \*aseg.mgz file'):
        setup_volume_source_space(
            'sample', mri='T1.mgz', volume_label=volume_label,
            subjects_dir=subjects_dir)

    # Test invalid volume label
    mri = aseg_short
    with pytest.raises(ValueError, match="'Left-Cerebral' not found.*Did you"):
        setup_volume_source_space(
            'sample', volume_label='Left-Cerebral', mri=mri,
            subjects_dir=subjects_dir)

    # These should be equivalent
    if pass_ids:
        use_volume_label = {volume_label: atlas_ids[volume_label]}
    else:
        use_volume_label = volume_label

    # ensure it works even when not provided (detect that it should be aseg)
    src = setup_volume_source_space(
        'sample', volume_label=use_volume_label, add_interpolator=False,
        subjects_dir=subjects_dir)
    assert_equal(volume_label, src[0]['seg_name'])
    assert src[0]['nuse'] == 404  # for our given pos and label

    # test reading and writing
    out_name = tmp_path / 'temp-src.fif'
    write_source_spaces(out_name, src)
    src_from_file = read_source_spaces(out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')


@testing.requires_testing_data
@requires_nibabel()
def test_source_space_exclusive_complete(src_volume_labels):
    """Test that we produce exclusive and complete labels."""
    # these two are neighbors and are quite large, so let's use them to
    # ensure no overlaps
    src, volume_labels, _ = src_volume_labels
    ii = volume_labels.index('Left-Cerebral-White-Matter')
    jj = volume_labels.index('Left-Cerebral-Cortex')
    assert src[ii]['nuse'] == 755  # 2034 with pos=5, was 2832
    assert src[jj]['nuse'] == 616  # 1520 with pos=5, was 2623
    src_full = read_source_spaces(fname_vol)
    # This implicitly checks for overlap because np.sort would preserve
    # duplicates, and it checks for completeness because the sets should match
    assert_array_equal(src_full[0]['vertno'],
                       np.sort(np.concatenate([s['vertno'] for s in src])))
    for si, s in enumerate(src):
        assert_allclose(src_full[0]['rr'], s['rr'], atol=1e-6)
    # also check single_volume=True -- should be the same result
    with pytest.warns(RuntimeWarning, match='Found no usable.*Left-vessel.*'):
        src_single = setup_volume_source_space(
            src[0]['subject_his_id'], 7., 'aseg.mgz', bem=fname_bem,
            volume_label=volume_labels, single_volume=True,
            add_interpolator=False, subjects_dir=subjects_dir)
    assert len(src_single) == 1
    assert 'Unknown+Left-Cerebral-White-Matter+Left-' in repr(src_single)
    assert_array_equal(src_full[0]['vertno'], src_single[0]['vertno'])


@pytest.mark.timeout(60)  # ~24 sec on Travis
@pytest.mark.slowtest
@testing.requires_testing_data
@requires_nibabel()
def test_read_volume_from_src():
    """Test reading volumes from a mixed source space."""
    labels_vol = ['Left-Amygdala',
                  'Brain-Stem',
                  'Right-Amygdala']

    src = read_source_spaces(fname)

    # Setup a volume source space
    vol_src = setup_volume_source_space('sample', mri=aseg_fname,
                                        pos=5.0,
                                        bem=fname_bem,
                                        volume_label=labels_vol,
                                        subjects_dir=subjects_dir)
    # Generate the mixed source space, testing some list methods
    assert src.kind == 'surface'
    assert vol_src.kind == 'volume'
    src += vol_src
    assert src.kind == 'mixed'
    assert vol_src.kind == 'volume'
    assert src[:2].kind == 'surface'
    assert src[2:].kind == 'volume'
    assert src[:].kind == 'mixed'
    with pytest.raises(RuntimeError, match='Invalid source space'):
        src[::2]

    volume_src = get_volume_labels_from_src(src, 'sample', subjects_dir)
    volume_label = volume_src[0].name
    volume_label = 'Left-' + volume_label.replace('-lh', '')

    # Test
    assert_equal(volume_label, src[2]['seg_name'])

    assert_equal(src[2]['type'], 'vol')


@testing.requires_testing_data
@requires_nibabel()
def test_combine_source_spaces(tmp_path):
    """Test combining source spaces."""
    import nibabel as nib
    rng = np.random.RandomState(2)
    volume_labels = ['Brain-Stem', 'Right-Hippocampus']  # two fairly large

    # create a sparse surface source space to ensure all get mapped
    # when mri_resolution=False
    srf = setup_source_space('sample', 'oct3', add_dist=False,
                             subjects_dir=subjects_dir)

    # setup 2 volume source spaces
    vol = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                    volume_label=volume_labels[0],
                                    mri=aseg_fname, add_interpolator=False)

    # setup a discrete source space
    rr = rng.randint(0, 11, (20, 3)) * 5e-3
    nn = np.zeros(rr.shape)
    nn[:, -1] = 1
    pos = {'rr': rr, 'nn': nn}
    disc = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                     pos=pos, verbose='error')

    # combine source spaces
    assert srf.kind == 'surface'
    assert vol.kind == 'volume'
    assert disc.kind == 'discrete'
    src = srf + vol + disc
    assert src.kind == 'mixed'
    assert srf.kind == 'surface'
    assert vol.kind == 'volume'
    assert disc.kind == 'discrete'

    # test addition of source spaces
    assert len(src) == 4

    # test reading and writing
    src_out_name = tmp_path / 'temp-src.fif'
    src.save(src_out_name)
    src_from_file = read_source_spaces(src_out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')
    assert repr(src).split('~')[0] == repr(src_from_file).split('~')[0]
    assert_equal(src.kind, 'mixed')

    # test that all source spaces are in MRI coordinates
    coord_frames = np.array([s['coord_frame'] for s in src])
    assert (coord_frames == FIFF.FIFFV_COORD_MRI).all()

    # test errors for export_volume
    image_fname = tmp_path / 'temp-image.mgz'

    # source spaces with no volume
    with pytest.raises(ValueError, match='at least one volume'):
        srf.export_volume(image_fname, verbose='error')

    # unrecognized source type
    disc2 = disc.copy()
    disc2[0]['type'] = 'kitty'
    with pytest.raises(ValueError, match='Invalid value'):
        src + disc2
    del disc2

    # unrecognized file type
    bad_image_fname = tmp_path / 'temp-image.png'
    # vertices outside vol space warning
    pytest.raises(ValueError, src.export_volume, bad_image_fname,
                  verbose='error')

    # mixed coordinate frames
    disc3 = disc.copy()
    disc3[0]['coord_frame'] = 10
    src_mixed_coord = src + disc3
    with pytest.raises(ValueError, match='must be in head coordinates'):
        src_mixed_coord.export_volume(image_fname, verbose='error')

    # now actually write it
    fname_img = tmp_path / 'img.nii'
    for mri_resolution in (False, 'sparse', True):
        for src, up in ((vol, 705),
                        (srf + vol, 27272),
                        (disc + vol, 705)):
            src.export_volume(
                fname_img, use_lut=False,
                mri_resolution=mri_resolution, overwrite=True)
            img_data = _get_img_fdata(nib.load(str(fname_img)))
            n_src = img_data.astype(bool).sum()
            n_want = sum(s['nuse'] for s in src)
            if mri_resolution is True:
                n_want += up
            assert n_src == n_want, src

    # gh-8004
    temp_aseg = tmp_path / 'aseg.mgz'
    aseg_img = nib.load(aseg_fname)
    aseg_affine = aseg_img.affine
    aseg_affine[:3, :3] *= 0.7
    new_aseg = nib.MGHImage(aseg_img.dataobj, aseg_affine)
    nib.save(new_aseg, str(temp_aseg))
    lh_cereb = mne.setup_volume_source_space(
        "sample", mri=temp_aseg, volume_label="Left-Cerebellum-Cortex",
        add_interpolator=False, subjects_dir=subjects_dir)
    src = srf + lh_cereb
    with pytest.warns(RuntimeWarning, match='2 surf vertices lay outside'):
        src.export_volume(image_fname, mri_resolution="sparse", overwrite=True)


@testing.requires_testing_data
def test_morph_source_spaces():
    """Test morphing of source spaces."""
    src = read_source_spaces(fname_fs)
    src_morph = read_source_spaces(fname_morph)
    src_morph_py = morph_source_spaces(src, 'sample',
                                       subjects_dir=subjects_dir)
    _compare_source_spaces(src_morph, src_morph_py, mode='approx')


@pytest.mark.timeout(60)  # can be slow on OSX Travis
@pytest.mark.slowtest
@testing.requires_testing_data
def test_morphed_source_space_return():
    """Test returning a morphed source space to the original subject."""
    # let's create some random data on fsaverage
    data = rng.randn(20484, 1)
    tmin, tstep = 0, 1.
    src_fs = read_source_spaces(fname_fs)
    stc_fs = SourceEstimate(data, [s['vertno'] for s in src_fs],
                            tmin, tstep, 'fsaverage')
    n_verts_fs = sum(len(s['vertno']) for s in src_fs)

    # Create our morph source space
    src_morph = morph_source_spaces(src_fs, 'sample',
                                    subjects_dir=subjects_dir)
    n_verts_sample = sum(len(s['vertno']) for s in src_morph)
    assert n_verts_fs == n_verts_sample

    # Morph the data over using standard methods
    stc_morph = compute_source_morph(
        src_fs, 'fsaverage', 'sample',
        spacing=[s['vertno'] for s in src_morph], smooth=1,
        subjects_dir=subjects_dir, warn=False).apply(stc_fs)
    assert stc_morph.data.shape[0] == n_verts_sample

    # We can now pretend like this was real data we got e.g. from an inverse.
    # To be complete, let's remove some vertices
    keeps = [np.sort(rng.permutation(np.arange(len(v)))[:len(v) - 10])
             for v in stc_morph.vertices]
    stc_morph = SourceEstimate(
        np.concatenate([stc_morph.lh_data[keeps[0]],
                        stc_morph.rh_data[keeps[1]]]),
        [v[k] for v, k in zip(stc_morph.vertices, keeps)], tmin, tstep,
        'sample')

    # Return it to the original subject
    stc_morph_return = stc_morph.to_original_src(
        src_fs, subjects_dir=subjects_dir)

    # This should fail (has too many verts in SourceMorph)
    with pytest.warns(RuntimeWarning, match='vertices not included'):
        morph = compute_source_morph(
            src_morph, subject_from='sample',
            spacing=stc_morph_return.vertices, smooth=1,
            subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='vertices do not match'):
        morph.apply(stc_morph)

    # Compare to the original data
    with pytest.warns(RuntimeWarning, match='vertices not included'):
        stc_morph_morph = compute_source_morph(
            src=stc_morph, subject_from='sample',
            spacing=stc_morph_return.vertices, smooth=1,
            subjects_dir=subjects_dir).apply(stc_morph)

    assert_equal(stc_morph_return.subject, stc_morph_morph.subject)
    for ii in range(2):
        assert_array_equal(stc_morph_return.vertices[ii],
                           stc_morph_morph.vertices[ii])
    # These will not match perfectly because morphing pushes data around
    corr = np.corrcoef(stc_morph_return.data[:, 0],
                       stc_morph_morph.data[:, 0])[0, 1]
    assert corr > 0.99, corr

    # Explicitly test having two vertices map to the same target vertex. We
    # simulate this by having two vertices be at the same position.
    src_fs2 = src_fs.copy()
    vert1, vert2 = src_fs2[0]['vertno'][:2]
    src_fs2[0]['rr'][vert1] = src_fs2[0]['rr'][vert2]
    stc_morph_return = stc_morph.to_original_src(
        src_fs2, subjects_dir=subjects_dir)

    # test to_original_src method result equality
    for ii in range(2):
        assert_array_equal(stc_morph_return.vertices[ii],
                           stc_morph_morph.vertices[ii])

    # These will not match perfectly because morphing pushes data around
    corr = np.corrcoef(stc_morph_return.data[:, 0],
                       stc_morph_morph.data[:, 0])[0, 1]
    assert corr > 0.99, corr

    # Degenerate cases
    stc_morph.subject = None  # no .subject provided
    pytest.raises(ValueError, stc_morph.to_original_src,
                  src_fs, subject_orig='fsaverage', subjects_dir=subjects_dir)
    stc_morph.subject = 'sample'
    del src_fs[0]['subject_his_id']  # no name in src_fsaverage
    pytest.raises(ValueError, stc_morph.to_original_src,
                  src_fs, subjects_dir=subjects_dir)
    src_fs[0]['subject_his_id'] = 'fsaverage'  # name mismatch
    pytest.raises(ValueError, stc_morph.to_original_src,
                  src_fs, subject_orig='foo', subjects_dir=subjects_dir)
    src_fs[0]['subject_his_id'] = 'sample'
    src = read_source_spaces(fname)  # wrong source space
    pytest.raises(RuntimeError, stc_morph.to_original_src,
                  src, subjects_dir=subjects_dir)


# The following code was used to generate small-src.fif.gz.
# Unfortunately the C code bombs when trying to add source space distances,
# possibly due to incomplete "faking" of a smaller surface on our part here.
"""
# -*- coding: utf-8 -*-

import os
import numpy as np
import mne

data_path = mne.datasets.sample.data_path()
src = mne.setup_source_space('sample', fname=None, spacing='oct5')
hemis = ['lh', 'rh']
fnames = [data_path + '/subjects/sample/surf/%s.decimated' % h for h in hemis]

vs = list()
for s, fname in zip(src, fnames):
    coords = s['rr'][s['vertno']]
    vs.append(s['vertno'])
    idx = -1 * np.ones(len(s['rr']))
    idx[s['vertno']] = np.arange(s['nuse'])
    faces = s['use_tris']
    faces = idx[faces]
    mne.write_surface(fname, coords, faces)

# we need to move sphere surfaces
spheres = [data_path + '/subjects/sample/surf/%s.sphere' % h for h in hemis]
for s in spheres:
    os.rename(s, s + '.bak')
try:
    for s, v in zip(spheres, vs):
        coords, faces = mne.read_surface(s + '.bak')
        coords = coords[v]
        mne.write_surface(s, coords, faces)
    src = mne.setup_source_space('sample', fname=None, spacing='oct4',
                                 surface='decimated')
finally:
    for s in spheres:
        os.rename(s + '.bak', s)

fname = 'small-src.fif'
fname_gz = fname + '.gz'
mne.write_source_spaces(fname, src)
mne.utils.run_subprocess(['mne_add_patch_info', '--src', fname,
                          '--srcp', fname])
mne.write_source_spaces(fname_gz, mne.read_source_spaces(fname))
"""
