# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
from shutil import copytree

import pytest
import scipy
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from mne.datasets import testing
import mne
from mne import (read_source_spaces, vertex_to_mni, write_source_spaces,
                 setup_source_space, setup_volume_source_space,
                 add_source_space_distances, read_bem_surfaces,
                 morph_source_spaces, SourceEstimate, make_sphere_model,
                 head_to_mni, read_trans, compute_source_morph,
                 read_bem_solution)
from mne.utils import (requires_nibabel, run_subprocess,
                       modified_env, requires_mne, run_tests_if_main,
                       check_version)
from mne.surface import _accumulate_normals, _triangle_neighbors
from mne.source_space import _get_mgz_header, _read_talxfm
from mne.source_estimate import _get_src_type
from mne.transforms import apply_trans, invert_transform
from mne.source_space import (get_volume_labels_from_aseg, SourceSpaces,
                              get_volume_labels_from_src,
                              _compare_source_spaces)
from mne.io.constants import FIFF

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
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
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_small = op.join(base_dir, 'small-src.fif.gz')
rng = np.random.RandomState(0)


@testing.requires_testing_data
@requires_nibabel()
def test_mgz_header():
    """Test MGZ header reading."""
    import nibabel
    header = _get_mgz_header(fname_mri)
    mri_hdr = nibabel.load(fname_mri).header
    assert_allclose(mri_hdr.get_data_shape(), header['dims'])
    assert_allclose(mri_hdr.get_vox2ras_tkr(), header['vox2ras_tkr'])
    assert_allclose(mri_hdr.get_ras2vox(), np.linalg.inv(header['vox2ras']))


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
def test_add_source_space_distances_limited(tmpdir):
    """Test adding distances to source space with a dist_limit."""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 200  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = tmpdir.join('temp-src.fif')
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
def test_add_source_space_distances(tmpdir):
    """Test adding distances to source space."""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 19  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = tmpdir.join('temp-src.fif')
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
def test_discrete_source_space(tmpdir):
    """Test setting up (and reading/writing) discrete source spaces."""
    src = read_source_spaces(fname)
    v = src[0]['vertno']

    # let's make a discrete version with the C code, and with ours
    temp_name = tmpdir.join('temp-src.fif')
    # save
    temp_pos = tmpdir.join('temp-pos.txt')
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
    pytest.raises(ValueError, setup_volume_source_space, 'sample',
                  pos=pos_dict, mri=fname_mri)
    assert repr(src_new) == repr(src_c)
    assert src_new.kind == 'discrete'
    assert _get_src_type(src_new, None) == 'discrete'


@pytest.mark.slowtest
@testing.requires_testing_data
def test_volume_source_space(tmpdir):
    """Test setting up volume source spaces."""
    src = read_source_spaces(fname_vol)
    temp_name = tmpdir.join('temp-src.fif')
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
    assert src.kind == 'volume'
    # Spheres
    sphere = make_sphere_model(r0=(0., 0., 0.), head_radius=0.1,
                               relative_radii=(0.9, 1.0), sigmas=(0.33, 1.0))
    with pytest.deprecated_call(match='sphere_units'):
        src = setup_volume_source_space(pos=10, sphere=(0., 0., 0., 90))
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
def test_other_volume_source_spaces(tmpdir):
    """Test setting up other volume source spaces."""
    # these are split off because they require the MNE tools, and
    # Travis doesn't seem to like them

    # let's try the spherical one (no bem or surf supplied)
    temp_name = tmpdir.join('temp-src.fif')
    run_subprocess(['mne_volume_source_space',
                    '--grid', '7.0',
                    '--src', temp_name,
                    '--mri', fname_mri])
    src = read_source_spaces(temp_name)
    src_new = setup_volume_source_space(None, pos=7.0, mri=fname_mri,
                                        subjects_dir=subjects_dir)
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
def test_setup_source_space(tmpdir):
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
    assert_equal(repr(src), repr(src_new))
    assert_equal(repr(src).count('surface ('), 2)
    assert_array_equal(src[0]['vertno'], np.arange(10242))
    assert_array_equal(src[1]['vertno'], np.arange(10242))

    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = tmpdir.join('temp-src.fif')
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
def test_setup_source_space_spacing(tmpdir, spacing):
    """Test setting up surface source spaces using a given spacing."""
    copytree(op.join(subjects_dir, 'sample'), str(tmpdir.join('sample')))
    args = [] if spacing == 7 else ['--spacing', str(spacing)]
    with modified_env(SUBJECTS_DIR=str(tmpdir), SUBJECT='sample'):
        run_subprocess(['mne_setup_source_space'] + args)
    src = read_source_spaces(tmpdir.join('sample', 'bem',
                                         'sample-%d-src.fif' % spacing))
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
def test_write_source_space(tmpdir):
    """Test reading and writing of source spaces."""
    src0 = read_source_spaces(fname, patch_stats=False)
    temp_fname = tmpdir.join('tmp-src.fif')
    write_source_spaces(temp_fname, src0)
    src1 = read_source_spaces(temp_fname, patch_stats=False)
    _compare_source_spaces(src0, src1)

    # test warnings on bad filenames
    src_badname = tmpdir.join('test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='-src.fif'):
        write_source_spaces(src_badname, src0)
    with pytest.warns(RuntimeWarning, match='-src.fif'):
        read_source_spaces(src_badname)


@testing.requires_testing_data
def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates."""
    # obtained using "tksurfer (sample) (l/r)h white"
    vertices = [100960, 7620, 150549, 96761]
    coords = np.array([[-60.86, -11.18, -3.19], [-36.46, -93.18, -2.36],
                       [-38.00, 50.08, -10.61], [47.14, 8.01, 46.93]])
    hemis = [0, 0, 0, 1]
    coords_2 = vertex_to_mni(vertices, hemis, 'sample', subjects_dir)
    # less than 1mm error
    assert_allclose(coords, coords_2, atol=1.0)


@testing.requires_testing_data
def test_head_to_mni():
    """Test conversion of aseg vertices to MNI coordinates."""
    # obtained using freeview
    coords = np.array([[22.52, 11.24, 17.72], [22.52, 5.46, 21.58],
                       [16.10, 5.46, 22.23], [21.24, 8.36, 22.23]])

    xfm = _read_talxfm('sample', subjects_dir)
    coords_MNI = apply_trans(xfm['trans'], coords)

    trans = read_trans(trans_fname)  # head->MRI (surface RAS)
    mri_head_t = invert_transform(trans)  # MRI (surface RAS)->head matrix

    # obtained from sample_audvis-meg-oct-6-mixed-fwd.fif
    coo_right_amygdala = np.array([[0.01745682, 0.02665809, 0.03281873],
                                   [0.01014125, 0.02496262, 0.04233755],
                                   [0.01713642, 0.02505193, 0.04258181],
                                   [0.01720631, 0.03073877, 0.03850075]])
    coords_MNI_2 = head_to_mni(coo_right_amygdala, 'sample', mri_head_t,
                               subjects_dir)
    # less than 1mm error
    assert_allclose(coords_MNI, coords_MNI_2, atol=10.0)


@testing.requires_testing_data
def test_vertex_to_mni_fs_nibabel(monkeypatch):
    """Test equivalence of vert_to_mni for nibabel and freesurfer."""
    n_check = 1000
    subject = 'sample'
    vertices = rng.randint(0, 100000, n_check)
    hemis = rng.randint(0, 1, n_check)
    coords = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    monkeypatch.setattr(mne.source_space, 'has_nibabel', lambda: False)
    coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir)
    # less than 0.1 mm error
    assert_allclose(coords, coords_2, atol=0.1)


@testing.requires_testing_data
@requires_nibabel()
def test_get_volume_label_names():
    """Test reading volume label names."""
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names, label_colors = get_volume_labels_from_aseg(aseg_fname,
                                                            return_colors=True)
    assert_equal(label_names.count('Brain-Stem'), 1)

    assert_equal(len(label_colors), len(label_names))


@testing.requires_testing_data
@requires_nibabel()
def test_source_space_from_label(tmpdir):
    """Test generating a source space from volume label."""
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names = get_volume_labels_from_aseg(aseg_fname)
    volume_label = label_names[int(np.random.rand() * len(label_names))]

    # Test pos as dict
    pos = dict()
    pytest.raises(ValueError, setup_volume_source_space, 'sample', pos=pos,
                  volume_label=volume_label, mri=aseg_fname)

    # Test no mri provided
    pytest.raises(RuntimeError, setup_volume_source_space, 'sample', mri=None,
                  volume_label=volume_label)

    # Test invalid volume label
    pytest.raises(ValueError, setup_volume_source_space, 'sample',
                  volume_label='Hello World!', mri=aseg_fname)

    src = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                    volume_label=volume_label, mri=aseg_fname,
                                    add_interpolator=False)
    assert_equal(volume_label, src[0]['seg_name'])

    # test reading and writing
    out_name = tmpdir.join('temp-src.fif')
    write_source_spaces(out_name, src)
    src_from_file = read_source_spaces(out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')


@testing.requires_testing_data
def test_source_space_exclusive():
    """Test that we produce exclusive labels."""
    # these two are neighbors and are quite large, so let's use them to
    # ensure no overlaps
    volume_label = ['Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex']
    src = setup_volume_source_space(
        'sample', subjects_dir=subjects_dir, volume_label=volume_label,
        mri='aseg.mgz', add_interpolator=False, sphere=(0, 0, 0, 0.095),
        sphere_units='m')
    assert src[0]['nuse'] == 2034  # was 2832
    assert src[1]['nuse'] == 1520  # was 2623
    overlap = np.in1d(src[0]['vertno'], src[1]['vertno']).mean()
    assert overlap == 0.


@pytest.mark.timeout(60)  # ~24 sec on Travis
@pytest.mark.slowtest
@testing.requires_testing_data
@requires_nibabel()
def test_read_volume_from_src():
    """Test reading volumes from a mixed source space."""
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
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
    # Generate the mixed source space
    src += vol_src

    volume_src = get_volume_labels_from_src(src, 'sample', subjects_dir)
    volume_label = volume_src[0].name
    volume_label = 'Left-' + volume_label.replace('-lh', '')

    # Test
    assert_equal(volume_label, src[2]['seg_name'])

    assert_equal(src[2]['type'], 'vol')


@testing.requires_testing_data
@requires_nibabel()
def test_combine_source_spaces(tmpdir):
    """Test combining source spaces."""
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names = get_volume_labels_from_aseg(aseg_fname)
    volume_labels = [label_names[int(np.random.rand() * len(label_names))]
                     for ii in range(2)]

    # get a surface source space (no need to test creation here)
    srf = read_source_spaces(fname, patch_stats=False)

    # setup 2 volume source spaces
    vol = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                    volume_label=volume_labels[0],
                                    mri=aseg_fname, add_interpolator=False)

    # setup a discrete source space
    rr = rng.randint(0, 20, (100, 3)) * 1e-3
    nn = np.zeros(rr.shape)
    nn[:, -1] = 1
    pos = {'rr': rr, 'nn': nn}
    disc = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                     pos=pos, verbose='error')

    # combine source spaces
    src = srf + vol + disc

    # test addition of source spaces
    assert_equal(type(src), SourceSpaces)
    assert_equal(len(src), 4)

    # test reading and writing
    src_out_name = tmpdir.join('temp-src.fif')
    src.save(src_out_name)
    src_from_file = read_source_spaces(src_out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')
    assert_equal(repr(src), repr(src_from_file))
    assert_equal(src.kind, 'mixed')

    # test that all source spaces are in MRI coordinates
    coord_frames = np.array([s['coord_frame'] for s in src])
    assert (coord_frames == FIFF.FIFFV_COORD_MRI).all()

    # test errors for export_volume
    image_fname = tmpdir.join('temp-image.mgz')

    # source spaces with no volume
    pytest.raises(ValueError, srf.export_volume, image_fname, verbose='error')

    # unrecognized source type
    disc2 = disc.copy()
    disc2[0]['type'] = 'kitty'
    src_unrecognized = src + disc2
    pytest.raises(ValueError, src_unrecognized.export_volume, image_fname,
                  verbose='error')

    # unrecognized file type
    bad_image_fname = tmpdir.join('temp-image.png')
    # vertices outside vol space warning
    pytest.raises(ValueError, src.export_volume, bad_image_fname,
                  verbose='error')

    # mixed coordinate frames
    disc3 = disc.copy()
    disc3[0]['coord_frame'] = 10
    src_mixed_coord = src + disc3
    pytest.raises(ValueError, src_mixed_coord.export_volume, image_fname,
                  verbose='error')


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


run_tests_if_main()

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
