from __future__ import print_function

import os
import os.path as op
from nose.tools import assert_true, assert_raises
from nose.plugins.skip import SkipTest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
import warnings
from scipy.spatial.distance import cdist

from mne.datasets import sample
from mne import (read_source_spaces, vertex_to_mni, write_source_spaces,
                 setup_source_space, setup_volume_source_space,
                 add_source_space_distances)
from mne.utils import (_TempDir, requires_fs_or_nibabel, requires_nibabel,
                       requires_freesurfer, run_subprocess,
                       requires_mne, requires_scipy_version)
from mne.surface import _accumulate_normals, _triangle_neighbors
from mne.source_space import _get_mgz_header
from mne.externals.six.moves import zip
from mne.source_space import get_volume_labels_from_aseg, SourceSpaces
from mne.io.constants import FIFF

warnings.simplefilter('always')

# WARNING: test_source_space is imported by forward, so download=False
# is critical here, otherwise on first import of MNE users will have to
# download the whole sample dataset!
base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_small = op.join(base_dir, 'small-src.fif.gz')
fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-5120-bem.fif')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')

tempdir = _TempDir()


@requires_nibabel(vox2ras_tkr=True)
def test_mgz_header():
    import nibabel as nib
    header = _get_mgz_header(fname_mri)
    mri_hdr = nib.load(fname_mri).get_header()
    assert_allclose(mri_hdr.get_data_shape(), header['dims'])
    assert_allclose(mri_hdr.get_vox2ras_tkr(), header['vox2ras_tkr'])
    assert_allclose(mri_hdr.get_ras2vox(), header['ras2vox'])


@requires_scipy_version('0.11')
def test_add_patch_info():
    """Test adding patch info to source space"""
    # let's setup a small source space
    src = read_source_spaces(fname_small)
    src_new = read_source_spaces(fname_small)
    for s in src_new:
        s['nearest'] = None
        s['nearest_dist'] = None
        s['pinfo'] = None

    # test that no patch info is added for small dist_limit
    try:
        add_source_space_distances(src_new, dist_limit=0.00001)
    except RuntimeError:  # what we throw when scipy version is wrong
        pass
    else:
        assert_true(all(s['nearest'] is None for s in src_new))
        assert_true(all(s['nearest_dist'] is None for s in src_new))
        assert_true(all(s['pinfo'] is None for s in src_new))

    # now let's use one that works
    add_source_space_distances(src_new)

    for s1, s2 in zip(src, src_new):
        assert_array_equal(s1['nearest'], s2['nearest'])
        assert_allclose(s1['nearest_dist'], s2['nearest_dist'], atol=1e-7)
        assert_equal(len(s1['pinfo']), len(s2['pinfo']))
        for p1, p2 in zip(s1['pinfo'], s2['pinfo']):
            assert_array_equal(p1, p2)


@sample.requires_sample_data
@requires_scipy_version('0.11')
def test_add_source_space_distances_limited():
    """Test adding distances to source space with a dist_limit"""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 200  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = op.join(tempdir, 'temp-src.fif')
    try:
        add_source_space_distances(src_new, dist_limit=0.007)
    except RuntimeError:  # what we throw when scipy version is wrong
        raise SkipTest('dist_limit requires scipy > 0.13')
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
        assert_true(np.sum(do.data < 0.007) > 400)

        # do comparison over the region computed
        d = (do - dn)[:sn['vertno'][n_do - 1]][:, :sn['vertno'][n_do - 1]]
        assert_allclose(np.zeros_like(d.data), d.data, rtol=0, atol=1e-6)


@sample.requires_sample_data
@requires_scipy_version('0.11')
def test_add_source_space_distances():
    """Test adding distances to source space"""
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 20  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = op.join(tempdir, 'temp-src.fif')
    add_source_space_distances(src_new)
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
        assert_true(np.sum(ds[0].data < 0.007) > 10)

        # do comparison
        d = ds[0] - ds[1]
        assert_allclose(np.zeros_like(d.data), d.data, rtol=0, atol=1e-9)


@sample.requires_sample_data
@requires_mne
def test_discrete_source_space():
    """Test setting up (and reading/writing) discrete source spaces
    """
    src = read_source_spaces(fname)
    v = src[0]['vertno']

    # let's make a discrete version with the C code, and with ours
    temp_name = op.join(tempdir, 'temp-src.fif')
    try:
        # save
        temp_pos = op.join(tempdir, 'temp-pos.txt')
        np.savetxt(temp_pos, np.c_[src[0]['rr'][v], src[0]['nn'][v]])
        # let's try the spherical one (no bem or surf supplied)
        run_subprocess(['mne_volume_source_space', '--meters',
                        '--pos', temp_pos, '--src', temp_name])
        src_c = read_source_spaces(temp_name)
        pos_dict = dict(rr=src[0]['rr'][v], nn=src[0]['nn'][v])
        src_new = setup_volume_source_space('sample', None,
                                            pos=pos_dict,
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src_c, src_new, mode='approx')
        assert_allclose(src[0]['rr'][v], src_new[0]['rr'],
                        rtol=1e-3, atol=1e-6)
        assert_allclose(src[0]['nn'][v], src_new[0]['nn'],
                        rtol=1e-3, atol=1e-6)

        # now do writing
        write_source_spaces(temp_name, src_c)
        src_c2 = read_source_spaces(temp_name)
        _compare_source_spaces(src_c, src_c2)

        # now do MRI
        assert_raises(ValueError, setup_volume_source_space, 'sample',
                      pos=pos_dict, mri=fname_mri)
    finally:
        if op.isfile(temp_name):
            os.remove(temp_name)


@sample.requires_sample_data
@requires_mne
def test_volume_source_space():
    """Test setting up volume source spaces
    """
    fname_vol = op.join(data_path, 'subjects', 'sample', 'bem',
                        'volume-7mm-src.fif')
    src = read_source_spaces(fname_vol)
    temp_name = op.join(tempdir, 'temp-src.fif')
    try:
        # The one in the sample dataset (uses bem as bounds)
        src_new = setup_volume_source_space('sample', temp_name, pos=7.0,
                                            bem=fname_bem, mri=fname_mri,
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src, src_new, mode='approx')
        src_new = read_source_spaces(temp_name)
        _compare_source_spaces(src, src_new, mode='approx')

        # let's try the spherical one (no bem or surf supplied)
        run_subprocess(['mne_volume_source_space',
                        '--grid', '15.0',
                        '--src', temp_name,
                        '--mri', fname_mri])
        src = read_source_spaces(temp_name)
        src_new = setup_volume_source_space('sample', temp_name, pos=15.0,
                                            mri=fname_mri,
                                            subjects_dir=subjects_dir)
        _compare_source_spaces(src, src_new, mode='approx')

        # now without MRI argument, it should give an error when we try
        # to read it
        run_subprocess(['mne_volume_source_space',
                        '--grid', '15.0',
                        '--src', temp_name])
        assert_raises(ValueError, read_source_spaces, temp_name)
    finally:
        if op.isfile(temp_name):
            os.remove(temp_name)


@sample.requires_sample_data
def test_triangle_neighbors():
    """Test efficient vertex neighboring triangles for surfaces"""
    this = read_source_spaces(fname)[0]
    this['neighbor_tri'] = [list() for _ in range(this['np'])]
    for p in range(this['ntri']):
        verts = this['tris'][p]
        this['neighbor_tri'][verts[0]].append(p)
        this['neighbor_tri'][verts[1]].append(p)
        this['neighbor_tri'][verts[2]].append(p)
    this['neighbor_tri'] = [np.array(nb, int) for nb in this['neighbor_tri']]

    neighbor_tri = _triangle_neighbors(this['tris'], this['np'])
    assert_true(np.array_equal(nt1, nt2)
                for nt1, nt2 in zip(neighbor_tri, this['neighbor_tri']))


def test_accumulate_normals():
    """Test efficient normal accumulation for surfaces"""
    # set up comparison
    rng = np.random.RandomState(0)
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


@sample.requires_sample_data
def test_setup_source_space():
    """Test setting up ico, oct, and all source spaces
    """
    fname_all = op.join(data_path, 'subjects', 'sample', 'bem',
                        'sample-all-src.fif')
    fname_ico = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')
    # first lets test some input params
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct',
                  add_dist=False)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='octo',
                  add_dist=False)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct6e',
                  add_dist=False)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='7emm',
                  add_dist=False)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='alls',
                  add_dist=False)
    assert_raises(IOError, setup_source_space, 'sample', spacing='oct6',
                  subjects_dir=subjects_dir, add_dist=False)

    # ico 5 (fsaverage) - write to temp file
    src = read_source_spaces(fname_ico)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(record=True):  # sklearn equiv neighbors
        warnings.simplefilter('always')
        src_new = setup_source_space('fsaverage', temp_name, spacing='ico5',
                                     subjects_dir=subjects_dir, add_dist=False)
    _compare_source_spaces(src, src_new, mode='approx')

    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(record=True):  # sklearn equiv neighbors
        warnings.simplefilter('always')
        src_new = setup_source_space('sample', temp_name, spacing='oct6',
                                     subjects_dir=subjects_dir,
                                     overwrite=True, add_dist=False)
    _compare_source_spaces(src, src_new, mode='approx')
    src_new = read_source_spaces(temp_name)
    _compare_source_spaces(src, src_new, mode='approx')

    # all source points - no file writing
    src = read_source_spaces(fname_all)
    src_new = setup_source_space('sample', None, spacing='all',
                                 subjects_dir=subjects_dir, add_dist=False)
    _compare_source_spaces(src, src_new, mode='approx')


@sample.requires_sample_data
def test_read_source_spaces():
    """Test reading of source space meshes
    """
    src = read_source_spaces(fname, add_geom=True)

    # 3D source space
    lh_points = src[0]['rr']
    lh_faces = src[0]['tris']
    lh_use_faces = src[0]['use_tris']
    rh_points = src[1]['rr']
    rh_faces = src[1]['tris']
    rh_use_faces = src[1]['use_tris']
    assert_true(lh_faces.min() == 0)
    assert_true(lh_faces.max() == lh_points.shape[0] - 1)
    assert_true(lh_use_faces.min() >= 0)
    assert_true(lh_use_faces.max() <= lh_points.shape[0] - 1)
    assert_true(rh_faces.min() == 0)
    assert_true(rh_faces.max() == rh_points.shape[0] - 1)
    assert_true(rh_use_faces.min() >= 0)
    assert_true(rh_use_faces.max() <= rh_points.shape[0] - 1)


@sample.requires_sample_data
def test_write_source_space():
    """Test writing and reading of source spaces
    """
    src0 = read_source_spaces(fname, add_geom=False)
    write_source_spaces(op.join(tempdir, 'tmp-src.fif'), src0)
    src1 = read_source_spaces(op.join(tempdir, 'tmp-src.fif'), add_geom=False)
    _compare_source_spaces(src0, src1)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        src_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        write_source_spaces(src_badname, src0)
        read_source_spaces(src_badname)
        print([ww.message for ww in w])
    assert_equal(len(w), 2)


def _compare_source_spaces(src0, src1, mode='exact'):
    """Compare two source spaces

    Note: this function is also used by forward/tests/test_make_forward.py
    """
    for s0, s1 in zip(src0, src1):
        for name in ['nuse', 'ntri', 'np', 'type', 'id']:
            print(name)
            assert_equal(s0[name], s1[name])
        for name in ['subject_his_id']:
            if name in s0 or name in s1:
                print(name)
                assert_equal(s0[name], s1[name])
        for name in ['interpolator']:
            if name in s0 or name in s1:
                print(name)
                diffs = (s0['interpolator'] - s1['interpolator']).data
                if len(diffs) > 0:
                    assert_true(np.sqrt(np.mean(diffs ** 2)) < 0.05)  # 5%
        for name in ['nn', 'rr', 'nuse_tri', 'coord_frame', 'tris']:
            print(name)
            if s0[name] is None:
                assert_true(s1[name] is None)
            else:
                if mode == 'exact':
                    assert_array_equal(s0[name], s1[name])
                elif mode == 'approx':
                    assert_allclose(s0[name], s1[name], rtol=1e-3, atol=1e-4)
                else:
                    raise RuntimeError('unknown mode')
        for name in ['seg_name']:
            if name in s0 or name in s1:
                print(name)
                assert_equal(s0[name], s1[name])
        if mode == 'exact':
            for name in ['inuse', 'vertno', 'use_tris']:
                assert_array_equal(s0[name], s1[name])
            # these fields will exist if patch info was added, these are
            # not tested in mode == 'approx'
            for name in ['nearest', 'nearest_dist']:
                print(name)
                if s0[name] is None:
                    assert_true(s1[name] is None)
                else:
                    assert_array_equal(s0[name], s1[name])
            for name in ['dist_limit']:
                print(name)
                assert_true(s0[name] == s1[name])
            for name in ['dist']:
                if s0[name] is not None:
                    assert_equal(s1[name].shape, s0[name].shape)
                    assert_true(len((s0['dist'] - s1['dist']).data) == 0)
            for name in ['pinfo']:
                if s0[name] is not None:
                    assert_true(len(s0[name]) == len(s1[name]))
                    for p1, p2 in zip(s0[name], s1[name]):
                        assert_true(all(p1 == p2))
        elif mode == 'approx':
            # deal with vertno, inuse, and use_tris carefully
            assert_array_equal(s0['vertno'], np.where(s0['inuse'])[0])
            assert_array_equal(s1['vertno'], np.where(s1['inuse'])[0])
            assert_equal(len(s0['vertno']), len(s1['vertno']))
            agreement = np.mean(s0['inuse'] == s1['inuse'])
            assert_true(agreement > 0.99)
            if agreement < 1.0:
                # make sure mismatched vertno are within 1.5mm
                v0 = np.setdiff1d(s0['vertno'], s1['vertno'])
                v1 = np.setdiff1d(s1['vertno'], s0['vertno'])
                dists = cdist(s0['rr'][v0], s1['rr'][v1])
                assert_allclose(np.min(dists, axis=1), np.zeros(len(v0)),
                                atol=1.5e-3)
            if s0['use_tris'] is not None:  # for "spacing"
                assert_array_equal(s0['use_tris'].shape, s1['use_tris'].shape)
            else:
                assert_true(s1['use_tris'] is None)
            assert_true(np.mean(s0['use_tris'] == s1['use_tris']) > 0.99)
    # The above "if s0[name] is not None" can be removed once the sample
    # dataset is updated to have a source space with distance info
    for name in ['working_dir', 'command_line']:
        if mode == 'exact':
            assert_equal(src0.info[name], src1.info[name])
        elif mode == 'approx':
            print(name)
            if name in src0.info:
                assert_true(name in src1.info)
            else:
                assert_true(name not in src1.info)


@sample.requires_sample_data
@requires_fs_or_nibabel
def test_vertex_to_mni():
    """Test conversion of vertices to MNI coordinates
    """
    # obtained using "tksurfer (sample/fsaverage) (l/r)h white"
    vertices = [100960, 7620, 150549, 96761]
    coords_s = np.array([[-60.86, -11.18, -3.19], [-36.46, -93.18, -2.36],
                         [-38.00, 50.08, -10.61], [47.14, 8.01, 46.93]])
    coords_f = np.array([[-41.28, -40.04, 18.20], [-6.05, 49.74, -18.15],
                         [-61.71, -14.55, 20.52], [21.70, -60.84, 25.02]])
    hemis = [0, 0, 0, 1]
    for coords, subject in zip([coords_s, coords_f], ['sample', 'fsaverage']):
        coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir)
        # less than 1mm error
        assert_allclose(coords, coords_2, atol=1.0)


@sample.requires_sample_data
@requires_freesurfer
@requires_nibabel()
def test_vertex_to_mni_fs_nibabel():
    """Test equivalence of vert_to_mni for nibabel and freesurfer
    """
    n_check = 1000
    for subject in ['sample', 'fsaverage']:
        vertices = np.random.randint(0, 100000, n_check)
        hemis = np.random.randint(0, 1, n_check)
        coords = vertex_to_mni(vertices, hemis, subject, subjects_dir,
                               'nibabel')
        coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir,
                                 'freesurfer')
        # less than 0.1 mm error
        assert_allclose(coords, coords_2, atol=0.1)


@sample.requires_sample_data
@requires_freesurfer
@requires_nibabel()
def test_get_volume_label_names():
    """Test reading volume label names
    """
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')

    label_names = get_volume_labels_from_aseg(aseg_fname)

    assert_equal(label_names.count('Brain-Stem'), 1)


@sample.requires_sample_data
@requires_freesurfer
@requires_nibabel()
def test_source_space_from_label():
    """Test generating a source space from volume label
    """
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names = get_volume_labels_from_aseg(aseg_fname)
    volume_label = np.random.choice(label_names, 1)[0]

    # Test pos as dict
    pos = dict()
    assert_raises(ValueError, setup_volume_source_space, 'sample', pos=pos,
                  volume_label=volume_label, mri=aseg_fname)

    # Test no mri provided
    assert_raises(RuntimeError, setup_volume_source_space, 'sample', mri=None,
                  volume_label=volume_label)

    # Test invalid volume label
    assert_raises(ValueError, setup_volume_source_space, 'sample',
                  volume_label='Hello World!', mri=aseg_fname)

    src = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                    volume_label=volume_label, mri=aseg_fname)
    assert_equal(volume_label, src[0]['seg_name'])

    # test reading and writing
    out_name = op.join(tempdir, 'temp-src.fif')
    write_source_spaces(out_name, src)
    src_from_file = read_source_spaces(out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')


@sample.requires_sample_data
@requires_freesurfer
@requires_nibabel()
def test_combine_source_spaces():
    """Test combining two source spaces
    """
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names = get_volume_labels_from_aseg(aseg_fname)
    volume_labels = np.random.choice(label_names, 2)

    # setup a surface source space
    srf = setup_source_space('sample', subjects_dir=subjects_dir,
                             overwrite=True)

    # setup 2 volume source spaces
    vol1 = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                     volume_label=volume_labels[0],
                                     mri=aseg_fname)
    vol2 = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                     volume_label=volume_labels[1],
                                     mri=aseg_fname)

    # setup a discrete source space
    rr = np.random.randint(0, 20, (100, 3)) * 1e-3
    nn = np.zeros(rr.shape)
    nn[:, -1] = 1
    pos = {'rr': rr, 'nn': nn}
    disc = setup_volume_source_space('sample', subjects_dir=subjects_dir,
                                     pos=pos)

    # combine source spaces
    src = srf + vol1 + vol2 + disc

    # test addition of source spaces
    assert_equal(type(src), SourceSpaces)
    assert_equal(len(src), 5)

    # test reading and writing
    src_out_name = op.join(tempdir, 'temp-src.fif')
    src.save(src_out_name)
    src_from_file = read_source_spaces(src_out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')

    # test that all source spaces are in MRI coordinates
    coord_frames = np.array([s['coord_frame'] for s in src])
    assert_true((coord_frames == FIFF.FIFFV_COORD_MRI).all())

    # test errors for export_volume
    image_fname = op.join(tempdir, 'temp-image.mgz')

    # source spaces with no volume
    assert_raises(ValueError, srf.export_volume, image_fname)

    # unrecognized source type
    disc2 = disc.copy()
    disc2[0]['type'] = 'kitty'
    src_unrecognized = src + disc2
    assert_raises(ValueError, src_unrecognized.export_volume, image_fname)

    # unrecognized file type
    bad_image_fname = op.join(tempdir, 'temp-image.png')
    assert_raises(ValueError, src.export_volume, bad_image_fname)

    # mixed coordinate frames
    disc3 = disc.copy()
    disc3[0]['coord_frame'] = 10
    src_mixed_coord = src + disc3
    assert_raises(ValueError, src_mixed_coord.export_volume, image_fname)

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
