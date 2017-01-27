from __future__ import print_function

import os
import os.path as op
from nose.tools import assert_true, assert_raises
from nose.plugins.skip import SkipTest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
import warnings

from mne.datasets import testing
from mne import (read_source_spaces, vertex_to_mni, write_source_spaces,
                 setup_source_space, setup_volume_source_space,
                 add_source_space_distances, read_bem_surfaces,
                 morph_source_spaces, SourceEstimate)
from mne.utils import (_TempDir, requires_fs_or_nibabel, requires_nibabel,
                       requires_freesurfer, run_subprocess, slow_test,
                       requires_mne, requires_version, run_tests_if_main)
from mne.surface import _accumulate_normals, _triangle_neighbors
from mne.source_space import _get_mri_header, _get_mgz_header
from mne.externals.six.moves import zip
from mne.source_space import (get_volume_labels_from_aseg, SourceSpaces,
                              get_volume_labels_from_src,
                              _compare_source_spaces)
from mne.tests.common import assert_naming
from mne.io.constants import FIFF

warnings.simplefilter('always')

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_mri = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
fname_vol = op.join(subjects_dir, 'sample', 'bem',
                    'sample-volume-7mm-src.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-bem.fif')
fname_fs = op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif')
fname_morph = op.join(subjects_dir, 'sample', 'bem',
                      'sample-fsaverage-ico-5-src.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_small = op.join(base_dir, 'small-src.fif.gz')
rng = np.random.RandomState(0)


@testing.requires_testing_data
@requires_nibabel(vox2ras_tkr=True)
def test_mgz_header():
    """Test MGZ header reading."""
    header = _get_mgz_header(fname_mri)
    mri_hdr = _get_mri_header(fname_mri)
    assert_allclose(mri_hdr.get_data_shape(), header['dims'])
    assert_allclose(mri_hdr.get_vox2ras_tkr(), header['vox2ras_tkr'])
    assert_allclose(mri_hdr.get_ras2vox(), header['ras2vox'])


@requires_version('scipy', '0.11')
def test_add_patch_info():
    """Test adding patch info to source space."""
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


@testing.requires_testing_data
@requires_version('scipy', '0.11')
def test_add_source_space_distances_limited():
    """Test adding distances to source space with a dist_limit."""
    tempdir = _TempDir()
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


@slow_test
@testing.requires_testing_data
@requires_version('scipy', '0.11')
def test_add_source_space_distances():
    """Test adding distances to source space."""
    tempdir = _TempDir()
    src = read_source_spaces(fname)
    src_new = read_source_spaces(fname)
    del src_new[0]['dist']
    del src_new[1]['dist']
    n_do = 19  # limit this for speed
    src_new[0]['vertno'] = src_new[0]['vertno'][:n_do].copy()
    src_new[1]['vertno'] = src_new[1]['vertno'][:n_do].copy()
    out_name = op.join(tempdir, 'temp-src.fif')
    n_jobs = 2
    assert_true(n_do % n_jobs != 0)
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
        assert_true(np.sum(ds[0].data < 0.007) > 10)

        # do comparison
        d = ds[0] - ds[1]
        assert_allclose(np.zeros_like(d.data), d.data, rtol=0, atol=1e-9)


@testing.requires_testing_data
@requires_mne
def test_discrete_source_space():
    """Test setting up (and reading/writing) discrete source spaces."""
    tempdir = _TempDir()
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
        src_new = setup_volume_source_space(None, None, pos=pos_dict)
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
        assert_equal(repr(src_new), repr(src_c))
        assert_equal(src_new.kind, 'discrete')
    finally:
        if op.isfile(temp_name):
            os.remove(temp_name)


@slow_test
@testing.requires_testing_data
def test_volume_source_space():
    """Test setting up volume source spaces."""
    tempdir = _TempDir()
    src = read_source_spaces(fname_vol)
    temp_name = op.join(tempdir, 'temp-src.fif')
    surf = read_bem_surfaces(fname_bem, s_id=FIFF.FIFFV_BEM_SURF_ID_BRAIN)
    surf['rr'] *= 1e3  # convert to mm
    # The one in the testing dataset (uses bem as bounds)
    for bem, surf in zip((fname_bem, None), (None, surf)):
        src_new = setup_volume_source_space('sample', None, pos=7.0,
                                            bem=bem, surface=surf,
                                            mri='T1.mgz',
                                            subjects_dir=subjects_dir)
        write_source_spaces(temp_name, src_new, overwrite=True)
        src[0]['subject_his_id'] = 'sample'  # XXX: to make comparison pass
        _compare_source_spaces(src, src_new, mode='approx')
        del src_new
        src_new = read_source_spaces(temp_name)
        _compare_source_spaces(src, src_new, mode='approx')
    assert_raises(IOError, setup_volume_source_space, 'sample', None,
                  pos=7.0, bem=None, surface='foo',  # bad surf
                  mri=fname_mri, subjects_dir=subjects_dir)
    assert_equal(repr(src), repr(src_new))
    assert_equal(src.kind, 'volume')


@testing.requires_testing_data
@requires_mne
def test_other_volume_source_spaces():
    """Test setting up other volume source spaces."""
    # these are split off because they require the MNE tools, and
    # Travis doesn't seem to like them

    # let's try the spherical one (no bem or surf supplied)
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp-src.fif')
    run_subprocess(['mne_volume_source_space',
                    '--grid', '7.0',
                    '--src', temp_name,
                    '--mri', fname_mri])
    src = read_source_spaces(temp_name)
    src_new = setup_volume_source_space(None, pos=7.0, mri=fname_mri,
                                        subjects_dir=subjects_dir)
    _compare_source_spaces(src, src_new, mode='approx')
    assert_true('volume, shape' in repr(src))
    del src
    del src_new
    assert_raises(ValueError, setup_volume_source_space, 'sample', temp_name,
                  pos=7.0, sphere=[1., 1.], mri=fname_mri,  # bad sphere
                  subjects_dir=subjects_dir)

    # now without MRI argument, it should give an error when we try
    # to read it
    run_subprocess(['mne_volume_source_space',
                    '--grid', '7.0',
                    '--src', temp_name])
    assert_raises(ValueError, read_source_spaces, temp_name)


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
    assert_true(np.array_equal(nt1, nt2)
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


@slow_test
@testing.requires_testing_data
def test_setup_source_space():
    """Test setting up ico, oct, and all source spaces."""
    tempdir = _TempDir()
    fname_ico = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')
    # first lets test some input params
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct',
                  add_dist=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='octo',
                  add_dist=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='oct6e',
                  add_dist=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='7emm',
                  add_dist=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, setup_source_space, 'sample', spacing='alls',
                  add_dist=False, subjects_dir=subjects_dir)
    assert_raises(IOError, setup_source_space, 'sample', spacing='oct6',
                  subjects_dir=subjects_dir, add_dist=False)

    # ico 5 (fsaverage) - write to temp file
    src = read_source_spaces(fname_ico)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(record=True):  # sklearn equiv neighbors
        warnings.simplefilter('always')
        src_new = setup_source_space('fsaverage', temp_name, spacing='ico5',
                                     subjects_dir=subjects_dir, add_dist=False,
                                     overwrite=True)
    _compare_source_spaces(src, src_new, mode='approx')
    assert_equal(repr(src), repr(src_new))
    assert_equal(repr(src).count('surface ('), 2)
    assert_array_equal(src[0]['vertno'], np.arange(10242))
    assert_array_equal(src[1]['vertno'], np.arange(10242))

    # oct-6 (sample) - auto filename + IO
    src = read_source_spaces(fname)
    temp_name = op.join(tempdir, 'temp-src.fif')
    with warnings.catch_warnings(record=True):  # sklearn equiv neighbors
        warnings.simplefilter('always')
        src_new = setup_source_space('sample', None, spacing='oct6',
                                     subjects_dir=subjects_dir,
                                     overwrite=True, add_dist=False)
        write_source_spaces(temp_name, src_new, overwrite=True)
    _compare_source_spaces(src, src_new, mode='approx', nearest=False)
    src_new = read_source_spaces(temp_name)
    _compare_source_spaces(src, src_new, mode='approx', nearest=False)

    # all source points - no file writing
    src_new = setup_source_space('sample', None, spacing='all',
                                 subjects_dir=subjects_dir, add_dist=False)
    assert_true(src_new[0]['nuse'] == len(src_new[0]['rr']))
    assert_true(src_new[1]['nuse'] == len(src_new[1]['rr']))

    # dense source space to hit surf['inuse'] lines of _create_surf_spacing
    assert_raises(RuntimeError, setup_source_space, 'sample', None,
                  spacing='ico6', subjects_dir=subjects_dir, add_dist=False)


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
    assert_true(lh_faces.min() == 0)
    assert_true(lh_faces.max() == lh_points.shape[0] - 1)
    assert_true(lh_use_faces.min() >= 0)
    assert_true(lh_use_faces.max() <= lh_points.shape[0] - 1)
    assert_true(rh_faces.min() == 0)
    assert_true(rh_faces.max() == rh_points.shape[0] - 1)
    assert_true(rh_use_faces.min() >= 0)
    assert_true(rh_use_faces.max() <= rh_points.shape[0] - 1)


@slow_test
@testing.requires_testing_data
def test_write_source_space():
    """Test reading and writing of source spaces."""
    tempdir = _TempDir()
    src0 = read_source_spaces(fname, patch_stats=False)
    write_source_spaces(op.join(tempdir, 'tmp-src.fif'), src0)
    src1 = read_source_spaces(op.join(tempdir, 'tmp-src.fif'),
                              patch_stats=False)
    _compare_source_spaces(src0, src1)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        src_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        write_source_spaces(src_badname, src0)
        read_source_spaces(src_badname)
    assert_naming(w, 'test_source_space.py', 2)


@testing.requires_testing_data
@requires_fs_or_nibabel
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
@requires_freesurfer
@requires_nibabel()
def test_vertex_to_mni_fs_nibabel():
    """Test equivalence of vert_to_mni for nibabel and freesurfer."""
    n_check = 1000
    subject = 'sample'
    vertices = rng.randint(0, 100000, n_check)
    hemis = rng.randint(0, 1, n_check)
    coords = vertex_to_mni(vertices, hemis, subject, subjects_dir,
                           'nibabel')
    coords_2 = vertex_to_mni(vertices, hemis, subject, subjects_dir,
                             'freesurfer')
    # less than 0.1 mm error
    assert_allclose(coords, coords_2, atol=0.1)


@testing.requires_testing_data
@requires_freesurfer
@requires_nibabel()
def test_get_volume_label_names():
    """Test reading volume label names."""
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names, label_colors = get_volume_labels_from_aseg(aseg_fname,
                                                            return_colors=True)
    assert_equal(label_names.count('Brain-Stem'), 1)

    assert_equal(len(label_colors), len(label_names))


@testing.requires_testing_data
@requires_freesurfer
@requires_nibabel()
def test_source_space_from_label():
    """Test generating a source space from volume label."""
    tempdir = _TempDir()
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    label_names = get_volume_labels_from_aseg(aseg_fname)
    volume_label = label_names[int(np.random.rand() * len(label_names))]

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
                                    volume_label=volume_label, mri=aseg_fname,
                                    add_interpolator=False)
    assert_equal(volume_label, src[0]['seg_name'])

    # test reading and writing
    out_name = op.join(tempdir, 'temp-src.fif')
    write_source_spaces(out_name, src)
    src_from_file = read_source_spaces(out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')


@testing.requires_testing_data
@requires_freesurfer
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
@requires_freesurfer
@requires_nibabel()
def test_combine_source_spaces():
    """Test combining source spaces."""
    tempdir = _TempDir()
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
    src_out_name = op.join(tempdir, 'temp-src.fif')
    src.save(src_out_name)
    src_from_file = read_source_spaces(src_out_name)
    _compare_source_spaces(src, src_from_file, mode='approx')
    assert_equal(repr(src), repr(src_from_file))
    assert_equal(src.kind, 'combined')

    # test that all source spaces are in MRI coordinates
    coord_frames = np.array([s['coord_frame'] for s in src])
    assert_true((coord_frames == FIFF.FIFFV_COORD_MRI).all())

    # test errors for export_volume
    image_fname = op.join(tempdir, 'temp-image.mgz')

    # source spaces with no volume
    assert_raises(ValueError, srf.export_volume, image_fname, verbose='error')

    # unrecognized source type
    disc2 = disc.copy()
    disc2[0]['type'] = 'kitty'
    src_unrecognized = src + disc2
    assert_raises(ValueError, src_unrecognized.export_volume, image_fname,
                  verbose='error')

    # unrecognized file type
    bad_image_fname = op.join(tempdir, 'temp-image.png')
    # vertices outside vol space warning
    assert_raises(ValueError, src.export_volume, bad_image_fname,
                  verbose='error')

    # mixed coordinate frames
    disc3 = disc.copy()
    disc3[0]['coord_frame'] = 10
    src_mixed_coord = src + disc3
    assert_raises(ValueError, src_mixed_coord.export_volume, image_fname,
                  verbose='error')


@testing.requires_testing_data
def test_morph_source_spaces():
    """Test morphing of source spaces."""
    src = read_source_spaces(fname_fs)
    src_morph = read_source_spaces(fname_morph)
    src_morph_py = morph_source_spaces(src, 'sample',
                                       subjects_dir=subjects_dir)
    _compare_source_spaces(src_morph, src_morph_py, mode='approx')


@slow_test
@testing.requires_testing_data
def test_morphed_source_space_return():
    """Test returning a morphed source space to the original subject."""
    # let's create some random data on fsaverage
    data = rng.randn(20484, 1)
    tmin, tstep = 0, 1.
    src_fs = read_source_spaces(fname_fs)
    stc_fs = SourceEstimate(data, [s['vertno'] for s in src_fs],
                            tmin, tstep, 'fsaverage')

    # Create our morph source space
    src_morph = morph_source_spaces(src_fs, 'sample',
                                    subjects_dir=subjects_dir)

    # Morph the data over using standard methods
    stc_morph = stc_fs.morph('sample', [s['vertno'] for s in src_morph],
                             smooth=1, subjects_dir=subjects_dir)

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

    # Compare to the original data
    stc_morph_morph = stc_morph.morph('fsaverage', stc_morph_return.vertices,
                                      smooth=1,
                                      subjects_dir=subjects_dir)
    assert_equal(stc_morph_return.subject, stc_morph_morph.subject)
    for ii in range(2):
        assert_array_equal(stc_morph_return.vertices[ii],
                           stc_morph_morph.vertices[ii])
    # These will not match perfectly because morphing pushes data around
    corr = np.corrcoef(stc_morph_return.data[:, 0],
                       stc_morph_morph.data[:, 0])[0, 1]
    assert_true(corr > 0.99, corr)

    # Degenerate cases
    stc_morph.subject = None  # no .subject provided
    assert_raises(ValueError, stc_morph.to_original_src,
                  src_fs, subject_orig='fsaverage', subjects_dir=subjects_dir)
    stc_morph.subject = 'sample'
    del src_fs[0]['subject_his_id']  # no name in src_fsaverage
    assert_raises(ValueError, stc_morph.to_original_src,
                  src_fs, subjects_dir=subjects_dir)
    src_fs[0]['subject_his_id'] = 'fsaverage'  # name mismatch
    assert_raises(ValueError, stc_morph.to_original_src,
                  src_fs, subject_orig='foo', subjects_dir=subjects_dir)
    src_fs[0]['subject_his_id'] = 'sample'
    src = read_source_spaces(fname)  # wrong source space
    assert_raises(RuntimeError, stc_morph.to_original_src,
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
