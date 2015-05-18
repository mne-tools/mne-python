import os
import os.path as op
import shutil
import glob
import warnings
import sys

import numpy as np
from scipy import sparse

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_true, assert_false, assert_raises

from mne.datasets import testing
from mne import (read_label, stc_to_label, read_source_estimate,
                 read_source_spaces, grow_labels, read_labels_from_annot,
                 write_labels_to_annot, split_label, spatial_tris_connectivity,
                 read_surface)
from mne.label import Label, _blend_colors
from mne.utils import (_TempDir, requires_sklearn, get_subjects_dir,
                       run_tests_if_main, slow_test)
from mne.fixes import digitize, in1d, assert_is, assert_is_not
from mne.label import _n_colors
from mne.source_space import SourceSpaces
from mne.source_estimate import mesh_edges
from mne.externals.six import string_types
from mne.externals.six.moves import cPickle as pickle


warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
src_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
stc_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-lh.stc')
real_label_fname = op.join(data_path, 'MEG', 'sample', 'labels',
                           'Aud-lh.label')
real_label_rh_fname = op.join(data_path, 'MEG', 'sample', 'labels',
                              'Aud-rh.label')
v1_label_fname = op.join(subjects_dir, 'sample', 'label', 'lh.V1.label')

fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
src_bad_fname = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')
label_dir = op.join(subjects_dir, 'sample', 'label', 'aparc')

test_path = op.join(op.split(__file__)[0], '..', 'io', 'tests', 'data')
label_fname = op.join(test_path, 'test-lh.label')
label_rh_fname = op.join(test_path, 'test-rh.label')

# This code was used to generate the "fake" test labels:
# for hemi in ['lh', 'rh']:
#    label = Label(np.unique((np.random.rand(100) * 10242).astype(int)),
#                  hemi=hemi, comment='Test ' + hemi, subject='fsaverage')
#    label.save(op.join(test_path, 'test-%s.label' % hemi))


# XXX : this was added for backward compat and keep the old test_label_in_src
def _stc_to_label(stc, src, smooth, subjects_dir=None):
    """Compute a label from the non-zero sources in an stc object.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates.
    src : SourceSpaces | str | None
        The source space over which the source estimates are defined.
        If it's a string it should the subject name (e.g. fsaverage).
        Can be None if stc.subject is not None.
    smooth : int
        Number of smoothing iterations.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    labels : list of Labels | list of list of Labels
        The generated labels. If connected is False, it returns
        a list of Labels (one per hemisphere). If no Label is available
        in a hemisphere, None is returned. If connected is True,
        it returns for each hemisphere a list of connected labels
        ordered in decreasing order depending of the maximum value in the stc.
        If no Label is available in an hemisphere, an empty list is returned.
    """
    src = stc.subject if src is None else src

    if isinstance(src, string_types):
        subject = src
    else:
        subject = stc.subject

    if isinstance(src, string_types):
        subjects_dir = get_subjects_dir(subjects_dir)
        surf_path_from = op.join(subjects_dir, src, 'surf')
        rr_lh, tris_lh = read_surface(op.join(surf_path_from,
                                      'lh.white'))
        rr_rh, tris_rh = read_surface(op.join(surf_path_from,
                                      'rh.white'))
        rr = [rr_lh, rr_rh]
        tris = [tris_lh, tris_rh]
    else:
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or a set of source spaces')
        if len(src) != 2:
            raise ValueError('source space should contain the 2 hemispheres')
        rr = [1e3 * src[0]['rr'], 1e3 * src[1]['rr']]
        tris = [src[0]['tris'], src[1]['tris']]

    labels = []
    cnt = 0
    for hemi_idx, (hemi, this_vertno, this_tris, this_rr) in enumerate(
            zip(['lh', 'rh'], stc.vertices, tris, rr)):
        this_data = stc.data[cnt:cnt + len(this_vertno)]
        e = mesh_edges(this_tris)
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)

        clusters = [this_vertno[np.any(this_data, axis=1)]]

        cnt += len(this_vertno)

        clusters = [c for c in clusters if len(c) > 0]

        if len(clusters) == 0:
            this_labels = None
        else:
            this_labels = []
            colors = _n_colors(len(clusters))
            for c, color in zip(clusters, colors):
                idx_use = c
                for k in range(smooth):
                    e_use = e[:, idx_use]
                    data1 = e_use * np.ones(len(idx_use))
                    idx_use = np.where(data1)[0]

                label = Label(idx_use, this_rr[idx_use], None, hemi,
                              'Label from stc', subject=subject,
                              color=color)

                this_labels.append(label)

            this_labels = this_labels[0]

        labels.append(this_labels)

    return labels


def assert_labels_equal(l0, l1, decimal=5, comment=True, color=True):
    if comment:
        assert_equal(l0.comment, l1.comment)
    if color:
        assert_equal(l0.color, l1.color)

    for attr in ['hemi', 'subject']:
        attr0 = getattr(l0, attr)
        attr1 = getattr(l1, attr)
        msg = "label.%s: %r != %r" % (attr, attr0, attr1)
        assert_equal(attr0, attr1, msg)
    for attr in ['vertices', 'pos', 'values']:
        a0 = getattr(l0, attr)
        a1 = getattr(l1, attr)
        assert_array_almost_equal(a0, a1, decimal)


def test_label_subject():
    """Test label subject name extraction
    """
    label = read_label(label_fname)
    assert_is(label.subject, None)
    assert_true('unknown' in repr(label))
    label = read_label(label_fname, subject='fsaverage')
    assert_true(label.subject == 'fsaverage')
    assert_true('fsaverage' in repr(label))


def test_label_addition():
    """Test label addition
    """
    pos = np.random.rand(10, 3)
    values = np.arange(10.) / 10
    idx0 = list(range(7))
    idx1 = list(range(7, 10))  # non-overlapping
    idx2 = list(range(5, 10))  # overlapping
    l0 = Label(idx0, pos[idx0], values[idx0], 'lh', color='red')
    l1 = Label(idx1, pos[idx1], values[idx1], 'lh')
    l2 = Label(idx2, pos[idx2], values[idx2], 'lh', color=(0, 1, 0, .5))

    assert_equal(len(l0), len(idx0))

    # adding non-overlapping labels
    l01 = l0 + l1
    assert_equal(len(l01), len(l0) + len(l1))
    assert_array_equal(l01.values[:len(l0)], l0.values)
    assert_equal(l01.color, l0.color)
    # subtraction
    assert_labels_equal(l01 - l0, l1, comment=False, color=False)
    assert_labels_equal(l01 - l1, l0, comment=False, color=False)

    # adding overlappig labels
    l = l0 + l2
    i0 = np.where(l0.vertices == 6)[0][0]
    i2 = np.where(l2.vertices == 6)[0][0]
    i = np.where(l.vertices == 6)[0][0]
    assert_equal(l.values[i], l0.values[i0] + l2.values[i2])
    assert_equal(l.values[0], l0.values[0])
    assert_array_equal(np.unique(l.vertices), np.unique(idx0 + idx2))
    assert_equal(l.color, _blend_colors(l0.color, l2.color))

    # adding lh and rh
    l2.hemi = 'rh'
    # this now has deprecated behavior
    bhl = l0 + l2
    assert_equal(bhl.hemi, 'both')
    assert_equal(len(bhl), len(l0) + len(l2))
    assert_equal(bhl.color, l.color)
    # subtraction
    assert_labels_equal(bhl - l0, l2)
    assert_labels_equal(bhl - l2, l0)

    bhl2 = l1 + bhl
    assert_labels_equal(bhl2.lh, l01)
    assert_equal(bhl2.color, _blend_colors(l1.color, bhl.color))
    # subtraction
    bhl_ = bhl2 - l1
    assert_labels_equal(bhl_.lh, bhl.lh, comment=False, color=False)
    assert_labels_equal(bhl_.rh, bhl.rh)
    assert_labels_equal(bhl2 - l2, l0 + l1)
    assert_labels_equal(bhl2 - l1 - l0, l2)


@testing.requires_testing_data
def test_label_in_src():
    """Test label in src"""
    src = read_source_spaces(src_fname)
    label = read_label(v1_label_fname)

    # construct label from source space vertices
    vert_in_src = np.intersect1d(label.vertices, src[0]['vertno'], True)
    where = in1d(label.vertices, vert_in_src)
    pos_in_src = label.pos[where]
    values_in_src = label.values[where]
    label_src = Label(vert_in_src, pos_in_src, values_in_src,
                      hemi='lh').fill(src)

    # check label vertices
    vertices_status = in1d(src[0]['nearest'], label.vertices)
    vertices_in = np.nonzero(vertices_status)[0]
    vertices_out = np.nonzero(np.logical_not(vertices_status))[0]
    assert_array_equal(label_src.vertices, vertices_in)
    assert_array_equal(in1d(vertices_out, label_src.vertices), False)

    # check values
    value_idx = digitize(src[0]['nearest'][vertices_in], vert_in_src, True)
    assert_array_equal(label_src.values, values_in_src[value_idx])

    # test exception
    vertices = np.append([-1], vert_in_src)
    assert_raises(ValueError, Label(vertices, hemi='lh').fill, src)


@testing.requires_testing_data
def test_label_io_and_time_course_estimates():
    """Test IO for label + stc files
    """
    stc = read_source_estimate(stc_fname)
    label = read_label(real_label_fname)
    stc_label = stc.in_label(label)

    assert_true(len(stc_label.times) == stc_label.data.shape[1])
    assert_true(len(stc_label.vertices[0]) == stc_label.data.shape[0])


@testing.requires_testing_data
def test_label_io():
    """Test IO of label files
    """
    tempdir = _TempDir()
    label = read_label(label_fname)

    # label attributes
    assert_equal(label.name, 'test-lh')
    assert_is(label.subject, None)
    assert_is(label.color, None)

    # save and reload
    label.save(op.join(tempdir, 'foo'))
    label2 = read_label(op.join(tempdir, 'foo-lh.label'))
    assert_labels_equal(label, label2)

    # pickling
    dest = op.join(tempdir, 'foo.pickled')
    with open(dest, 'wb') as fid:
        pickle.dump(label, fid, pickle.HIGHEST_PROTOCOL)
    with open(dest, 'rb') as fid:
        label2 = pickle.load(fid)
    assert_labels_equal(label, label2)


def _assert_labels_equal(labels_a, labels_b, ignore_pos=False):
    """Make sure two sets of labels are equal"""
    for label_a, label_b in zip(labels_a, labels_b):
        assert_array_equal(label_a.vertices, label_b.vertices)
        assert_true(label_a.name == label_b.name)
        assert_true(label_a.hemi == label_b.hemi)
        if not ignore_pos:
            assert_array_equal(label_a.pos, label_b.pos)


@testing.requires_testing_data
def test_annot_io():
    """Test I/O from and to *.annot files"""
    # copy necessary files from fsaverage to tempdir
    tempdir = _TempDir()
    subject = 'fsaverage'
    label_src = os.path.join(subjects_dir, 'fsaverage', 'label')
    surf_src = os.path.join(subjects_dir, 'fsaverage', 'surf')
    label_dir = os.path.join(tempdir, subject, 'label')
    surf_dir = os.path.join(tempdir, subject, 'surf')
    os.makedirs(label_dir)
    os.mkdir(surf_dir)
    shutil.copy(os.path.join(label_src, 'lh.PALS_B12_Lobes.annot'), label_dir)
    shutil.copy(os.path.join(label_src, 'rh.PALS_B12_Lobes.annot'), label_dir)
    shutil.copy(os.path.join(surf_src, 'lh.white'), surf_dir)
    shutil.copy(os.path.join(surf_src, 'rh.white'), surf_dir)

    # read original labels
    labels = read_labels_from_annot(subject, 'PALS_B12_Lobes',
                                    subjects_dir=tempdir)

    # test saving parcellation only covering one hemisphere
    parc = [l for l in labels if l.name == 'LOBE.TEMPORAL-lh']
    write_labels_to_annot(parc, subject, 'myparc', subjects_dir=tempdir)
    parc1 = read_labels_from_annot(subject, 'myparc', subjects_dir=tempdir)
    parc1 = [l for l in parc1 if not l.name.startswith('unknown')]
    assert_equal(len(parc1), len(parc))
    for l1, l in zip(parc1, parc):
        assert_labels_equal(l1, l)

    # test saving only one hemisphere
    parc = [l for l in labels if l.name.startswith('LOBE')]
    write_labels_to_annot(parc, subject, 'myparc2', hemi='lh',
                          subjects_dir=tempdir)
    annot_fname = os.path.join(tempdir, subject, 'label', '%sh.myparc2.annot')
    assert_true(os.path.isfile(annot_fname % 'l'))
    assert_false(os.path.isfile(annot_fname % 'r'))
    parc1 = read_labels_from_annot(subject, 'myparc2',
                                   annot_fname=annot_fname % 'l',
                                   subjects_dir=tempdir)
    parc_lh = [l for l in parc if l.name.endswith('lh')]
    for l1, l in zip(parc1, parc_lh):
        assert_labels_equal(l1, l)


@testing.requires_testing_data
def test_read_labels_from_annot():
    """Test reading labels from FreeSurfer parcellation
    """
    # test some invalid inputs
    assert_raises(ValueError, read_labels_from_annot, 'sample', hemi='bla',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, read_labels_from_annot, 'sample',
                  annot_fname='bla.annot', subjects_dir=subjects_dir)

    # read labels using hemi specification
    labels_lh = read_labels_from_annot('sample', hemi='lh',
                                       subjects_dir=subjects_dir)
    for label in labels_lh:
        assert_true(label.name.endswith('-lh'))
        assert_true(label.hemi == 'lh')
        # XXX fails on 2.6 for some reason...
        if sys.version_info[:2] > (2, 6):
            assert_is_not(label.color, None)

    # read labels using annot_fname
    annot_fname = op.join(subjects_dir, 'sample', 'label', 'rh.aparc.annot')
    labels_rh = read_labels_from_annot('sample', annot_fname=annot_fname,
                                       subjects_dir=subjects_dir)
    for label in labels_rh:
        assert_true(label.name.endswith('-rh'))
        assert_true(label.hemi == 'rh')
        assert_is_not(label.color, None)

    # combine the lh, rh, labels and sort them
    labels_lhrh = list()
    labels_lhrh.extend(labels_lh)
    labels_lhrh.extend(labels_rh)

    names = [label.name for label in labels_lhrh]
    labels_lhrh = [label for (name, label) in sorted(zip(names, labels_lhrh))]

    # read all labels at once
    labels_both = read_labels_from_annot('sample', subjects_dir=subjects_dir)

    # we have the same result
    _assert_labels_equal(labels_lhrh, labels_both)

    # aparc has 68 cortical labels
    assert_true(len(labels_both) == 68)

    # test regexp
    label = read_labels_from_annot('sample', parc='aparc.a2009s',
                                   regexp='Angu', subjects_dir=subjects_dir)[0]
    assert_true(label.name == 'G_pariet_inf-Angular-lh')
    # silly, but real regexp:
    label = read_labels_from_annot('sample', 'aparc.a2009s',
                                   regexp='.*-.{4,}_.{3,3}-L',
                                   subjects_dir=subjects_dir)[0]
    assert_true(label.name == 'G_oc-temp_med-Lingual-lh')
    assert_raises(RuntimeError, read_labels_from_annot, 'sample', parc='aparc',
                  annot_fname=annot_fname, regexp='JackTheRipper',
                  subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_read_labels_from_annot_annot2labels():
    """Test reading labels from parc. by comparing with mne_annot2labels
    """
    label_fnames = glob.glob(label_dir + '/*.label')
    label_fnames.sort()
    labels_mne = [read_label(fname) for fname in label_fnames]
    labels = read_labels_from_annot('sample', subjects_dir=subjects_dir)

    # we have the same result, mne does not fill pos, so ignore it
    _assert_labels_equal(labels, labels_mne, ignore_pos=True)


@testing.requires_testing_data
def test_write_labels_to_annot():
    """Test writing FreeSurfer parcellation from labels"""
    tempdir = _TempDir()

    labels = read_labels_from_annot('sample', subjects_dir=subjects_dir)

    # write left and right hemi labels:
    fnames = ['%s/%s-myparc' % (tempdir, hemi) for hemi in ['lh', 'rh']]

    for fname in fnames:
        write_labels_to_annot(labels, annot_fname=fname)

    # read it back
    labels2 = read_labels_from_annot('sample', subjects_dir=subjects_dir,
                                     annot_fname=fnames[0])
    labels22 = read_labels_from_annot('sample', subjects_dir=subjects_dir,
                                      annot_fname=fnames[1])
    labels2.extend(labels22)

    names = [label.name for label in labels2]

    for label in labels:
        idx = names.index(label.name)
        assert_labels_equal(label, labels2[idx])

    # same with label-internal colors
    for fname in fnames:
        write_labels_to_annot(labels, annot_fname=fname, overwrite=True)
    labels3 = read_labels_from_annot('sample', subjects_dir=subjects_dir,
                                     annot_fname=fnames[0])
    labels33 = read_labels_from_annot('sample', subjects_dir=subjects_dir,
                                      annot_fname=fnames[1])
    labels3.extend(labels33)
    names3 = [label.name for label in labels3]
    for label in labels:
        idx = names3.index(label.name)
        assert_labels_equal(label, labels3[idx])

    # make sure we can't overwrite things
    assert_raises(ValueError, write_labels_to_annot, labels,
                  annot_fname=fnames[0])

    # however, this works
    write_labels_to_annot(labels, annot_fname=fnames[0], overwrite=True)

    # label without color
    labels_ = labels[:]
    labels_[0] = labels_[0].copy()
    labels_[0].color = None
    write_labels_to_annot(labels_, annot_fname=fnames[0], overwrite=True)

    # duplicate color
    labels_[0].color = labels_[2].color
    assert_raises(ValueError, write_labels_to_annot, labels_,
                  annot_fname=fnames[0], overwrite=True)

    # invalid color inputs
    labels_[0].color = (1.1, 1., 1., 1.)
    assert_raises(ValueError, write_labels_to_annot, labels_,
                  annot_fname=fnames[0], overwrite=True)

    # overlapping labels
    labels_ = labels[:]
    cuneus_lh = labels[6]
    precuneus_lh = labels[50]
    labels_.append(precuneus_lh + cuneus_lh)
    assert_raises(ValueError, write_labels_to_annot, labels_,
                  annot_fname=fnames[0], overwrite=True)

    # unlabeled vertices
    labels_lh = [label for label in labels if label.name.endswith('lh')]
    write_labels_to_annot(labels_lh[1:], 'sample', annot_fname=fnames[0],
                          overwrite=True, subjects_dir=subjects_dir)
    labels_reloaded = read_labels_from_annot('sample', annot_fname=fnames[0],
                                             subjects_dir=subjects_dir)
    assert_equal(len(labels_lh), len(labels_reloaded))
    label0 = labels_lh[0]
    label1 = labels_reloaded[-1]
    assert_equal(label1.name, "unknown-lh")
    assert_true(np.all(in1d(label0.vertices, label1.vertices)))


@testing.requires_testing_data
def test_split_label():
    """Test splitting labels"""
    aparc = read_labels_from_annot('fsaverage', 'aparc', 'lh',
                                   regexp='lingual', subjects_dir=subjects_dir)
    lingual = aparc[0]

    # split with names
    parts = ('lingual_post', 'lingual_ant')
    post, ant = split_label(lingual, parts, subjects_dir=subjects_dir)

    # check output names
    assert_equal(post.name, parts[0])
    assert_equal(ant.name, parts[1])

    # check vertices add up
    lingual_reconst = post + ant
    lingual_reconst.name = lingual.name
    lingual_reconst.comment = lingual.comment
    lingual_reconst.color = lingual.color
    assert_labels_equal(lingual_reconst, lingual)

    # compare output of Label.split() method
    post1, ant1 = lingual.split(parts, subjects_dir=subjects_dir)
    assert_labels_equal(post1, post)
    assert_labels_equal(ant1, ant)

    # compare fs_like split with freesurfer split
    antmost = split_label(lingual, 40, None, subjects_dir, True)[-1]
    fs_vert = [210, 4401, 7405, 12079, 16276, 18956, 26356, 32713, 32716,
               32719, 36047, 36050, 42797, 42798, 42799, 59281, 59282, 59283,
               71864, 71865, 71866, 71874, 71883, 79901, 79903, 79910, 103024,
               107849, 107850, 122928, 139356, 139357, 139373, 139374, 139375,
               139376, 139377, 139378, 139381, 149117, 149118, 149120, 149127]
    assert_array_equal(antmost.vertices, fs_vert)

    # check default label name
    assert_equal(antmost.name, "lingual_div40-lh")


@slow_test
@testing.requires_testing_data
@requires_sklearn
def test_stc_to_label():
    """Test stc_to_label
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        src = read_source_spaces(fwd_fname)
    src_bad = read_source_spaces(src_bad_fname)
    stc = read_source_estimate(stc_fname, 'sample')
    os.environ['SUBJECTS_DIR'] = op.join(data_path, 'subjects')
    labels1 = _stc_to_label(stc, src='sample', smooth=3)
    labels2 = _stc_to_label(stc, src=src, smooth=3)
    assert_equal(len(labels1), len(labels2))
    for l1, l2 in zip(labels1, labels2):
        assert_labels_equal(l1, l2, decimal=4)

    with warnings.catch_warnings(record=True) as w:  # connectedness warning
        warnings.simplefilter('always')
        labels_lh, labels_rh = stc_to_label(stc, src=src, smooth=True,
                                            connected=True)

    assert_true(len(w) > 0)
    assert_raises(ValueError, stc_to_label, stc, 'sample', smooth=True,
                  connected=True)
    assert_raises(RuntimeError, stc_to_label, stc, smooth=True, src=src_bad,
                  connected=True)
    assert_equal(len(labels_lh), 1)
    assert_equal(len(labels_rh), 1)

    # test getting tris
    tris = labels_lh[0].get_tris(src[0]['use_tris'], vertices=stc.vertices[0])
    assert_raises(ValueError, spatial_tris_connectivity, tris,
                  remap_vertices=False)
    connectivity = spatial_tris_connectivity(tris, remap_vertices=True)
    assert_true(connectivity.shape[0] == len(stc.vertices[0]))

    # "src" as a subject name
    assert_raises(TypeError, stc_to_label, stc, src=1, smooth=False,
                  connected=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, stc_to_label, stc, src=SourceSpaces([src[0]]),
                  smooth=False, connected=False, subjects_dir=subjects_dir)
    assert_raises(ValueError, stc_to_label, stc, src='sample', smooth=False,
                  connected=True, subjects_dir=subjects_dir)
    assert_raises(ValueError, stc_to_label, stc, src='sample', smooth=True,
                  connected=False, subjects_dir=subjects_dir)
    labels_lh, labels_rh = stc_to_label(stc, src='sample', smooth=False,
                                        connected=False,
                                        subjects_dir=subjects_dir)
    assert_true(len(labels_lh) > 1)
    assert_true(len(labels_rh) > 1)

    # with smooth='patch'
    with warnings.catch_warnings(record=True) as w:  # connectedness warning
        warnings.simplefilter('always')
        labels_patch = stc_to_label(stc, src=src, smooth=True)
    assert_equal(len(w), 1)
    assert_equal(len(labels_patch), len(labels1))
    for l1, l2 in zip(labels1, labels2):
        assert_labels_equal(l1, l2, decimal=4)


@slow_test
@testing.requires_testing_data
def test_morph():
    """Test inter-subject label morphing
    """
    label_orig = read_label(real_label_fname)
    label_orig.subject = 'sample'
    # should work for specifying vertices for both hemis, or just the
    # hemi of the given label
    vals = list()
    for grade in [5, [np.arange(10242), np.arange(10242)], np.arange(10242)]:
        label = label_orig.copy()
        # this should throw an error because the label has all zero values
        assert_raises(ValueError, label.morph, 'sample', 'fsaverage')
        label.values.fill(1)
        label.morph(None, 'fsaverage', 5, grade, subjects_dir, 1,
                    copy=False)
        label.morph('fsaverage', 'sample', 5, None, subjects_dir, 2,
                    copy=False)
        assert_true(np.mean(in1d(label_orig.vertices, label.vertices)) == 1.0)
        assert_true(len(label.vertices) < 3 * len(label_orig.vertices))
        vals.append(label.vertices)
    assert_array_equal(vals[0], vals[1])
    # make sure label smoothing can run
    assert_equal(label.subject, 'sample')
    label.morph(label.subject, 'fsaverage', 5,
                [np.arange(10242), np.arange(10242)], subjects_dir, 2,
                copy=False)


@testing.requires_testing_data
def test_grow_labels():
    """Test generation of circular source labels"""
    seeds = [0, 50000]
    # these were chosen manually in mne_analyze
    should_be_in = [[49, 227], [51207, 48794]]
    hemis = [0, 1]
    names = ['aneurism', 'tumor']
    labels = grow_labels('sample', seeds, 3, hemis, subjects_dir, names=names)

    tgt_names = ['aneurism-lh', 'tumor-rh']
    tgt_hemis = ['lh', 'rh']
    for label, seed, hemi, sh, name in zip(labels, seeds, tgt_hemis,
                                           should_be_in, tgt_names):
        assert_true(np.any(label.vertices == seed))
        assert_true(np.all(in1d(sh, label.vertices)))
        assert_equal(label.hemi, hemi)
        assert_equal(label.name, name)

    # grow labels with and without overlap
    seeds = [57532, [58887, 6304]]
    l01, l02 = grow_labels('fsaverage', seeds, 20, [0, 0], subjects_dir)
    seeds = [57532, [58887, 6304]]
    l11, l12 = grow_labels('fsaverage', seeds, 20, [0, 0], subjects_dir,
                           overlap=False)

    # test label naming
    assert_equal(l01.name, 'Label_0-lh')
    assert_equal(l02.name, 'Label_1-lh')
    assert_equal(l11.name, 'Label_0-lh')
    assert_equal(l12.name, 'Label_1-lh')

    # make sure set 1 does not overlap
    overlap = np.intersect1d(l11.vertices, l12.vertices, True)
    assert_array_equal(overlap, [])

    # make sure both sets cover the same vertices
    l0 = l01 + l02
    l1 = l11 + l12
    assert_array_equal(l1.vertices, l0.vertices)


run_tests_if_main()
