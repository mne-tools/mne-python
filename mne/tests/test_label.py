import os
import os.path as op
from ..externals.six.moves import cPickle as pickle
import glob
import warnings

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_equal, assert_true, assert_raises

from mne.datasets import sample
from mne import (label_time_courses, read_label, stc_to_label,
                 read_source_estimate, read_source_spaces, grow_labels,
                 read_labels_from_annot, write_labels_to_annot, split_label)
from mne.label import Label, _blend_colors
from mne.utils import requires_mne, run_subprocess, _TempDir, requires_sklearn
from mne.fixes import digitize, in1d, assert_is, assert_is_not

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
src_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
stc_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')
real_label_fname = op.join(data_path, 'MEG', 'sample', 'labels',
                           'Aud-lh.label')
real_label_rh_fname = op.join(data_path, 'MEG', 'sample', 'labels',
                              'Aud-rh.label')
v1_label_fname = op.join(subjects_dir, 'sample', 'label', 'lh.V1.label')

fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-eeg-oct-6p-fwd.fif')
src_bad_fname = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                        'fsaverage-ico-5-src.fif')

test_path = op.join(op.split(__file__)[0], '..', 'io', 'tests', 'data')
label_fname = op.join(test_path, 'test-lh.label')
label_rh_fname = op.join(test_path, 'test-rh.label')
tempdir = _TempDir()

# This code was used to generate the "fake" test labels:
# for hemi in ['lh', 'rh']:
#    label = Label(np.unique((np.random.rand(100) * 10242).astype(int)),
#                  hemi=hemi, comment='Test ' + hemi, subject='fsaverage')
#    label.save(op.join(test_path, 'test-%s.label' % hemi))


def assert_labels_equal(l0, l1, decimal=5):
    for attr in ['comment', 'hemi', 'subject', 'color']:
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

    bhl2 = l1 + bhl
    assert_labels_equal(bhl2.lh, l01)
    assert_equal(bhl2.color, _blend_colors(l1.color, bhl.color))


@sample.requires_sample_data
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


@sample.requires_sample_data
def test_label_io_and_time_course_estimates():
    """Test IO for label + stc files
    """
    values, times, vertices = label_time_courses(real_label_fname, stc_fname)
    assert_true(len(times) == values.shape[1])
    assert_true(len(vertices) == values.shape[0])


def test_label_io():
    """Test IO of label files
    """
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


@sample.requires_sample_data
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


@sample.requires_sample_data
@requires_mne
def test_read_labels_from_annot_annot2labels():
    """Test reading labels from parc. by comparing with mne_annot2labels
    """

    def _mne_annot2labels(subject, subjects_dir, parc):
        """Get labels using mne_annot2lables"""
        label_dir = _TempDir()
        cwd = os.getcwd()
        try:
            os.chdir(label_dir)
            env = os.environ.copy()
            env['SUBJECTS_DIR'] = subjects_dir
            cmd = ['mne_annot2labels', '--subject', subject, '--parc', parc]
            run_subprocess(cmd, env=env)
            label_fnames = glob.glob(label_dir + '/*.label')
            label_fnames.sort()
            labels = [read_label(fname) for fname in label_fnames]
        finally:
            del label_dir
            os.chdir(cwd)

        return labels

    labels = read_labels_from_annot('sample', subjects_dir=subjects_dir)
    labels_mne = _mne_annot2labels('sample', subjects_dir, 'aparc')

    # we have the same result, mne does not fill pos, so ignore it
    _assert_labels_equal(labels, labels_mne, ignore_pos=True)


@sample.requires_sample_data
def test_write_labels_to_annot():
    """Test writing FreeSurfer parcellation from labels"""

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


@sample.requires_sample_data
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


@sample.requires_sample_data
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
    with warnings.catch_warnings(record=True) as w:  # connectedness warning
        warnings.simplefilter('always')
        labels1 = stc_to_label(stc, src='sample', smooth=3)
        labels2 = stc_to_label(stc, src=src, smooth=3)
    assert_true(len(w) > 0)
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

    # with smooth='patch'
    with warnings.catch_warnings(record=True) as w:  # connectedness warning
        warnings.simplefilter('always')
        labels_patch = stc_to_label(stc, src=src, smooth=True)
    assert_equal(len(w), 1)
    assert_equal(len(labels_patch), len(labels1))
    for l1, l2 in zip(labels1, labels2):
        assert_labels_equal(l1, l2, decimal=4)


@sample.requires_sample_data
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
        label.morph(None, 'fsaverage', 5, grade, subjects_dir, 2,
                    copy=False)
        label.morph('fsaverage', 'sample', 5, None, subjects_dir, 2,
                    copy=False)
        assert_true(np.mean(in1d(label_orig.vertices, label.vertices)) == 1.0)
        assert_true(len(label.vertices) < 3 * len(label_orig.vertices))
        vals.append(label.vertices)
    assert_array_equal(vals[0], vals[1])
    # make sure label smoothing can run
    label.morph(label.subject, 'fsaverage', 5,
                [np.arange(10242), np.arange(10242)], subjects_dir, 2,
                copy=False)
    # subject name should be inferred now
    label.smooth(subjects_dir=subjects_dir)


@sample.requires_sample_data
def test_grow_labels():
    """Test generation of circular source labels"""
    seeds = [0, 50000]
    # these were chosen manually in mne_analyze
    should_be_in = [[49, 227], [51207, 48794]]
    hemis = [0, 1]
    names = ['aneurism', 'tumor']
    labels = grow_labels('sample', seeds, 3, hemis, subjects_dir, n_jobs=2,
                         names=names)

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


@sample.requires_sample_data
def test_label_time_course():
    """Test extracting label data from SourceEstimate"""
    values, times, vertices = label_time_courses(real_label_fname, stc_fname)
    stc = read_source_estimate(stc_fname)
    label_lh = read_label(real_label_fname)
    stc_lh = stc.in_label(label_lh)
    assert_array_almost_equal(stc_lh.data, values)
    assert_array_almost_equal(stc_lh.times, times)
    assert_array_almost_equal(stc_lh.vertno[0], vertices)

    label_rh = read_label(real_label_rh_fname)
    stc_rh = stc.in_label(label_rh)
    label_bh = label_rh + label_lh
    stc_bh = stc.in_label(label_bh)
    assert_array_equal(stc_bh.data, np.vstack((stc_lh.data, stc_rh.data)))
