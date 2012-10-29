import os
import os.path as op
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne import label_time_courses, read_label, write_label, stc_to_label, \
               read_source_estimate, read_source_spaces, grow_labels
from mne.label import Label


examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
stc_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')
label = 'Aud-lh'
label_fname = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)
label_rh_fname = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-rh.label')
src_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-eeg-oct-6p-fwd.fif')


def assert_labels_equal(l0, l1, decimal=5):
    for attr in ['comment', 'hemi']:
        assert_true(getattr(l0, attr) == getattr(l1, attr))
    for attr in ['vertices', 'pos', 'values']:
        a0 = getattr(l0, attr)
        a1 = getattr(l1, attr)
        assert_array_almost_equal(a0, a1, decimal)


def test_label_addition():
    """Test label addition
    """
    pos = np.random.rand(10, 3)
    values = np.arange(10.) / 10
    idx0 = range(7)
    idx1 = range(7, 10) # non-overlapping
    idx2 = range(5, 10) # overlapping
    l0 = Label(idx0, pos[idx0], values[idx0], 'lh')
    l1 = Label(idx1, pos[idx1], values[idx1], 'lh')
    l2 = Label(idx2, pos[idx2], values[idx2], 'lh')

    assert len(l0) == len(idx0)

    # adding non-overlapping labels
    l01 = l0 + l1
    assert len(l01) == len(l0) + len(l1)
    assert_array_equal(l01.values[:len(l0)], l0.values)

    # adding overlappig labels
    l = l0 + l2
    i0 = np.where(l0.vertices == 6)[0][0]
    i2 = np.where(l2.vertices == 6)[0][0]
    i = np.where(l.vertices == 6)[0][0]
    assert l.values[i] == l0.values[i0] + l2.values[i2]
    assert l.values[0] == l0.values[0]
    assert_array_equal(np.unique(l.vertices), np.unique(idx0 + idx2))

    # adding lh and rh
    l2.hemi = 'rh'
    # this now has deprecated behavior
    bhl = l0 + l2
    assert bhl.hemi == 'both'
    assert len(bhl) == len(l0) + len(l2)

    bhl = l1 + bhl
    assert_labels_equal(bhl.lh, l01)


def test_label_io_and_time_course_estimates():
    """Test IO for label + stc files
    """

    values, times, vertices = label_time_courses(label_fname, stc_fname)

    assert_true(len(times) == values.shape[1])
    assert_true(len(vertices) == values.shape[0])


def test_label_io():
    """Test IO of label files
    """
    label = read_label(label_fname)
    label.save('foo')
    label2 = read_label('foo-lh.label')
    assert_labels_equal(label, label2)


def test_stc_to_label():
    """Test stc_to_label
    """
    src = read_source_spaces(src_fname)
    stc = read_source_estimate(stc_fname)
    os.environ['SUBJECTS_DIR'] = op.join(data_path, 'subjects')
    labels1 = stc_to_label(stc, src='sample', smooth=3)
    labels2 = stc_to_label(stc, src=src, smooth=3)
    assert_true(len(labels1) == len(labels2))
    for l1, l2 in zip(labels1, labels2):
        assert_labels_equal(l1, l2, decimal=4)


def test_grow_labels():
    """Test generation of circular source labels"""
    seeds = [0, 50000]
    hemis = [0, 1]
    labels = grow_labels('sample', seeds, 3, hemis)

    for label, seed, hemi in zip(labels, seeds, hemis):
        assert(np.any(label.vertices == seed))
        if hemi == 0:
            assert(label.hemi == 'lh')
        else:
            assert(label.hemi == 'rh')


def test_label_time_course():
    """Test extracting label data from SourceEstimate"""
    values, times, vertices = label_time_courses(label_fname, stc_fname)
    stc = read_source_estimate(stc_fname)
    label_lh = read_label(label_fname)
    stc_lh = stc.label_stc(label_lh)
    assert_array_almost_equal(stc_lh.data, values)
    assert_array_almost_equal(stc_lh.times, times)
    assert_array_almost_equal(stc_lh.vertno[0], vertices)

    label_rh = read_label(label_rh_fname)
    stc_rh = stc.label_stc(label_rh)
    label_bh = label_rh + label_lh
    stc_bh = stc.label_stc(label_bh)
    assert_array_equal(stc_bh.data, np.vstack((stc_lh.data, stc_rh.data)))
