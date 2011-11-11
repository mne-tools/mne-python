import os
import os.path as op
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true

from ..datasets import sample
from .. import label_time_courses, read_label, write_label, stc_to_label, \
               SourceEstimate, read_source_spaces


examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
stc_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')
label = 'Aud-lh'
label_fname = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)
src_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-eeg-oct-6p-fwd.fif')


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
    write_label('foo', label)
    label2 = read_label('foo-lh.label')

    for key in label.keys():
        if key in ['comment', 'hemi']:
            assert_true(label[key] == label2[key])
        else:
            assert_array_almost_equal(label[key], label2[key], 5)


def test_stc_to_label():
    """Test stc_to_label
    """
    src = read_source_spaces(src_fname)
    stc = SourceEstimate(stc_fname)
    os.environ['SUBJECTS_DIR'] = op.join(data_path, 'subjects')
    labels1 = stc_to_label(stc, src='sample', smooth=3)
    labels2 = stc_to_label(stc, src=src, smooth=3)
    assert_true(len(labels1) == len(labels2))
    for l1, l2 in zip(labels1, labels2):
        for key in l1.keys():
            if key in ['comment', 'hemi']:
                assert_true(l1[key] == l1[key])
            else:
                assert_array_almost_equal(l1[key], l2[key], 4)
