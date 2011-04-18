import os.path as op

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
stc_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')
label = 'Aud-lh'
label_fname = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)


def test_label_io_and_time_course_estimates():
    """Test IO for STC files
    """

    values, times, vertices = mne.label_time_courses(label_fname, stc_fname)

    assert len(times) == values.shape[1]
    assert len(vertices) == values.shape[0]
