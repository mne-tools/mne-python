# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

from datetime import datetime
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
import os.path as op

import numpy as np

from mne.io import Raw
from mne.annotations import Annotations, _combine_annotations
from mne.datasets import testing

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_annotations():
    """Test annotation class."""
    raw = Raw(fif_fname)
    onset = list(range(10))
    duration = np.ones(10)
    description = np.repeat('test', 10)
    dt = datetime.utcnow()
    meas_date = raw.info['meas_date']
    # Test time shifts.
    for orig_time in [None, dt, meas_date[0], meas_date]:
        annot = Annotations(onset, duration, description, orig_time)

    assert_raises(ValueError, Annotations, onset, duration, description[:9])
    assert_raises(ValueError, Annotations, [onset, 1], duration, description)
    assert_raises(ValueError, Annotations, onset, [duration, 1], description)

    # Test combining annotations
    annot1 = Annotations(onset, duration, description, dt)
    annot2 = Annotations(onset, duration * 5, description, None)
    last_samps = np.repeat(raw.last_samp, 2)
    sfreq = raw.info['sfreq']
    annot = _combine_annotations(np.array([annot1, annot2]), last_samps, sfreq)
    assert_array_equal(annot1.onset, annot.onset[:10])
    assert_array_equal(annot2.duration, annot.duration[10:])
    assert_array_equal(annot2.onset + raw.last_samp / sfreq, annot.onset[10:])
