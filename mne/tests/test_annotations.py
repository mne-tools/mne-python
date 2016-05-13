# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

from datetime import datetime
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
import os.path as op

import numpy as np

from mne.utils import run_tests_if_main
from mne.io import Raw, concatenate_raws
from mne.annotations import Annotations
from mne.datasets import testing

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_annotations():
    """Test annotation class."""
    raw = Raw(fif_fname)
    onset = np.array(range(10))
    duration = np.ones(10) + raw.first_samp
    description = np.repeat('test', 10)
    dt = datetime.utcnow()
    meas_date = raw.info['meas_date']
    # Test time shifts.
    for orig_time in [None, dt, meas_date[0], meas_date]:
        annot = Annotations(onset, duration, description, orig_time)

    assert_raises(ValueError, Annotations, onset, duration, description[:9])
    assert_raises(ValueError, Annotations, [onset, 1], duration, description)
    assert_raises(ValueError, Annotations, onset, [duration, 1], description)

    # Test combining annotations with concatenate_raws
    annot = Annotations(onset, duration, description, dt)
    sfreq = raw.info['sfreq']
    raw2 = raw.copy()
    raw2.annotations = annot
    concatenate_raws([raw, raw2])
    assert_array_equal(annot.onset, raw.annotations.onset)
    assert_array_equal(annot.duration, raw.annotations.duration)

    raw2.annotations = Annotations(onset, duration * 2, description, None)
    last_samp = raw.last_samp - 1
    concatenate_raws([raw, raw2])
    onsets = np.concatenate([onset,
                             onset + (last_samp - raw.first_samp) / sfreq])
    assert_array_equal(raw.annotations.onset, onsets)
    assert_array_equal(raw.annotations.onset[:10], onset)
    assert_array_equal(raw.annotations.duration[:10], duration)
    assert_array_equal(raw.annotations.duration[10:], duration * 2)
    assert_array_equal(raw.annotations.description, np.repeat('test', 20))

run_tests_if_main()
