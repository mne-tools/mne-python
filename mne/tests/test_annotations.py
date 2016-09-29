# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

from datetime import datetime
from nose.tools import assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import os.path as op

import numpy as np

from mne import create_info
from mne.utils import run_tests_if_main
from mne.io import read_raw_fif, RawArray, concatenate_raws
from mne.annotations import Annotations
from mne.datasets import testing

data_dir = op.join(testing.data_path(download=False), 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


@testing.requires_testing_data
def test_annotations():
    """Test annotation class."""
    raw = read_raw_fif(fif_fname, add_eeg_ref=False)
    onset = np.array(range(10))
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

    # Test combining annotations with concatenate_raws
    raw2 = raw.copy()
    orig_time = (meas_date[0] + meas_date[1] * 0.000001 +
                 raw2.first_samp / raw2.info['sfreq'])
    annot = Annotations(onset, duration, description, orig_time)
    raw2.annotations = annot
    assert_array_equal(raw2.annotations.onset, onset)
    concatenate_raws([raw, raw2])
    assert_array_almost_equal(onset + 20., raw.annotations.onset, decimal=2)
    assert_array_equal(annot.duration, raw.annotations.duration)
    assert_array_equal(raw.annotations.description, np.repeat('test', 10))

    # Test combining with RawArray and orig_times
    data = np.random.randn(2, 1000) * 10e-12
    sfreq = 100.
    info = create_info(ch_names=['MEG1', 'MEG2'], ch_types=['grad'] * 2,
                       sfreq=sfreq)
    info['meas_date'] = 0
    raws = []
    for i, fs in enumerate([1000, 100, 12]):
        raw = RawArray(data.copy(), info, first_samp=fs)
        ants = Annotations([1., 2.], [.5, .5], 'x', fs / sfreq)
        raw.annotations = ants
        raws.append(raw)
    raw = concatenate_raws(raws)
    assert_array_equal(raw.annotations.onset, [1., 2., 11., 12., 21., 22.])


run_tests_if_main()
