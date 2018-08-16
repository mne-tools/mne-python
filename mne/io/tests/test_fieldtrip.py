# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)

import mne
import numpy as np
import os.path
import types
from mne.datasets import testing
from mne.utils import requires_h5py


@testing.requires_testing_data
@requires_h5py
def test_whole_process_old():
    """Test the reader functions for FieldTrip data."""
    test_data_folder = os.path.join(mne.datasets.testing.data_path(),
                                    'fieldtrip/old')
    all_versions = ['v7', 'v73']
    for version in all_versions:
        f_name_raw = os.path.join(test_data_folder, 'raw_%s.mat' % (version,))
        f_name_epoched = os.path.join(test_data_folder,
                                      'epoched_%s.mat' % (version,))
        f_name_avg = os.path.join(test_data_folder,
                                  'averaged_%s.mat' % (version,))
        f_name_events = os.path.join(test_data_folder, 'events.eve')

        # load everything
        data_raw = mne.io.read_raw_fieldtrip(f_name_raw, data_name='data')
        data_epoched = mne.io.read_epochs_fieldtrip(f_name_epoched,
                                                    data_name='data_epoched')
        data_avg = mne.io.read_evoked_fieldtrip(f_name_avg,
                                                data_name='data_avg')
        events = mne.read_events(f_name_events)

        mne_epoched = mne.Epochs(data_raw, events, tmin=-0.05, tmax=0.05,
                                 preload=True, baseline=None)
        np.testing.assert_almost_equal(data_epoched.get_data(),
                                       mne_epoched.get_data()[:, :, :-1])

        mne_avg = mne_epoched.average(
            picks=np.arange(0, len(mne_epoched.ch_names)))
        np.testing.assert_almost_equal(data_avg.data, mne_avg.data[:, :-1])


@testing.requires_testing_data
@requires_h5py
def test_raw():
    """Test comparing reading a raw fiff file and the FieldTrip version."""
    test_data_folder_ft = os.path.join(mne.datasets.testing.data_path(),
                                       'fieldtrip/from_mne_sample')
    raw_fiff_file = os.path.join(mne.datasets.testing.data_path(),
                                 'MEG/sample', 'sample_audvis_trunc_raw.fif')

    # Load the raw fiff file with mne
    raw_fiff_mne = mne.io.read_raw_fif(raw_fiff_file, preload=True)

    all_versions = ['v7', 'v73']

    for version in all_versions:
        cur_fname = os.path.join(test_data_folder_ft,
                                 'raw_%s.mat' % (version, ))
        raw_fiff_ft = mne.io.read_raw_fieldtrip(cur_fname)

        # Check that the data was loaded correctly
        np.testing.assert_almost_equal(raw_fiff_mne.get_data(),
                                       raw_fiff_ft.get_data())
        pass


def _assert_deep_almost_equal(expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.

    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).

    This code has been adapted from
    https://github.com/larsbutler/oq-engine/blob/master/tests/utils/helpers.py
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')

    if isinstance(expected, np.ndarray) and expected.size == 0:
        expected = None

    if isinstance(actual, np.ndarray) and actual.size == 0:
        actual = None

    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_almost_equal(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray,
                                   types.GeneratorType)):
            if isinstance(expected, types.GeneratorType):
                expected = list(expected)
                actual = list(actual)

                np.testing.assert_equal(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                _assert_deep_almost_equal(v1, v2,
                                          __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            np.testing.assert_equal(set(expected), set(actual))
            for key in expected:
                _assert_deep_almost_equal(expected[key], actual[key],
                                          __trace=repr(key), *args, **kwargs)
        else:
            np.testing.assert_equal(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            message = ''
            try:
                message = exc.message
            except AttributeError:
                pass
            exc = AssertionError("%s\nTRACE: %s" % (message, trace))
        raise exc