# -*- coding: UTF-8 -*-
# Authors: Thomas Hartmann <thomas.hartmann@th-ht.de>
#          Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
# License: BSD (3-clause)
import types
import numpy as np
import copy

info_ignored_fields = ('file_id', 'hpi_results', 'hpi_meas', 'meas_id',
                       'meas_date', 'highpass', 'lowpass', 'subject_info',
                       'hpi_subsystem', 'experimenter', 'description',
                       'proj_id', 'proj_name', 'line_freq', 'gantry_angle',
                       'dev_head_t', 'dig', 'bads', 'projs')

ch_ignore_fields = ('logno', 'cal', 'range', 'loc', 'coord_frame') # TODO: remove loc and coord_frame from here


def _remove_ignored_ch_fields(info):
    if 'chs' in info:
        for cur_ch in info['chs']:
            for cur_field in ch_ignore_fields:
                if cur_field in cur_ch:
                    del cur_ch[cur_field]


def _remove_ignored_info_fields(info):
    for cur_field in info_ignored_fields:
        if cur_field in info:
            del info[cur_field]

    _remove_ignored_ch_fields(info)


def check_info_fields(expected, actual):
    """
    Check if info fields are equal.

    Some fields are ignored.
    """

    expected = copy.deepcopy(expected.info)
    actual = copy.deepcopy(actual.info)

    _remove_ignored_info_fields(expected)
    _remove_ignored_info_fields(actual)

    assert_deep_almost_equal(expected, actual)


def assert_deep_almost_equal(expected, actual, *args, **kwargs):
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
    is_root = '__trace' not in kwargs
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
                assert_deep_almost_equal(v1, v2,
                                         __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            np.testing.assert_equal(set(expected), set(actual))
            for key in expected:
                assert_deep_almost_equal(expected[key], actual[key],
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
