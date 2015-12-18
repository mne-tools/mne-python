# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from .. import pick_types, Evoked
from ..io import _BaseRaw
from ..io.constants import FIFF
from ..bem import fit_sphere_to_headshape


def _get_data(x, ch_idx):
    """Helper to get the (n_ch, n_times) data array"""
    if isinstance(x, _BaseRaw):
        return x[ch_idx][0]
    elif isinstance(x, Evoked):
        return x.data[ch_idx]


def assert_meg_snr(actual, desired, min_tol, med_tol=500., msg=None):
    """Helper to assert channel SNR of a certain level

    Mostly useful for operations like Maxwell filtering that modify
    MEG channels while leaving EEG and others intact.
    """
    from nose.tools import assert_true
    picks = pick_types(desired.info, meg=True, exclude=[])
    others = np.setdiff1d(np.arange(len(actual.ch_names)), picks)
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(_get_data(actual, others),
                        _get_data(desired, others), atol=1e-11, rtol=1e-5,
                        err_msg='non-MEG channel mismatch')
    actual_data = _get_data(actual, picks)
    desired_data = _get_data(desired, picks)
    bench_rms = np.sqrt(np.mean(desired_data * desired_data, axis=1))
    error = actual_data - desired_data
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    snrs = bench_rms / error_rms
    # min tol
    snr = snrs.min()
    bad_count = (snrs < min_tol).sum()
    msg = ' (%s)' % msg if msg != '' else msg
    assert_true(bad_count == 0, 'SNR (worst %0.2f) < %0.2f for %s/%s '
                'channels%s' % (snr, min_tol, bad_count, len(picks), msg))
    # median tol
    snr = np.median(snrs)
    assert_true(snr >= med_tol, 'SNR median %0.2f < %0.2f%s'
                % (snr, med_tol, msg))


def _dig_sort_key(dig):
    """Helper for sorting"""
    return 10000 * dig['kind'] + dig['ident']


def assert_dig_allclose(info_py, info_bin):
    # test dig positions
    dig_py = sorted(info_py['dig'], key=_dig_sort_key)
    dig_bin = sorted(info_bin['dig'], key=_dig_sort_key)
    assert_equal(len(dig_py), len(dig_bin))
    for ii, (d_py, d_bin) in enumerate(zip(dig_py, dig_bin)):
        for key in ('ident', 'kind', 'coord_frame'):
            assert_equal(d_py[key], d_bin[key])
        assert_allclose(d_py['r'], d_bin['r'], rtol=1e-5, atol=1e-5,
                        err_msg='Failure on %s:\n%s\n%s'
                        % (ii, d_py['r'], d_bin['r']))
    if any(d['kind'] == FIFF.FIFFV_POINT_EXTRA for d in dig_py):
        R_bin, o_head_bin, o_dev_bin = fit_sphere_to_headshape(info_bin)
        R_py, o_head_py, o_dev_py = fit_sphere_to_headshape(info_py)
        assert_allclose(R_py, R_bin)
        assert_allclose(o_dev_py, o_dev_bin, rtol=1e-5, atol=1e-3)  # mm
        assert_allclose(o_head_py, o_head_bin, rtol=1e-5, atol=1e-3)  # mm
