# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal

from scipy import linalg

from .. import pick_types, Evoked
from ..io import BaseRaw
from ..io.constants import FIFF
from ..bem import fit_sphere_to_headshape


def _get_data(x, ch_idx):
    """Helper to get the (n_ch, n_times) data array"""
    if isinstance(x, BaseRaw):
        return x[ch_idx][0]
    elif isinstance(x, Evoked):
        return x.data[ch_idx]


def _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind='MEG'):
    """Helper to check the SNR of a set of channels"""
    from nose.tools import assert_true
    actual_data = _get_data(actual, picks)
    desired_data = _get_data(desired, picks)
    bench_rms = np.sqrt(np.mean(desired_data * desired_data, axis=1))
    error = actual_data - desired_data
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    np.clip(error_rms, 1e-60, np.inf, out=error_rms)  # avoid division by zero
    snrs = bench_rms / error_rms
    # min tol
    snr = snrs.min()
    bad_count = (snrs < min_tol).sum()
    msg = ' (%s)' % msg if msg != '' else msg
    assert_true(bad_count == 0, 'SNR (worst %0.2f) < %0.2f for %s/%s '
                'channels%s' % (snr, min_tol, bad_count, len(picks), msg))
    # median tol
    snr = np.median(snrs)
    assert_true(snr >= med_tol, '%s SNR median %0.2f < %0.2f%s'
                % (kind, snr, med_tol, msg))


def assert_meg_snr(actual, desired, min_tol, med_tol=500., chpi_med_tol=500.,
                   msg=None):
    """Helper to assert channel SNR of a certain level

    Mostly useful for operations like Maxwell filtering that modify
    MEG channels while leaving EEG and others intact.
    """
    picks = pick_types(desired.info, meg=True, exclude=[])
    picks_desired = pick_types(desired.info, meg=True, exclude=[])
    assert_array_equal(picks, picks_desired, err_msg='MEG pick mismatch')
    chpis = pick_types(actual.info, meg=False, chpi=True, exclude=[])
    chpis_desired = pick_types(desired.info, meg=False, chpi=True, exclude=[])
    if chpi_med_tol is not None:
        assert_array_equal(chpis, chpis_desired, err_msg='cHPI pick mismatch')
    others = np.setdiff1d(np.arange(len(actual.ch_names)),
                          np.concatenate([picks, chpis]))
    others_desired = np.setdiff1d(np.arange(len(desired.ch_names)),
                                  np.concatenate([picks_desired,
                                                  chpis_desired]))
    assert_array_equal(others, others_desired, err_msg='Other pick mismatch')
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(_get_data(actual, others),
                        _get_data(desired, others), atol=1e-11, rtol=1e-5,
                        err_msg='non-MEG channel mismatch')
    _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind='MEG')
    if chpi_med_tol is not None and len(chpis) > 0:
        _check_snr(actual, desired, chpis, 0., chpi_med_tol, msg, kind='cHPI')


def assert_snr(actual, desired, tol):
    """Assert actual and desired arrays are within some SNR tolerance"""
    from nose.tools import assert_true
    snr = (linalg.norm(desired, ord='fro') /
           linalg.norm(desired - actual, ord='fro'))
    assert_true(snr >= tol, msg='%f < %f' % (snr, tol))


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
        r_bin, o_head_bin, o_dev_bin = fit_sphere_to_headshape(
            info_bin, units='m', verbose='error')
        r_py, o_head_py, o_dev_py = fit_sphere_to_headshape(
            info_py, units='m', verbose='error')
        assert_allclose(r_py, r_bin, atol=1e-6)
        assert_allclose(o_dev_py, o_dev_bin, rtol=1e-5, atol=1e-6)
        assert_allclose(o_head_py, o_head_bin, rtol=1e-5, atol=1e-6)


def assert_naming(warns, fname, n_warn):
    """Assert a non-standard naming scheme was used while saving or loading

    Parameters
    ----------
    warns : list
        List of warnings from ``warnings.catch_warnings(record=True)``.
    fname : str
        Filename that should appear in the warning message.
    n_warn : int
        Number of warnings that should have naming convention errors.
    """
    from nose.tools import assert_true
    assert_true(sum('naming conventions' in str(ww.message)
                    for ww in warns) == n_warn)
    # check proper stacklevel reporting
    for ww in warns:
        if 'naming conventions' in str(ww.message):
            assert_true(fname in ww.filename,
                        msg='"%s" not in "%s"' % (fname, ww.filename))
