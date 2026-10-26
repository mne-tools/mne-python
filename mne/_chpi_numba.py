"""Numba-accelerated helpers for :mod:`mne.chpi`.

Kept in its own module so that ``import mne.chpi`` does not import numba; they live
together because numba can only call other jitted functions in nopython mode.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ._numba import jit
from ._transforms_numba import _quat_to_rot


@jit()
def _magnetic_dipole_delta(fwd, whitener, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    fwd = fwd @ whitener.T
    u, s, v = np.linalg.svd(fwd, full_matrices=False)
    one = v @ B
    Bm2 = one @ one
    return B2 - Bm2, u, s, one


@jit()
def _chpi_objective(x, coil_dev_rrs, coil_head_rrs, weights):
    """Compute objective function."""
    d = coil_dev_rrs @ _quat_to_rot(x[:3]).T
    d += x[3:]
    d -= coil_head_rrs
    d *= d
    return d.sum(axis=1) @ weights  # sum over coils, weighted


@jit()
def _reorder_inv_model(inv_model, n_freqs):
    # Reorder for faster computation
    idx = np.arange(2 * n_freqs).reshape(2, n_freqs).T.ravel()
    return inv_model[idx]


@jit()
def _fast_fit(this_data, proj, n_freqs, model, inv_model_reord):
    # first or last window
    if this_data.shape[1] != model.shape[0]:
        model = model[: this_data.shape[1]]
        inv_model_reord = _reorder_inv_model(np.linalg.pinv(model), n_freqs)
    proj_data = proj @ this_data
    X = inv_model_reord @ proj_data.T

    sin_fit = np.zeros((n_freqs, X.shape[1]))
    for fi in range(n_freqs):
        # use SVD across all sensors to estimate the sinusoid phase
        u, s, vt = np.linalg.svd(X[2 * fi : 2 * fi + 2], full_matrices=False)
        # the first component holds the predominant phase direction
        # (so ignore the second, effectively doing s[1] = 0):
        sin_fit[fi] = vt[0] * s[0]
    return sin_fit


@jit()
def _fast_fit_snr(this_data, n_freqs, model, inv_model, mag_picks, grad_picks):
    # first or last window
    if this_data.shape[1] != model.shape[0]:
        model = model[: this_data.shape[1]]
        inv_model = np.linalg.pinv(model)
    coefs = np.ascontiguousarray(inv_model) @ np.ascontiguousarray(this_data.T)
    # average sin & cos terms (special property of sinusoids: power=A²/2)
    hpi_power = (coefs[:n_freqs] ** 2 + coefs[n_freqs : (2 * n_freqs)] ** 2) / 2
    resid = this_data - np.ascontiguousarray((model @ coefs).T)
    # can't use np.var(..., axis=1) with Numba, so do it manually:
    resid_mean = np.atleast_2d(resid.sum(axis=1) / resid.shape[1]).T
    squared_devs = np.abs(resid - resid_mean) ** 2
    resid_var = squared_devs.sum(axis=1) / squared_devs.shape[1]
    # output array will be (n_freqs, 3 * n_ch_types). The 3 columns for each
    # channel type are the SNR, the mean cHPI power and the residual variance
    # (which gets tiled to shape (n_freqs,) because it's a scalar).
    snrs = np.empty((n_freqs, 0))
    # average power & compute residual variance separately for each ch type
    for _picks in (mag_picks, grad_picks):
        if len(_picks):
            avg_power = hpi_power[:, _picks].sum(axis=1) / len(_picks)
            avg_resid = resid_var[_picks].mean() * np.ones(n_freqs)
            snr = 10 * np.log10(avg_power / avg_resid)
            snrs = np.hstack((snrs, np.stack((snr, avg_power, avg_resid), 1)))
    return snrs
