# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import math

import numpy as np
from scipy.signal import lfilter

from .._fiff.pick import pick_info
from ..cov import Covariance, compute_whitener
from ..epochs import BaseEpochs
from ..evoked import Evoked
from ..forward import apply_forward
from ..io import BaseRaw
from ..utils import (
    _check_preload,
    _legacy_rng,
    _validate_type,
    check_random_state,
    logger,
    verbose,
)


@_legacy_rng("random_state")
@verbose
def simulate_evoked(
    fwd,
    stc,
    info,
    cov=None,
    nave=30,
    iir_filter=None,
    use_cps=True,
    verbose=None,
    *,
    rng=None,
    random_state=None,
):
    """Generate noisy evoked data.

    .. note:: No projections from ``info`` will be present in the
              output ``evoked``. You can use e.g.
              :func:`evoked.add_proj <mne.Evoked.add_proj>` or
              :func:`evoked.set_eeg_reference <mne.Evoked.set_eeg_reference>`
              to add them afterward as necessary.

    Parameters
    ----------
    fwd : instance of Forward
        A forward solution.
    stc : SourceEstimate object
        The source time courses.
    %(info_not_none)s Used to generate the evoked.
    cov : Covariance object | None
        The noise covariance. If None, no noise is added.
    nave : int
        Number of averaged epochs (defaults to 30).

        .. versionadded:: 0.15.0
    iir_filter : None | array
        IIR filter coefficients (denominator) e.g. [1, -1, 0.2].
    %(use_cps)s

        .. versionadded:: 0.15
    %(verbose)s
    %(rng)s
    %(random_state_rng)s

    Returns
    -------
    evoked : Evoked object
        The simulated evoked data.

    See Also
    --------
    simulate_raw
    simulate_stc
    simulate_sparse_stc

    Notes
    -----
    To make the equivalence between snr and nave, when the snr is given
    instead of nave::

        nave = (1 / 10 ** ((actual_snr - snr)) / 20) ** 2

    where actual_snr is the snr to the generated noise before scaling.

    .. versionadded:: 0.10.0
    """
    evoked = apply_forward(fwd, stc, info, use_cps=use_cps)
    if cov is None:
        return evoked

    if nave < np.inf:
        noise = _simulate_noise_evoked(evoked, cov, iir_filter, rng)
        evoked.data += noise.data / math.sqrt(nave)
        evoked.nave = np.int64(nave)
    if cov.get("projs", None):
        evoked.add_proj(cov["projs"]).apply_proj()
    return evoked


def _simulate_noise_evoked(evoked, cov, iir_filter, rng):
    noise = evoked.copy()
    noise.data[:] = 0
    return _add_noise(noise, cov, iir_filter, rng, allow_subselection=False)


@_legacy_rng("random_state")
@verbose
def add_noise(inst, cov, iir_filter=None, verbose=None, *, rng=None, random_state=None):
    """Create noise as a multivariate Gaussian.

    The spatial covariance of the noise is given from the cov matrix.

    Parameters
    ----------
    inst : instance of Evoked, Epochs, or Raw
        Instance to which to add noise.
    cov : instance of Covariance
        The noise covariance.
    iir_filter : None | array-like
        IIR filter coefficients (denominator).
    %(verbose)s
    %(rng)s
    %(random_state_rng)s

    Returns
    -------
    inst : same type as the input data
        The instance, modified to have additional noise.

    Notes
    -----
    Only channels in both ``inst.info['ch_names']`` and
    ``cov['names']`` will have noise added to them.

    This function operates inplace on ``inst``.

    .. versionadded:: 0.18.0
    """
    # We always allow subselection here
    legacy_seed = random_state if isinstance(random_state, int | np.integer) else None
    return _add_noise(inst, cov, iir_filter, rng, legacy_seed=legacy_seed)


def _add_noise(
    inst, cov, iir_filter, rng, allow_subselection=True, *, legacy_seed=None
):
    """Add noise, possibly with channel subselection."""
    _validate_type(cov, Covariance, "cov")
    _validate_type(
        inst, (BaseRaw, BaseEpochs, Evoked), "inst", "Raw, Epochs, or Evoked"
    )
    _check_preload(inst, "Adding noise")
    data = inst._data
    assert data.ndim in (2, 3)
    if data.ndim == 2:
        data = data[np.newaxis]
    # Subselect if necessary
    info = inst.info
    info._check_consistency()
    picks = gen_picks = slice(None)
    if allow_subselection:
        use_chs = list(set(info["ch_names"]) & set(cov["names"]))
        picks = np.where(np.isin(info["ch_names"], use_chs))[0]
        logger.info(
            "Adding noise to %d/%d channels (%d channels in cov)",
            len(picks),
            len(info["chs"]),
            len(cov["names"]),
        )
        info = pick_info(inst.info, picks)
        info._check_consistency()

        gen_picks = np.arange(info["nchan"])
    for epoch in data:
        # An integer passed to the deprecated parameter historically restarted
        # the same stream for each epoch. ``rng`` intentionally advances.
        this_rng = check_random_state(legacy_seed) if legacy_seed is not None else rng
        epoch[picks] += _generate_noise(
            info, cov, iir_filter, this_rng, epoch.shape[1], picks=gen_picks
        )[0]
    return inst


def _generate_noise(info, cov, iir_filter, rng, n_samples, zi=None, picks=None):
    """Create spatially colored and temporally IIR-filtered noise."""
    _, _, colorer = compute_whitener(
        cov, info, pca=True, return_colorer=True, picks=picks, verbose=False
    )
    noise = np.dot(colorer, rng.standard_normal((colorer.shape[1], n_samples)))
    if iir_filter is not None:
        if zi is None:
            zi = np.zeros((len(colorer), len(iir_filter) - 1))
        noise, zf = lfilter([1], iir_filter, noise, axis=-1, zi=zi)
    else:
        zf = None
    return noise, zf
