# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import warnings
import math

import numpy as np

from ..io.pick import pick_channels_cov
from ..forward import apply_forward
from ..utils import check_random_state, verbose


@verbose
def simulate_evoked(fwd, stc, info, cov, nave=30, iir_filter=None,
                    random_state=None, use_cps=None, verbose=None):
    """Generate noisy evoked data.

    .. note:: No projections from ``info`` will be present in the
              output ``evoked``. You can use e.g.
              :func:`evoked.add_proj <mne.Evoked.add_proj>` or
              :func:`evoked.set_eeg_reference <mne.Evoked.set_eeg_reference>`
              to add them afterward as necessary.

    Parameters
    ----------
    fwd : Forward
        a forward solution.
    stc : SourceEstimate object
        The source time courses.
    info : dict
        Measurement info to generate the evoked.
    cov : Covariance object
        The noise covariance.
    nave : int
        Number of averaged epochs (defaults to 30).

        .. versionadded:: 0.15.0
    iir_filter : None | array
        IIR filter coefficients (denominator) e.g. [1, -1, 0.2].
    random_state : None | int | np.random.RandomState
        To specify the random generator state.
    use_cps : None | bool (default None)
        Whether to use cortical patch statistics to define normal
        orientations when converting to fixed orientation (if necessary).

        .. versionadded:: 0.15
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    evoked : Evoked object
        The simulated evoked data

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
    if nave < np.inf:
        noise = simulate_noise_evoked(evoked, cov, iir_filter, random_state)
        evoked.data += noise.data / math.sqrt(nave)
        evoked.nave = np.int(nave)
    return evoked


def simulate_noise_evoked(evoked, cov, iir_filter=None, random_state=None):
    """Create noise as a multivariate Gaussian.

    The spatial covariance of the noise is given from the cov matrix.

    Parameters
    ----------
    evoked : evoked object
        an instance of evoked used as template
    cov : Covariance object
        The noise covariance
    iir_filter : None | array
        IIR filter coefficients (denominator)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    noise : evoked object
        an instance of evoked

    Notes
    -----
    .. versionadded:: 0.10.0
    """
    noise = evoked.copy()
    noise.data = _generate_noise(evoked.info, cov, iir_filter, random_state,
                                 evoked.data.shape[1])[0]
    return noise


def _generate_noise(info, cov, iir_filter, random_state, n_samples, zi=None):
    """Create spatially colored and temporally IIR-filtered noise."""
    from scipy.signal import lfilter
    noise_cov = pick_channels_cov(cov, include=info['ch_names'], exclude=[])
    if set(info['ch_names']) != set(noise_cov.ch_names):
        raise ValueError('Evoked and covariance channel names are not '
                         'identical. Cannot generate the noise matrix. '
                         'Channels missing in covariance %s.' %
                         np.setdiff1d(info['ch_names'], noise_cov.ch_names))

    rng = check_random_state(random_state)
    c = np.diag(noise_cov.data) if noise_cov['diag'] else noise_cov.data
    mu_channels = np.zeros(len(c))
    # we almost always get a positive semidefinite warning here, so squash it
    with warnings.catch_warnings(record=True):
        noise = rng.multivariate_normal(mu_channels, c, n_samples).T
    if iir_filter is not None:
        if zi is None:
            zi = np.zeros((len(c), len(iir_filter) - 1))
        noise, zf = lfilter([1], iir_filter, noise, axis=-1, zi=zi)
    else:
        zf = None
    return noise, zf
