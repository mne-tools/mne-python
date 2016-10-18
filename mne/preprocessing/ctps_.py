# Authors: Juergen Dammers <j.dammers@fz-juelich.de>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: Simplified BSD
import math

import numpy as np


def _compute_normalized_phase(data):
    """Compute normalized phase angles.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_sources, n_times)
        The data to compute the phase angles for.

    Returns
    -------
    phase_angles : ndarray, shape (n_epochs, n_sources, n_times)
        The normalized phase angles.
    """
    from scipy.signal import hilbert
    return (np.angle(hilbert(data)) + np.pi) / (2 * np.pi)


def ctps(data, is_raw=True):
    """Compute cross-trial-phase-statistics [1].

    Note. It is assumed that the sources are already
    appropriately filtered

    Parameters
    ----------
    data: ndarray, shape (n_epochs, n_channels, n_times)
        Any kind of data of dimensions trials, traces, features.
    is_raw : bool
        If True it is assumed that data haven't been transformed to Hilbert
        space and phase angles haven't been normalized. Defaults to True.

    Returns
    -------
    ks_dynamics : ndarray, shape (n_sources, n_times)
        The kuiper statistics.
    pk_dynamics : ndarray, shape (n_sources, n_times)
        The normalized kuiper index for ICA sources and
        time slices.
    phase_angles : ndarray, shape (n_epochs, n_sources, n_times) | None
        The phase values for epochs, sources and time slices. If ``is_raw``
        is False, None is returned.

    References
    ----------
    [1] Dammers, J., Schiek, M., Boers, F., Silex, C., Zvyagintsev,
        M., Pietrzyk, U., Mathiak, K., 2008. Integration of amplitude
        and phase statistics for complete artifact removal in independent
        components of neuromagnetic recordings. Biomedical
        Engineering, IEEE Transactions on 55 (10), 2353-2362.
    """
    if not data.ndim == 3:
        ValueError('Data must have 3 dimensions, not %i.' % data.ndim)

    if is_raw:
        phase_angles = _compute_normalized_phase(data)
    else:
        phase_angles = data  # phase angles can be computed externally

    # initialize array for results
    ks_dynamics = np.zeros_like(phase_angles[0])
    pk_dynamics = np.zeros_like(phase_angles[0])

    # calculate Kuiper's statistic for each source
    for ii, source in enumerate(np.transpose(phase_angles, [1, 0, 2])):
        ks, pk = kuiper(source)
        pk_dynamics[ii, :] = pk
        ks_dynamics[ii, :] = ks

    return ks_dynamics, pk_dynamics, phase_angles if is_raw else None


def kuiper(data, dtype=np.float64):  # noqa: D401
    """Kuiper's test of uniform distribution.

    Parameters
    ----------
    data : ndarray, shape (n_sources,) | (n_sources, n_times)
           Empirical distribution.
    dtype : str | obj
        The data type to be used.

    Returns
    -------
    ks : ndarray
        Kuiper's statistic.
    pk : ndarray
        Normalized probability of Kuiper's statistic [0, 1].
    """
    # if data not numpy array, implicitly convert and make to use copied data
    # ! sort data array along first axis !
    data = np.sort(data, axis=0).astype(dtype)
    shape = data.shape
    n_dim = len(shape)
    n_trials = shape[0]

    # create uniform cdf
    j1 = (np.arange(n_trials, dtype=dtype) + 1.) / float(n_trials)
    j2 = np.arange(n_trials, dtype=dtype) / float(n_trials)
    if n_dim > 1:  # single phase vector (n_trials)
        j1 = j1[:, np.newaxis]
        j2 = j2[:, np.newaxis]
    d1 = (j1 - data).max(axis=0)
    d2 = (data - j2).max(axis=0)
    n_eff = n_trials

    d = d1 + d2  # Kuiper's statistic [n_time_slices]

    return d, _prob_kuiper(d, n_eff, dtype=dtype)


def _prob_kuiper(d, n_eff, dtype='f8'):
    """Test for statistical significance against uniform distribution.

    Parameters
    ----------
    d : float
        The kuiper distance value.
    n_eff : int
        The effective number of elements.
    dtype : str | obj
        The data type to be used. Defaults to double precision floats.

    Returns
    -------
    pk_norm : float
        The normalized Kuiper value such that 0 < ``pk_norm`` < 1.

    References
    ----------
    [1] Stephens MA 1970. Journal of the Royal Statistical Society, ser. B,
    vol 32, pp 115-122.

    [2] Kuiper NH 1962. Proceedings of the Koninklijke Nederlands Akademie
    van Wetenschappen, ser Vol 63 pp 38-47
    """
    n_time_slices = np.size(d)  # single value or vector
    n_points = 100

    en = math.sqrt(n_eff)
    k_lambda = (en + 0.155 + 0.24 / en) * d  # see [1]
    l2 = k_lambda ** 2.0
    j2 = (np.arange(n_points) + 1) ** 2
    j2 = j2.repeat(n_time_slices).reshape(n_points, n_time_slices)
    fact = 4. * j2 * l2 - 1.
    expo = np.exp(-2. * j2 * l2)
    term = 2. * fact * expo
    pk = term.sum(axis=0, dtype=dtype)

    # Normalized pK to range [0,1]
    pk_norm = np.zeros(n_time_slices)  # init pk_norm
    pk_norm[pk > 0] = -np.log(pk[pk > 0]) / (2. * n_eff)
    pk_norm[pk <= 0] = 1

    # check for no difference to uniform cdf
    pk_norm = np.where(k_lambda < 0.4, 0.0, pk_norm)

    # check for round off errors
    pk_norm = np.where(pk_norm > 1.0, 1.0, pk_norm)

    return pk_norm
