import numpy as np

from mne.utils import _validate_type


def compute_sme(epochs, start=None, stop=None):
    """Compute standardized measurement error (SME).

    The standardized measurement error :footcite:`LuckEtAl2021` can be used as a
    universal measure of data quality in ERP studies.

    Parameters
    ----------
    epochs: mne.Epochs
        The epochs containing the data for which to compute the SME.
    start : int | float | None
        Start time (in s) of the time window used for SME computation. If ``None``, use
        the start of the epoch.
    stop : int | float | None
        Stop time (in s) of the time window used for SME computation. If ``None``, use
        the end of the epoch.

    Returns
    -------
    sme : array, shape (n_channels,)
        SME in given time window for each channel.

    Notes
    -----
    Currently, only the mean value in the given time window is supported, meaning that
    the resulting SME is only valid in studies which quantify the amplitude of an ERP
    component as the mean within the time window (as opposed to e.g. the peak, which
    would require bootstrapping).

    References
    ----------
    .. footbibliography::
    """
    _validate_type(start, ("numeric", None), "start", "int or float")
    _validate_type(stop, ("numeric", None), "stop", "int or float")
    start = epochs.tmin if start is None else start
    stop = epochs.tmax if stop is None else stop
    if start < epochs.tmin:
        raise ValueError("start is out of bounds.")
    if stop > epochs.tmax:
        raise ValueError("stop is out of bounds.")

    data = epochs.get_data(tmin=start, tmax=stop)
    return data.mean(axis=2).std(axis=0) / np.sqrt(data.shape[0])
