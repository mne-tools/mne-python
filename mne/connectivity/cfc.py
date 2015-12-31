# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Voytek Lab <https://github.com/voytekresearch/pacpy>
#
# License: BSD (3-clause)
import numpy as np
from pacpy import pac as ppac
from mne.filter import band_pass_filter
from pacpy.pac import _range_sanity
from ..utils import _time_mask
from ..parallel import parallel_func
from scipy.signal import hilbert


def phase_amplitude_coupling(inst, f_phase, f_amp, ixs, pac_func='plv',
                             ev=None, tmin=None, tmax=None,
                             n_jobs=1, verbose=None):
    """ Compute phase-amplitude coupling between pairs of signals using pacpy.

    Parameters
    ----------
    inst : an instance of Raw or Epochs
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amplitude : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac
    ev : array-like, shape (n_events,) | None
        Indices for events. To be supplied if data is 2D and output should be
        split by events. In this case, tmin and tmax must be provided
    tmin : float | None
        If ev is not provided, it is the start time to use in inst. If ev
        is provided, it is the time (in seconds) to include before each
        event index.
    tmax : float | None
        If ev is not provided, it is the stop time to use in inst. If ev
        is provided, it is the time (in seconds) to include after each
        event index.
    n_jobs : int
        Number of CPUs to use in the computation.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs)
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs

    References
    ----------
    [1] This function uses the PacPy modulte developed by the Voytek lab.
        https://github.com/voytekresearch/pacpy
    """
    from ..io.base import _BaseRaw
    from..epochs import _BaseEpochs
    if not isinstance(inst, (_BaseEpochs, _BaseRaw)):
        raise ValueError('Must supply either Epochs or Raw')

    sfreq = inst.info['sfreq']
    time_mask = _time_mask(inst.times, tmin, tmax)
    if isinstance(inst, _BaseRaw):
        if ev is None:
            start, stop = np.where(time_mask)[0][[0, -1]]
            data = inst[:, start:(stop + 1)][0]
        else:
            # In this case tmin/tmax are for creating epochs later
            data = inst[:, :][0]
    else:
        data = inst.get_data()[..., time_mask]
    pac = _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                                    pac_func=pac_func, ev=ev, tmin=tmin,
                                    tmax=tmax, n_jobs=n_jobs,
                                    verbose=verbose)
    return pac


def _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='plv', ev=None, tmin=None, tmax=None,
                              n_jobs=1, verbose=None):
    """ Compute phase-amplitude coupling using pacpy.

    Parameters
    ----------
    data : array, shape ([n_epochs], n_channels, n_times)
        The data used to calculate PAC
    sfreq : float
        The sampling frequency of the data
    f_phase : array, dtype float, shape (2,)
        The frequency range to use for low-frequency phase carrier.
    f_amplitude : array, dtype float, shape (2,)
        The frequency range to use for high-frequency amplitude modulation.
    ixs : array-like, shape (n_pairs x 2)
        The indices for low/high frequency channels. PAC will be estimated
        between n_pairs of channels. Indices correspond to rows of `data`.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac

    Returns
    -------
    pac_out : array, dtype float, shape (n_pairs, [n_events])
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs.
    """
    func = getattr(ppac, pac_func)
    ixs = np.array(ixs, ndmin=2)

    if data.ndim < 2 or data.ndim > 3:
        raise ValueError('Data must be 2D or 3D')
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if ixs.shape[1] != 2:
        raise ValueError('Indices must have have a 2nd dimension of length 2')
    if len(f_phase) != 2 or len(f_amp) != 2:
        raise ValueError('Frequencies must be specified w/ a low/hi tuple')
    if pac_func not in ppac.__dict__.keys():
        raise ValueError("That PAC function doesn't exist in PacPy")

    ndim = data.ndim
    n_epochs, n_channels, n_times = data.shape
    parallel, my_pac, n_jobs = parallel_func(_my_pac, n_jobs=n_jobs,
                                             verbose=verbose)
    pacs = np.array(parallel(my_pac(data, ixf, ixa, f_phase, f_amp, func,
                                    ev=ev, tmin=tmin, tmax=tmax, sfreq=sfreq)
                             for ixf, ixa in ixs))
    if ndim < 3:
        pacs = pacs.squeeze()
    return pacs


def _filter_ph_am(xph, xam, f_ph, f_am, sfreq, filterfn=None, kws_filt=None):
    """Aux function for phase/amplitude filtering"""
    filterfn = band_pass_filter if filterfn is None else filterfn
    kws_filt = {} if kws_filt is None else kws_filt

    # Filter the two signals + hilbert/phase
    _range_sanity(f_ph, f_am)
    xph = filterfn(xph, sfreq, *f_ph)
    xam = filterfn(xam, sfreq, *f_am)

    xph = np.angle(hilbert(xph))
    xam = np.abs(hilbert(xam))
    return xph, xam


def _my_pac(x, ix_phase, ix_amp, f_phase, f_amp, func,
            ev=None, tmin=None, tmax=None, sfreq=None):
    """Aux function for PAC.

    This includes support for epochs-like shapes, as well as for the user
    to provide a list of event indices (ev) in order to do all filtering before
    the epochs are created."""
    pac = []
    for ep in x:
        xph = ep[ix_phase]
        xam = ep[ix_amp]
        # If we have events, assume <3D shape
        if ev is not None:
            # Checks for proper inputs/shape
            ev = np.array(ev)
            if x.shape[0] > 1:
                raise ValueError("If ev is given, input must have"
                                 " first dim length 1")
            if ev.ndim > 1:
                raise ValueError('Events must be a 1-d array')
            if any([tmin is None, tmax is None]):
                raise ValueError('If ev is given,'
                                 ' tmin/tmax must be given')
            if not isinstance(sfreq, (int, float)):
                raise ValueError('If ev is given, sfreq must be given')
            xph, xam = _filter_ph_am(xph, xam, f_phase, f_amp, sfreq)

            # Turn into events and pass through func
            epochs = np.vstack([xph, xam])
            epochs = _array_raw_to_epochs(epochs, sfreq, ev, tmin, tmax)
            for ep_f in epochs:
                # f_phase and f_amp won't be used in this case
                pac.append(func(ep_f[0], ep_f[1],
                           f_phase, f_amp, filterfn=False))
        else:
            pac.append(func(xph, xam, f_phase, f_amp))
    return np.array(pac)


def _array_raw_to_epochs(x, sfreq, ev, tmin, tmax):
    """Aux function to create epochs from a 2D array"""
    win_size = sfreq * (tmax - tmin)
    msk_remove = np.logical_or(ev < win_size, (ev > (x.shape[-1] - win_size)))
    if any(msk_remove):
        print('raise a warning!')
        ev = ev[~msk_remove]
    times = np.arange(x.shape[-1]) / float(sfreq)
    epochs = []
    for ix in ev:
        start, stop = times[ix] - tmin, times[ix] + tmax
        msk = _time_mask(times, start, stop)
        epochs.append(x[np.newaxis, :, msk])
    epochs = np.concatenate(epochs, axis=0)
    return epochs
