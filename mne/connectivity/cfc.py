# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Voytek Lab <https://github.com/voytekresearch/pacpy>
#
# License: BSD (3-clause)
import numpy as np
from pacpy import pac as ppac
from ..utils import _time_mask
from ..parallel import parallel_func
from ..io.pick import pick_types


def phase_amplitude_coupling(inst, f_phase, f_amp, ixs, tmin=None, tmax=None,
                             pac_func='plv', picks=None, n_jobs=1,
                             verbose=None):
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
    tmin : float
        Minimum time instant to consider (in seconds).
    tmax : float | None
        Maximum time instant to consider (in seconds). None will use the
        end of the dataset.
    pac_func : string, ['plv', 'glm', 'mi_canolty', 'mi_tort', 'ozkurt']
        The function for estimating PAC. Corresponds to functions in pacpy.pac
    picks : array-like of int | None
        The selection of channels to include in the computation.
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
    if picks is None:
        picks = pick_types(inst.info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    sfreq = inst.info['sfreq']
    time_mask = _time_mask(inst.times, tmin, tmax)
    if isinstance(inst, _BaseRaw):
        start, stop = np.where(time_mask)[0][[0, -1]]
        data = inst[picks, start:(stop + 1)][0]
    else:
        data = inst.get_data()[:, picks][..., time_mask]
    pac = _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                                    pac_func=pac_func, n_jobs=n_jobs,
                                    verbose=verbose)
    return pac


def _phase_amplitude_coupling(data, sfreq, f_phase, f_amp, ixs,
                              pac_func='plv', n_jobs=1, verbose=None):
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
    pac_out : array, dtype float, shape (n_pairs)
        The computed phase-amplitude coupling between each pair of data sources
        given in ixs
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
    pacs = np.array(parallel(my_pac(data, ixf, ixa, f_phase, f_amp, func)
                             for ixf, ixa in ixs))
    if ndim < 3:
        pacs = pacs.squeeze()
    return pacs


def _my_pac(x, ix_phase, ix_amp, f_phase, f_amp, func):
    """Aux function for PAC."""
    pac = [func(ep[ix_phase], ep[ix_amp], f_phase, f_amp) for ep in x]
    return np.array(pac)
