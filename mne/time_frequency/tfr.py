"""A module which implements the time-frequency estimation.

Morlet code inspired by Matlab code from Sheraz Khan & Brainstorm & SPM
"""
# Authors : Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#           Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#           Clement Moutard <clement.moutard@polytechnique.org>
#           Jean-Remi King <jeanremi.king@gmail.com>
#
# License : BSD (3-clause)

from copy import deepcopy
from functools import partial
from math import sqrt

import numpy as np
from scipy import linalg
from scipy.fftpack import fft, ifft

from ..baseline import rescale
from ..parallel import parallel_func
from ..utils import logger, verbose, _time_mask, check_fname, sizeof_fmt
from ..channels.channels import ContainsMixin, UpdateChannelsMixin
from ..channels.layout import _pair_grad_sensors
from ..io.pick import pick_info, pick_types
from ..io.meas_info import Info
from ..utils import SizeMixin
from .multitaper import dpss_windows
from ..viz.utils import figure_nobar, plt_show, _setup_cmap
from ..externals.h5io import write_hdf5, read_hdf5
from ..externals.six import string_types


# Make wavelet

def morlet(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        frequency range of interest (1 x Frequencies)
    n_cycles: float | array of float, defaults to 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, defaults to None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, defaults to False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
        W /= sqrt(0.5) * linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


def _make_dpss(sfreq, freqs, n_cycles=7., time_bandwidth=4.0, zero_mean=False):
    """Compute DPSS tapers for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling frequency.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,), defaults to 7.
        The number of cycles globally or for each frequency.
    time_bandwidth : float, defaults to 4.0
        Time x Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1).
        Default is 4.0, giving 3 good tapers.
    zero_mean : bool | None, , defaults to False
        Make sure the wavelet has a mean of zero.


    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    if time_bandwidth < 2.0:
        raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
    n_taps = int(np.floor(time_bandwidth - 1))
    n_cycles = np.atleast_1d(n_cycles)

    if n_cycles.size != 1 and n_cycles.size != len(freqs):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")

    for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles) != 1:
                this_n_cycles = n_cycles[k]
            else:
                this_n_cycles = n_cycles[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0., t_win, 1.0 / sfreq)
            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.))

            # Get dpss tapers
            tapers, conc = dpss_windows(t.shape[0], time_bandwidth / 2.,
                                        n_taps)

            Wk = oscillation * tapers[m]
            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= sqrt(0.5) * linalg.norm(Wk.ravel())

            Wm.append(Wk)

        Ws.append(Wm)

    return Ws


# Low level convolution

def _cwt(X, Ws, mode="same", decim=1, use_fft=True):
    """Compute cwt with fft based convolutions or temporal convolutions.

    Parameters
    ----------
    X : array of shape (n_signals, n_times)
        The data.
    Ws : list of array
        Wavelets time series.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : int | slice, defaults to 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

    use_fft : bool, defaults to True
        Use the FFT for convolutions or not.

    Returns
    -------
    out : array, shape (n_signals, n_freqs, n_time_decim)
        The time-frequency transform of the signals.
    """
    if mode not in ['same', 'valid', 'full']:
        raise ValueError("`mode` must be 'same', 'valid' or 'full', "
                         "got %s instead." % mode)
    if mode == 'full' and (not use_fft):
        # XXX JRK: full wavelet decomposition needs to be implemented
        raise ValueError('`full` decomposition with convolution is currently' +
                         ' not supported.')
    decim = _check_decim(decim)
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_times_out = X[:, decim].shape[1]
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))

    # precompute FFTs of Ws
    if use_fft:
        fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('At least one of the wavelets is longer than the '
                             'signal. Use a longer signal or shorter '
                             'wavelets.')
        if use_fft:
            fft_Ws[i] = fft(W, fsize)

    # Make generator looping across signals
    tfr = np.zeros((n_freqs, n_times_out), dtype=np.complex128)
    for x in X:
        if use_fft:
            fft_x = fft(x, fsize)

        # Loop across wavelets
        for ii, W in enumerate(Ws):
            if use_fft:
                ret = ifft(fft_x * fft_Ws[ii])[:n_times + W.size - 1]
            else:
                ret = np.convolve(x, W, mode=mode)

            # Center and decimate decomposition
            if mode == "valid":
                sz = int(abs(W.size - n_times)) + 1
                offset = (n_times - sz) // 2
                this_slice = slice(offset // decim.step,
                                   (offset + sz) // decim.step)
                if use_fft:
                    ret = _centered(ret, sz)
                tfr[ii, this_slice] = ret[decim]
            else:
                if use_fft:
                    ret = _centered(ret, n_times)
                tfr[ii, :] = ret[decim]
        yield tfr


# Loop of convolution: single trial


def _compute_tfr(epoch_data, frequencies, sfreq=1.0, method='morlet',
                 n_cycles=7.0, zero_mean=None, time_bandwidth=None,
                 use_fft=True, decim=1, output='complex', n_jobs=1,
                 verbose=None):
    """Compute time-frequency transforms.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    frequencies : array-like of floats, shape (n_freqs)
        The frequencies.
    sfreq : float | int, defaults to 1.0
        Sampling frequency of the data.
    method : 'multitaper' | 'morlet', defaults to 'morlet'
        The time-frequency method. 'morlet' convolves a Morlet wavelet.
        'multitaper' uses Morlet wavelets windowed with multiple DPSS
        multitapers.
    n_cycles : float | array of float, defaults to 7.0
        Number of cycles  in the Morlet wavelet. Fixed number
        or one per frequency.
    zero_mean : bool | None, defaults to None
        None means True for method='multitaper' and False for method='morlet'.
        If True, make sure the wavelets have a mean of zero.
    time_bandwidth : float, defaults to None
        If None and method=multitaper, will be set to 4.0 (3 tapers).
        Time x (Full) Bandwidth product. Only applies if
        method == 'multitaper'. The number of good tapers (low-bias) is
        chosen automatically based on this to equal floor(time_bandwidth - 1).
    use_fft : bool, defaults to True
        Use the FFT for convolutions or not.
    decim : int | slice, defaults to 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.

    output : str, defaults to 'complex'

        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.

    n_jobs : int, defaults to 1
        The number of epochs to process at the same time. The parallelization
        is implemented across channels.
    verbose : bool, str, int, or None, defaults to None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc
    """
    # Check data
    epoch_data = np.asarray(epoch_data)
    if epoch_data.ndim != 3:
        raise ValueError('epoch_data must be of shape '
                         '(n_epochs, n_chans, n_times)')

    # Check params
    frequencies, sfreq, zero_mean, n_cycles, time_bandwidth, decim = \
        _check_tfr_param(frequencies, sfreq, method, zero_mean, n_cycles,
                         time_bandwidth, use_fft, decim, output)

    # Setup wavelet
    if method == 'morlet':
        W = morlet(sfreq, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)
        Ws = [W]  # to have same dimensionality as the 'multitaper' case

    elif method == 'multitaper':
        Ws = _make_dpss(sfreq, frequencies, n_cycles=n_cycles,
                        time_bandwidth=time_bandwidth, zero_mean=zero_mean)

    # Check wavelets
    if len(Ws[0][0]) > epoch_data.shape[2]:
        raise ValueError('At least one of the wavelets is longer than the '
                         'signal. Use a longer signal or shorter wavelets.')

    # Initialize output
    decim = _check_decim(decim)
    n_freqs = len(frequencies)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim].shape
    if output in ('power', 'phase', 'avg_power', 'itc'):
        dtype = np.float
    elif output in ('complex', 'avg_power_itc'):
        # avg_power_itc is stored as power + 1i * itc to keep a
        # simple dimensionality
        dtype = np.complex

    if ('avg_' in output) or ('itc' in output):
        out = np.empty((n_chans, n_freqs, n_times), dtype)
    else:
        out = np.empty((n_chans, n_epochs, n_freqs, n_times), dtype)

    # Parallel computation
    parallel, my_cwt, _ = parallel_func(_time_frequency_loop, n_jobs)

    # Parallelization is applied across channels.
    tfrs = parallel(
        my_cwt(channel, Ws, output, use_fft, 'same', decim)
        for channel in epoch_data.transpose(1, 0, 2))

    # FIXME: to avoid overheads we should use np.array_split()
    for channel_idx, tfr in enumerate(tfrs):
        out[channel_idx] = tfr

    if ('avg_' not in output) and ('itc' not in output):
        # This is to enforce that the first dimension is for epochs
        out = out.transpose(1, 0, 2, 3)
    return out


def _check_tfr_param(frequencies, sfreq, method, zero_mean, n_cycles,
                     time_bandwidth, use_fft, decim, output):
    """Aux. function to _compute_tfr to check the params validity."""
    # Check frequencies
    if not isinstance(frequencies, (list, np.ndarray)):
        raise ValueError('frequencies must be an array-like, got %s '
                         'instead.' % type(frequencies))
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1:
        raise ValueError('frequencies must be of shape (n_freqs,), got %s '
                         'instead.' % np.array(frequencies.shape))

    # Check sfreq
    if not isinstance(sfreq, (float, int)):
        raise ValueError('sfreq must be a float or an int, got %s '
                         'instead.' % type(sfreq))
    sfreq = float(sfreq)

    # Default zero_mean = True if multitaper else False
    zero_mean = method == 'multitaper' if zero_mean is None else zero_mean
    if not isinstance(zero_mean, bool):
        raise ValueError('zero_mean should be of type bool, got %s. instead'
                         % type(zero_mean))
    frequencies = np.asarray(frequencies)

    if (method == 'multitaper') and (output == 'phase'):
        raise NotImplementedError(
            'This function is not optimized to compute the phase using the '
            'multitaper method. Use np.angle of the complex output instead.')

    # Check n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles = float(n_cycles)
    elif isinstance(n_cycles, (list, np.ndarray)):
        n_cycles = np.array(n_cycles)
        if len(n_cycles) != len(frequencies):
            raise ValueError('n_cycles must be a float or an array of length '
                             '%i frequencies, got %i cycles instead.' %
                             (len(frequencies), len(n_cycles)))
    else:
        raise ValueError('n_cycles must be a float or an array, got %s '
                         'instead.' % type(n_cycles))

    # Check time_bandwidth
    if (method == 'morlet') and (time_bandwidth is not None):
        raise ValueError('time_bandwidth only applies to "multitaper" method.')
    elif method == 'multitaper':
        time_bandwidth = (4.0 if time_bandwidth is None
                          else float(time_bandwidth))

    # Check use_fft
    if not isinstance(use_fft, bool):
        raise ValueError('use_fft must be a boolean, got %s '
                         'instead.' % type(use_fft))
    # Check decim
    if isinstance(decim, int):
        decim = slice(None, None, decim)
    if not isinstance(decim, slice):
        raise ValueError('decim must be an integer or a slice, '
                         'got %s instead.' % type(decim))

    # Check output
    allowed_ouput = ('complex', 'power', 'phase',
                     'avg_power_itc', 'avg_power', 'itc')
    if output not in allowed_ouput:
        raise ValueError("Unknown output type. Allowed are %s but "
                         "got %s." % (allowed_ouput, output))

    if method not in ('multitaper', 'morlet'):
        raise ValueError('method must be "morlet" or "multitaper", got %s '
                         'instead.' % type(method))

    return frequencies, sfreq, zero_mean, n_cycles, time_bandwidth, decim


def _time_frequency_loop(X, Ws, output, use_fft, mode, decim):
    """Aux. function to _compute_tfr.

    Loops time-frequency transform across wavelets and epochs.

    Parameters
    ----------
    X : array, shape (n_epochs, n_times)
        The epochs data of a single channel.
    Ws : list, shape (n_tapers, n_wavelets, n_times)
        The wavelets.
    output : str

        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.

    use_fft : bool
        Use the FFT for convolutions or not.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : slice
        The decimation slice: e.g. power[:, decim]
    """
    # Set output type
    dtype = np.float
    if output in ['complex', 'avg_power_itc']:
        dtype = np.complex

    # Init outputs
    decim = _check_decim(decim)
    n_epochs, n_times = X[:, decim].shape
    n_freqs = len(Ws[0])
    if ('avg_' in output) or ('itc' in output):
        tfrs = np.zeros((n_freqs, n_times), dtype=dtype)
    else:
        tfrs = np.zeros((n_epochs, n_freqs, n_times), dtype=dtype)

    # Loops across tapers.
    for W in Ws:
        coefs = _cwt(X, W, mode, decim=decim, use_fft=use_fft)

        # Inter-trial phase locking is apparently computed per taper...
        if 'itc' in output:
            plf = np.zeros((n_freqs, n_times), dtype=np.complex)

        # Loop across epochs
        for epoch_idx, tfr in enumerate(coefs):
            # Transform complex values
            if output in ['power', 'avg_power']:
                tfr = (tfr * tfr.conj()).real  # power
            elif output == 'phase':
                tfr = np.angle(tfr)
            elif output == 'avg_power_itc':
                tfr_abs = np.abs(tfr)
                plf += tfr / tfr_abs  # phase
                tfr = tfr_abs ** 2  # power
            elif output == 'itc':
                plf += tfr / np.abs(tfr)  # phase
                continue  # not need to stack anything else than plf

            # Stack or add
            if ('avg_' in output) or ('itc' in output):
                tfrs += tfr
            else:
                tfrs[epoch_idx] += tfr

        # Compute inter trial coherence
        if output == 'avg_power_itc':
            tfrs += 1j * np.abs(plf)
        elif output == 'itc':
            tfrs += np.abs(plf)

    # Normalization of average metrics
    if ('avg_' in output) or ('itc' in output):
        tfrs /= n_epochs

    # Normalization by number of taper
    tfrs /= len(Ws)
    return tfrs


def cwt(X, Ws, use_fft=True, mode='same', decim=1):
    """Compute time freq decomposition with continuous wavelet transform.

    Parameters
    ----------
    X : array, shape (n_signals, n_times)
        The signals.
    Ws : list of array
        Wavelets time series.
    use_fft : bool
        Use FFT for convolutions. Defaults to True.
    mode : 'same' | 'valid' | 'full'
        Convention for convolution. 'full' is currently not implemented with
        `use_fft=False`. Defaults to 'same'.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

        Defaults to 1.

    Returns
    -------
    tfr : array, shape (n_signals, n_frequencies, n_times)
        The time-frequency decompositions.

    See Also
    --------
    mne.time_frequency.tfr_morlet : Compute time-frequency decomposition
                                    with Morlet wavelets
    """
    decim = _check_decim(decim)
    n_signals, n_times = X[:, decim].shape

    coefs = _cwt(X, Ws, mode, decim=decim, use_fft=use_fft)

    tfrs = np.empty((n_signals, len(Ws), n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def _tfr_aux(method, inst, freqs, decim, return_itc, picks, average,
             **tfr_params):
    """Help reduce redundancy between tfr_morlet and tfr_multitaper."""
    decim = _check_decim(decim)
    data = _get_data(inst, return_itc)
    info = inst.info

    info, data, picks = _prepare_picks(info, data, picks)
    data = data[:, picks, :]

    if average:
        if return_itc:
            output = 'avg_power_itc'
        else:
            output = 'avg_power'
    else:
        output = 'power'
        if return_itc:
            raise ValueError('Inter-trial coherence is not supported'
                             ' with average=False')

    out = _compute_tfr(data, freqs, info['sfreq'], method=method,
                       output=output, decim=decim, **tfr_params)
    times = inst.times[decim].copy()

    if average:
        if return_itc:
            power, itc = out.real, out.imag
        else:
            power = out
        nave = len(data)
        out = AverageTFR(info, power, times, freqs, nave,
                         method='%s-power' % method)
        if return_itc:
            out = (out, AverageTFR(info, itc, times, freqs, nave,
                                   method='%s-itc' % method))
    else:
        power = out
        out = EpochsTFR(info, power, times, freqs, method='%s-power' % method)

    return out


@verbose
def tfr_morlet(inst, freqs, n_cycles, use_fft=False, return_itc=True, decim=1,
               n_jobs=1, picks=None, zero_mean=True, average=True,
               verbose=None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
    use_fft : bool, defaults to False
        The fft based convolution or not.
    return_itc : bool, defaults to True
        Return inter-trial coherence (ITC) as well as averaged power.
        Must be ``False`` for evoked data.
    decim : int | slice, defaults to 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

    n_jobs : int, defaults to 1
        The number of jobs to run in parallel.
    picks : array-like of int | None, defaults to None
        The indices of the channels to decompose. If None, all available
        channels are decomposed.
    zero_mean : bool, defaults to True
        Make sure the wavelet has a mean of zero.

        .. versionadded:: 0.13.0
    average : bool, defaults to True
        If True average across Epochs.

        .. versionadded:: 0.13.0
    verbose : bool, str, int, or None, defaults to None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    power : AverageTFR | EpochsTFR
        The averaged or single-trial power.
    itc : AverageTFR | EpochsTFR
        The inter-trial coherence (ITC). Only returned if return_itc
        is True.

    See Also
    --------
    mne.time_frequency.tfr_array_morlet
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell
    """
    tfr_params = dict(n_cycles=n_cycles, n_jobs=n_jobs, use_fft=use_fft,
                      zero_mean=zero_mean)
    return _tfr_aux('morlet', inst, freqs, decim, return_itc, picks,
                    average, **tfr_params)


@verbose
def tfr_array_morlet(epoch_data, sfreq, frequencies, n_cycles=7.0,
                     zero_mean=False, use_fft=True, decim=1, output='complex',
                     n_jobs=1, verbose=None):
    """Compute time-frequency transform using Morlet wavelets.

    Convolves epoch data with selected Morlet wavelets.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float | int
        Sampling frequency of the data.
    frequencies : array-like of floats, shape (n_freqs)
        The frequencies.
    n_cycles : float | array of float, defaults to 7.0
        Number of cycles in the Morlet wavelet. Fixed number or one per
        frequency.
    zero_mean : bool | False
        If True, make sure the wavelets have a mean of zero. Defaults to False.
    use_fft : bool
        Use the FFT for convolutions or not. Defaults to True.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition. Defaults to 1
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.

    output : str, defaults to 'complex'

        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.

    n_jobs : int
        The number of epochs to process at the same time. The parallelization
        is implemented across channels. Defaults to 1
    verbose : bool, str, int, or None, defaults to None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc

    See Also
    --------
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell

    Notes
    -----
    .. versionadded:: 0.14.0
    """
    return _compute_tfr(epoch_data=epoch_data, frequencies=frequencies,
                        sfreq=sfreq, method='morlet', n_cycles=n_cycles,
                        zero_mean=zero_mean, time_bandwidth=None,
                        use_fft=use_fft, decim=decim, output=output,
                        n_jobs=n_jobs, verbose=verbose)


@verbose
def tfr_multitaper(inst, freqs, n_cycles, time_bandwidth=4.0,
                   use_fft=True, return_itc=True, decim=1,
                   n_jobs=1, picks=None, average=True, verbose=None):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
        The time-window length is thus T = n_cycles / freq.
    time_bandwidth : float, (optional), defaults to 4.0 (3 good tapers).
        Time x (Full) Bandwidth product. Should be >= 2.0.
        Choose this along with n_cycles to get desired frequency resolution.
        The number of good tapers (least leakage from far away frequencies)
        is chosen automatically based on this to floor(time_bandwidth - 1).
        E.g., With freq = 20 Hz and n_cycles = 10, we get time = 0.5 s.
        If time_bandwidth = 4., then frequency smoothing is (4 / time) = 8 Hz.
    use_fft : bool, defaults to True
        The fft based convolution or not.
    return_itc : bool, defaults to True
        Return inter-trial coherence (ITC) as well as averaged (or
        single-trial) power.
    decim : int | slice, defaults to 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

    n_jobs : int,  defaults to 1
        The number of jobs to run in parallel.
    picks : array-like of int | None, defaults to None
        The indices of the channels to decompose. If None, all available
        channels are decomposed.
    average : bool, defaults to True
        If True average across Epochs.

        .. versionadded:: 0.13.0
    verbose : bool, str, int, or None, defaults to None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    power : AverageTFR | EpochsTFR
        The averaged or single-trial power.
    itc : AverageTFR | EpochsTFR
        The inter-trial coherence (ITC). Only returned if return_itc
        is True.

    See Also
    --------
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_array_morlet

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    tfr_params = dict(n_cycles=n_cycles, n_jobs=n_jobs, use_fft=use_fft,
                      zero_mean=True, time_bandwidth=time_bandwidth)
    return _tfr_aux('multitaper', inst, freqs, decim, return_itc, picks,
                    average, **tfr_params)


# TFR(s) class

class _BaseTFR(ContainsMixin, UpdateChannelsMixin, SizeMixin):
    """Base TFR class."""

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    def crop(self, tmin=None, tmax=None):
        """Crop data to a given time interval in place.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.

        Returns
        -------
        inst : instance of AverageTFR
            The modified instance.
        """
        mask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'])
        self.times = self.times[mask]
        self.data = self.data[..., mask]
        return self

    def copy(self):
        """Return a copy of the instance."""
        return deepcopy(self)

    @verbose
    def apply_baseline(self, baseline, mode='mean', verbose=None):
        """Baseline correct the data.

        Parameters
        ----------
        baseline : tuple or list of length 2
            The time interval to apply rescaling / baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)),
            mean simply subtracts the mean power, percent is the same as
            applying ratio then mean, logratio is the same as mean but then
            rendered in log-scale, zlogratio is the same as zscore but data
            is rendered in log-scale first.
            If None no baseline correction is applied.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see :func:`mne.verbose`).

        Returns
        -------
        inst : instance of AverageTFR
            The modified instance.

        """  # noqa: E501
        self.data = rescale(self.data, self.times, baseline, mode,
                            copy=False)
        return self


class AverageTFR(_BaseTFR):
    """Container for Time-Frequency data.

    Can for example store induced power at sensor level or inter-trial
    coherence.

    Parameters
    ----------
    info : Info
        The measurement info.
    data : ndarray, shape (n_channels, n_freqs, n_times)
        The data.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    nave : int
        The number of averaged TFRs.
    comment : str | None, defaults to None
        Comment on the data, e.g., the experimental condition.
    method : str | None, defaults to None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Attributes
    ----------
    ch_names : list
        The names of the channels.
    """

    @verbose
    def __init__(self, info, data, times, freqs, nave, comment=None,
                 method=None, verbose=None):  # noqa: D102
        self.info = info
        if data.ndim != 3:
            raise ValueError('data should be 3d. Got %d.' % data.ndim)
        n_channels, n_freqs, n_times = data.shape
        if n_channels != len(info['chs']):
            raise ValueError("Number of channels and data size don't match"
                             " (%d != %d)." % (n_channels, len(info['chs'])))
        if n_freqs != len(freqs):
            raise ValueError("Number of frequencies and data size don't match"
                             " (%d != %d)." % (n_freqs, len(freqs)))
        if n_times != len(times):
            raise ValueError("Number of times and data size don't match"
                             " (%d != %d)." % (n_times, len(times)))
        self.data = data
        self.times = np.array(times, dtype=float)
        self.freqs = np.array(freqs, dtype=float)
        self.nave = nave
        self.comment = comment
        self.method = method
        self.preload = True

    @verbose
    def plot(self, picks, baseline=None, mode='mean', tmin=None, tmax=None,
             fmin=None, fmax=None, vmin=None, vmax=None, cmap='RdBu_r',
             dB=False, colorbar=True, show=True, title=None, axes=None,
             layout=None, yscale='auto', verbose=None):
        """Plot TFRs as a two-dimensional image(s).

        Parameters
        ----------
        picks : array-like of int
            The indices of the channels to plot, one figure per channel.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)),
            mean simply subtracts the mean power, percent is the same as
            applying ratio then mean, logratio is the same as mean but then
            rendered in log-scale, zlogratio is the same as zscore but data
            is rendered in log-scale first.
            If None no baseline correction is applied.
        tmin : None | float
            The first time instant to display. If None the first time point
            available is used.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used.
        vmin : float | None
            The mininum value an the color scale. If vmin is None, the data
            minimum value is used.
        vmax : float | None
            The maxinum value an the color scale. If vmax is None, the data
            maximum value is used.
        cmap : matplotlib colormap | 'interactive' | (colormap, bool)
            The colormap to use. If tuple, the first value indicates the
            colormap to use and the second value is a boolean defining
            interactivity. In interactive mode the colors are adjustable by
            clicking and dragging the colorbar with left and right mouse
            button. Left mouse button moves the scale up and down and right
            mouse button adjusts the range. Hitting space bar resets the range.
            Up and down arrows can be used to change the colormap. If
            'interactive', translates to ('RdBu_r', True). Defaults to
            'RdBu_r'.

            .. warning:: Interactive mode works smoothly only for a small
                amount of images.

        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot. For user defined axes,
            the colorbar cannot be drawn. Defaults to True.
        show : bool
            Call pyplot.show() at the end.
        title : str | None
            String for title. Defaults to None (blank/no title).
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as the number of channels. If instance of Axes,
            there must be only one channel plotted.
        layout : Layout | None
            Layout instance specifying sensor positions. Used for interactive
            plotting of topographies on rectangle selection. If possible, the
            correct layout is inferred from the data.
        yscale : 'auto' (default) | 'linear' | 'log'
            The scale of y (frequency) axis. 'linear' gives linear y axis,
            'log' leads to log-spaced y axis and 'auto' detects if frequencies
            are log-spaced and only then sets the y axis to 'log'.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see :func:`mne.verbose`).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.
        """  # noqa: E501
        from ..viz.topo import _imshow_tfr
        import matplotlib.pyplot as plt
        times, freqs = self.times.copy(), self.freqs.copy()
        info = self.info
        data = self.data

        n_picks = len(picks)
        info, data, picks = _prepare_picks(info, data, picks)
        data = data[picks]

        data, times, freqs, vmin, vmax = \
            _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax, mode,
                         baseline, vmin, vmax, dB, info['sfreq'])

        tmin, tmax = times[0], times[-1]
        if isinstance(axes, plt.Axes):
            axes = [axes]
        if isinstance(axes, list) or isinstance(axes, np.ndarray):
            if len(axes) != n_picks:
                raise RuntimeError('There must be an axes for each picked '
                                   'channel.')

        cmap = _setup_cmap(cmap)
        for idx in range(len(data)):
            if axes is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            else:
                ax = axes[idx]
                fig = ax.get_figure()
            onselect_callback = partial(self._onselect, baseline=baseline,
                                        mode=mode, layout=layout)
            _imshow_tfr(ax, 0, tmin, tmax, vmin, vmax, onselect_callback,
                        ylim=None, tfr=data[idx: idx + 1], freq=freqs,
                        x_label='Time (ms)', y_label='Frequency (Hz)',
                        colorbar=colorbar, cmap=cmap, yscale=yscale)
            if title:
                fig.suptitle(title)

        plt_show(show)
        return fig

    def _onselect(self, eclick, erelease, baseline, mode, layout):
        """Callback function called by rubber band selector in channel tfr."""
        import matplotlib.pyplot as plt
        from ..viz import plot_tfr_topomap
        if abs(eclick.x - erelease.x) < .1 or abs(eclick.y - erelease.y) < .1:
            return
        plt.ion()  # turn interactive mode on
        tmin = round(min(eclick.xdata, erelease.xdata) / 1000., 5)  # ms to s
        tmax = round(max(eclick.xdata, erelease.xdata) / 1000., 5)
        fmin = round(min(eclick.ydata, erelease.ydata), 5)  # Hz
        fmax = round(max(eclick.ydata, erelease.ydata), 5)
        tmin = min(self.times, key=lambda x: abs(x - tmin))  # find closest
        tmax = min(self.times, key=lambda x: abs(x - tmax))
        fmin = min(self.freqs, key=lambda x: abs(x - fmin))
        fmax = min(self.freqs, key=lambda x: abs(x - fmax))
        if tmin == tmax or fmin == fmax:
            logger.info('The selected area is too small. '
                        'Select a larger time-frequency window.')
            return

        types = list()
        if 'eeg' in self:
            types.append('eeg')
        if 'mag' in self:
            types.append('mag')
        if 'grad' in self:
            if len(_pair_grad_sensors(self.info, topomap_coords=False,
                                      raise_error=False)) >= 2:
                types.append('grad')
            elif len(types) == 0:
                return  # Don't draw a figure for nothing.
        fig = figure_nobar()
        fig.suptitle('{0:.2f} s - {1:.2f} s, {2:.2f} Hz - {3:.2f} Hz'.format(
            tmin, tmax, fmin, fmax), y=0.04)
        for idx, ch_type in enumerate(types):
            ax = plt.subplot(1, len(types), idx + 1)
            plot_tfr_topomap(self, ch_type=ch_type, tmin=tmin, tmax=tmax,
                             fmin=fmin, fmax=fmax, layout=layout,
                             baseline=baseline, mode=mode, cmap=None,
                             title=ch_type, vmin=None, vmax=None,
                             axes=ax)

    def plot_topo(self, picks=None, baseline=None, mode='mean', tmin=None,
                  tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
                  layout=None, cmap='RdBu_r', title=None, dB=False,
                  colorbar=True, layout_scale=0.945, show=True,
                  border='none', fig_facecolor='k', fig_background=None,
                  font_color='w', yscale='auto'):
        """Plot TFRs in a topography with images.

        Parameters
        ----------
        picks : array-like of int | None
            The indices of the channels to plot. If None, all available
            channels are displayed.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)),
            mean simply subtracts the mean power, percent is the same as
            applying ratio then mean, logratio is the same as mean but then
            rendered in log-scale, zlogratio is the same as zscore but data
            is rendered in log-scale first.
            If None no baseline correction is applied.
        tmin : None | float
            The first time instant to display. If None the first time point
            available is used.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used.
        vmin : float | None
            The mininum value an the color scale. If vmin is None, the data
            minimum value is used.
        vmax : float | None
            The maxinum value an the color scale. If vmax is None, the data
            maximum value is used.
        layout : Layout | None
            Layout instance specifying sensor positions. If possible, the
            correct layout is inferred from the data.
        cmap : matplotlib colormap | str
            The colormap to use. Defaults to 'RdBu_r'.
        title : str
            Title of the figure.
        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas.
        show : bool
            Call pyplot.show() at the end.
        border : str
            matplotlib borders style to be used for each sensor plot.
        fig_facecolor : str | obj
            The figure face color. Defaults to black.
        fig_background : None | array
            A background image for the figure. This must be a valid input to
            `matplotlib.pyplot.imshow`. Defaults to None.
        font_color: str | obj
            The color of tick labels in the colorbar. Defaults to white.
        yscale : 'auto' (default) | 'linear' | 'log'
            The scale of y (frequency) axis. 'linear' gives linear y axis,
            'log' leads to log-spaced y axis and 'auto' detects if frequencies
            are log-spaced and only then sets the y axis to 'log'.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.
        """  # noqa: E501
        from ..viz.topo import _imshow_tfr, _plot_topo, _imshow_tfr_unified
        from ..viz import add_background_image
        times = self.times.copy()
        freqs = self.freqs
        data = self.data
        info = self.info

        info, data, picks = _prepare_picks(info, data, picks)
        data = data[picks]

        data, times, freqs, vmin, vmax = \
            _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax,
                         mode, baseline, vmin, vmax, dB, info['sfreq'])

        if layout is None:
            from mne import find_layout
            layout = find_layout(self.info)
        onselect_callback = partial(self._onselect, baseline=baseline,
                                    mode=mode, layout=layout)

        click_fun = partial(_imshow_tfr, tfr=data, freq=freqs, yscale=yscale,
                            cmap=(cmap, True), onselect=onselect_callback)
        imshow = partial(_imshow_tfr_unified, tfr=data, freq=freqs, cmap=cmap,
                         onselect=onselect_callback)

        fig = _plot_topo(info=info, times=times, show_func=imshow,
                         click_func=click_fun, layout=layout,
                         colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                         layout_scale=layout_scale, title=title, border=border,
                         x_label='Time (ms)', y_label='Frequency (Hz)',
                         fig_facecolor=fig_facecolor, font_color=font_color,
                         unified=True, img=True)

        add_background_image(fig, fig_background)
        plt_show(show)
        return fig

    def plot_topomap(self, tmin=None, tmax=None, fmin=None, fmax=None,
                     ch_type=None, baseline=None, mode='mean',
                     layout=None, vmin=None, vmax=None, cmap=None,
                     sensors=True, colorbar=True, unit=None, res=64, size=2,
                     cbar_fmt='%1.1e', show_names=False, title=None,
                     axes=None, show=True, outlines='head', head_pos=None):
        """Plot topographic maps of time-frequency intervals of TFR data.

        Parameters
        ----------
        tmin : None | float
            The first time instant to display. If None the first time point
            available is used.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used.
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For 'grad', the gradiometers are
            collected in pairs and the RMS for each pair is plotted.
            If None, then first available channel type from order given
            above is used. Defaults to None.
        baseline : tuple or list of length 2
            The time interval to apply rescaling / baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)),
            mean simply subtracts the mean power, percent is the same as
            applying ratio then mean, logratio is the same as mean but then
            rendered in log-scale, zlogratio is the same as zscore but data
            is rendered in log-scale first.
            If None no baseline correction is applied.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to
            be specified for Neuromag data). If possible, the correct layout
            file is inferred from the data; if no appropriate layout file was
            found, the layout is automatically generated from the sensor
            locations.
        vmin : float | callable | None
            The value specifying the lower bound of the color range. If None,
            and vmax is None, -vmax is used. Else np.min(data) or in case
            data contains only positive values 0. If callable, the output
            equals vmin(data). Defaults to None.
        vmax : float | callable | None
            The value specifying the upper bound of the color range. If None,
            the maximum value is used. If callable, the output equals
            vmax(data). Defaults to None.
        cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
            Colormap to use. If tuple, the first value indicates the colormap
            to use and the second value is a boolean defining interactivity. In
            interactive mode the colors are adjustable by clicking and dragging
            the colorbar with left and right mouse button. Left mouse button
            moves the scale up and down and right mouse button adjusts the
            range. Hitting space bar resets the range. Up and down arrows can
            be used to change the colormap. If None (default), 'Reds' is used
            for all positive data, otherwise defaults to 'RdBu_r'. If
            'interactive', translates to (None, True).
        sensors : bool | str
            Add markers for sensor locations to the plot. Accepts matplotlib
            plot format string (e.g., 'r+' for red plusses). If True, a circle
            will be used (via .add_artist). Defaults to True.
        colorbar : bool
            Plot a colorbar.
        unit : dict | str | None
            The unit of the channel type used for colorbar label. If
            scale is None the unit is automatically determined.
        res : int
            The resolution of the topomap image (n pixels along each side).
        size : float
            Side length per topomap in inches.
        cbar_fmt : str
            String format for colorbar values.
        show_names : bool | callable
            If True, show channel names on top of the map. If a callable is
            passed, channel names will be formatted using the callable; e.g.,
            to delete the prefix 'MEG ' from all channel names, pass the
            function lambda x: x.replace('MEG ', ''). If `mask` is not None,
            only significant sensors will be shown.
        title : str | None
            Title. If None (default), no title is displayed.
        axes : instance of Axes | None
            The axes to plot to. If None the axes is defined automatically.
        show : bool
            Call pyplot.show() at the end.
        outlines : 'head' | 'skirt' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will
            be drawn. If 'skirt' the head scheme will be drawn, but sensors are
            allowed to be plotted outside of the head circle. If dict, each key
            refers to a tuple of x and y positions, the values in 'mask_pos'
            will serve as image mask, and the 'autoshrink' (bool) field will
            trigger automated shrinking of the positions due to points outside
            the outline. Alternatively, a matplotlib patch object can be passed
            for advanced masking options, either directly or as a function that
            returns patches (required for multi-axis plots). If None, nothing
            will be drawn. Defaults to 'head'.
        head_pos : dict | None
            If None (default), the sensors are positioned such that they span
            the head circle. If dict, can have entries 'center' (tuple) and
            'scale' (tuple) for what the center and scale of the head should be
            relative to the electrode locations.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.
        """  # noqa: E501
        from ..viz import plot_tfr_topomap
        return plot_tfr_topomap(self, tmin=tmin, tmax=tmax, fmin=fmin,
                                fmax=fmax, ch_type=ch_type, baseline=baseline,
                                mode=mode, layout=layout, vmin=vmin, vmax=vmax,
                                cmap=cmap, sensors=sensors, colorbar=colorbar,
                                unit=unit, res=res, size=size,
                                cbar_fmt=cbar_fmt, show_names=show_names,
                                title=title, axes=axes, show=show,
                                outlines=outlines, head_pos=head_pos)

    def _check_compat(self, tfr):
        """Check that self and tfr have the same time-frequency ranges."""
        assert np.all(tfr.times == self.times)
        assert np.all(tfr.freqs == self.freqs)

    def __add__(self, tfr):  # noqa: D105
        """Add instances."""
        self._check_compat(tfr)
        out = self.copy()
        out.data += tfr.data
        return out

    def __iadd__(self, tfr):  # noqa: D105
        self._check_compat(tfr)
        self.data += tfr.data
        return self

    def __sub__(self, tfr):  # noqa: D105
        """Subtract instances."""
        self._check_compat(tfr)
        out = self.copy()
        out.data -= tfr.data
        return out

    def __isub__(self, tfr):  # noqa: D105
        self._check_compat(tfr)
        self.data -= tfr.data
        return self

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.nave
        s += ', channels : %d' % self.data.shape[0]
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<AverageTFR  |  %s>" % s

    def save(self, fname, overwrite=False):
        """Save TFR object to hdf5 file.

        Parameters
        ----------
        fname : str
            The file name, which should end with -tfr.h5 .
        overwrite : bool
            If True, overwrite file (if it exists). Defaults to false
        """
        write_tfrs(fname, self, overwrite=overwrite)


class EpochsTFR(_BaseTFR):
    """Container for Time-Frequency data on epochs.

    Can for example store induced power at sensor level.

    Parameters
    ----------
    info : Info
        The measurement info.
    data : ndarray, shape (n_epochs, n_channels, n_freqs, n_times)
        The data.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    comment : str | None, defaults to None
        Comment on the data, e.g., the experimental condition.
    method : str | None, defaults to None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Attributes
    ----------
    ch_names : list
        The names of the channels.

    Notes
    -----
    .. versionadded:: 0.13.0
    """

    @verbose
    def __init__(self, info, data, times, freqs, comment=None,
                 method=None, verbose=None):  # noqa: D102
        self.info = info
        if data.ndim != 4:
            raise ValueError('data should be 4d. Got %d.' % data.ndim)
        n_epochs, n_channels, n_freqs, n_times = data.shape
        if n_channels != len(info['chs']):
            raise ValueError("Number of channels and data size don't match"
                             " (%d != %d)." % (n_channels, len(info['chs'])))
        if n_freqs != len(freqs):
            raise ValueError("Number of frequencies and data size don't match"
                             " (%d != %d)." % (n_freqs, len(freqs)))
        if n_times != len(times):
            raise ValueError("Number of times and data size don't match"
                             " (%d != %d)." % (n_times, len(times)))
        self.data = data
        self.times = np.array(times, dtype=float)
        self.freqs = np.array(freqs, dtype=float)
        self.comment = comment
        self.method = method
        self.preload = True

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", epochs : %d" % self.data.shape[0]
        s += ', channels : %d' % self.data.shape[1]
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochsTFR  |  %s>" % s

    def average(self):
        """Average the data across epochs.

        Returns
        -------
        ave : instance of AverageTFR
            The averaged data.
        """
        data = np.mean(self.data, axis=0)
        return AverageTFR(info=self.info.copy(), data=data,
                          times=self.times.copy(), freqs=self.freqs.copy(),
                          nave=self.data.shape[0],
                          method=self.method)


def combine_tfr(all_tfr, weights='nave'):
    """Merge AverageTFR data by weighted addition.

    Create a new AverageTFR instance, using a combination of the supplied
    instances as its data. By default, the mean (weighted by trials) is used.
    Subtraction can be performed by passing negative weights (e.g., [1, -1]).
    Data must have the same channels and the same time instants.

    Parameters
    ----------
    all_tfr : list of AverageTFR
        The tfr datasets.
    weights : list of float | str
        The weights to apply to the data of each AverageTFR instance.
        Can also be ``'nave'`` to weight according to tfr.nave,
        or ``'equal'`` to use equal weighting (each weighted as ``1/N``).

    Returns
    -------
    tfr : AverageTFR
        The new TFR data.

    Notes
    -----
    .. versionadded:: 0.11.0
    """
    tfr = all_tfr[0].copy()
    if isinstance(weights, string_types):
        if weights not in ('nave', 'equal'):
            raise ValueError('Weights must be a list of float, or "nave" or '
                             '"equal"')
        if weights == 'nave':
            weights = np.array([e.nave for e in all_tfr], float)
            weights /= weights.sum()
        else:  # == 'equal'
            weights = [1. / len(all_tfr)] * len(all_tfr)
    weights = np.array(weights, float)
    if weights.ndim != 1 or weights.size != len(all_tfr):
        raise ValueError('Weights must be the same size as all_tfr')

    ch_names = tfr.ch_names
    for t_ in all_tfr[1:]:
        assert t_.ch_names == ch_names, ValueError("%s and %s do not contain "
                                                   "the same channels"
                                                   % (tfr, t_))
        assert np.max(np.abs(t_.times - tfr.times)) < 1e-7, \
            ValueError("%s and %s do not contain the same time instants"
                       % (tfr, t_))

    # use union of bad channels
    bads = list(set(tfr.info['bads']).union(*(t_.info['bads']
                                              for t_ in all_tfr[1:])))
    tfr.info['bads'] = bads

    # XXX : should be refactored with combined_evoked function
    tfr.data = sum(w * t_.data for w, t_ in zip(weights, all_tfr))
    tfr.nave = max(int(1. / sum(w ** 2 / e.nave
                                for w, e in zip(weights, all_tfr))), 1)
    return tfr


# Utils


def _get_data(inst, return_itc):
    """Get data from Epochs or Evoked instance as epochs x ch x time."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    if not isinstance(inst, (BaseEpochs, Evoked)):
        raise TypeError('inst must be Epochs or Evoked')
    if isinstance(inst, BaseEpochs):
        data = inst.get_data()
    else:
        if return_itc:
            raise ValueError('return_itc must be False for evoked data')
        data = inst.data[np.newaxis].copy()
    return data


def _prepare_picks(info, data, picks):
    """Prepare the picks."""
    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')
    if np.array_equal(picks, np.arange(len(data))):
        picks = slice(None)
    else:
        info = pick_info(info, picks)

    return info, data, picks


def _centered(arr, newsize):
    """Aux Function to center data."""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax, mode,
                 baseline, vmin, vmax, dB, sfreq):
    """Aux Function to prepare tfr computation."""
    from ..viz.utils import _setup_vmin_vmax

    copy = baseline is not None
    data = rescale(data, times, baseline, mode, copy=copy)

    # crop time
    itmin, itmax = None, None
    idx = np.where(_time_mask(times, tmin, tmax, sfreq=sfreq))[0]
    if tmin is not None:
        itmin = idx[0]
    if tmax is not None:
        itmax = idx[-1] + 1

    times = times[itmin:itmax]

    # crop freqs
    ifmin, ifmax = None, None
    idx = np.where(_time_mask(freqs, fmin, fmax, sfreq=sfreq))[0]
    if fmin is not None:
        ifmin = idx[0]
    if fmax is not None:
        ifmax = idx[-1] + 1

    freqs = freqs[ifmin:ifmax]

    # crop data
    data = data[:, ifmin:ifmax, itmin:itmax]

    times *= 1e3
    if dB:
        data = 10 * np.log10((data * data.conj()).real)

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)
    return data, times, freqs, vmin, vmax


def _check_decim(decim):
    """Aux function checking the decim parameter."""
    if isinstance(decim, int):
        decim = slice(None, None, decim)
    elif not isinstance(decim, slice):
        raise(TypeError, '`decim` must be int or slice, got %s instead'
                         % type(decim))
    return decim


# i/o


def write_tfrs(fname, tfr, overwrite=False):
    """Write a TFR dataset to hdf5.

    Parameters
    ----------
    fname : string
        The file name, which should end with -tfr.h5
    tfr : AverageTFR instance, or list of AverageTFR instances
        The TFR dataset, or list of TFR datasets, to save in one file.
        Note. If .comment is not None, a name will be generated on the fly,
        based on the order in which the TFR objects are passed
    overwrite : bool
        If True, overwrite file (if it exists). Defaults to False.

    See Also
    --------
    read_tfrs

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    out = []
    if not isinstance(tfr, (list, tuple)):
        tfr = [tfr]
    for ii, tfr_ in enumerate(tfr):
        comment = ii if tfr_.comment is None else tfr_.comment
        out.append(_prepare_write_tfr(tfr_, condition=comment))
    write_hdf5(fname, out, overwrite=overwrite, title='mnepython')


def _prepare_write_tfr(tfr, condition):
    """Aux function."""
    return (condition, dict(times=tfr.times, freqs=tfr.freqs,
                            data=tfr.data, info=tfr.info,
                            nave=tfr.nave, comment=tfr.comment,
                            method=tfr.method))


def read_tfrs(fname, condition=None):
    """Read TFR datasets from hdf5 file.

    Parameters
    ----------
    fname : string
        The file name, which should end with -tfr.h5 .
    condition : int or str | list of int or str | None
        The condition to load. If None, all conditions will be returned.
        Defaults to None.

    See Also
    --------
    write_tfrs

    Returns
    -------
    tfrs : list of instances of AverageTFR | instance of AverageTFR
        Depending on `condition` either the TFR object or a list of multiple
        TFR objects.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    check_fname(fname, 'tfr', ('-tfr.h5',))

    logger.info('Reading %s ...' % fname)
    tfr_data = read_hdf5(fname, title='mnepython')
    for k, tfr in tfr_data:
        tfr['info'] = Info(tfr['info'])

    if condition is not None:
        tfr_dict = dict(tfr_data)
        if condition not in tfr_dict:
            keys = ['%s' % k for k in tfr_dict]
            raise ValueError('Cannot find condition ("{0}") in this file. '
                             'The file contains "{1}""'
                             .format(condition, " or ".join(keys)))
        out = AverageTFR(**tfr_dict[condition])
    else:
        out = [AverageTFR(**d) for d in list(zip(*tfr_data))[1]]
    return out
