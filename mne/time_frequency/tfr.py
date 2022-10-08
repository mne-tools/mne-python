"""A module which implements the time-frequency estimation.

Morlet code inspired by Matlab code from Sheraz Khan & Brainstorm & SPM
"""
# Authors : Alexandre Gramfort <alexandre.gramfort@inria.fr>
#           Hari Bharadwaj <hari@nmr.mgh.harvard.edu>
#           Clement Moutard <clement.moutard@polytechnique.org>
#           Jean-Remi King <jeanremi.king@gmail.com>
#
# License : BSD-3-Clause

from copy import deepcopy
from functools import partial

import numpy as np

from .multitaper import dpss_windows

from ..baseline import rescale, _check_baseline
from ..filter import next_fast_len
from ..parallel import parallel_func
from ..utils import (logger, verbose, _time_mask, _freq_mask, check_fname,
                     sizeof_fmt, GetEpochsMixin, TimeMixin,
                     _prepare_read_metadata, fill_doc, _prepare_write_metadata,
                     _check_event_id, _gen_events, SizeMixin, _is_numeric,
                     _check_option, _validate_type, _check_combine,
                     _check_pandas_installed, _check_pandas_index_arguments,
                     _check_time_format, _convert_times, _build_data_frame,
                     warn, _import_h5io_funcs)
from ..channels.channels import UpdateChannelsMixin
from ..channels.layout import _merge_ch_data, _pair_grad_sensors
from ..defaults import (_INTERPOLATION_DEFAULT, _EXTRAPOLATE_DEFAULT,
                        _BORDER_DEFAULT)
from ..io.pick import (pick_info, _picks_to_idx, channel_type, _pick_inst,
                       _get_channel_types)
from ..io.meas_info import Info, ContainsMixin
from ..viz.utils import (figure_nobar, plt_show, _setup_cmap,
                         _connection_line, _prepare_joint_axes,
                         _setup_vmin_vmax, _set_title_multiple_electrodes,
                         _warn_deprecated_vmin_vmax)


def morlet(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        Frequency range of interest (1 x Frequencies).
    n_cycles : float | array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, default None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, default False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be "
                         "greater than 0.")

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
        W /= np.sqrt(0.5) * np.linalg.norm(W.ravel())
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
    n_cycles : float | ndarray, shape (n_freqs,), default 7.
        The number of cycles globally or for each frequency.
    time_bandwidth : float, default 4.0
        Time x Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1).
        Default is 4.0, giving 3 good tapers.
    zero_mean : bool | None, , default False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be "
                         "greater than 0.")

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
            Wk /= np.sqrt(0.5) * np.linalg.norm(Wk.ravel())

            Wm.append(Wk)

        Ws.append(Wm)

    return Ws


# Low level convolution

def _get_nfft(wavelets, X, use_fft=True, check=True):
    n_times = X.shape[-1]
    max_size = max(w.size for w in wavelets)
    if max_size > n_times:
        msg = (f'At least one of the wavelets ({max_size}) is longer than the '
               f'signal ({n_times}). Consider using a longer signal or '
               'shorter wavelets.')
        if check:
            if use_fft:
                warn(msg, UserWarning)
            else:
                raise ValueError(msg)
    nfft = n_times + max_size - 1
    nfft = next_fast_len(nfft)  # 2 ** int(np.ceil(np.log2(nfft)))
    return nfft


def _cwt_gen(X, Ws, *, fsize=0, mode="same", decim=1, use_fft=True):
    """Compute cwt with fft based convolutions or temporal convolutions.

    Parameters
    ----------
    X : array of shape (n_signals, n_times)
        The data.
    Ws : list of array
        Wavelets time series.
    fsize : int
        FFT length.
    mode : {'full', 'valid', 'same'}
        See numpy.convolve.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

    use_fft : bool, default True
        Use the FFT for convolutions or not.

    Returns
    -------
    out : array, shape (n_signals, n_freqs, n_time_decim)
        The time-frequency transform of the signals.
    """
    from scipy.fft import fft, ifft
    _check_option('mode', mode, ['same', 'valid', 'full'])
    decim = _check_decim(decim)
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    _, n_times = X.shape
    n_times_out = X[:, decim].shape[1]
    n_freqs = len(Ws)

    # precompute FFTs of Ws
    if use_fft:
        fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
        for i, W in enumerate(Ws):
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
            if mode == 'valid':
                sz = int(abs(W.size - n_times)) + 1
                offset = (n_times - sz) // 2
                this_slice = slice(offset // decim.step,
                                   (offset + sz) // decim.step)
                if use_fft:
                    ret = _centered(ret, sz)
                tfr[ii, this_slice] = ret[decim]
            elif mode == 'full' and not use_fft:
                start = (W.size - 1) // 2
                end = len(ret) - (W.size // 2)
                ret = ret[start:end]
                tfr[ii, :] = ret[decim]
            else:
                if use_fft:
                    ret = _centered(ret, n_times)
                tfr[ii, :] = ret[decim]
        yield tfr


# Loop of convolution: single trial


def _compute_tfr(epoch_data, freqs, sfreq=1.0, method='morlet',
                 n_cycles=7.0, zero_mean=None, time_bandwidth=None,
                 use_fft=True, decim=1, output='complex', n_jobs=None,
                 verbose=None):
    """Compute time-frequency transforms.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    freqs : array-like of floats, shape (n_freqs)
        The frequencies.
    sfreq : float | int, default 1.0
        Sampling frequency of the data.
    method : 'multitaper' | 'morlet', default 'morlet'
        The time-frequency method. 'morlet' convolves a Morlet wavelet.
        'multitaper' uses complex exponentials windowed with multiple DPSS
        tapers.
    n_cycles : float | array of float, default 7.0
        Number of cycles in the wavelet. Fixed number
        or one per frequency.
    zero_mean : bool | None, default None
        None means True for method='multitaper' and False for method='morlet'.
        If True, make sure the wavelets have a mean of zero.
    time_bandwidth : float, default None
        If None and method=multitaper, will be set to 4.0 (3 tapers).
        Time x (Full) Bandwidth product. Only applies if
        method == 'multitaper'. The number of good tapers (low-bias) is
        chosen automatically based on this to equal floor(time_bandwidth - 1).
    use_fft : bool, default True
        Use the FFT for convolutions or not.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.

    output : str, default 'complex'

        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.

    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels.
    %(verbose)s

    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of ``out`` is ``(n_epochs, n_chans,
        n_freqs, n_times)``, else it is ``(n_chans, n_freqs, n_times)``.
        However, using multitaper method and output ``'complex'`` or
        ``'phase'`` results in shape of ``out`` being ``(n_epochs, n_chans,
        n_tapers, n_freqs, n_times)``. If output is ``'avg_power_itc'``, the
        real values in the ``output`` contain average power' and the imaginary
        values contain the ITC: ``out = avg_power + i * itc``.
    """
    # Check data
    epoch_data = np.asarray(epoch_data)
    if epoch_data.ndim != 3:
        raise ValueError('epoch_data must be of shape (n_epochs, n_chans, '
                         'n_times), got %s' % (epoch_data.shape,))

    # Check params
    freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim = \
        _check_tfr_param(freqs, sfreq, method, zero_mean, n_cycles,
                         time_bandwidth, use_fft, decim, output)

    decim = _check_decim(decim)
    if (freqs > sfreq / 2.).any():
        raise ValueError('Cannot compute freq above Nyquist freq of the data '
                         '(%0.1f Hz), got %0.1f Hz'
                         % (sfreq / 2., freqs.max()))

    # We decimate *after* decomposition, so we need to create our kernels
    # for the original sfreq
    if method == 'morlet':
        W = morlet(sfreq, freqs, n_cycles=n_cycles, zero_mean=zero_mean)
        Ws = [W]  # to have same dimensionality as the 'multitaper' case

    elif method == 'multitaper':
        Ws = _make_dpss(sfreq, freqs, n_cycles=n_cycles,
                        time_bandwidth=time_bandwidth, zero_mean=zero_mean)

    # Check wavelets
    if len(Ws[0][0]) > epoch_data.shape[2]:
        raise ValueError('At least one of the wavelets is longer than the '
                         'signal. Use a longer signal or shorter wavelets.')

    # Initialize output
    n_freqs = len(freqs)
    n_tapers = len(Ws)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim].shape
    if output in ('power', 'phase', 'avg_power', 'itc'):
        dtype = np.float64
    elif output in ('complex', 'avg_power_itc'):
        # avg_power_itc is stored as power + 1i * itc to keep a
        # simple dimensionality
        dtype = np.complex128

    if ('avg_' in output) or ('itc' in output):
        out = np.empty((n_chans, n_freqs, n_times), dtype)
    elif output in ['complex', 'phase'] and method == 'multitaper':
        out = np.empty((n_chans, n_tapers, n_epochs, n_freqs, n_times), dtype)
    else:
        out = np.empty((n_chans, n_epochs, n_freqs, n_times), dtype)

    # Parallel computation
    all_Ws = sum([list(W) for W in Ws], list())
    _get_nfft(all_Ws, epoch_data, use_fft)
    parallel, my_cwt, n_jobs = parallel_func(_time_frequency_loop, n_jobs)

    # Parallelization is applied across channels.
    tfrs = parallel(
        my_cwt(channel, Ws, output, use_fft, 'same', decim, method)
        for channel in epoch_data.transpose(1, 0, 2))

    # FIXME: to avoid overheads we should use np.array_split()
    for channel_idx, tfr in enumerate(tfrs):
        out[channel_idx] = tfr

    if ('avg_' not in output) and ('itc' not in output):
        # This is to enforce that the first dimension is for epochs
        if output in ['complex', 'phase'] and method == 'multitaper':
            out = out.transpose(2, 0, 1, 3, 4)
        else:
            out = out.transpose(1, 0, 2, 3)
    return out


def _check_tfr_param(freqs, sfreq, method, zero_mean, n_cycles,
                     time_bandwidth, use_fft, decim, output):
    """Aux. function to _compute_tfr to check the params validity."""
    # Check freqs
    if not isinstance(freqs, (list, np.ndarray)):
        raise ValueError('freqs must be an array-like, got %s '
                         'instead.' % type(freqs))
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError('freqs must be of shape (n_freqs,), got %s '
                         'instead.' % np.array(freqs.shape))

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
    freqs = np.asarray(freqs)

    # Check n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles = float(n_cycles)
    elif isinstance(n_cycles, (list, np.ndarray)):
        n_cycles = np.array(n_cycles)
        if len(n_cycles) != len(freqs):
            raise ValueError('n_cycles must be a float or an array of length '
                             '%i frequencies, got %i cycles instead.' %
                             (len(freqs), len(n_cycles)))
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
    _check_option('output', output, ['complex', 'power', 'phase',
                                     'avg_power_itc', 'avg_power', 'itc'])
    _check_option('method', method, ['multitaper', 'morlet'])

    return freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim


def _time_frequency_loop(X, Ws, output, use_fft, mode, decim,
                         method=None):
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
    method : str | None
        Used only for multitapering to create tapers dimension in the output
        if ``output in ['complex', 'phase']``.
    """
    # Set output type
    dtype = np.float64
    if output in ['complex', 'avg_power_itc']:
        dtype = np.complex128

    # Init outputs
    decim = _check_decim(decim)
    n_tapers = len(Ws)
    n_epochs, n_times = X[:, decim].shape
    n_freqs = len(Ws[0])
    if ('avg_' in output) or ('itc' in output):
        tfrs = np.zeros((n_freqs, n_times), dtype=dtype)
    elif output in ['complex', 'phase'] and method == 'multitaper':
        tfrs = np.zeros((n_tapers, n_epochs, n_freqs, n_times),
                        dtype=dtype)
    else:
        tfrs = np.zeros((n_epochs, n_freqs, n_times), dtype=dtype)

    # Loops across tapers.
    for taper_idx, W in enumerate(Ws):
        # No need to check here, it's done earlier (outside parallel part)
        nfft = _get_nfft(W, X, use_fft, check=False)
        coefs = _cwt_gen(
            X, W, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)

        # Inter-trial phase locking is apparently computed per taper...
        if 'itc' in output:
            plf = np.zeros((n_freqs, n_times), dtype=np.complex128)

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
            elif output in ['complex', 'phase'] and method == 'multitaper':
                tfrs[taper_idx, epoch_idx] += tfr
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
    if n_tapers > 1 and output not in ['complex', 'phase']:
        tfrs /= n_tapers
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
        ``use_fft=False``. Defaults to ``'same'``.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.

        Defaults to 1.

    Returns
    -------
    tfr : array, shape (n_signals, n_freqs, n_times)
        The time-frequency decompositions.

    See Also
    --------
    mne.time_frequency.tfr_morlet : Compute time-frequency decomposition
                                    with Morlet wavelets.
    """
    nfft = _get_nfft(Ws, X, use_fft)
    return _cwt_array(X, Ws, nfft, mode, decim, use_fft)


def _cwt_array(X, Ws, nfft, mode, decim, use_fft):
    decim = _check_decim(decim)
    coefs = _cwt_gen(
        X, Ws, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)

    n_signals, n_times = X[:, decim].shape
    tfrs = np.empty((n_signals, len(Ws), n_times), dtype=np.complex128)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def _tfr_aux(method, inst, freqs, decim, return_itc, picks, average,
             output=None, **tfr_params):
    from ..epochs import BaseEpochs
    """Help reduce redundancy between tfr_morlet and tfr_multitaper."""
    decim = _check_decim(decim)
    data = _get_data(inst, return_itc)
    info = inst.info.copy()  # make a copy as sfreq can be altered

    info, data = _prepare_picks(info, data, picks, axis=1)
    del picks

    if average:
        if output == 'complex':
            raise ValueError('output must be "power" if average=True')
        if return_itc:
            output = 'avg_power_itc'
        else:
            output = 'avg_power'
    else:
        output = 'power' if output is None else output
        if return_itc:
            raise ValueError('Inter-trial coherence is not supported'
                             ' with average=False')

    out = _compute_tfr(data, freqs, info['sfreq'], method=method,
                       output=output, decim=decim, **tfr_params)
    times = inst.times[decim].copy()
    with info._unlock():
        info['sfreq'] /= decim.step

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
        if isinstance(inst, BaseEpochs):
            meta = deepcopy(inst._metadata)
            evs = deepcopy(inst.events)
            ev_id = deepcopy(inst.event_id)
            selection = deepcopy(inst.selection)
            drop_log = deepcopy(inst.drop_log)
        else:
            # if the input is of class Evoked
            meta = evs = ev_id = selection = drop_log = None

        out = EpochsTFR(info, power, times, freqs, method='%s-power' % method,
                        events=evs, event_id=ev_id, selection=selection,
                        drop_log=drop_log, metadata=meta)

    return out


@verbose
def tfr_morlet(inst, freqs, n_cycles, use_fft=False, return_itc=True, decim=1,
               n_jobs=None, picks=None, zero_mean=True, average=True,
               output='power', verbose=None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.

    Same computation as `~mne.time_frequency.tfr_array_morlet`, but
    operates on `~mne.Epochs` objects instead of
    :class:`NumPy arrays <numpy.ndarray>`.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
    use_fft : bool, default False
        The fft based convolution or not.
    return_itc : bool, default True
        Return inter-trial coherence (ITC) as well as averaged power.
        Must be ``False`` for evoked data.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.
    %(n_jobs)s
    picks : array-like of int | None, default None
        The indices of the channels to decompose. If None, all available
        good data channels are decomposed.
    zero_mean : bool, default True
        Make sure the wavelet has a mean of zero.

        .. versionadded:: 0.13.0
    %(average_tfr)s
    output : str
        Can be "power" (default) or "complex". If "complex", then
        average must be False.

        .. versionadded:: 0.15.0
    %(verbose)s

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
                      zero_mean=zero_mean, output=output)
    return _tfr_aux('morlet', inst, freqs, decim, return_itc, picks,
                    average, **tfr_params)


@verbose
def tfr_array_morlet(epoch_data, sfreq, freqs, n_cycles=7.0,
                     zero_mean=False, use_fft=True, decim=1, output='complex',
                     n_jobs=None, verbose=None):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.

    Same computation as `~mne.time_frequency.tfr_morlet`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float | int
        Sampling frequency of the data.
    freqs : array-like of float, shape (n_freqs,)
        The frequencies.
    n_cycles : float | array of float, default 7.0
        Number of cycles in the Morlet wavelet. Fixed number or one per
        frequency.
    zero_mean : bool | False
        If True, make sure the wavelets have a mean of zero. default False.
    use_fft : bool
        Use the FFT for convolutions or not. default True.
    decim : int | slice
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note::
            Decimation may create aliasing artifacts, yet decimation
            is done after the convolutions.
    output : str, default 'complex'

        * 'complex' : single trial complex.
        * 'power' : single trial power.
        * 'phase' : single trial phase.
        * 'avg_power' : average of single trial power.
        * 'itc' : inter-trial coherence.
        * 'avg_power_itc' : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels. Default 1.
    %(verbose)s

    Returns
    -------
    out : array
        Time frequency transform of epoch_data. If output is in ['complex',
        'phase', 'power'], then shape of out is (n_epochs, n_chans, n_freqs,
        n_times), else it is (n_chans, n_freqs, n_times). If output is
        'avg_power_itc', the real values code for 'avg_power' and the
        imaginary values code for the 'itc': out = avg_power + i * itc.

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
    return _compute_tfr(epoch_data=epoch_data, freqs=freqs,
                        sfreq=sfreq, method='morlet', n_cycles=n_cycles,
                        zero_mean=zero_mean, time_bandwidth=None,
                        use_fft=use_fft, decim=decim, output=output,
                        n_jobs=n_jobs, verbose=verbose)


@verbose
def tfr_multitaper(inst, freqs, n_cycles, time_bandwidth=4.0,
                   use_fft=True, return_itc=True, decim=1,
                   n_jobs=None, picks=None, average=True, verbose=None):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers.

    Same computation as `~mne.time_frequency.tfr_array_multitaper`, but
    operates on `~mne.Epochs` objects instead of
    :class:`NumPy arrays <numpy.ndarray>`.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
        The time-window length is thus T = n_cycles / freq.
    time_bandwidth : float, (optional), default 4.0 (n_tapers=3)
        Time x (Full) Bandwidth product. Should be >= 2.0.
        Choose this along with n_cycles to get desired frequency resolution.
        The number of good tapers (least leakage from far away frequencies)
        is chosen automatically based on this to floor(time_bandwidth - 1).
        E.g., With freq = 20 Hz and n_cycles = 10, we get time = 0.5 s.
        If time_bandwidth = 4., then frequency smoothing is (4 / time) = 8 Hz.
    use_fft : bool, default True
        The fft based convolution or not.
    return_itc : bool, default True
        Return inter-trial coherence (ITC) as well as averaged (or
        single-trial) power.
    decim : int | slice, default 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition.
        If `int`, returns tfr[..., ::decim].
        If `slice`, returns tfr[..., decim].

        .. note:: Decimation may create aliasing artifacts.
    %(n_jobs)s
    %(picks_good_data)s
    %(average_tfr)s
    %(verbose)s

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

class _BaseTFR(ContainsMixin, UpdateChannelsMixin, SizeMixin, TimeMixin):
    """Base TFR class."""

    def __init__(self):
        self.baseline = None

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

    @fill_doc
    def crop(self, tmin=None, tmax=None, fmin=None, fmax=None,
             include_tmax=True):
        """Crop data to a given time interval in place.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        fmin : float | None
            Lowest frequency of selection in Hz.

            .. versionadded:: 0.18.0
        fmax : float | None
            Highest frequency of selection in Hz.

            .. versionadded:: 0.18.0
        %(include_tmax)s

        Returns
        -------
        inst : instance of AverageTFR
            The modified instance.
        """
        super().crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)

        if fmin is not None or fmax is not None:
            freq_mask = _freq_mask(self.freqs, sfreq=self.info['sfreq'],
                                   fmin=fmin, fmax=fmax)
        else:
            freq_mask = slice(None)

        self.freqs = self.freqs[freq_mask]
        # Deal with broadcasting (boolean arrays do not broadcast, but indices
        # do, so we need to convert freq_mask to make use of broadcasting)
        if isinstance(freq_mask, np.ndarray):
            freq_mask = np.where(freq_mask)[0]
        self._data = self._data[..., freq_mask, :]
        return self

    def copy(self):
        """Return a copy of the instance.

        Returns
        -------
        copy : instance of EpochsTFR | instance of AverageTFR
            A copy of the instance.
        """
        return deepcopy(self)

    @verbose
    def apply_baseline(self, baseline, mode='mean', verbose=None):
        """Baseline correct the data.

        Parameters
        ----------
        baseline : array-like, shape (2,)
            The time interval to apply rescaling / baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
            Perform baseline correction by

            - subtracting the mean of baseline values ('mean')
            - dividing by the mean of baseline values ('ratio')
            - dividing by the mean of baseline values and taking the log
              ('logratio')
            - subtracting the mean of baseline values followed by dividing by
              the mean of baseline values ('percent')
            - subtracting the mean of baseline values and dividing by the
              standard deviation of baseline values ('zscore')
            - dividing by the mean of baseline values, taking the log, and
              dividing by the standard deviation of log baseline values
              ('zlogratio')
        %(verbose)s

        Returns
        -------
        inst : instance of AverageTFR
            The modified instance.
        """  # noqa: E501
        self.baseline = _check_baseline(baseline, times=self.times,
                                        sfreq=self.info['sfreq'])
        rescale(self.data, self.times, self.baseline, mode, copy=False)
        return self

    @verbose
    def save(self, fname, overwrite=False, *, verbose=None):
        """Save TFR object to hdf5 file.

        Parameters
        ----------
        fname : str
            The file name, which should end with ``-tfr.h5``.
        %(overwrite)s
        %(verbose)s

        See Also
        --------
        read_tfrs, write_tfrs
        """
        write_tfrs(fname, self, overwrite=overwrite)

    @verbose
    def to_data_frame(self, picks=None, index=None, long_format=False,
                      time_format=None, *, verbose=None):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        additional columns ``'time'``, ``'freq'``, ``'epoch'``, and
        ``'condition'`` (epoch event description) are added, unless ``index``
        is not ``None`` (in which case the columns specified in ``index`` will
        be used to form the DataFrame's index instead). ``'epoch'``, and
        ``'condition'`` are not supported for ``AverageTFR``.

        Parameters
        ----------
        %(picks_all)s
        %(index_df_epo)s
            Valid string values are ``'time'``, ``'freq'``, ``'epoch'``, and
            ``'condition'`` for ``EpochsTFR`` and ``'time'`` and ``'freq'``
            for ``AverageTFR``.
            Defaults to ``None``.
        %(long_format_df_epo)s
        %(time_format_df)s

            .. versionadded:: 0.23
        %(verbose)s

        Returns
        -------
        %(df_return)s
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ['time', 'freq']
        if isinstance(self, EpochsTFR):
            valid_index_args.extend(['epoch', 'condition'])
        valid_time_formats = ['ms', 'timedelta']
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        times = self.times
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        if isinstance(self, EpochsTFR):
            data = self.data[:, picks, :, :]
        else:
            data = self.data[np.newaxis, picks]  # add singleton "epochs" axis
        n_epochs, n_picks, n_freqs, n_times = data.shape
        # reshape to (epochs*freqs*times) x signals
        data = np.moveaxis(data, 1, -1)
        data = data.reshape(n_epochs * n_freqs * n_times, n_picks)
        # prepare extra columns / multiindex
        mindex = list()
        times = np.tile(times, n_epochs * n_freqs)
        times = _convert_times(self, times, time_format)
        mindex.append(('time', times))
        freqs = self.freqs
        freqs = np.tile(np.repeat(freqs, n_times), n_epochs)
        mindex.append(('freq', freqs))
        if isinstance(self, EpochsTFR):
            mindex.append(('epoch', np.repeat(self.selection,
                                              n_times * n_freqs)))
            rev_event_id = {v: k for k, v in self.event_id.items()}
            conditions = [rev_event_id[k] for k in self.events[:, 2]]
            mindex.append(('condition', np.repeat(conditions,
                                                  n_times * n_freqs)))
        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)
        # build DataFrame
        if isinstance(self, EpochsTFR):
            default_index = ['condition', 'epoch', 'freq', 'time']
        else:
            default_index = ['freq', 'time']
        df = _build_data_frame(self, data, picks, long_format, mindex, index,
                               default_index=default_index)
        return df


@fill_doc
class AverageTFR(_BaseTFR):
    """Container for Time-Frequency data.

    Can for example store induced power at sensor level or inter-trial
    coherence.

    Parameters
    ----------
    %(info_not_none)s
    data : ndarray, shape (n_channels, n_freqs, n_times)
        The data.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    nave : int
        The number of averaged TFRs.
    comment : str | None, default None
        Comment on the data, e.g., the experimental condition.
    method : str | None, default None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    %(verbose)s

    Attributes
    ----------
    %(info_not_none)s
    ch_names : list
        The names of the channels.
    nave : int
        Number of averaged epochs.
    data : ndarray, shape (n_channels, n_freqs, n_times)
        The data array.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    comment : str
        Comment on dataset. Can be the condition.
    method : str | None, default None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    """

    @verbose
    def __init__(self, info, data, times, freqs, nave, comment=None,
                 method=None, verbose=None):  # noqa: D102
        super().__init__()
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
        self._set_times(np.array(times, dtype=float))
        self._raw_times = self.times.copy()
        self.freqs = np.array(freqs, dtype=float)
        self.nave = nave
        self.comment = comment
        self.method = method
        self.preload = True

    @verbose
    def plot(self, picks=None, baseline=None, mode='mean', tmin=None,
             tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
             cmap='RdBu_r', dB=False, colorbar=True, show=True, title=None,
             axes=None, layout=None, yscale='auto', mask=None,
             mask_style=None, mask_cmap="Greys", mask_alpha=0.1, combine=None,
             exclude=[], cnorm=None, verbose=None):
        """Plot TFRs as a two-dimensional image(s).

        Parameters
        ----------
        %(picks_good_data)s
        baseline : None (default) or tuple, shape (2,)
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
            Perform baseline correction by

            - subtracting the mean of baseline values ('mean') (default)
            - dividing by the mean of baseline values ('ratio')
            - dividing by the mean of baseline values and taking the log
              ('logratio')
            - subtracting the mean of baseline values followed by dividing by
              the mean of baseline values ('percent')
            - subtracting the mean of baseline values and dividing by the
              standard deviation of baseline values ('zscore')
            - dividing by the mean of baseline values, taking the log, and
              dividing by the standard deviation of log baseline values
              ('zlogratio')

        tmin : None | float
            The first time instant to display. If None the first time point
            available is used. Defaults to None.
        tmax : None | float
            The last time instant to display. If None the last time point
            available is used. Defaults to None.
        fmin : None | float
            The first frequency to display. If None the first frequency
            available is used. Defaults to None.
        fmax : None | float
            The last frequency to display. If None the last frequency
            available is used. Defaults to None.
        vmin : float | None
            The minimum value an the color scale. If vmin is None, the data
            minimum value is used. Defaults to None.
        vmax : float | None
            The maximum value an the color scale. If vmax is None, the data
            maximum value is used. Defaults to None.
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
            If True, 10*log10 is applied to the data to get dB.
            Defaults to False.
        colorbar : bool
            If true, colorbar will be added to the plot. Defaults to True.
        show : bool
            Call pyplot.show() at the end. Defaults to True.
        title : str | 'auto' | None
            String for ``title``. Defaults to None (blank/no title). If
            'auto', and ``combine`` is None, the title for each figure
            will be the channel name. If 'auto' and ``combine`` is not None,
            ``title`` states how many channels were combined into that figure
            and the method that was used for ``combine``. If str, that String
            will be the title for each figure.
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as ``picks``. If instance of Axes, there must be
            only one channel plotted. If ``combine`` is not None, ``axes``
            must either be an instance of Axes, or a list of length 1.
        layout : Layout | None
            Layout instance specifying sensor positions. Used for interactive
            plotting of topographies on rectangle selection. If possible, the
            correct layout is inferred from the data.
        yscale : 'auto' (default) | 'linear' | 'log'
            The scale of y (frequency) axis. 'linear' gives linear y axis,
            'log' leads to log-spaced y axis and 'auto' detects if frequencies
            are log-spaced and only then sets the y axis to 'log'.

            .. versionadded:: 0.14.0
        mask : ndarray | None
            An array of booleans of the same shape as the data. Entries of the
            data that correspond to False in the mask are plotted
            transparently. Useful for, e.g., masking for statistical
            significance.

            .. versionadded:: 0.16.0
        mask_style : None | 'both' | 'contour' | 'mask'
            If ``mask`` is not None: if ``'contour'``, a contour line is drawn
            around the masked areas (``True`` in ``mask``). If ``'mask'``,
            entries not ``True`` in ``mask`` are shown transparently. If
            ``'both'``, both a contour and transparency are used.
            If ``None``, defaults to ``'both'`` if ``mask`` is not None, and is
            ignored otherwise.

            .. versionadded:: 0.17
        mask_cmap : matplotlib colormap | (colormap, bool) | 'interactive'
            The colormap chosen for masked parts of the image (see below), if
            ``mask`` is not ``None``. If None, ``cmap`` is reused. Defaults to
            ``'Greys'``. Not interactive. Otherwise, as ``cmap``.

            .. versionadded:: 0.17
        mask_alpha : float
            A float between 0 and 1. If ``mask`` is not None, this sets the
            alpha level (degree of transparency) for the masked-out segments.
            I.e., if 0, masked-out segments are not visible at all.
            Defaults to 0.1.

            .. versionadded:: 0.16.0
        combine : 'mean' | 'rms' | None
            Type of aggregation to perform across selected channels. If
            None, plot one figure per selected channel.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the
            bad channels are excluded. Defaults to an empty list.
        %(cnorm)s

            .. versionadded:: 0.24
        %(verbose)s

        Returns
        -------
        figs : list of instances of matplotlib.figure.Figure
            A list of figures containing the time-frequency power.
        """  # noqa: E501
        return self._plot(picks=picks, baseline=baseline, mode=mode,
                          tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                          vmin=vmin, vmax=vmax, cmap=cmap, dB=dB,
                          colorbar=colorbar, show=show, title=title,
                          axes=axes, layout=layout, yscale=yscale, mask=mask,
                          mask_style=mask_style, mask_cmap=mask_cmap,
                          mask_alpha=mask_alpha, combine=combine,
                          exclude=exclude, cnorm=cnorm, verbose=verbose)

    @verbose
    def _plot(self, picks=None, baseline=None, mode='mean', tmin=None,
              tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
              cmap='RdBu_r', dB=False, colorbar=True, show=True, title=None,
              axes=None, layout=None, yscale='auto', mask=None,
              mask_style=None, mask_cmap="Greys", mask_alpha=.25,
              combine=None, exclude=None, copy=True,
              source_plot_joint=False, topomap_args=dict(), ch_type=None,
              cnorm=None, verbose=None):
        """Plot TFRs as a two-dimensional image(s).

        See self.plot() for parameters description.
        """
        import matplotlib.pyplot as plt
        from ..viz.topo import _imshow_tfr

        # channel selection
        # simply create a new tfr object(s) with the desired channel selection
        tfr = _preproc_tfr_instance(
            self, picks, tmin, tmax, fmin, fmax, vmin, vmax, dB, mode,
            baseline, exclude, copy)
        del picks

        data = tfr.data
        n_picks = len(tfr.ch_names) if combine is None else 1

        # combine picks
        if combine == 'mean':
            data = data.mean(axis=0, keepdims=True)
        elif combine == 'rms':
            data = np.sqrt((data ** 2).mean(axis=0, keepdims=True))
        elif combine is not None:
            raise ValueError('combine must be None, mean or rms.')

        # figure overhead
        # set plot dimension
        tmin, tmax = tfr.times[[0, -1]]
        if vmax is None:
            vmax = np.abs(data).max()
        if vmin is None:
            vmin = -np.abs(data).max()

        # set colorbar
        cmap = _setup_cmap(cmap)

        # make sure there are as many axes as there will be channels to plot
        if isinstance(axes, list) or isinstance(axes, np.ndarray):
            figs_and_axes = [(ax.get_figure(), ax) for ax in axes]
        elif isinstance(axes, plt.Axes):
            figs_and_axes = [(ax.get_figure(), ax) for ax in [axes]]
        elif axes is None:
            figs = [plt.figure() for i in range(n_picks)]
            figs_and_axes = [(fig, fig.add_subplot(111)) for fig in figs]
        else:
            raise ValueError('axes must be None, plt.Axes, or list '
                             'of plt.Axes.')
        if len(figs_and_axes) != n_picks:
            raise RuntimeError('There must be an axes for each picked '
                               'channel.')

        for idx in range(n_picks):
            fig = figs_and_axes[idx][0]
            ax = figs_and_axes[idx][1]
            onselect_callback = partial(
                tfr._onselect, cmap=cmap, source_plot_joint=source_plot_joint,
                topomap_args={k: v for k, v in topomap_args.items()
                              if k not in {"vmin", "vmax", "cmap", "axes"}})
            _imshow_tfr(
                ax, 0, tmin, tmax, vmin, vmax, onselect_callback, ylim=None,
                tfr=data[idx: idx + 1], freq=tfr.freqs, x_label='Time (s)',
                y_label='Frequency (Hz)', colorbar=colorbar, cmap=cmap,
                yscale=yscale, mask=mask, mask_style=mask_style,
                mask_cmap=mask_cmap, mask_alpha=mask_alpha, cnorm=cnorm)

            if title == 'auto':
                if len(tfr.info['ch_names']) == 1 or combine is None:
                    subtitle = tfr.info['ch_names'][idx]
                else:
                    subtitle = _set_title_multiple_electrodes(
                        None, combine, tfr.info["ch_names"], all=True,
                        ch_type=ch_type)
            else:
                subtitle = title
            fig.suptitle(subtitle)

        plt_show(show)
        return [fig for (fig, ax) in figs_and_axes]

    @verbose
    def plot_joint(self, timefreqs=None, picks=None, baseline=None,
                   mode='mean', tmin=None, tmax=None, fmin=None, fmax=None,
                   vmin=None, vmax=None, cmap='RdBu_r', dB=False,
                   colorbar=True, show=True, title=None,
                   yscale='auto', combine='mean', exclude=[],
                   topomap_args=None, image_args=None, verbose=None):
        """Plot TFRs as a two-dimensional image with topomaps.

        Parameters
        ----------
        timefreqs : None | list of tuple | dict of tuple
            The time-frequency point(s) for which topomaps will be plotted.
            See Notes.
        %(picks_good_data)s
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None, the beginning of the data is used.
            If b is None, then b is set to the end of the interval.
            If baseline is equal to (None, None), the  entire time
            interval is used.
        mode : None | str
            If str, must be one of 'ratio', 'zscore', 'mean', 'percent',
            'logratio' and 'zlogratio'.
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)),
            mean simply subtracts the mean power, percent is the same as
            applying ratio then mean, logratio is the same as mean but then
            rendered in log-scale, zlogratio is the same as zscore but data
            is rendered in log-scale first.
            If None no baseline correction is applied.
        %(tmin_tmax_psd)s
        %(fmin_fmax_psd)s
        vmin : float | None
            The minimum value of the color scale for the image (for
            topomaps, see ``topomap_args``). If vmin is None, the data
            absolute minimum value is used.
        vmax : float | None
            The maximum value of the color scale for the image (for
            topomaps, see ``topomap_args``). If vmax is None, the data
            absolute maximum value is used.
        cmap : matplotlib colormap
            The colormap to use.
        dB : bool
            If True, 10*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot (relating to the
            topomaps). For user defined axes, the colorbar cannot be drawn.
            Defaults to True.
        show : bool
            Call pyplot.show() at the end.
        title : str | None
            String for title. Defaults to None (blank/no title).
        yscale : 'auto' (default) | 'linear' | 'log'
            The scale of y (frequency) axis. 'linear' gives linear y axis,
            'log' leads to log-spaced y axis and 'auto' detects if frequencies
            are log-spaced and only then sets the y axis to 'log'.
        combine : 'mean' | 'rms'
            Type of aggregation to perform across selected channels.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the
            bad channels are excluded. Defaults to an empty list, i.e., ``[]``.
        topomap_args : None | dict
            A dict of ``kwargs`` that are forwarded to
            :func:`mne.viz.plot_topomap` to style the topomaps. ``axes`` and
            ``show`` are ignored. If ``times`` is not in this dict, automatic
            peak detection is used. Beyond that, if ``None``, no customizable
            arguments will be passed.
            Defaults to ``None``.
        image_args : None | dict
            A dict of ``kwargs`` that are forwarded to :meth:`AverageTFR.plot`
            to style the image. ``axes`` and ``show`` are ignored. Beyond that,
            if ``None``, no customizable arguments will be passed.
            Defaults to ``None``.
        %(verbose)s

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.

        Notes
        -----
        ``timefreqs`` has three different modes: tuples, dicts, and auto.
        For (list of) tuple(s) mode, each tuple defines a pair
        (time, frequency) in s and Hz on the TFR plot. For example, to
        look at 10 Hz activity 1 second into the epoch and 3 Hz activity
        300 msec into the epoch, ::

            timefreqs=((1, 10), (.3, 3))

        If provided as a dictionary, (time, frequency) tuples are keys and
        (time_window, frequency_window) tuples are the values - indicating the
        width of the windows (centered on the time and frequency indicated by
        the key) to be averaged over. For example, ::

            timefreqs={(1, 10): (0.1, 2)}

        would translate into a window that spans 0.95 to 1.05 seconds, as
        well as 9 to 11 Hz. If None, a single topomap will be plotted at the
        absolute peak across the time-frequency representation.

        .. versionadded:: 0.16.0
        """  # noqa: E501
        from ..viz.topomap import (_set_contour_locator, plot_topomap,
                                   _get_pos_outlines, _find_topomap_coords)
        import matplotlib.pyplot as plt

        #####################################
        # Handle channels (picks and types) #
        #####################################

        # it would be nicer to let this happen in self._plot,
        # but we need it here to do the loop over the remaining channel
        # types in case a user supplies `picks` that pre-select only one
        # channel type.
        # Nonetheless, it should be refactored for code reuse.
        copy = any(var is not None for var in (exclude, picks, baseline))
        tfr = _pick_inst(self, picks, exclude, copy=copy)
        del picks
        ch_types = _get_channel_types(tfr.info, unique=True)

        # if multiple sensor types: one plot per channel type, recursive call
        if len(ch_types) > 1:
            logger.info("Multiple channel types selected, returning one "
                        "figure per type.")
            figs = list()
            for this_type in ch_types:  # pick corresponding channel type
                type_picks = [idx for idx in range(tfr.info['nchan'])
                              if channel_type(tfr.info, idx) == this_type]
                tf_ = _pick_inst(tfr, type_picks, None, copy=True)
                if len(_get_channel_types(tf_.info, unique=True)) > 1:
                    raise RuntimeError(
                        'Possibly infinite loop due to channel selection '
                        'problem. This should never happen! Please check '
                        'your channel types.')
                figs.append(
                    tf_.plot_joint(
                        timefreqs=timefreqs, picks=None, baseline=baseline,
                        mode=mode, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
                        vmin=vmin, vmax=vmax, cmap=cmap, dB=dB,
                        colorbar=colorbar, show=False, title=title,
                        yscale=yscale, combine=combine,
                        exclude=None, topomap_args=topomap_args,
                        verbose=verbose))
            return figs
        else:
            ch_type = ch_types.pop()

        # Handle timefreqs
        timefreqs = _get_timefreqs(tfr, timefreqs)
        n_timefreqs = len(timefreqs)

        if topomap_args is None:
            topomap_args = dict()
        topomap_args_pass = {k: v for k, v in topomap_args.items() if
                             k not in ('axes', 'show', 'colorbar')}
        topomap_args_pass['outlines'] = topomap_args.get('outlines', 'head')
        topomap_args_pass["contours"] = topomap_args.get('contours', 6)
        topomap_args_pass['ch_type'] = ch_type

        ##############
        # Image plot #
        ##############

        fig, tf_ax, map_ax, cbar_ax = _prepare_joint_axes(n_timefreqs)

        cmap = _setup_cmap(cmap)

        # image plot
        # we also use this to baseline and truncate (times and freqs)
        # (a copy of) the instance
        if image_args is None:
            image_args = dict()
        fig = tfr._plot(
            picks=None, baseline=baseline, mode=mode, tmin=tmin, tmax=tmax,
            fmin=fmin, fmax=fmax, vmin=vmin, vmax=vmax, cmap=cmap, dB=dB,
            colorbar=False, show=False, title=title, axes=tf_ax,
            yscale=yscale, combine=combine, exclude=None, copy=False,
            source_plot_joint=True, topomap_args=topomap_args_pass,
            ch_type=ch_type, **image_args)[0]

        # set and check time and freq limits ...
        # can only do this after the tfr plot because it may change these
        # parameters
        tmax, tmin = tfr.times.max(), tfr.times.min()
        fmax, fmin = tfr.freqs.max(), tfr.freqs.min()
        for time, freq in timefreqs.keys():
            if not (tmin <= time <= tmax):
                error_value = "time point (" + str(time) + " s)"
            elif not (fmin <= freq <= fmax):
                error_value = "frequency (" + str(freq) + " Hz)"
            else:
                continue
            raise ValueError("Requested " + error_value + " exceeds the range"
                             "of the data. Choose different `timefreqs`.")

        ############
        # Topomaps #
        ############

        titles, all_data, all_pos, vlims = [], [], [], []

        # the structure here is a bit complicated to allow aggregating vlims
        # over all topomaps. First, one loop over all timefreqs to collect
        # vlims. Then, find the max vlims and in a second loop over timefreqs,
        # do the actual plotting.
        timefreqs_array = np.array([np.array(keys) for keys in timefreqs])
        order = timefreqs_array[:, 0].argsort()  # sort by time

        for ii, (time, freq) in enumerate(timefreqs_array[order]):
            avg = timefreqs[(time, freq)]
            # set up symmetric windows
            time_half_range, freq_half_range = avg / 2.

            if time_half_range == 0:
                time = tfr.times[np.argmin(np.abs(tfr.times - time))]
            if freq_half_range == 0:
                freq = tfr.freqs[np.argmin(np.abs(tfr.freqs - freq))]

            if (time_half_range == 0) and (freq_half_range == 0):
                sub_map_title = '(%.2f s,\n%.1f Hz)' % (time, freq)
            else:
                sub_map_title = \
                    '(%.1f \u00B1 %.1f s,\n%.1f \u00B1 %.1f Hz)' % \
                    (time, time_half_range, freq, freq_half_range)

            tmin = time - time_half_range
            tmax = time + time_half_range
            fmin = freq - freq_half_range
            fmax = freq + freq_half_range

            data = tfr.data

            # merging grads here before rescaling makes ERDs visible

            sphere = topomap_args.get('sphere')
            if ch_type == 'grad':
                picks = _pair_grad_sensors(tfr.info, topomap_coords=False)
                pos = _find_topomap_coords(
                    tfr.info, picks=picks[::2], sphere=sphere)
                method = combine or 'rms'
                data, _ = _merge_ch_data(data[picks], ch_type, [],
                                         method=method)
                del picks, method
            else:
                pos, _ = _get_pos_outlines(tfr.info, None, sphere)
            del sphere

            all_pos.append(pos)

            data, times, freqs, _, _ = _preproc_tfr(
                data, tfr.times, tfr.freqs, tmin, tmax, fmin, fmax,
                mode, baseline, vmin, vmax, None, tfr.info['sfreq'])

            vlims.append(np.abs(data).max())
            titles.append(sub_map_title)
            all_data.append(data)
            new_t = tfr.times[np.abs(tfr.times - np.median([times])).argmin()]
            new_f = tfr.freqs[np.abs(tfr.freqs - np.median([freqs])).argmin()]
            timefreqs_array[ii] = (new_t, new_f)

        # passing args to the topomap calls
        max_lim = max(vlims)
        _vlim = topomap_args.get('vlim', (None, None))
        # TODO v1.3: remove next 3 lines (vmin/vmax gone from plot_topomap)
        _vmin = topomap_args.get('vmin', None)
        _vmax = topomap_args.get('vmax', None)
        _vlim = list(_warn_deprecated_vmin_vmax(_vlim, _vmin, _vmax))
        # fall back on ± max_lim
        for sign, index in zip((-1, 1), (0, 1)):
            if _vlim[index] is None:
                _vlim[index] = sign * max_lim
        topomap_args_pass['vlim'] = tuple(_vlim)
        locator, contours = _set_contour_locator(
            *_vlim, topomap_args_pass["contours"])
        topomap_args_pass['contours'] = contours

        for ax, title, data, pos in zip(map_ax, titles, all_data, all_pos):
            ax.set_title(title)
            plot_topomap(data.mean(axis=(-1, -2)), pos,
                         cmap=cmap[0], axes=ax, show=False,
                         **topomap_args_pass)

        #############
        # Finish up #
        #############

        if colorbar:
            from matplotlib import ticker
            cbar = plt.colorbar(ax.images[0], cax=cbar_ax)
            if locator is None:
                locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = locator
            cbar.update_ticks()

        plt.subplots_adjust(left=.12, right=.925, bottom=.14,
                            top=1. if title is not None else 1.2)

        # draw the connection lines between time series and topoplots
        lines = [_connection_line(time_, fig, tf_ax, map_ax_, y=freq_,
                                  y_source_transform="transData")
                 for (time_, freq_), map_ax_ in zip(timefreqs_array, map_ax)]
        fig.lines.extend(lines)

        plt_show(show)
        return fig

    @verbose
    def _onselect(self, eclick, erelease, baseline=None, mode=None,
                  cmap=None, source_plot_joint=False, topomap_args=None,
                  verbose=None):
        """Handle rubber band selector in channel tfr."""
        from ..viz.topomap import plot_tfr_topomap, plot_topomap, _add_colorbar
        if abs(eclick.x - erelease.x) < .1 or abs(eclick.y - erelease.y) < .1:
            return
        tmin = round(min(eclick.xdata, erelease.xdata), 5)  # s
        tmax = round(max(eclick.xdata, erelease.xdata), 5)
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
        fig.suptitle('{:.2f} s - {:.2f} s, {:.2f} Hz - {:.2f} Hz'.format(
            tmin, tmax, fmin, fmax), y=0.04)

        if source_plot_joint:
            ax = fig.add_subplot(111)
            data = _preproc_tfr(
                self.data, self.times, self.freqs, tmin, tmax, fmin, fmax,
                None, None, None, None, None, self.info['sfreq'])[0]
            data = data.mean(-1).mean(-1)
            vmax = np.abs(data).max()
            im, _ = plot_topomap(data, self.info, vlim=(-vmax, vmax),
                                 cmap=cmap[0], axes=ax, show=False,
                                 **topomap_args)
            _add_colorbar(ax, im, cmap, title="AU", pad=.1)
            fig.show()
        else:
            for idx, ch_type in enumerate(types):
                ax = fig.add_subplot(1, len(types), idx + 1)
                plot_tfr_topomap(self, ch_type=ch_type, tmin=tmin, tmax=tmax,
                                 fmin=fmin, fmax=fmax,
                                 baseline=baseline, mode=mode, cmap=None,
                                 title=ch_type, vmin=None, vmax=None, axes=ax)

    @verbose
    def plot_topo(self, picks=None, baseline=None, mode='mean', tmin=None,
                  tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
                  layout=None, cmap='RdBu_r', title=None, dB=False,
                  colorbar=True, layout_scale=0.945, show=True,
                  border='none', fig_facecolor='k', fig_background=None,
                  font_color='w', yscale='auto', verbose=None):
        """Plot TFRs in a topography with images.

        Parameters
        ----------
        %(picks_good_data)s
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
            Perform baseline correction by

            - subtracting the mean of baseline values ('mean')
            - dividing by the mean of baseline values ('ratio')
            - dividing by the mean of baseline values and taking the log
              ('logratio')
            - subtracting the mean of baseline values followed by dividing by
              the mean of baseline values ('percent')
            - subtracting the mean of baseline values and dividing by the
              standard deviation of baseline values ('zscore')
            - dividing by the mean of baseline values, taking the log, and
              dividing by the standard deviation of log baseline values
              ('zlogratio')

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
            The minimum value of the color scale. If vmin is None, the data
            minimum value is used.
        vmax : float | None
            The maximum value of the color scale. If vmax is None, the data
            maximum value is used.
        layout : Layout | None
            Layout instance specifying sensor positions. If possible, the
            correct layout is inferred from the data.
        cmap : matplotlib colormap | str
            The colormap to use. Defaults to 'RdBu_r'.
        title : str
            Title of the figure.
        dB : bool
            If True, 10*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot.
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas.
        show : bool
            Call pyplot.show() at the end.
        border : str
            Matplotlib borders style to be used for each sensor plot.
        fig_facecolor : color
            The figure face color. Defaults to black.
        fig_background : None | array
            A background image for the figure. This must be a valid input to
            `matplotlib.pyplot.imshow`. Defaults to None.
        font_color : color
            The color of tick labels in the colorbar. Defaults to white.
        yscale : 'auto' (default) | 'linear' | 'log'
            The scale of y (frequency) axis. 'linear' gives linear y axis,
            'log' leads to log-spaced y axis and 'auto' detects if frequencies
            are log-spaced and only then sets the y axis to 'log'.
        %(verbose)s

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

        info, data = _prepare_picks(info, data, picks, axis=0)
        del picks

        data, times, freqs, vmin, vmax = \
            _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax,
                         mode, baseline, vmin, vmax, dB, info['sfreq'])

        if layout is None:
            from mne import find_layout
            layout = find_layout(self.info)
        onselect_callback = partial(self._onselect, baseline=baseline,
                                    mode=mode)

        click_fun = partial(_imshow_tfr, tfr=data, freq=freqs, yscale=yscale,
                            cmap=(cmap, True), onselect=onselect_callback)
        imshow = partial(_imshow_tfr_unified, tfr=data, freq=freqs, cmap=cmap,
                         onselect=onselect_callback)

        fig = _plot_topo(info=info, times=times, show_func=imshow,
                         click_func=click_fun, layout=layout,
                         colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                         layout_scale=layout_scale, title=title, border=border,
                         x_label='Time (s)', y_label='Frequency (Hz)',
                         fig_facecolor=fig_facecolor, font_color=font_color,
                         unified=True, img=True)

        add_background_image(fig, fig_background)
        plt_show(show)
        return fig

    @fill_doc
    def plot_topomap(
            self, tmin=None, tmax=None, fmin=0., fmax=np.inf, *, ch_type=None,
            baseline=None, mode='mean', sensors=True, show_names=False,
            mask=None, mask_params=None, contours=6, outlines='head',
            sphere=None, image_interp=_INTERPOLATION_DEFAULT,
            extrapolate=_EXTRAPOLATE_DEFAULT, border=_BORDER_DEFAULT, res=64,
            size=2, cmap=None, vlim=(None, None), vmin=None, vmax=None,
            cnorm=None, colorbar=True, cbar_fmt='%1.1e', unit=None, units=None,
            axes=None, title=None, show=True):
        """Plot topographic maps of time-frequency intervals of TFR data.

        Parameters
        ----------
        %(tmin_tmax_psd)s
        %(fmin_fmax_psd)s
        %(ch_type_topomap_psd)s
        baseline : tuple or list of length 2
            The time interval to apply rescaling / baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal to (None, None) all the time
            interval is used.
        mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
            Perform baseline correction by

            - subtracting the mean of baseline values ('mean')
            - dividing by the mean of baseline values ('ratio')
            - dividing by the mean of baseline values and taking the log
              ('logratio')
            - subtracting the mean of baseline values followed by dividing by
              the mean of baseline values ('percent')
            - subtracting the mean of baseline values and dividing by the
              standard deviation of baseline values ('zscore')
            - dividing by the mean of baseline values, taking the log, and
              dividing by the standard deviation of log baseline values
              ('zlogratio')
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap)s

            .. versionadded:: 1.2
        %(vmin_vmax_topomap)s

            .. deprecated:: v1.2
               The ``vmin`` and ``vmax`` parameters will be removed in version 1.3.
               Please use the ``vlim`` parameter instead.
        %(cnorm)s

            .. versionadded:: 1.2
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        unit : str | None
            The unit of the channel type used for colorbar labels.

            .. deprecated:: v1.2
               The "unit" parameter is deprecated and will be removed in v1.3.
               Use "units" instead.
        %(units_topomap)s
        %(axes_plot_topomap)s
        %(title_none)s

            .. deprecated:: v1.2
               The ``title`` parameter will be removed in version 1.3. Please
               use :meth:`fig.suptitle()<matplotlib.figure.Figure.suptitle>`
               instead.
        %(show)s

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.
        """  # noqa: E501
        from ..viz import plot_tfr_topomap
        return plot_tfr_topomap(
            self, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, ch_type=ch_type,
            baseline=baseline, mode=mode, sensors=sensors,
            show_names=show_names, mask=mask, mask_params=mask_params,
            contours=contours, outlines=outlines, sphere=sphere,
            image_interp=image_interp, extrapolate=extrapolate, border=border,
            res=res, size=size, cmap=cmap, vlim=vlim, vmin=vmin, vmax=vmax,
            cnorm=cnorm, colorbar=colorbar, cbar_fmt=cbar_fmt, unit=unit,
            axes=axes, title=title, show=show)

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

    def __truediv__(self, a):  # noqa: D105
        """Divide instances."""
        out = self.copy()
        out /= a
        return out

    def __itruediv__(self, a):  # noqa: D105
        self.data /= a
        return self

    def __mul__(self, a):
        """Multiply source instances."""
        out = self.copy()
        out *= a
        return out

    def __imul__(self, a):  # noqa: D105
        self.data *= a
        return self

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.nave
        s += ', channels : %d' % self.data.shape[0]
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<AverageTFR | %s>" % s


@fill_doc
class EpochsTFR(_BaseTFR, GetEpochsMixin):
    """Container for Time-Frequency data on epochs.

    Can for example store induced power at sensor level.

    Parameters
    ----------
    %(info_not_none)s
    data : ndarray, shape (n_epochs, n_channels, n_freqs, n_times)
        The data.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    comment : str | None, default None
        Comment on the data, e.g., the experimental condition.
    method : str | None, default None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    events : ndarray, shape (n_events, 3) | None
        The events as stored in the Epochs class. If None (default), all event
        values are set to 1 and event time-samples are set to range(n_epochs).
    event_id : dict | None
        Example: dict(auditory=1, visual=3). They keys can be used to access
        associated events. If None, all events will be used and a dict is
        created with string integer names corresponding to the event id
        integers.
    selection : iterable | None
        Iterable of indices of selected epochs. If ``None``, will be
        automatically generated, corresponding to all non-zero events.

        .. versionadded:: 0.23
    drop_log : tuple | None
        Tuple of tuple of strings indicating which epochs have been marked to
        be ignored.

        .. versionadded:: 0.23
    metadata : instance of pandas.DataFrame | None
        A :class:`pandas.DataFrame` containing pertinent information for each
        trial. See :class:`mne.Epochs` for further details.
    %(verbose)s

    Attributes
    ----------
    %(info_not_none)s
    ch_names : list
        The names of the channels.
    data : ndarray, shape (n_epochs, n_channels, n_freqs, n_times)
        The data array.
    times : ndarray, shape (n_times,)
        The time values in seconds.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    comment : string
        Comment on dataset. Can be the condition.
    method : str | None, default None
        Comment on the method used to compute the data, e.g., morlet wavelet.
    events : ndarray, shape (n_events, 3) | None
        Array containing sample information as event_id
    event_id : dict | None
        Names of conditions correspond to event_ids
    selection : array
        List of indices of selected events (not dropped or ignored etc.). For
        example, if the original event array had 4 events and the second event
        has been dropped, this attribute would be np.array([0, 2, 3]).
    drop_log : tuple of tuple
        A tuple of the same length as the event array used to initialize the
        ``EpochsTFR`` object. If the i-th original event is still part of the
        selection, drop_log[i] will be an empty tuple; otherwise it will be
        a tuple of the reasons the event is not longer in the selection, e.g.:

        - ``'IGNORED'``
            If it isn't part of the current subset defined by the user
        - ``'NO_DATA'`` or ``'TOO_SHORT'``
            If epoch didn't contain enough data names of channels that
            exceeded the amplitude threshold
        - ``'EQUALIZED_COUNTS'``
            See :meth:`~mne.Epochs.equalize_event_counts`
        - ``'USER'``
            For user-defined reasons (see :meth:`~mne.Epochs.drop`).

    metadata : pandas.DataFrame, shape (n_events, n_cols) | None
        DataFrame containing pertinent information for each trial
    Notes
    -----
    .. versionadded:: 0.13.0
    """

    @verbose
    def __init__(self, info, data, times, freqs, comment=None, method=None,
                 events=None, event_id=None, selection=None,
                 drop_log=None, metadata=None, verbose=None):
        # noqa: D102
        super().__init__()
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
        if events is None:
            n_epochs = len(data)
            events = _gen_events(n_epochs)
        if selection is None:
            n_epochs = len(data)
            selection = np.arange(n_epochs)
        if drop_log is None:
            n_epochs_prerejection = max(len(events), max(selection) + 1)
            drop_log = tuple(
                () if k in selection else ('IGNORED',)
                for k in range(n_epochs_prerejection))
        else:
            drop_log = drop_log
        # check consistency:
        assert len(selection) == len(events)
        assert len(drop_log) >= len(events)
        assert len(selection) == sum(
            (len(dl) == 0 for dl in drop_log))
        event_id = _check_event_id(event_id, events)
        self.data = data
        self._set_times(np.array(times, dtype=float))
        self._raw_times = self.times.copy()  # needed for decimate
        self._decim = 1
        self.freqs = np.array(freqs, dtype=float)
        self.events = events
        self.event_id = event_id
        self.selection = selection
        self.drop_log = drop_log
        self.comment = comment
        self.method = method
        self.preload = True
        self.metadata = metadata

    @property
    def _detrend_picks(self):
        return list()

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", epochs : %d" % self.data.shape[0]
        s += ', channels : %d' % self.data.shape[1]
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochsTFR | %s>" % s

    def __abs__(self):
        """Take the absolute value."""
        epochs = self.copy()
        epochs.data = np.abs(self.data)
        return epochs

    def average(self, method='mean', dim='epochs', copy=False):
        """Average the data across epochs.

        Parameters
        ----------
        method : str | callable
            How to combine the data. If "mean"/"median", the mean/median
            are returned. Otherwise, must be a callable which, when passed
            an array of shape (n_epochs, n_channels, n_freqs, n_time)
            returns an array of shape (n_channels, n_freqs, n_time).
            Note that due to file type limitations, the kind for all
            these will be "average".
        dim : 'epochs' | 'freqs' | 'times'
            The dimension along which to combine the data.
        copy : bool
            Whether to return a copy of the modified instance,
            or modify in place. Ignored when ``dim='epochs'``
            because a new instance must be returned.

        Returns
        -------
        ave : instance of AverageTFR | EpochsTFR
            The averaged data.

        Notes
        -----
        Passing in ``np.median`` is considered unsafe when there is complex
        data because NumPy doesn't compute the marginal median. Numpy currently
        sorts the complex values by real part and return whatever value is
        computed. Use with caution. We use the marginal median in the
        complex case (i.e. the median of each component separately) if
        one passes in ``median``. See a discussion in scipy:

        https://github.com/scipy/scipy/pull/12676#issuecomment-783370228
        """
        _check_option('dim', dim, ('epochs', 'freqs', 'times'))
        axis = dict(epochs=0, freqs=2, times=self.data.ndim - 1)[dim]

        # return a lambda function for computing a combination metric
        # over epochs
        func = _check_combine(mode=method, axis=axis)
        data = func(self.data)

        n_epochs, n_channels, n_freqs, n_times = self.data.shape
        freqs, times = self.freqs, self.times

        if dim == 'freqs':
            freqs = np.mean(self.freqs, keepdims=True)
            n_freqs = 1
        elif dim == 'times':
            times = np.mean(self.times, keepdims=True)
            n_times = 1
        if dim == 'epochs':
            expected_shape = self._data.shape[1:]
        else:
            expected_shape = (n_epochs, n_channels, n_freqs, n_times)
            data = np.expand_dims(data, axis=axis)

        if data.shape != expected_shape:
            raise RuntimeError(
                f'You passed a function that resulted in data of shape '
                f'{data.shape}, but it should be {expected_shape}.')

        if dim == 'epochs':
            return AverageTFR(info=self.info.copy(), data=data,
                              times=times, freqs=freqs,
                              nave=self.data.shape[0], method=self.method,
                              comment=self.comment)
        elif copy:
            return EpochsTFR(info=self.info.copy(), data=data,
                             times=times, freqs=freqs, method=self.method,
                             comment=self.comment, metadata=self.metadata,
                             events=self.events, event_id=self.event_id)
        else:
            self.data = data
            self._set_times(times)
            self.freqs = freqs
            return self


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
    if isinstance(weights, str):
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


def _prepare_picks(info, data, picks, axis):
    """Prepare the picks."""
    picks = _picks_to_idx(info, picks, exclude='bads')
    info = pick_info(info, picks)
    sl = [slice(None)] * data.ndim
    sl[axis] = picks
    data = data[tuple(sl)]
    return info, data


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
                 baseline, vmin, vmax, dB, sfreq, copy=None):
    """Aux Function to prepare tfr computation."""
    if copy is None:
        copy = baseline is not None
    data = rescale(data, times, baseline, mode, copy=copy)

    if np.iscomplexobj(data):
        # complex amplitude → real power (for plotting); if data are
        # real-valued they should already be power
        data = (data * data.conj()).real

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

    if dB:
        data = 10 * np.log10(data)

    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax)
    return data, times, freqs, vmin, vmax


def _check_decim(decim):
    """Aux function checking the decim parameter."""
    _validate_type(decim, ('int-like', slice), 'decim')
    if not isinstance(decim, slice):
        decim = slice(None, None, int(decim))
    # ensure that we can actually use `decim.step`
    if decim.step is None:
        decim = slice(decim.start, decim.stop, 1)
    return decim


# i/o


@verbose
def write_tfrs(fname, tfr, overwrite=False, *, verbose=None):
    """Write a TFR dataset to hdf5.

    Parameters
    ----------
    fname : str
        The file name, which should end with ``-tfr.h5``.
    tfr : AverageTFR | list of AverageTFR | EpochsTFR
        The TFR dataset, or list of TFR datasets, to save in one file.
        Note. If .comment is not None, a name will be generated on the fly,
        based on the order in which the TFR objects are passed.
    %(overwrite)s
    %(verbose)s

    See Also
    --------
    read_tfrs

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    _, write_hdf5 = _import_h5io_funcs()
    out = []
    if not isinstance(tfr, (list, tuple)):
        tfr = [tfr]
    for ii, tfr_ in enumerate(tfr):
        comment = ii if tfr_.comment is None else tfr_.comment
        out.append(_prepare_write_tfr(tfr_, condition=comment))
    write_hdf5(fname, out, overwrite=overwrite, title='mnepython',
               slash='replace')


def _prepare_write_tfr(tfr, condition):
    """Aux function."""
    attributes = dict(times=tfr.times, freqs=tfr.freqs, data=tfr.data,
                      info=tfr.info, comment=tfr.comment, method=tfr.method)
    if hasattr(tfr, 'nave'):  # if AverageTFR
        attributes['nave'] = tfr.nave
    elif hasattr(tfr, 'events'):  # if EpochsTFR
        attributes['events'] = tfr.events
        attributes['event_id'] = tfr.event_id
        attributes['selection'] = tfr.selection
        attributes['drop_log'] = tfr.drop_log
        attributes['metadata'] = _prepare_write_metadata(tfr.metadata)
    return condition, attributes


def read_tfrs(fname, condition=None):
    """Read TFR datasets from hdf5 file.

    Parameters
    ----------
    fname : str
        The file name, which should end with -tfr.h5 .
    condition : int or str | list of int or str | None
        The condition to load. If None, all conditions will be returned.
        Defaults to None.

    Returns
    -------
    tfr : AverageTFR | list of AverageTFR | EpochsTFR
        Depending on ``condition`` either the TFR object or a list of multiple
        TFR objects.

    See Also
    --------
    write_tfrs

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    check_fname(fname, 'tfr', ('-tfr.h5', '_tfr.h5'))
    read_hdf5, _ = _import_h5io_funcs()

    logger.info('Reading %s ...' % fname)
    tfr_data = read_hdf5(fname, title='mnepython', slash='replace')
    for k, tfr in tfr_data:
        tfr['info'] = Info(tfr['info'])
        tfr['info']._check_consistency()
        if 'metadata' in tfr:
            tfr['metadata'] = _prepare_read_metadata(tfr['metadata'])
    is_average = 'nave' in tfr
    if condition is not None:
        if not is_average:
            raise NotImplementedError('condition not supported when reading '
                                      'EpochsTFR.')
        tfr_dict = dict(tfr_data)
        if condition not in tfr_dict:
            keys = ['%s' % k for k in tfr_dict]
            raise ValueError('Cannot find condition ("{}") in this file. '
                             'The file contains "{}""'
                             .format(condition, " or ".join(keys)))
        out = AverageTFR(**tfr_dict[condition])
    else:
        inst = AverageTFR if is_average else EpochsTFR
        out = [inst(**d) for d in list(zip(*tfr_data))[1]]
    return out


def _get_timefreqs(tfr, timefreqs):
    """Find and/or setup timefreqs for `tfr.plot_joint`."""
    # Input check
    timefreq_error_msg = (
        "Supplied `timefreqs` are somehow malformed. Please supply None, "
        "a list of tuple pairs, or a dict of such tuple pairs, not: ")
    if isinstance(timefreqs, dict):
        for k, v in timefreqs.items():
            for item in (k, v):
                if len(item) != 2 or any((not _is_numeric(n) for n in item)):
                    raise ValueError(timefreq_error_msg, item)
    elif timefreqs is not None:
        if not hasattr(timefreqs, "__len__"):
            raise ValueError(timefreq_error_msg, timefreqs)
        if len(timefreqs) == 2 and all((_is_numeric(v) for v in timefreqs)):
            timefreqs = [tuple(timefreqs)]  # stick a pair of numbers in a list
        else:
            for item in timefreqs:
                if (hasattr(item, "__len__") and len(item) == 2 and
                        all((_is_numeric(n) for n in item))):
                    pass
                else:
                    raise ValueError(timefreq_error_msg, item)

    # If None, automatic identification of max peak
    else:
        from scipy.signal import argrelmax

        order = max((1, tfr.data.shape[2] // 30))
        peaks_idx = argrelmax(tfr.data, order=order, axis=2)
        if peaks_idx[0].size == 0:
            _, p_t, p_f = np.unravel_index(tfr.data.argmax(), tfr.data.shape)
            timefreqs = [(tfr.times[p_t], tfr.freqs[p_f])]
        else:
            peaks = [tfr.data[0, f, t] for f, t in
                     zip(peaks_idx[1], peaks_idx[2])]
            peakmax_idx = np.argmax(peaks)
            peakmax_time = tfr.times[peaks_idx[2][peakmax_idx]]
            peakmax_freq = tfr.freqs[peaks_idx[1][peakmax_idx]]

            timefreqs = [(peakmax_time, peakmax_freq)]

    timefreqs = {
        tuple(k): np.asarray(timefreqs[k]) if isinstance(timefreqs, dict)
        else np.array([0, 0]) for k in timefreqs}

    return timefreqs


def _preproc_tfr_instance(tfr, picks, tmin, tmax, fmin, fmax, vmin, vmax, dB,
                          mode, baseline, exclude, copy=True):
    """Baseline and truncate (times and freqs) a TFR instance."""
    tfr = tfr.copy() if copy else tfr

    exclude = None if picks is None else exclude
    picks = _picks_to_idx(tfr.info, picks, exclude='bads')
    pick_names = [tfr.info['ch_names'][pick] for pick in picks]
    tfr.pick_channels(pick_names)

    if exclude == 'bads':
        exclude = [ch for ch in tfr.info['bads']
                   if ch in tfr.info['ch_names']]
    if exclude is not None:
        tfr.drop_channels(exclude)

    data, times, freqs, _, _ = _preproc_tfr(
        tfr.data, tfr.times, tfr.freqs, tmin, tmax, fmin, fmax, mode,
        baseline, vmin, vmax, dB, tfr.info['sfreq'], copy=False)

    tfr._set_times(times)
    tfr.freqs = freqs
    tfr.data = data

    return tfr


def _check_tfr_complex(tfr, reason='source space estimation'):
    """Check that time-frequency epochs or average data is complex."""
    if not np.iscomplexobj(tfr.data):
        raise RuntimeError(f'Time-frequency data must be complex for {reason}')
