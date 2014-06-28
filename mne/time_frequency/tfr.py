"""A module which implements the continuous wavelet transform
with complex Morlet wavelets.

Author : Alexandre Gramfort, alexandre.gramfort@telecom-paristech.fr (2011)
License : BSD 3-clause

inspired by Matlab code from Sheraz Khan & Brainstorm & SPM
"""

from math import sqrt
from copy import deepcopy
from functools import partial
import numpy as np
from scipy import linalg
from scipy.fftpack import fftn, ifftn

from ..baseline import rescale
from ..parallel import parallel_func
from ..utils import logger, verbose
from ..channels import ContainsMixin, PickDropChannelsMixin
from ..io.pick import pick_info, pick_types
from ..utils import deprecated


def morlet(Fs, freqs, n_cycles=7, sigma=None, zero_mean=False):
    """Compute Wavelets for the given frequency range

    Parameters
    ----------
    Fs : float
        Sampling Frequency
    freqs : array
        frequency range of interest (1 x Frequencies)
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    sigma : float, (optional)
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool
        Make sure the wavelet is zero mean

    Returns
    -------
    Ws : list of array
        Wavelets time series
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
        t = np.arange(0, 5 * sigma_t, 1.0 / Fs)
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


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _cwt_fft(X, Ws, mode="same"):
    """Compute cwt with fft based convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))

    # precompute FFTs of Ws
    fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('Wavelet is too long for such a short signal. '
                             'Reduce the number of cycles.')
        fft_Ws[i] = fftn(W, [fsize])

    for k, x in enumerate(X):
        if mode == "full":
            tfr = np.zeros((n_freqs, fsize), dtype=np.complex128)
        elif mode == "same" or mode == "valid":
            tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)

        fft_x = fftn(x, [fsize])
        for i, W in enumerate(Ws):
            ret = ifftn(fft_x * fft_Ws[i])[:n_times + W.size - 1]
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = _centered(ret, sz)
            else:
                tfr[i, :] = _centered(ret, n_times)
        yield tfr


def _cwt_convolve(X, Ws, mode='same'):
    """Compute time freq decomposition with temporal convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    # Compute convolutions
    for x in X:
        tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)
        for i, W in enumerate(Ws):
            ret = np.convolve(x, W, mode=mode)
            if len(W) > len(x):
                raise ValueError('Wavelet is too long for such a short '
                                 'signal. Reduce the number of cycles.')
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = ret
            else:
                tfr[i] = ret
        yield tfr


def cwt_morlet(X, Fs, freqs, use_fft=True, n_cycles=7.0, zero_mean=False):
    """Compute time freq decomposition with Morlet wavelets

    Parameters
    ----------
    X : array of shape [n_signals, n_times]
        signals (one per line)
    Fs : float
        sampling Frequency
    freqs : array
        Array of frequencies of interest
    use_fft : bool
        Compute convolution with FFT or temoral convolution.
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    zero_mean : bool
        Make sure the wavelets are zero mean.

    Returns
    -------
    tfr : 3D array
        Time Frequency Decompositions (n_signals x n_frequencies x n_times)
    """
    mode = 'same'
    # mode = "valid"
    n_signals, n_times = X.shape
    n_frequencies = len(freqs)

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, freqs, n_cycles=n_cycles, zero_mean=zero_mean)

    if use_fft:
        coefs = _cwt_fft(X, Ws, mode)
    else:
        coefs = _cwt_convolve(X, Ws, mode)

    tfrs = np.empty((n_signals, n_frequencies, n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def cwt(X, Ws, use_fft=True, mode='same', decim=1):
    """Compute time freq decomposition with continuous wavelet transform

    Parameters
    ----------
    X : array of shape [n_signals, n_times]
        signals (one per line)
    Ws : list of array
        Wavelets time series
    use_fft : bool
        Use FFT for convolutions
    mode : 'same' | 'valid' | 'full'
        Convention for convolution
    decim : int
        Temporal decimation factor

    Returns
    -------
    tfr : 3D array
        Time Frequency Decompositions (n_signals x n_frequencies x n_times)
    """
    n_signals, n_times = X[:, ::decim].shape
    n_frequencies = len(Ws)

    if use_fft:
        coefs = _cwt_fft(X, Ws, mode)
    else:
        coefs = _cwt_convolve(X, Ws, mode)

    tfrs = np.empty((n_signals, n_frequencies, n_times), dtype=np.complex)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr[..., ::decim]

    return tfrs


def _time_frequency(X, Ws, use_fft):
    """Aux of time_frequency for parallel computing over channels
    """
    n_epochs, n_times = X.shape
    n_frequencies = len(Ws)
    psd = np.zeros((n_frequencies, n_times))  # PSD
    plf = np.zeros((n_frequencies, n_times), dtype=np.complex)  # phase lock

    mode = 'same'
    if use_fft:
        tfrs = _cwt_fft(X, Ws, mode)
    else:
        tfrs = _cwt_convolve(X, Ws, mode)

    for tfr in tfrs:
        tfr_abs = np.abs(tfr)
        psd += tfr_abs ** 2
        plf += tfr / tfr_abs

    return psd, plf


@verbose
def single_trial_power(data, Fs, frequencies, use_fft=True, n_cycles=7,
                       baseline=None, baseline_mode='ratio', times=None,
                       decim=1, n_jobs=1, zero_mean=False, verbose=None):
    """Compute time-frequency power on single epochs

    Parameters
    ----------
    data : array of shape [n_epochs, n_channels, n_times]
        The epochs
    Fs : float
        Sampling rate
    frequencies : array-like
        The frequencies
    use_fft : bool
        Use the FFT for convolutions or not.
    n_cycles : float | array of float
        Number of cycles  in the Morlet wavelet. Fixed number
        or one per frequency.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'ratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline))
    times : array
        Required to define baseline
    decim : int
        Temporal decimation factor
    n_jobs : int
        The number of epochs to process at the same time
    zero_mean : bool
        Make sure the wavelets are zero mean.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    power : 4D array
        Power estimate (Epochs x Channels x Frequencies x Timepoints).
    """
    mode = 'same'
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data[:, :, ::decim].shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    parallel, my_cwt, _ = parallel_func(cwt, n_jobs)

    logger.info("Computing time-frequency power on single epochs...")

    power = np.empty((n_epochs, n_channels, n_frequencies, n_times),
                     dtype=np.float)

    # Package arguments for `cwt` here to minimize omissions where only one of
    # the two calls below is updated with new function arguments.
    cwt_kw = dict(Ws=Ws, use_fft=use_fft, mode=mode, decim=decim)
    if n_jobs == 1:
        for k, e in enumerate(data):
            power[k] = np.abs(cwt(e, **cwt_kw)) ** 2
    else:
        # Precompute tf decompositions in parallel
        tfrs = parallel(my_cwt(e, **cwt_kw) for e in data)
        for k, tfr in enumerate(tfrs):
            power[k] = np.abs(tfr) ** 2

    # Run baseline correction.  Be sure to decimate the times array as well if
    # needed.
    if times is not None:
        times = times[::decim]
    power = rescale(power, times, baseline, baseline_mode, copy=False)
    return power


def _induced_power(data, Fs, frequencies, use_fft=True, n_cycles=7,
                  decim=1, n_jobs=1, zero_mean=False):
    """Compute time induced power and inter-trial phase-locking factor

    The time frequency decomposition is done with Morlet wavelets

    Parameters
    ----------
    data : array
        3D array of shape [n_epochs, n_channels, n_times]
    Fs : float
        sampling Frequency
    frequencies : array
        Array of frequencies of interest
    use_fft : bool
        Compute transform with fft based convolutions or temporal
        convolutions.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim: int
        Temporal decimation factor
    n_jobs : int
        The number of CPUs used in parallel. All CPUs are used in -1.
        Requires joblib package.
    zero_mean : bool
        Make sure the wavelets are zero mean.

    Returns
    -------
    power : 2D array
        Induced power (Channels x Frequencies x Timepoints).
        Squared amplitude of time-frequency coefficients.
    phase_lock : 2D array
        Phase locking factor in [0, 1] (Channels x Frequencies x Timepoints)
    """
    n_frequencies = len(frequencies)
    n_epochs, n_channels, n_times = data[:, :, ::decim].shape

    # Precompute wavelets for given frequency range to save time
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)

    if n_jobs == 1:
        psd = np.empty((n_channels, n_frequencies, n_times))
        plf = np.empty((n_channels, n_frequencies, n_times), dtype=np.complex)

        for c in range(n_channels):
            X = np.squeeze(data[:, c, :])
            this_psd, this_plf = _time_frequency(X, Ws, use_fft)
            psd[c], plf[c] = this_psd[:, ::decim], this_plf[:, ::decim]
    else:
        parallel, my_time_frequency, _ = parallel_func(_time_frequency, n_jobs)

        psd_plf = parallel(my_time_frequency(np.squeeze(data[:, c, :]),
                                             Ws, use_fft)
                           for c in range(n_channels))

        psd = np.zeros((n_channels, n_frequencies, n_times))
        plf = np.zeros((n_channels, n_frequencies, n_times), dtype=np.complex)
        for c, (psd_c, plf_c) in enumerate(psd_plf):
            psd[c, :, :], plf[c, :, :] = psd_c[:, ::decim], plf_c[:, ::decim]

    psd /= n_epochs
    plf = np.abs(plf) / n_epochs
    return psd, plf


@deprecated("induced_power will be removed in release 0.9. Use "
            "tfr_morlet instead.")
def induced_power(data, Fs, frequencies, use_fft=True, n_cycles=7,
                  decim=1, n_jobs=1, zero_mean=False):
    """Compute time induced power and inter-trial phase-locking factor

    The time frequency decomposition is done with Morlet wavelets

    Parameters
    ----------
    data : array
        3D array of shape [n_epochs, n_channels, n_times]
    Fs : float
        sampling Frequency
    frequencies : array
        Array of frequencies of interest
    use_fft : bool
        Compute transform with fft based convolutions or temporal
        convolutions.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim: int
        Temporal decimation factor
    n_jobs : int
        The number of CPUs used in parallel. All CPUs are used in -1.
        Requires joblib package.
    zero_mean : bool
        Make sure the wavelets are zero mean.

    Returns
    -------
    power : 2D array
        Induced power (Channels x Frequencies x Timepoints).
        Squared amplitude of time-frequency coefficients.
    phase_lock : 2D array
        Phase locking factor in [0, 1] (Channels x Frequencies x Timepoints)
    """
    return _induced_power(data, Fs, frequencies, use_fft=use_fft,
                          n_cycles=n_cycles, decim=decim, n_jobs=n_jobs,
                          zero_mean=zero_mean)


def _preproc_tfr(data, times, freqs, tmin, tmax, fmin, fmax, mode,
                 baseline, vmin, vmax, dB):
    if mode is not None and baseline is not None:
        logger.info("Applying baseline correction '%s' during %s" %
                    (mode, baseline))
        data = rescale(data.copy(), times, baseline, mode)

    # crop time
    itmin, itmax = None, None
    if tmin is not None:
        itmin = np.where(times >= tmin)[0][0]
    if tmax is not None:
        itmax = np.where(times <= tmax)[0][-1]

    times = times[itmin:itmax]

    # crop freqs
    ifmin, ifmax = None, None
    if fmin is not None:
        ifmin = np.where(freqs >= fmin)[0][0]
    if fmax is not None:
        ifmax = np.where(freqs <= fmax)[0][-1]

    freqs = freqs[ifmin:ifmax]

    times *= 1e3
    if dB:
        data = 20 * np.log10(data)
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    return data, times, freqs


# XXX : todo IO of TFRs
class AverageTFR(ContainsMixin, PickDropChannelsMixin):
    """Container for Time-Frequency data

    Can for example store induced power at sensor level or intertrial
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

    Attributes
    ----------
    ch_names : list
        The names of the channels.
    """
    @verbose
    def __init__(self, info, data, times, freqs, verbose=None):
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
        self.times = times
        self.freqs = freqs

    @property
    def ch_names(self):
        return self.info['ch_names']

    @verbose
    def plot(self, picks, baseline=None, mode='mean', tmin=None, tmax=None,
             fmin=None, fmax=None, vmin=None, vmax=None, cmap=None, dB=False,
             colorbar=True, show=True, verbose=None):
        """Plot TFRs in a topography with images

        Parameters
        ----------
        picks : array-like of int
            The indices of the channels to plot.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)).
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
        cmap : matplotlib colormap
            The colormap to use.
        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas
        show : bool
            Call pyplot.show() at the end.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        from ..viz import _imshow_tfr
        import matplotlib.pyplot as plt
        times, freqs = self.times, self.freqs
        data = self.data[picks]
        data, times, freqs = _preproc_tfr(data, times, freqs, tmin, tmax,
                                          fmin, fmax, mode, baseline, vmin,
                                          vmax, dB)

        if mode is not None and baseline is not None:
            logger.info("Applying baseline correction '%s' during %s" %
                        (mode, baseline))
            data = rescale(data, times, baseline, mode)

        tmin, tmax = times[0], times[-1]
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        for k, p in zip(range(len(data)), picks):
            plt.figure()
            _imshow_tfr(plt, 0, tmin, tmax, vmin, vmax, ylim=None,
                        tfr=data[k: k + 1], freq=freqs, x_label='Time (ms)',
                        y_label='Frequency (Hz)', colorbar=colorbar,
                        picker=False)

        if show:
            import matplotlib.pyplot as plt
            plt.show()

    def plot_topo(self, picks=None, baseline=None, mode='mean', tmin=None,
                  tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
                  layout=None, cmap=None, title=None, dB=False, colorbar=True,
                  layout_scale=0.945, show=True):
        """Plot TFRs in a topography with images

        Parameters
        ----------
        picks : array-like of int | None
            The indices of the channels to plot. If None all available
            channels are displayed.
        baseline : None (default) or tuple of length 2
            The time interval to apply baseline correction.
            If None do not apply it. If baseline is (a, b)
            the interval is between "a (s)" and "b (s)".
            If a is None the beginning of the data is used
            and if b is None then b is set to the end of the interval.
            If baseline is equal ot (None, None) all the time
            interval is used.
        mode : None | 'logratio' | 'ratio' | 'zscore' | 'mean' | 'percent'
            Do baseline correction with ratio (power is divided by mean
            power during baseline) or zscore (power is divided by standard
            deviation of power during baseline after subtracting the mean,
            power = [power - mean(power_baseline)] / std(power_baseline)).
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
        cmap : matplotlib colormap
            The colormap to use.
        title : str
            Title of the figure.
        dB : bool
            If True, 20*log10 is applied to the data to get dB.
        colorbar : bool
            If true, colorbar will be added to the plot
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas
        show : bool
            Call pyplot.show() at the end.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        """
        from ..viz import _imshow_tfr, _plot_topo
        times = self.times.copy()
        freqs = self.freqs
        data = self.data
        info = self.info

        if picks is not None:
            data = data[picks]
            info = pick_info(info, picks)

        data, times, freqs = _preproc_tfr(data, times, freqs, tmin, tmax,
                                          fmin, fmax, mode, baseline, vmin,
                                          vmax, dB)

        if layout is None:
            from mne.layouts.layout import find_layout
            layout = find_layout(self.info)

        imshow = partial(_imshow_tfr, tfr=data, freq=freqs)

        fig = _plot_topo(info=info, times=times,
                         show_func=imshow, layout=layout,
                         colorbar=colorbar, vmin=vmin, vmax=vmax, cmap=cmap,
                         layout_scale=layout_scale, title=title, border='w',
                         x_label='Time (ms)', y_label='Frequency (Hz)')

        if show:
            import matplotlib.pyplot as plt
            plt.show()

        return fig

    def _check_compat(self, tfr):
        assert np.all(tfr.times == self.times)
        assert np.all(tfr.freqs == self.freqs)

    def __add__(self, tfr):
        self._check_compat(tfr)
        out = self.copy()
        out.data += tfr.data
        return out

    def __iadd__(self, tfr):
        self._check_compat(tfr)
        self.data += tfr.data
        return self

    def __sub__(self, tfr):
        self._check_compat(tfr)
        out = self.copy()
        out.data -= tfr.data
        return out

    def __isub__(self, tfr):
        self._check_compat(tfr)
        self.data -= tfr.data
        return self

    def copy(self):
        """Return a copy of the instance."""
        return deepcopy(self)

    def __repr__(self):
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ', channels: %d' % self.data.shape[1]
        return "<AverageTFR  |  %s>" % s


def tfr_morlet(epochs, freqs, n_cycles, use_fft=False,
               return_itc=True, decim=1, n_jobs=1):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets

    Parameters
    ----------
    epochs : Epochs
        The epochs.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
    use_fft : bool
        The fft based convolution or not.
    return_itc : bool
        Return intertrial coherence (ITC) as well as averaged power.
    decim : int
        The decimation factor on the time axis. To reduce memory usage.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    power : AverageTFR
        The averaged power.
    itc : AverageTFR
        The intertrial coherence (ITC). Only returned if return_itc
        is True.
    """
    data = epochs.get_data()
    picks = pick_types(epochs.info, meg=True, eeg=True)
    info = pick_info(epochs.info, picks)
    data = data[:, picks, :]
    power, itc = _induced_power(data, Fs=info['sfreq'], frequencies=freqs,
                               n_cycles=n_cycles, n_jobs=n_jobs,
                               use_fft=use_fft, decim=decim,
                               zero_mean=True)
    times = epochs.times[::decim].copy()
    out = AverageTFR(info, power, times, freqs)
    if return_itc:
        out = (out, AverageTFR(info, itc, times, freqs))
    return out
