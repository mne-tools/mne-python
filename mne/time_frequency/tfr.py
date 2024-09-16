"""A module which implements the time-frequency estimation.

Morlet code inspired by Matlab code from Sheraz Khan & Brainstorm & SPM
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal import argrelmax

from .._fiff.meas_info import ContainsMixin, Info
from .._fiff.pick import _picks_to_idx, pick_info
from ..baseline import _check_baseline, rescale
from ..channels.channels import UpdateChannelsMixin
from ..channels.layout import _find_topomap_coords, _merge_ch_data, _pair_grad_sensors
from ..defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from ..filter import next_fast_len
from ..parallel import parallel_func
from ..utils import (
    ExtendedTimeMixin,
    GetEpochsMixin,
    SizeMixin,
    _build_data_frame,
    _check_combine,
    _check_event_id,
    _check_fname,
    _check_method_kwargs,
    _check_option,
    _check_pandas_index_arguments,
    _check_pandas_installed,
    _check_time_format,
    _convert_times,
    _ensure_events,
    _freq_mask,
    _import_h5io_funcs,
    _is_numeric,
    _pl,
    _prepare_read_metadata,
    _prepare_write_metadata,
    _time_mask,
    _validate_type,
    check_fname,
    copy_doc,
    copy_function_doc_to_method_doc,
    fill_doc,
    legacy,
    logger,
    object_diff,
    repr_html,
    sizeof_fmt,
    verbose,
    warn,
)
from ..utils.spectrum import _get_instance_type_string
from ..viz.topo import _imshow_tfr, _imshow_tfr_unified, _plot_topo
from ..viz.topomap import (
    _add_colorbar,
    _get_pos_outlines,
    _set_contour_locator,
    plot_tfr_topomap,
    plot_topomap,
)
from ..viz.utils import (
    _make_combine_callable,
    _prepare_joint_axes,
    _set_title_multiple_electrodes,
    _setup_cmap,
    _setup_vmin_vmax,
    add_background_image,
    figure_nobar,
    plt_show,
)
from .multitaper import dpss_windows, tfr_array_multitaper
from .spectrum import EpochsSpectrum


@fill_doc
def morlet(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : float | array-like, shape (n_freqs,)
        Frequencies to compute Morlet wavelets for.
    n_cycles : float | array-like, shape (n_freqs,)
        Number of cycles. Can be a fixed number (float) or one per frequency
        (array-like).
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
    Ws : list of ndarray | ndarray
        The wavelets time series. If ``freqs`` was a float, a single
        ndarray is returned instead of a list of ndarray.

    See Also
    --------
    mne.time_frequency.fwhm

    Notes
    -----
    %(morlet_reference)s
    %(fwhm_morlet_notes)s

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Let's show a simple example of the relationship between ``n_cycles`` and
    the FWHM using :func:`mne.time_frequency.fwhm`:

    .. plot::

        import numpy as np
        import matplotlib.pyplot as plt
        from mne.time_frequency import morlet, fwhm

        sfreq, freq, n_cycles = 1000., 10, 7  # i.e., 700 ms
        this_fwhm = fwhm(freq, n_cycles)
        wavelet = morlet(sfreq=sfreq, freqs=freq, n_cycles=n_cycles)
        M, w = len(wavelet), n_cycles # convert to SciPy convention
        s = w * sfreq / (2 * freq * np.pi)  # from SciPy docs

        _, ax = plt.subplots(layout="constrained")
        colors = dict(real="#66CCEE", imag="#EE6677")
        t = np.arange(-M // 2 + 1, M // 2 + 1) / sfreq
        for kind in ('real', 'imag'):
            ax.plot(
                t, getattr(wavelet, kind), label=kind, color=colors[kind],
            )
        ax.plot(t, np.abs(wavelet), label=f'abs', color='k', lw=1., zorder=6)
        half_max = np.max(np.abs(wavelet)) / 2.
        ax.plot([-this_fwhm / 2., this_fwhm / 2.], [half_max, half_max],
                color='k', linestyle='-', label='FWHM', zorder=6)
        ax.legend(loc='upper right')
        ax.set(xlabel='Time (s)', ylabel='Amplitude')
    """  # noqa: E501
    Ws = list()
    n_cycles = np.array(n_cycles, float).ravel()

    freqs = np.array(freqs, float)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be greater than 0.")

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for each frequency.")
    _check_option("freqs.ndim", freqs.ndim, [0, 1])
    singleton = freqs.ndim == 0
    if singleton:
        freqs = freqs[np.newaxis]
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # sigma_t is the stddev of gaussian window in the time domain; can be
        # scale-dependent or fixed across freqs
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # time vector. We go 5 standard deviations out to make sure we're
        # *very* close to zero at the ends. We also make sure that there's a
        # sample at exactly t=0
        t = np.arange(0.0, 5.0 * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        if zero_mean:
            # this offset is equivalent to the κ_σ term in Wikipedia's
            # equations, and satisfies the "admissibility criterion" for CWTs
            real_offset = np.exp(-2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        gaussian_envelope = np.exp(-(t**2) / (2.0 * sigma_t**2))
        W = oscillation * gaussian_envelope
        # the scaling factor here is proportional to what is used in
        # Tallon-Baudry 1997: (sigma_t*sqrt(pi))^(-1/2).  It yields a wavelet
        # with norm sqrt(2) for the full wavelet / norm 1 for the real part
        W /= np.sqrt(0.5) * np.linalg.norm(W.ravel())
        Ws.append(W)
    if singleton:
        Ws = Ws[0]
    return Ws


def fwhm(freq, n_cycles):
    """Compute the full-width half maximum of a Morlet wavelet.

    Uses the formula from :footcite:t:`Cohen2019`.

    Parameters
    ----------
    freq : float
        The oscillation frequency of the wavelet.
    n_cycles : float
        The duration of the wavelet, expressed as the number of oscillation
        cycles.

    Returns
    -------
    fwhm : float
        The full-width half maximum of the wavelet.

    Notes
    -----
     .. versionadded:: 1.3

    References
    ----------
    .. footbibliography::
    """
    return n_cycles * np.sqrt(2 * np.log(2)) / (np.pi * freq)


def _make_dpss(
    sfreq,
    freqs,
    n_cycles=7.0,
    time_bandwidth=4.0,
    zero_mean=False,
    return_weights=False,
):
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
    return_weights : bool
        Whether to return the concentration weights.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be greater than 0.")

    if time_bandwidth < 2.0:
        raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
    n_taps = int(np.floor(time_bandwidth - 1))
    n_cycles = np.atleast_1d(n_cycles)

    if n_cycles.size != 1 and n_cycles.size != len(freqs):
        raise ValueError("n_cycles should be fixed or defined for each frequency.")

    for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles) != 1:
                this_n_cycles = n_cycles[k]
            else:
                this_n_cycles = n_cycles[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0.0, t_win, 1.0 / sfreq)
            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.0))

            # Get dpss tapers
            tapers, conc = dpss_windows(
                t.shape[0], time_bandwidth / 2.0, n_taps, sym=False
            )

            Wk = oscillation * tapers[m]
            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= np.sqrt(0.5) * np.linalg.norm(Wk.ravel())

            Wm.append(Wk)

        Ws.append(Wm)
    if return_weights:
        return Ws, conc
    return Ws


# Low level convolution


def _get_nfft(wavelets, X, use_fft=True, check=True):
    n_times = X.shape[-1]
    max_size = max(w.size for w in wavelets)
    if max_size > n_times:
        msg = (
            f"At least one of the wavelets ({max_size}) is longer than the "
            f"signal ({n_times}). Consider using a longer signal or "
            "shorter wavelets."
        )
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
    _check_option("mode", mode, ["same", "valid", "full"])
    decim = _ensure_slice(decim)
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
                ret = ifft(fft_x * fft_Ws[ii])[: n_times + W.size - 1]
            else:
                # Work around multarray.correlate->OpenBLAS bug on ppc64le
                # ret = np.correlate(x, W, mode=mode)
                ret = np.convolve(x, W.real, mode=mode) + 1j * np.convolve(
                    x, W.imag, mode=mode
                )

            # Center and decimate decomposition
            if mode == "valid":
                sz = int(abs(W.size - n_times)) + 1
                offset = (n_times - sz) // 2
                this_slice = slice(offset // decim.step, (offset + sz) // decim.step)
                if use_fft:
                    ret = _centered(ret, sz)
                tfr[ii, this_slice] = ret[decim]
            elif mode == "full" and not use_fft:
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


def _compute_tfr(
    epoch_data,
    freqs,
    sfreq=1.0,
    method="morlet",
    n_cycles=7.0,
    zero_mean=None,
    time_bandwidth=None,
    use_fft=True,
    decim=1,
    output="complex",
    n_jobs=None,
    *,
    verbose=None,
):
    """Compute time-frequency transforms.

    Parameters
    ----------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        The epochs.default ``'complex'``
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

    output : str

        * 'complex' (default) : single trial complex.
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
        raise ValueError(
            "epoch_data must be of shape (n_epochs, n_chans, "
            f"n_times), got {epoch_data.shape}"
        )

    # Check params
    freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim = _check_tfr_param(
        freqs,
        sfreq,
        method,
        zero_mean,
        n_cycles,
        time_bandwidth,
        use_fft,
        decim,
        output,
    )

    decim = _ensure_slice(decim)
    if (freqs > sfreq / 2.0).any():
        raise ValueError(
            "Cannot compute freq above Nyquist freq of the data "
            f"({sfreq / 2.0:0.1f} Hz), got {freqs.max():0.1f} Hz"
        )

    # We decimate *after* decomposition, so we need to create our kernels
    # for the original sfreq
    if method == "morlet":
        W = morlet(sfreq, freqs, n_cycles=n_cycles, zero_mean=zero_mean)
        Ws = [W]  # to have same dimensionality as the 'multitaper' case

    elif method == "multitaper":
        Ws = _make_dpss(
            sfreq,
            freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
            zero_mean=zero_mean,
        )

    # Check wavelets
    if len(Ws[0][0]) > epoch_data.shape[2]:
        raise ValueError(
            "At least one of the wavelets is longer than the "
            "signal. Use a longer signal or shorter wavelets."
        )

    # Initialize output
    n_freqs = len(freqs)
    n_tapers = len(Ws)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim].shape
    if output in ("power", "phase", "avg_power", "itc"):
        dtype = np.float64
    elif output in ("complex", "avg_power_itc"):
        # avg_power_itc is stored as power + 1i * itc to keep a
        # simple dimensionality
        dtype = np.complex128

    if ("avg_" in output) or ("itc" in output):
        out = np.empty((n_chans, n_freqs, n_times), dtype)
    elif output in ["complex", "phase"] and method == "multitaper":
        out = np.empty((n_chans, n_tapers, n_epochs, n_freqs, n_times), dtype)
    else:
        out = np.empty((n_chans, n_epochs, n_freqs, n_times), dtype)

    # Parallel computation
    all_Ws = sum([list(W) for W in Ws], list())
    _get_nfft(all_Ws, epoch_data, use_fft)
    parallel, my_cwt, n_jobs = parallel_func(_time_frequency_loop, n_jobs)

    # Parallelization is applied across channels.
    tfrs = parallel(
        my_cwt(channel, Ws, output, use_fft, "same", decim, method)
        for channel in epoch_data.transpose(1, 0, 2)
    )

    # FIXME: to avoid overheads we should use np.array_split()
    for channel_idx, tfr in enumerate(tfrs):
        out[channel_idx] = tfr

    if ("avg_" not in output) and ("itc" not in output):
        # This is to enforce that the first dimension is for epochs
        if output in ["complex", "phase"] and method == "multitaper":
            out = out.transpose(2, 0, 1, 3, 4)
        else:
            out = out.transpose(1, 0, 2, 3)
    return out


def _check_tfr_param(
    freqs, sfreq, method, zero_mean, n_cycles, time_bandwidth, use_fft, decim, output
):
    """Aux. function to _compute_tfr to check the params validity."""
    # Check freqs
    if not isinstance(freqs, list | np.ndarray):
        raise ValueError(f"freqs must be an array-like, got {type(freqs)} instead.")
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError(
            f"freqs must be of shape (n_freqs,), got {np.array(freqs.shape)} "
            "instead."
        )

    # Check sfreq
    if not isinstance(sfreq, float | int):
        raise ValueError(f"sfreq must be a float or an int, got {type(sfreq)} instead.")
    sfreq = float(sfreq)

    # Default zero_mean = True if multitaper else False
    zero_mean = method == "multitaper" if zero_mean is None else zero_mean
    if not isinstance(zero_mean, bool):
        raise ValueError(
            f"zero_mean should be of type bool, got {type(zero_mean)}. instead"
        )
    freqs = np.asarray(freqs)

    # Check n_cycles
    if isinstance(n_cycles, int | float):
        n_cycles = float(n_cycles)
    elif isinstance(n_cycles, list | np.ndarray):
        n_cycles = np.array(n_cycles)
        if len(n_cycles) != len(freqs):
            raise ValueError(
                "n_cycles must be a float or an array of length "
                f"{len(freqs)} frequencies, got {len(n_cycles)} cycles instead."
            )
    else:
        raise ValueError(
            f"n_cycles must be a float or an array, got {type(n_cycles)} instead."
        )

    # Check time_bandwidth
    if (method == "morlet") and (time_bandwidth is not None):
        raise ValueError('time_bandwidth only applies to "multitaper" method.')
    elif method == "multitaper":
        time_bandwidth = 4.0 if time_bandwidth is None else float(time_bandwidth)

    # Check use_fft
    if not isinstance(use_fft, bool):
        raise ValueError(f"use_fft must be a boolean, got {type(use_fft)} instead.")
    # Check decim
    if isinstance(decim, int):
        decim = slice(None, None, decim)
    if not isinstance(decim, slice):
        raise ValueError(
            f"decim must be an integer or a slice, got {type(decim)} instead."
        )

    # Check output
    _check_option(
        "output",
        output,
        ["complex", "power", "phase", "avg_power_itc", "avg_power", "itc"],
    )
    _check_option("method", method, ["multitaper", "morlet"])

    return freqs, sfreq, zero_mean, n_cycles, time_bandwidth, decim


def _time_frequency_loop(X, Ws, output, use_fft, mode, decim, method=None):
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
    if output in ["complex", "avg_power_itc"]:
        dtype = np.complex128

    # Init outputs
    decim = _ensure_slice(decim)
    n_tapers = len(Ws)
    n_epochs, n_times = X[:, decim].shape
    n_freqs = len(Ws[0])
    if ("avg_" in output) or ("itc" in output):
        tfrs = np.zeros((n_freqs, n_times), dtype=dtype)
    elif output in ["complex", "phase"] and method == "multitaper":
        tfrs = np.zeros((n_tapers, n_epochs, n_freqs, n_times), dtype=dtype)
    else:
        tfrs = np.zeros((n_epochs, n_freqs, n_times), dtype=dtype)

    # Loops across tapers.
    for taper_idx, W in enumerate(Ws):
        # No need to check here, it's done earlier (outside parallel part)
        nfft = _get_nfft(W, X, use_fft, check=False)
        coefs = _cwt_gen(X, W, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)

        # Inter-trial phase locking is apparently computed per taper...
        if "itc" in output:
            plf = np.zeros((n_freqs, n_times), dtype=np.complex128)

        # Loop across epochs
        for epoch_idx, tfr in enumerate(coefs):
            # Transform complex values
            if output in ["power", "avg_power"]:
                tfr = (tfr * tfr.conj()).real  # power
            elif output == "phase":
                tfr = np.angle(tfr)
            elif output == "avg_power_itc":
                tfr_abs = np.abs(tfr)
                plf += tfr / tfr_abs  # phase
                tfr = tfr_abs**2  # power
            elif output == "itc":
                plf += tfr / np.abs(tfr)  # phase
                continue  # not need to stack anything else than plf

            # Stack or add
            if ("avg_" in output) or ("itc" in output):
                tfrs += tfr
            elif output in ["complex", "phase"] and method == "multitaper":
                tfrs[taper_idx, epoch_idx] += tfr
            else:
                tfrs[epoch_idx] += tfr

        # Compute inter trial coherence
        if output == "avg_power_itc":
            tfrs += 1j * np.abs(plf)
        elif output == "itc":
            tfrs += np.abs(plf)

    # Normalization of average metrics
    if ("avg_" in output) or ("itc" in output):
        tfrs /= n_epochs

    # Normalization by number of taper
    if n_tapers > 1 and output not in ["complex", "phase"]:
        tfrs /= n_tapers
    return tfrs


@fill_doc
def cwt(X, Ws, use_fft=True, mode="same", decim=1):
    """Compute time-frequency decomposition with continuous wavelet transform.

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
    %(decim_tfr)s

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
    decim = _ensure_slice(decim)
    coefs = _cwt_gen(X, Ws, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)

    n_signals, n_times = X[:, decim].shape
    tfrs = np.empty((n_signals, len(Ws), n_times), dtype=np.complex128)
    for k, tfr in enumerate(coefs):
        tfrs[k] = tfr

    return tfrs


def _tfr_aux(
    method, inst, freqs, decim, return_itc, picks, average, output, **tfr_params
):
    from ..epochs import BaseEpochs

    kwargs = dict(
        method=method,
        freqs=freqs,
        picks=picks,
        decim=decim,
        output=output,
        **tfr_params,
    )
    if isinstance(inst, BaseEpochs):
        kwargs.update(average=average, return_itc=return_itc)
    elif average:
        logger.info("inst is Evoked, setting `average=False`")
        average = False
    if average and output == "complex":
        raise ValueError('output must be "power" if average=True')
    if not average and return_itc:
        raise ValueError("Inter-trial coherence is not supported with average=False")
    return inst.compute_tfr(**kwargs)


@legacy(alt='.compute_tfr(method="morlet")')
@verbose
def tfr_morlet(
    inst,
    freqs,
    n_cycles,
    use_fft=False,
    return_itc=True,
    decim=1,
    n_jobs=None,
    picks=None,
    zero_mean=True,
    average=True,
    output="power",
    verbose=None,
):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.

    Same computation as `~mne.time_frequency.tfr_array_morlet`, but
    operates on `~mne.Epochs` or `~mne.Evoked` objects instead of
    :class:`NumPy arrays <numpy.ndarray>`.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    %(freqs_tfr_array)s
    %(n_cycles_tfr)s
    use_fft : bool, default False
        The fft based convolution or not.
    return_itc : bool, default True
        Return inter-trial coherence (ITC) as well as averaged power.
        Must be ``False`` for evoked data.
    %(decim_tfr)s
    %(n_jobs)s
    picks : array-like of int | None, default None
        The indices of the channels to decompose. If None, all available
        good data channels are decomposed.
    zero_mean : bool, default True
        Make sure the wavelet has a mean of zero.

        .. versionadded:: 0.13.0
    %(average_tfr)s
    output : str
        Can be ``"power"`` (default) or ``"complex"``. If ``"complex"``, then
        ``average`` must be ``False``.

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

    Notes
    -----
    %(morlet_reference)s
    %(temporal_window_tfr_intro)s
    %(temporal_window_tfr_morlet_notes)s

    See :func:`mne.time_frequency.morlet` for more information about the
    Morlet wavelet.

    References
    ----------
    .. footbibliography::
    """
    tfr_params = dict(
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        use_fft=use_fft,
        zero_mean=zero_mean,
        output=output,
    )
    return _tfr_aux(
        "morlet", inst, freqs, decim, return_itc, picks, average, **tfr_params
    )


@verbose
def tfr_array_morlet(
    data,
    sfreq,
    freqs,
    n_cycles=7.0,
    zero_mean=True,
    use_fft=True,
    decim=1,
    output="complex",
    n_jobs=None,
    *,
    verbose=None,
):
    """Compute Time-Frequency Representation (TFR) using Morlet wavelets.

    Same computation as `~mne.time_frequency.tfr_morlet`, but operates on
    :class:`NumPy arrays <numpy.ndarray>` instead of `~mne.Epochs` objects.

    Parameters
    ----------
    data : array of shape (n_epochs, n_channels, n_times)
        The epochs.
    sfreq : float | int
        Sampling frequency of the data.
    %(freqs_tfr_array)s
    %(n_cycles_tfr)s
    zero_mean : bool | None
        If True, make sure the wavelets have a mean of zero. default False.

        .. versionchanged:: 1.8
            The default will change from ``zero_mean=False`` in 1.6 to ``True`` in
            1.8.

    use_fft : bool
        Use the FFT for convolutions or not. default True.
    %(decim_tfr)s
    output : str, default ``'complex'``

        * ``'complex'`` : single trial complex.
        * ``'power'`` : single trial power.
        * ``'phase'`` : single trial phase.
        * ``'avg_power'`` : average of single trial power.
        * ``'itc'`` : inter-trial coherence.
        * ``'avg_power_itc'`` : average of single trial power and inter-trial
          coherence across trials.
    %(n_jobs)s
        The number of epochs to process at the same time. The parallelization
        is implemented across channels. Default 1.
    %(verbose)s

    Returns
    -------
    out : array
        Time frequency transform of ``data``.

        - if ``output in ('complex', 'phase', 'power')``, array of shape
          ``(n_epochs, n_chans, n_freqs, n_times)``
        - else, array of shape ``(n_chans, n_freqs, n_times)``

        If ``output`` is ``'avg_power_itc'``, the real values in ``out``
        contain the average power and the imaginary values contain the ITC:
        :math:`out = power_{avg} + i * itc`.

    See Also
    --------
    mne.time_frequency.tfr_morlet
    mne.time_frequency.tfr_multitaper
    mne.time_frequency.tfr_array_multitaper
    mne.time_frequency.tfr_stockwell
    mne.time_frequency.tfr_array_stockwell

    Notes
    -----
    %(morlet_reference)s
    %(temporal_window_tfr_intro)s
    %(temporal_window_tfr_morlet_notes)s

    .. versionadded:: 0.14.0

    References
    ----------
    .. footbibliography::
    """
    return _compute_tfr(
        epoch_data=data,
        freqs=freqs,
        sfreq=sfreq,
        method="morlet",
        n_cycles=n_cycles,
        zero_mean=zero_mean,
        time_bandwidth=None,
        use_fft=use_fft,
        decim=decim,
        output=output,
        n_jobs=n_jobs,
        verbose=verbose,
    )


@legacy(alt='.compute_tfr(method="multitaper")')
@verbose
def tfr_multitaper(
    inst,
    freqs,
    n_cycles,
    time_bandwidth=4.0,
    use_fft=True,
    return_itc=True,
    decim=1,
    n_jobs=None,
    picks=None,
    average=True,
    *,
    verbose=None,
):
    """Compute Time-Frequency Representation (TFR) using DPSS tapers.

    Same computation as :func:`~mne.time_frequency.tfr_array_multitaper`, but
    operates on :class:`~mne.Epochs` or :class:`~mne.Evoked` objects instead of
    :class:`NumPy arrays <numpy.ndarray>`.

    Parameters
    ----------
    inst : Epochs | Evoked
        The epochs or evoked object.
    %(freqs_tfr_array)s
    %(n_cycles_tfr)s
    %(time_bandwidth_tfr)s
    use_fft : bool, default True
        The fft based convolution or not.
    return_itc : bool, default True
        Return inter-trial coherence (ITC) as well as averaged (or
        single-trial) power.
    %(decim_tfr)s
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
    %(temporal_window_tfr_intro)s
    %(temporal_window_tfr_multitaper_notes)s
    %(time_bandwidth_tfr_notes)s

    .. versionadded:: 0.9.0
    """
    from ..epochs import EpochsArray
    from ..evoked import Evoked

    tfr_params = dict(
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        use_fft=use_fft,
        zero_mean=True,
        time_bandwidth=time_bandwidth,
    )
    if isinstance(inst, Evoked) and not average:
        # convert AverageTFR to EpochsTFR for backwards compatibility
        inst = EpochsArray(inst.data[np.newaxis], inst.info, tmin=inst.tmin, proj=False)
    return _tfr_aux(
        method="multitaper",
        inst=inst,
        freqs=freqs,
        decim=decim,
        return_itc=return_itc,
        picks=picks,
        average=average,
        output="power",
        **tfr_params,
    )


# TFR(s) class


@fill_doc
class BaseTFR(ContainsMixin, UpdateChannelsMixin, SizeMixin, ExtendedTimeMixin):
    """Base class for RawTFR, EpochsTFR, and AverageTFR (for type checking only).

    .. note::
        This class should not be instantiated directly; it is provided in the public API
        only for type-checking purposes (e.g., ``isinstance(my_obj, BaseTFR)``). To
        create TFR objects, use the ``.compute_tfr()`` methods on :class:`~mne.io.Raw`,
        :class:`~mne.Epochs`, or :class:`~mne.Evoked`, or use the constructors listed
        below under "See Also".

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The data from which to compute the time-frequency representation.
    %(method_tfr)s
    %(freqs_tfr)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(decim_tfr)s
    %(n_jobs)s
    %(reject_by_annotation_tfr)s
    %(verbose)s
    %(method_kw_tfr)s

    See Also
    --------
    mne.time_frequency.RawTFR
    mne.time_frequency.RawTFRArray
    mne.time_frequency.EpochsTFR
    mne.time_frequency.EpochsTFRArray
    mne.time_frequency.AverageTFR
    mne.time_frequency.AverageTFRArray
    """

    def __init__(
        self,
        inst,
        method,
        freqs,
        tmin,
        tmax,
        picks,
        proj,
        *,
        decim,
        n_jobs,
        reject_by_annotation=None,
        verbose=None,
        **method_kw,
    ):
        from ..epochs import BaseEpochs
        from ._stockwell import tfr_array_stockwell

        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        if method is None or freqs is None:
            problem = [
                f"{k}=None"
                for k, v in dict(method=method, freqs=freqs).items()
                if v is None
            ]
            # TODO when py3.11 is min version, replace if/elif/else block with
            # classname = inspect.currentframe().f_back.f_code.co_qualname.split(".")[0]
            _varnames = inspect.currentframe().f_back.f_code.co_varnames
            if "BaseRaw" in _varnames:
                classname = "RawTFR"
            elif "Evoked" in _varnames:
                classname = "AverageTFR"
            else:
                assert "BaseEpochs" in _varnames and "Evoked" not in _varnames
                classname = "EpochsTFR"
            # end TODO
            raise ValueError(
                f'{classname} got unsupported parameter value{_pl(problem)} '
                f'{" and ".join(problem)}.'
            )
        # shim for tfr_array_morlet deprecation warning (TODO: remove after 1.7 release)
        if method == "morlet":
            method_kw.setdefault("zero_mean", True)
        # check method
        valid_methods = ["morlet", "multitaper"]
        if isinstance(inst, BaseEpochs):
            valid_methods.append("stockwell")
        method = _check_option("method", method, valid_methods)
        # for stockwell, `tmin, tmax` already added to `method_kw` by calling method,
        # and `freqs` vector has been pre-computed
        if method != "stockwell":
            method_kw.update(freqs=freqs)
            # ↓↓↓ if constructor called directly, prevents key error
            method_kw.setdefault("output", "power")
        self._freqs = np.asarray(freqs, dtype=np.float64)
        del freqs
        # check validity of kwargs manually to save compute time if any are invalid
        tfr_funcs = dict(
            morlet=tfr_array_morlet,
            multitaper=tfr_array_multitaper,
            stockwell=tfr_array_stockwell,
        )
        _check_method_kwargs(tfr_funcs[method], method_kw, msg=f'TFR method "{method}"')
        self._tfr_func = partial(tfr_funcs[method], **method_kw)
        # apply proj if desired
        if proj:
            inst = inst.copy().apply_proj()
        self.inst = inst

        # prep picks and add the info object. bads and non-data channels are dropped by
        # _picks_to_idx() so we update the info accordingly:
        self._picks = _picks_to_idx(inst.info, picks, "data", with_ref_meg=False)
        self.info = pick_info(inst.info, sel=self._picks, copy=True)
        # assign some attributes
        self._method = method
        self._inst_type = type(inst)
        self._baseline = None
        self.preload = True  # needed for __getitem__, never False for TFRs
        # self._dims may also get updated by child classes
        self._dims = ["channel", "freq", "time"]
        self._needs_taper_dim = method == "multitaper" and method_kw["output"] in (
            "complex",
            "phase",
        )
        if self._needs_taper_dim:
            self._dims.insert(1, "taper")
        self._dims = tuple(self._dims)
        # get the instance data.
        time_mask = _time_mask(inst.times, tmin, tmax, sfreq=self.sfreq)
        get_instance_data_kw = dict(time_mask=time_mask)
        if reject_by_annotation is not None:
            get_instance_data_kw.update(reject_by_annotation=reject_by_annotation)
        data = self._get_instance_data(**get_instance_data_kw)
        # compute the TFR
        self._decim = _ensure_slice(decim)
        self._raw_times = inst.times[time_mask]
        self._compute_tfr(data, n_jobs, verbose)
        self._update_epoch_attributes()
        # "apply" decim to the rest of the object (data is decimated in _compute_tfr)
        with self.info._unlock():
            self.info["sfreq"] /= self._decim.step
        _decim_times = inst.times[self._decim]
        _decim_time_mask = _time_mask(_decim_times, tmin, tmax, sfreq=self.sfreq)
        self._raw_times = _decim_times[_decim_time_mask].copy()
        self._set_times(self._raw_times)
        self._decim = 1
        # record data type (for repr and html_repr). ITC handled in the calling method.
        if method == "stockwell":
            self._data_type = "Power Estimates"
        else:
            data_types = dict(
                power="Power Estimates",
                avg_power="Average Power Estimates",
                avg_power_itc="Average Power Estimates",
                phase="Phase",
                complex="Complex Amplitude",
            )
            self._data_type = data_types[method_kw["output"]]
        # check for correct shape and bad values. `tfr_array_stockwell` doesn't take kw
        # `output` so it may be missing here, so use `.get()`
        negative_ok = method_kw.get("output", "") in ("complex", "phase")
        # if method_kw.get("output", None) in ("phase", "complex"):
        #     raise RuntimeError
        self._check_values(negative_ok=negative_ok)
        # we don't need these anymore, and they make save/load harder
        del self._picks
        del self._tfr_func
        del self._needs_taper_dim
        del self._shape  # calculated from self._data henceforth
        del self.inst  # save memory

    def __abs__(self):
        """Return the absolute value."""
        tfr = self.copy()
        tfr.data = np.abs(tfr.data)
        return tfr

    @fill_doc
    def __add__(self, other):
        """Add two TFR instances.

        %(__add__tfr)s
        """
        self._check_compatibility(other)
        out = self.copy()
        out.data += other.data
        return out

    @fill_doc
    def __iadd__(self, other):
        """Add a TFR instance to another, in-place.

        %(__iadd__tfr)s
        """
        self._check_compatibility(other)
        self.data += other.data
        return self

    @fill_doc
    def __sub__(self, other):
        """Subtract two TFR instances.

        %(__sub__tfr)s
        """
        self._check_compatibility(other)
        out = self.copy()
        out.data -= other.data
        return out

    @fill_doc
    def __isub__(self, other):
        """Subtract a TFR instance from another, in-place.

        %(__isub__tfr)s
        """
        self._check_compatibility(other)
        self.data -= other.data
        return self

    @fill_doc
    def __mul__(self, num):
        """Multiply a TFR instance by a scalar.

        %(__mul__tfr)s
        """
        out = self.copy()
        out.data *= num
        return out

    @fill_doc
    def __imul__(self, num):
        """Multiply a TFR instance by a scalar, in-place.

        %(__imul__tfr)s
        """
        self.data *= num
        return self

    @fill_doc
    def __truediv__(self, num):
        """Divide a TFR instance by a scalar.

        %(__truediv__tfr)s
        """
        out = self.copy()
        out.data /= num
        return out

    @fill_doc
    def __itruediv__(self, num):
        """Divide a TFR instance by a scalar, in-place.

        %(__itruediv__tfr)s
        """
        self.data /= num
        return self

    def __eq__(self, other):
        """Test equivalence of two TFR instances."""
        return object_diff(vars(self), vars(other)) == ""

    def __getstate__(self):
        """Prepare object for serialization."""
        return dict(
            method=self.method,
            data=self._data,
            sfreq=self.sfreq,
            dims=self._dims,
            freqs=self.freqs,
            times=self.times,
            inst_type_str=_get_instance_type_string(self),
            data_type=self._data_type,
            info=self.info,
            baseline=self._baseline,
            decim=self._decim,
        )

    def __setstate__(self, state):
        """Unpack from serialized format."""
        from ..epochs import Epochs
        from ..evoked import Evoked
        from ..io import Raw

        defaults = dict(
            method="unknown",
            dims=("epoch", "channel", "freq", "time")[-state["data"].ndim :],
            baseline=None,
            decim=1,
            data_type="TFR",
            inst_type_str="Unknown",
        )
        defaults.update(**state)
        self._method = defaults["method"]
        self._data = defaults["data"]
        self._freqs = np.asarray(defaults["freqs"], dtype=np.float64)
        self._dims = defaults["dims"]
        self._raw_times = np.asarray(defaults["times"], dtype=np.float64)
        self._baseline = defaults["baseline"]
        self.info = Info(**defaults["info"])
        self._data_type = defaults["data_type"]
        self._decim = defaults["decim"]
        self.preload = True
        self._set_times(self._raw_times)
        # Handle instance type. Prior to gh-11282, Raw was not a possibility so if
        # `inst_type_str` is missing it must be Epochs or Evoked
        unknown_class = Epochs if "epoch" in self._dims else Evoked
        inst_types = dict(Raw=Raw, Epochs=Epochs, Evoked=Evoked, Unknown=unknown_class)
        self._inst_type = inst_types[defaults["inst_type_str"]]
        # sanity check data/freqs/times/info agreement
        self._check_state()

    def __repr__(self):
        """Build string representation of the TFR object."""
        inst_type_str = _get_instance_type_string(self)
        nave = f" (nave={self.nave})" if hasattr(self, "nave") else ""
        # shape & dimension names
        dims = " × ".join(
            [f"{size} {dim}s" for size, dim in zip(self.shape, self._dims)]
        )
        freq_range = f"{self.freqs[0]:0.1f} - {self.freqs[-1]:0.1f} Hz"
        time_range = f"{self.times[0]:0.2f} - {self.times[-1]:0.2f} s"
        return (
            f"<{self._data_type} from {inst_type_str}{nave}, "
            f"{self.method} method | {dims}, {freq_range}, {time_range}, "
            f"{sizeof_fmt(self._size)}>"
        )

    @repr_html
    def _repr_html_(self, caption=None):
        """Build HTML representation of the TFR object."""
        from ..html_templates import _get_html_template

        inst_type_str = _get_instance_type_string(self)
        nave = getattr(self, "nave", 0)
        t = _get_html_template("repr", "tfr.html.jinja")
        t = t.render(tfr=self, inst_type=inst_type_str, nave=nave, caption=caption)
        return t

    def _check_compatibility(self, other):
        """Check compatibility of two TFR instances, in preparation for arithmetic."""
        operation = inspect.currentframe().f_back.f_code.co_name.strip("_")
        if operation.startswith("i"):
            operation = operation[1:]
        msg = f"Cannot {operation} the two TFR instances: {{}} do not match{{}}."
        extra = ""
        if not isinstance(other, type(self)):
            problem = "types"
            extra = f" (self is {type(self)}, other is {type(other)})"
        elif not self.times.shape == other.times.shape or np.any(
            self.times != other.times
        ):
            problem = "times"
        elif not self.freqs.shape == other.freqs.shape or np.any(
            self.freqs != other.freqs
        ):
            problem = "freqs"
        else:  # should be OK
            return
        raise RuntimeError(msg.format(problem, extra))

    def _check_state(self):
        """Check data/freqs/times/info agreement during __setstate__."""
        msg = "{} axis of data ({}) doesn't match {} attribute ({})"
        n_chan_info = len(self.info["chs"])
        n_chan = self._data.shape[self._dims.index("channel")]
        n_freq = self._data.shape[self._dims.index("freq")]
        n_time = self._data.shape[self._dims.index("time")]
        if n_chan_info != n_chan:
            msg = msg.format("Channel", n_chan, "info", n_chan_info)
        elif n_freq != len(self.freqs):
            msg = msg.format("Frequency", n_freq, "freqs", self.freqs.size)
        elif n_time != len(self.times):
            msg = msg.format("Time", n_time, "times", self.times.size)
        else:
            return
        raise ValueError(msg)

    def _check_values(self, negative_ok=False):
        """Check TFR results for correct shape and bad values."""
        assert len(self._dims) == self._data.ndim
        assert self._data.shape == self._shape
        # Check for implausible power values: take min() across all but the channel axis
        # TODO: should this be more fine-grained (report "chan X in epoch Y")?
        ch_dim = self._dims.index("channel")
        dims = np.arange(self._data.ndim).tolist()
        dims.pop(ch_dim)
        negative_values = self._data.min(axis=tuple(dims)) < 0
        if negative_values.any() and not negative_ok:
            chs = np.array(self.ch_names)[negative_values].tolist()
            s = _pl(negative_values.sum())
            warn(
                f"Negative value in time-frequency decomposition for channel{s} "
                f'{", ".join(chs)}',
                UserWarning,
            )

    def _compute_tfr(self, data, n_jobs, verbose):
        result = self._tfr_func(
            data,
            self.sfreq,
            decim=self._decim,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # assign ._data and maybe ._itc
        # tfr_array_stockwell always returns ITC (sometimes it's None)
        if self.method == "stockwell":
            self._data, self._itc, freqs = result
            assert np.array_equal(self._freqs, freqs)
        elif self._tfr_func.keywords.get("output", "").endswith("_itc"):
            self._data, self._itc = result.real, result.imag
        else:
            self._data = result
        # remove fake "epoch" dimension
        if self.method != "stockwell" and _get_instance_type_string(self) != "Epochs":
            self._data = np.squeeze(self._data, axis=0)

        # this is *expected* shape, it gets asserted later in _check_values()
        # (and then deleted afterwards)
        expected_shape = [
            len(self.ch_names),
            len(self.freqs),
            len(self._raw_times[self._decim]),  # don't use self.times, not set yet
        ]
        # deal with the "taper" dimension
        if self._needs_taper_dim:
            tapers_dim = 1 if _get_instance_type_string(self) != "Epochs" else 2
            expected_shape.insert(1, self._data.shape[tapers_dim])
        self._shape = tuple(expected_shape)

    @verbose
    def _onselect(
        self,
        eclick,
        erelease,
        picks=None,
        exclude="bads",
        combine="mean",
        baseline=None,
        mode=None,
        cmap=None,
        source_plot_joint=False,
        topomap_args=None,
        verbose=None,
    ):
        """Respond to rectangle selector in TFR image plots with a topomap plot."""
        if abs(eclick.x - erelease.x) < 0.1 or abs(eclick.y - erelease.y) < 0.1:
            return
        t_range = (min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata))
        f_range = (min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata))
        # snap to nearest measurement point
        t_idx = np.abs(self.times - np.atleast_2d(t_range).T).argmin(axis=1)
        f_idx = np.abs(self.freqs - np.atleast_2d(f_range).T).argmin(axis=1)
        tmin, tmax = self.times[t_idx]
        fmin, fmax = self.freqs[f_idx]
        # immutable → mutable default
        if topomap_args is None:
            topomap_args = dict()
        topomap_args.setdefault("cmap", cmap)
        topomap_args.setdefault("vlim", (None, None))
        # figure out which channel types we're dealing with
        types = list()
        if "eeg" in self:
            types.append("eeg")
        if "mag" in self:
            types.append("mag")
        if "grad" in self:
            grad_picks = _pair_grad_sensors(
                self.info, topomap_coords=False, raise_error=False
            )
            if len(grad_picks) > 1:
                types.append("grad")
            elif len(types) == 0:
                logger.info(
                    "Need at least 2 gradiometer pairs to plot a gradiometer topomap."
                )
                return  # Don't draw a figure for nothing.

        fig = figure_nobar()
        t_range = f"{tmin:.3f}" if tmin == tmax else f"{tmin:.3f} - {tmax:.3f}"
        f_range = f"{fmin:.2f}" if fmin == fmax else f"{fmin:.2f} - {fmax:.2f}"
        fig.suptitle(f"{t_range} s,\n{f_range} Hz")

        if source_plot_joint:
            ax = fig.add_subplot()
            data, times, freqs = self.get_data(
                picks=picks, exclude=exclude, return_times=True, return_freqs=True
            )
            # merge grads before baselining (makes ERDs visible)
            ch_types = np.array(self.get_channel_types(unique=True))
            ch_type = ch_types.item()  # will error if there are more than one
            data, pos = _merge_if_grads(
                data=data,
                info=self.info,
                ch_type=ch_type,
                sphere=topomap_args.get("sphere"),
                combine=combine,
            )
            # baseline and crop
            data, *_ = _prep_data_for_plot(
                data,
                times,
                freqs,
                tmin=tmin,
                tmax=tmax,
                fmin=fmin,
                fmax=fmax,
                baseline=baseline,
                mode=mode,
                verbose=verbose,
            )
            # average over times and freqs
            data = data.mean((-2, -1))

            im, _ = plot_topomap(data, pos, axes=ax, show=False, **topomap_args)
            _add_colorbar(ax, im, topomap_args["cmap"], title="AU")
            plt_show(fig=fig)
        else:
            for idx, ch_type in enumerate(types):
                ax = fig.add_subplot(1, len(types), idx + 1)
                plot_tfr_topomap(
                    self,
                    ch_type=ch_type,
                    tmin=tmin,
                    tmax=tmax,
                    fmin=fmin,
                    fmax=fmax,
                    baseline=baseline,
                    mode=mode,
                    axes=ax,
                    **topomap_args,
                )
                ax.set_title(ch_type)

    def _update_epoch_attributes(self):
        # overwritten in EpochsTFR; adds things needed for to_data_frame and __getitem__
        pass

    @property
    def _detrend_picks(self):
        """Provide compatibility with __iter__."""
        return list()

    @property
    def baseline(self):
        """Start and end of the baseline period (in seconds)."""
        return self._baseline

    @property
    def ch_names(self):
        """The channel names."""
        return self.info["ch_names"]

    @property
    def data(self):
        """The time-frequency-resolved power estimates."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def freqs(self):
        """The frequencies at which power estimates were computed."""
        return self._freqs

    @property
    def method(self):
        """The method used to compute the time-frequency power estimates."""
        return self._method

    @property
    def sfreq(self):
        """Sampling frequency of the data."""
        return self.info["sfreq"]

    @property
    def shape(self):
        """Data shape."""
        return self._data.shape

    @property
    def times(self):
        """The time points present in the data (in seconds)."""
        return self._times_readonly

    @fill_doc
    def crop(self, tmin=None, tmax=None, fmin=None, fmax=None, include_tmax=True):
        """Crop data to a given time interval in place.

        Parameters
        ----------
        %(tmin_tmax_psd)s
        fmin : float | None
            Lowest frequency of selection in Hz.

            .. versionadded:: 0.18.0
        fmax : float | None
            Highest frequency of selection in Hz.

            .. versionadded:: 0.18.0
        %(include_tmax)s

        Returns
        -------
        %(inst_tfr)s
            The modified instance.
        """
        super().crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)

        if fmin is not None or fmax is not None:
            freq_mask = _freq_mask(
                self.freqs, sfreq=self.info["sfreq"], fmin=fmin, fmax=fmax
            )
        else:
            freq_mask = slice(None)

        self._freqs = self.freqs[freq_mask]
        # Deal with broadcasting (boolean arrays do not broadcast, but indices
        # do, so we need to convert freq_mask to make use of broadcasting)
        if isinstance(freq_mask, np.ndarray):
            freq_mask = np.where(freq_mask)[0]
        self._data = self._data[..., freq_mask, :]
        return self

    def copy(self):
        """Return copy of the TFR instance.

        Returns
        -------
        %(inst_tfr)s
            A copy of the object.
        """
        return deepcopy(self)

    @verbose
    def apply_baseline(self, baseline, mode="mean", verbose=None):
        """Baseline correct the data.

        Parameters
        ----------
        %(baseline_rescale)s

            How baseline is computed is determined by the ``mode`` parameter.
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
        %(inst_tfr)s
            The modified instance.
        """
        self._baseline = _check_baseline(baseline, times=self.times, sfreq=self.sfreq)
        rescale(self.data, self.times, self.baseline, mode, copy=False, verbose=verbose)
        return self

    @fill_doc
    def get_data(
        self,
        picks=None,
        exclude="bads",
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        return_times=False,
        return_freqs=False,
    ):
        """Get time-frequency data in NumPy array format.

        Parameters
        ----------
        %(picks_good_data_noref)s
        %(exclude_spectrum_get_data)s
        %(fmin_fmax_tfr)s
        %(tmin_tmax_psd)s
        return_times : bool
            Whether to return the time values for the requested time range.
            Default is ``False``.
        return_freqs : bool
            Whether to return the frequency bin values for the requested
            frequency range. Default is ``False``.

        Returns
        -------
        data : array
            The requested data in a NumPy array.
        times : array
            The time values for the requested data range. Only returned if
            ``return_times`` is ``True``.
        freqs : array
            The frequency values for the requested data range. Only returned if
            ``return_freqs`` is ``True``.

        Notes
        -----
        Returns a copy of the underlying data (not a view).
        """
        tmin = self.times[0] if tmin is None else tmin
        tmax = self.times[-1] if tmax is None else tmax
        fmin = 0 if fmin is None else fmin
        fmax = np.inf if fmax is None else fmax
        picks = _picks_to_idx(
            self.info, picks, "data_or_ica", exclude=exclude, with_ref_meg=False
        )
        fmin_idx = np.searchsorted(self.freqs, fmin)
        fmax_idx = np.searchsorted(self.freqs, fmax, side="right")
        tmin_idx = np.searchsorted(self.times, tmin)
        tmax_idx = np.searchsorted(self.times, tmax, side="right")
        freq_picks = np.arange(fmin_idx, fmax_idx)
        time_picks = np.arange(tmin_idx, tmax_idx)
        freq_axis = self._dims.index("freq")
        time_axis = self._dims.index("time")
        chan_axis = self._dims.index("channel")
        # normally there's a risk of np.take reducing array dimension if there
        # were only one channel or frequency selected, but `_picks_to_idx`
        # and np.arange both always return arrays, so we're safe; the result
        # will always have the same `ndim` as it started with.
        data = (
            self._data.take(picks, chan_axis)
            .take(freq_picks, freq_axis)
            .take(time_picks, time_axis)
        )
        out = [data]
        if return_times:
            times = self._raw_times[tmin_idx:tmax_idx]
            out.append(times)
        if return_freqs:
            freqs = self._freqs[fmin_idx:fmax_idx]
            out.append(freqs)
        if not return_times and not return_freqs:
            return out[0]
        return tuple(out)

    @verbose
    def plot(
        self,
        picks=None,
        *,
        exclude=(),
        tmin=None,
        tmax=None,
        fmin=0.0,
        fmax=np.inf,
        baseline=None,
        mode="mean",
        dB=False,
        combine=None,
        layout=None,  # TODO deprecate? not used in orig implementation either
        yscale="auto",
        vlim=(None, None),
        cnorm=None,
        cmap=None,
        colorbar=True,
        title=None,  # don't deprecate this one; has (useful) option title="auto"
        mask=None,
        mask_style=None,
        mask_cmap="Greys",
        mask_alpha=0.1,
        axes=None,
        show=True,
        verbose=None,
    ):
        """Plot TFRs as two-dimensional time-frequency images.

        Parameters
        ----------
        %(picks_good_data)s
        %(exclude_spectrum_plot)s
        %(tmin_tmax_psd)s
        %(fmin_fmax_tfr)s
        %(baseline_rescale)s

            How baseline is computed is determined by the ``mode`` parameter.
        %(mode_tfr_plot)s
        %(dB_spectrum_plot)s
        %(combine_tfr_plot)s

            .. versionchanged:: 1.3
               Added support for ``callable``.
        %(layout_spectrum_plot_topo)s
        %(yscale_tfr_plot)s

            .. versionadded:: 0.14.0
        %(vlim_tfr_plot)s
        %(cnorm)s

            .. versionadded:: 0.24
        %(cmap_topomap)s
        %(colorbar)s
        %(title_tfr_plot)s
        %(mask_tfr_plot)s

            .. versionadded:: 0.16.0
        %(mask_style_tfr_plot)s

            .. versionadded:: 0.17
        %(mask_cmap_tfr_plot)s

            .. versionadded:: 0.17
        %(mask_alpha_tfr_plot)s

            .. versionadded:: 0.16.0
        %(axes_tfr_plot)s
        %(show)s
        %(verbose)s

        Returns
        -------
        figs : list of instances of matplotlib.figure.Figure
            A list of figures containing the time-frequency power.
        """
        # the rectangle selector plots topomaps, which needs all channels uncombined,
        # so we keep a reference to that state here, and (because the topomap plotting
        # function wants an AverageTFR) update it with `comment` and `nave` values in
        # case we started out with a singleton EpochsTFR or RawTFR
        initial_state = self.__getstate__()
        initial_state.setdefault("comment", "")
        initial_state.setdefault("nave", 1)
        # `_picks_to_idx` also gets done inside `get_data()`` below, but we do it here
        # because we need the indices later
        idx_picks = _picks_to_idx(
            self.info, picks, "data_or_ica", exclude=exclude, with_ref_meg=False
        )
        pick_names = np.array(self.ch_names)[idx_picks].tolist()  # for titles
        ch_types = self.get_channel_types(idx_picks)
        # get data arrays
        data, times, freqs = self.get_data(
            picks=idx_picks, exclude=(), return_times=True, return_freqs=True
        )
        # pass tmin/tmax here ↓↓↓, not here ↑↑↑; we want to crop *after* baselining
        data, times, freqs = _prep_data_for_plot(
            data,
            times,
            freqs,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            baseline=baseline,
            mode=mode,
            dB=dB,
            verbose=verbose,
        )
        # shape
        ch_axis = self._dims.index("channel")
        freq_axis = self._dims.index("freq")
        time_axis = self._dims.index("time")
        want_shape = list(self.shape)
        want_shape[ch_axis] = len(idx_picks) if combine is None else 1
        want_shape[freq_axis] = len(freqs)  # in case there was fmin/fmax cropping
        want_shape[time_axis] = len(times)  # in case there was tmin/tmax cropping
        want_shape = tuple(want_shape)
        # combine
        combine_was_none = combine is None
        combine = _make_combine_callable(
            combine, axis=ch_axis, valid=("mean", "rms"), keepdims=True
        )
        try:
            data = combine(data)  # no need to copy; get_data() never returns a view
        except Exception as e:
            msg = (
                "Something went wrong with the callable passed to 'combine'; see "
                "traceback."
            )
            raise ValueError(msg) from e
        # call succeeded, check type and shape
        mismatch = False
        if not isinstance(data, np.ndarray):
            mismatch = "type"
            extra = ""
        elif data.shape not in (want_shape, want_shape[1:]):
            mismatch = "shape"
            extra = f" of shape {data.shape}"
        if mismatch:
            raise RuntimeError(
                f"Wrong {mismatch} yielded by callable passed to 'combine'. Make sure "
                "your function takes a single argument (an array of shape "
                "(n_channels, n_freqs, n_times)) and returns an array of shape "
                f"(n_freqs, n_times); yours yielded: {type(data)}{extra}."
            )
        # restore singleton collapsed axis (removed by user-provided callable):
        # (n_freqs, n_times) → (1, n_freqs, n_times)
        if data.shape == (len(freqs), len(times)):
            data = data[np.newaxis]

        assert data.shape == want_shape
        # cmap handling. power may be negative depending on baseline strategy so set
        # `norm` empirically — but only if user didn't set limits explicitly.
        norm = False if vlim == (None, None) else data.min() >= 0.0
        vmin, vmax = _setup_vmin_vmax(data, *vlim, norm=norm)
        cmap = _setup_cmap(cmap, norm=norm)
        # prepare figure(s)
        if axes is None:
            figs = [plt.figure(layout="constrained") for _ in range(data.shape[0])]
            axes = [fig.add_subplot() for fig in figs]
        elif isinstance(axes, plt.Axes):
            figs = [axes.get_figure()]
            axes = [axes]
        elif isinstance(axes, np.ndarray):  # allow plotting into a grid of axes
            figs = [ax.get_figure() for ax in axes.flat]
        elif hasattr(axes, "__iter__") and len(axes):
            figs = [ax.get_figure() for ax in axes]
        else:
            raise ValueError(
                f"axes must be None, Axes, or list/array of Axes, got {type(axes)}"
            )
        if len(axes) != data.shape[0]:
            raise RuntimeError(
                f"Mismatch between picked channels ({data.shape[0]}) and axes "
                f"({len(axes)}); there must be one axes for each picked channel."
            )
        # check if we're being called from within plot_joint(). If so, get the
        # `topomap_args` from the calling context and pass it to the onselect handler.
        # (we need 2 `f_back` here because of the verbose decorator)
        calling_frame = inspect.currentframe().f_back.f_back
        source_plot_joint = calling_frame.f_code.co_name == "plot_joint"
        topomap_args = (
            dict()
            if not source_plot_joint
            else calling_frame.f_locals.get("topomap_args", dict())
        )
        # plot
        for ix, _fig in enumerate(figs):
            # restrict the onselect instance to the channel type of the picks used in
            # the image plot
            uniq_types = np.unique(ch_types)
            ch_type = None if len(uniq_types) > 1 else uniq_types.item()
            this_tfr = AverageTFR(inst=initial_state).pick(ch_type, verbose=verbose)
            _onselect_callback = partial(
                this_tfr._onselect,
                picks=None,  # already restricted the picks in `this_tfr`
                exclude=(),
                baseline=baseline,
                mode=mode,
                cmap=cmap,
                source_plot_joint=source_plot_joint,
                topomap_args=topomap_args,
            )
            # draw the image plot
            _imshow_tfr(
                ax=axes[ix],
                tfr=data[[ix]],
                ch_idx=0,
                tmin=times[0],
                tmax=times[-1],
                vmin=vmin,
                vmax=vmax,
                onselect=_onselect_callback,
                ylim=None,
                freq=freqs,
                x_label="Time (s)",
                y_label="Frequency (Hz)",
                colorbar=colorbar,
                cmap=cmap,
                yscale=yscale,
                mask=mask,
                mask_style=mask_style,
                mask_cmap=mask_cmap,
                mask_alpha=mask_alpha,
                cnorm=cnorm,
            )
            # handle title. automatic title is:
            #   f"{Baselined} {power} ({ch_name})" or
            #   f"{Baselined} {power} ({combination} of {N} {ch_type}s)"
            if title == "auto":
                if combine_was_none:  # one plot per channel
                    which_chs = pick_names[ix]
                elif len(pick_names) == 1:  # there was only one pick anyway
                    which_chs = pick_names[0]
                else:  # one plot for all chs combined
                    which_chs = _set_title_multiple_electrodes(
                        None, combine, pick_names, all_=True, ch_type=ch_type
                    )
                _prefix = "Power" if baseline is None else "Baselined power"
                _title = f"{_prefix} ({which_chs})"
            else:
                _title = title
            _fig.suptitle(_title)
        plt_show(show)
        return figs

    @verbose
    def plot_joint(
        self,
        *,
        timefreqs=None,
        picks=None,
        exclude=(),
        combine="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        baseline=None,
        mode="mean",
        dB=False,
        yscale="auto",
        vlim=(None, None),
        cnorm=None,
        cmap=None,
        colorbar=True,
        title=None,  # TODO consider deprecating this one, or adding an "auto" option
        show=True,
        topomap_args=None,
        image_args=None,
        verbose=None,
    ):
        """Plot TFRs as a two-dimensional image with topomap highlights.

        Parameters
        ----------
        %(timefreqs)s
        %(picks_good_data)s
        %(exclude_psd)s
            Default is an empty :class:`tuple` which includes all channels.
        %(combine_tfr_plot_joint)s

            .. versionchanged:: 1.3
                Added support for ``callable``.
        %(tmin_tmax_psd)s
        %(fmin_fmax_tfr)s
        %(baseline_rescale)s

            How baseline is computed is determined by the ``mode`` parameter.
        %(mode_tfr_plot)s
        %(dB_tfr_plot_topo)s
        %(yscale_tfr_plot)s
        %(vlim_tfr_plot_joint)s
        %(cnorm)s
        %(cmap_tfr_plot_topo)s
        %(colorbar_tfr_plot_joint)s
        %(title_none)s
        %(show)s
        %(topomap_args)s
        %(image_args)s
        %(verbose)s

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.

        Notes
        -----
        %(notes_timefreqs_tfr_plot_joint)s

        .. versionadded:: 0.16.0
        """
        from matplotlib import ticker
        from matplotlib.patches import ConnectionPatch

        # handle recursion
        picks = _picks_to_idx(
            self.info, picks, "data_or_ica", exclude=exclude, with_ref_meg=False
        )
        all_ch_types = np.array(self.get_channel_types())
        uniq_ch_types = sorted(set(all_ch_types[picks]))
        if len(uniq_ch_types) > 1:
            msg = "Multiple channel types selected, returning one figure per type."
            logger.info(msg)
            figs = list()
            for this_type in uniq_ch_types:
                this_picks = np.intersect1d(
                    picks,
                    np.nonzero(np.isin(all_ch_types, this_type))[0],
                    assume_unique=True,
                )
                # TODO might be nice to not "copy first, then pick"; alternative might
                # be to subset the data with `this_picks` and then construct the "copy"
                # using __getstate__ and __setstate__
                _tfr = self.copy().pick(this_picks)
                figs.append(
                    _tfr.plot_joint(
                        timefreqs=timefreqs,
                        picks=None,
                        baseline=baseline,
                        mode=mode,
                        tmin=tmin,
                        tmax=tmax,
                        fmin=fmin,
                        fmax=fmax,
                        vlim=vlim,
                        cmap=cmap,
                        dB=dB,
                        colorbar=colorbar,
                        show=False,
                        title=title,
                        yscale=yscale,
                        combine=combine,
                        exclude=(),
                        topomap_args=topomap_args,
                        verbose=verbose,
                    )
                )
            return figs
        else:
            ch_type = uniq_ch_types[0]

        # handle defaults
        _validate_type(combine, ("str", "callable"), item_name="combine")  # no `None`
        image_args = dict() if image_args is None else image_args
        topomap_args = dict() if topomap_args is None else topomap_args.copy()
        # make sure if topomap_args["ch_type"] is set, it matches what is in `self.info`
        topomap_args.setdefault("ch_type", ch_type)
        if topomap_args["ch_type"] != ch_type:
            raise ValueError(
                f"topomap_args['ch_type'] is {topomap_args['ch_type']} which does not "
                f"match the channel type present in the object ({ch_type})."
            )
        # some necessary defaults
        topomap_args.setdefault("outlines", "head")
        topomap_args.setdefault("contours", 6)
        # don't pass these:
        topomap_args.pop("axes", None)
        topomap_args.pop("show", None)
        topomap_args.pop("colorbar", None)

        # get the time/freq limits of the image plot, to make sure requested annotation
        # times/freqs are in range
        _, times, freqs = self.get_data(
            picks=picks,
            exclude=(),
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            return_times=True,
            return_freqs=True,
        )
        # validate requested annotation times and freqs
        timefreqs = _get_timefreqs(self, timefreqs)
        valid_timefreqs = dict()
        while timefreqs:
            (_time, _freq), (t_win, f_win) = timefreqs.popitem()
            # convert to half-windows
            t_win /= 2
            f_win /= 2
            # make sure the times / freqs are in-bounds
            msg = (
                "Requested {} exceeds the range of the data ({}). Choose different "
                "`timefreqs`."
            )
            if (times > _time).all() or (times < _time).all():
                _var = f"time point ({_time:0.3f} s)"
                _range = f"{times[0]:0.3f} - {times[-1]:0.3f} s"
                raise ValueError(msg.format(_var, _range))
            elif (freqs > _freq).all() or (freqs < _freq).all():
                _var = f"frequency ({_freq:0.1f} Hz)"
                _range = f"{freqs[0]:0.1f} - {freqs[-1]:0.1f} Hz"
                raise ValueError(msg.format(_var, _range))
            # snap the times/freqs to the nearest point we have an estimate for, and
            # store the validated points
            if t_win == 0:
                _time = times[np.argmin(np.abs(times - _time))]
            if f_win == 0:
                _freq = freqs[np.argmin(np.abs(freqs - _freq))]
            valid_timefreqs[(_time, _freq)] = (t_win, f_win)

        # prep data for topomaps (unlike image plot, must include all channels of the
        # current ch_type). Don't pass tmin/tmax here (crop later after baselining)
        topomap_picks = _picks_to_idx(self.info, ch_type)
        data, times, freqs = self.get_data(
            picks=topomap_picks, exclude=(), return_times=True, return_freqs=True
        )
        # merge grads before baselining (makes ERDS visible)
        info = pick_info(self.info, sel=topomap_picks, copy=True)
        data, pos = _merge_if_grads(
            data=data,
            info=info,
            ch_type=ch_type,
            sphere=topomap_args.get("sphere"),
            combine=combine,
        )
        # loop over intended topomap locations, to find one vlim that works for all.
        tf_array = np.array(list(valid_timefreqs))  # each row is [time, freq]
        tf_array = tf_array[tf_array[:, 0].argsort()]  # sort by time
        _vmin, _vmax = (np.inf, -np.inf)
        topomap_arrays = list()
        topomap_titles = list()
        for _time, _freq in tf_array:
            # reduce data to the range of interest in the TF plane (i.e., finally crop)
            t_win, f_win = valid_timefreqs[(_time, _freq)]
            _tmin, _tmax = np.array([-1, 1]) * t_win + _time
            _fmin, _fmax = np.array([-1, 1]) * f_win + _freq
            _data, *_ = _prep_data_for_plot(
                data,
                times,
                freqs,
                tmin=_tmin,
                tmax=_tmax,
                fmin=_fmin,
                fmax=_fmax,
                baseline=baseline,
                mode=mode,
                verbose=verbose,
            )
            _data = _data.mean(axis=(-1, -2))  # avg over times and freqs
            topomap_arrays.append(_data)
            _vmin = min(_data.min(), _vmin)
            _vmax = max(_data.max(), _vmax)
            # construct topopmap subplot title
            t_pm = "" if t_win == 0 else f" ± {t_win:0.2f}"
            f_pm = "" if f_win == 0 else f" ± {f_win:0.1f}"
            _title = f"{_time:0.2f}{t_pm} s,\n{_freq:0.1f}{f_pm} Hz"
            topomap_titles.append(_title)
        # handle cmap. Power may be negative depending on baseline strategy so set
        # `norm` empirically. vmin/vmax will be handled separately within the `plot()`
        # call for the image plot.
        norm = np.min(topomap_arrays) >= 0.0
        cmap = _setup_cmap(cmap, norm=norm)
        topomap_args.setdefault("cmap", cmap[0])  # prevent interactive cbar
        # finalize topomap vlims and compute contour locations.
        # By passing `data=None` here ↓↓↓↓ we effectively assert vmin & vmax aren't None
        _vlim = _setup_vmin_vmax(data=None, vmin=_vmin, vmax=_vmax, norm=norm)
        topomap_args.setdefault("vlim", _vlim)
        locator, topomap_args["contours"] = _set_contour_locator(
            *topomap_args["vlim"], topomap_args["contours"]
        )
        # initialize figure and do the image plot. `self.plot()` needed to wait to be
        # called until after `topomap_args` was fully populated --- we don't pass the
        # dict through to `self.plot()` explicitly here, but we do "reach back" and get
        # it if it's needed by the interactive rectangle selector.
        fig, image_ax, topomap_axes = _prepare_joint_axes(len(valid_timefreqs))
        fig = self.plot(
            picks=picks,
            exclude=(),
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            baseline=baseline,
            mode=mode,
            dB=dB,
            combine=combine,
            yscale=yscale,
            vlim=vlim,
            cnorm=cnorm,
            cmap=cmap,
            colorbar=False,
            title=title,
            # mask, mask_style, mask_cmap, mask_alpha
            axes=image_ax,
            show=False,
            verbose=verbose,
            **image_args,
        )[0]  # [0] because `.plot()` always returns a list
        # now, actually plot the topomaps
        for ax, title, _data in zip(topomap_axes, topomap_titles, topomap_arrays):
            ax.set_title(title)
            plot_topomap(_data, pos, axes=ax, show=False, **topomap_args)
        # draw colorbar
        if colorbar:
            cbar = fig.colorbar(ax.images[0])
            cbar.locator = ticker.MaxNLocator(nbins=5) if locator is None else locator
            cbar.update_ticks()
        # draw the connection lines between time-frequency image and topoplots
        for (time_, freq_), topo_ax in zip(tf_array, topomap_axes):
            con = ConnectionPatch(
                xyA=[time_, freq_],
                xyB=[0.5, 0],
                coordsA="data",
                coordsB="axes fraction",
                axesA=image_ax,
                axesB=topo_ax,
                color="grey",
                linestyle="-",
                linewidth=1.5,
                alpha=0.66,
                zorder=1,
                clip_on=False,
            )
            fig.add_artist(con)

        plt_show(show)
        return fig

    @verbose
    def plot_topo(
        self,
        picks=None,
        baseline=None,
        mode="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        vmin=None,  # TODO deprecate in favor of `vlim` (needs helper func refactor)
        vmax=None,
        layout=None,
        cmap="RdBu_r",
        title=None,  # don't deprecate; topo titles aren't standard (color, size, just.)
        dB=False,
        colorbar=True,
        layout_scale=0.945,
        show=True,
        border="none",
        fig_facecolor="k",
        fig_background=None,
        font_color="w",
        yscale="auto",
        verbose=None,
    ):
        """Plot a TFR image for each channel in a sensor layout arrangement.

        Parameters
        ----------
        %(picks_good_data)s
        %(baseline_rescale)s

            How baseline is computed is determined by the ``mode`` parameter.
        %(mode_tfr_plot)s
        %(tmin_tmax_psd)s
        %(fmin_fmax_tfr)s
        %(vmin_vmax_tfr_plot_topo)s
        %(layout_spectrum_plot_topo)s
        %(cmap_tfr_plot_topo)s
        %(title_none)s
        %(dB_tfr_plot_topo)s
        %(colorbar)s
        %(layout_scale)s
        %(show)s
        %(border_topo)s
        %(fig_facecolor)s
        %(fig_background)s
        %(font_color)s
        %(yscale_tfr_plot)s
        %(verbose)s

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the topography.
        """
        # convenience vars
        times = self.times.copy()
        freqs = self.freqs
        data = self.data
        info = self.info

        info, data = _prepare_picks(info, data, picks, axis=0)
        del picks

        # TODO this is the only remaining call to _preproc_tfr; should be refactored
        #      (to use _prep_data_for_plot?)
        data, times, freqs, vmin, vmax = _preproc_tfr(
            data,
            times,
            freqs,
            tmin,
            tmax,
            fmin,
            fmax,
            mode,
            baseline,
            vmin,
            vmax,
            dB,
            info["sfreq"],
        )

        if layout is None:
            from mne import find_layout

            layout = find_layout(self.info)
        onselect_callback = partial(self._onselect, baseline=baseline, mode=mode)

        click_fun = partial(
            _imshow_tfr,
            tfr=data,
            freq=freqs,
            yscale=yscale,
            cmap=(cmap, True),
            onselect=onselect_callback,
        )
        imshow = partial(
            _imshow_tfr_unified,
            tfr=data,
            freq=freqs,
            cmap=cmap,
            onselect=onselect_callback,
        )

        fig = _plot_topo(
            info=info,
            times=times,
            show_func=imshow,
            click_func=click_fun,
            layout=layout,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            layout_scale=layout_scale,
            title=title,
            border=border,
            x_label="Time (s)",
            y_label="Frequency (Hz)",
            fig_facecolor=fig_facecolor,
            font_color=font_color,
            unified=True,
            img=True,
        )

        add_background_image(fig, fig_background)
        plt_show(show)
        return fig

    @copy_function_doc_to_method_doc(plot_tfr_topomap)
    def plot_topomap(
        self,
        tmin=None,
        tmax=None,
        fmin=0.0,
        fmax=np.inf,
        *,
        ch_type=None,
        baseline=None,
        mode="mean",
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=2,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%1.1e",
        units=None,
        axes=None,
        show=True,
    ):
        return plot_tfr_topomap(
            self,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            ch_type=ch_type,
            baseline=baseline,
            mode=mode,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            units=units,
            axes=axes,
            show=show,
        )

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save time-frequency data to disk (in HDF5 format).

        Parameters
        ----------
        fname : path-like
            Path of file to save to, which should end with ``-tfr.h5`` or ``-tfr.hdf5``.
        %(overwrite)s
        %(verbose)s

        See Also
        --------
        mne.time_frequency.read_tfrs
        """
        _, write_hdf5 = _import_h5io_funcs()
        check_fname(fname, "time-frequency object", (".h5", ".hdf5"))
        fname = _check_fname(fname, overwrite=overwrite, verbose=verbose)
        out = self.__getstate__()
        if "metadata" in out:
            out["metadata"] = _prepare_write_metadata(out["metadata"])
        write_hdf5(fname, out, overwrite=overwrite, title="mnepython", slash="replace")

    @verbose
    def to_data_frame(
        self,
        picks=None,
        index=None,
        long_format=False,
        time_format=None,
        *,
        verbose=None,
    ):
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
        valid_index_args = ["time", "freq"]
        if isinstance(self, EpochsTFR):
            valid_index_args.extend(["epoch", "condition"])
        valid_time_formats = ["ms", "timedelta"]
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        picks = _picks_to_idx(self.info, picks, "all", exclude=())
        data, times, freqs = self.get_data(picks, return_times=True, return_freqs=True)
        axis = self._dims.index("channel")
        if not isinstance(self, EpochsTFR):
            data = data[np.newaxis]  # add singleton "epochs" axis
            axis += 1
        n_epochs, n_picks, n_freqs, n_times = data.shape
        # reshape to (epochs*freqs*times) x signals
        data = np.moveaxis(data, axis, -1)
        data = data.reshape(n_epochs * n_freqs * n_times, n_picks)
        # prepare extra columns / multiindex
        mindex = list()
        times = _convert_times(times, time_format, self.info["meas_date"])
        times = np.tile(times, n_epochs * n_freqs)
        freqs = np.tile(np.repeat(freqs, n_times), n_epochs)
        mindex.append(("time", times))
        mindex.append(("freq", freqs))
        if isinstance(self, EpochsTFR):
            mindex.append(("epoch", np.repeat(self.selection, n_times * n_freqs)))
            rev_event_id = {v: k for k, v in self.event_id.items()}
            conditions = [rev_event_id[k] for k in self.events[:, 2]]
            mindex.append(("condition", np.repeat(conditions, n_times * n_freqs)))
        assert all(len(mdx) == len(mindex[0]) for mdx in mindex[1:])
        # build DataFrame
        if isinstance(self, EpochsTFR):
            default_index = ["condition", "epoch", "freq", "time"]
        else:
            default_index = ["freq", "time"]
        df = _build_data_frame(
            self, data, picks, long_format, mindex, index, default_index=default_index
        )
        return df


@fill_doc
class AverageTFR(BaseTFR):
    """Data object for spectrotemporal representations of averaged data.

    .. warning:: The preferred means of creating AverageTFR objects is via the
                 instance methods :meth:`mne.Epochs.compute_tfr` and
                 :meth:`mne.Evoked.compute_tfr`, or via
                 :meth:`mne.time_frequency.EpochsTFR.average`. Direct class
                 instantiation is discouraged.

    Parameters
    ----------
    inst : instance of Evoked | instance of Epochs | dict
        The data from which to compute the time-frequency representation. Passing a
        :class:`dict` will create the AverageTFR using the ``__setstate__`` interface
        and is not recommended for typical use cases.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    %(method_tfr)s
    %(freqs_tfr)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(decim_tfr)s
    %(comment_averagetfr)s
    %(n_jobs)s
    %(verbose)s
    %(method_kw_tfr)s

    Attributes
    ----------
    %(baseline_tfr_attr)s
    %(ch_names_tfr_attr)s
    %(comment_averagetfr_attr)s
    %(freqs_tfr_attr)s
    %(info_not_none)s
    %(method_tfr_attr)s
    %(nave_tfr_attr)s
    %(sfreq_tfr_attr)s
    %(shape_tfr_attr)s

    See Also
    --------
    RawTFR
    EpochsTFR
    AverageTFRArray
    mne.Evoked.compute_tfr
    mne.time_frequency.EpochsTFR.average

    Notes
    -----
    The old API (prior to version 1.7) was::

        AverageTFR(info, data, times, freqs, nave, comment=None, method=None)

    That API is still available via :class:`~mne.time_frequency.AverageTFRArray` for
    cases where the data are precomputed or do not originate from MNE-Python objects.
    The preferred new API uses instance methods::

        evoked.compute_tfr(method, freqs, ...)
        epochs.compute_tfr(method, freqs, average=True, ...)

    The new API also supports AverageTFR instantiation from a :class:`dict`, but this
    is primarily for save/load and internal purposes, and wraps ``__setstate__``.
    During the transition from the old to the new API, it may be expedient to use
    :class:`~mne.time_frequency.AverageTFRArray` as a "quick-fix" approach to updating
    scripts under active development.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        *,
        inst=None,
        freqs=None,
        method=None,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        decim=1,
        comment=None,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        from ..epochs import BaseEpochs
        from ..evoked import Evoked
        from ._stockwell import _check_input_st, _compute_freqs_st

        # dict is allowed for __setstate__ compatibility, and Epochs.compute_tfr() can
        # return an AverageTFR depending on its parameters, so Epochs input is allowed
        _validate_type(
            inst, (BaseEpochs, Evoked, dict), "object passed to AverageTFR constructor"
        )
        # stockwell API is very different from multitaper/morlet
        if method == "stockwell" and not isinstance(inst, dict):
            if isinstance(freqs, str) and freqs == "auto":
                fmin, fmax = None, None
            elif len(freqs) == 2:
                fmin, fmax = freqs
            else:
                raise ValueError(
                    "for Stockwell method, freqs must be a length-2 iterable "
                    f'or "auto", got {freqs}.'
                )
            method_kw.update(fmin=fmin, fmax=fmax)
            # Compute freqs. We need a couple lines of code dupe here (also in
            # BaseTFR.__init__) to get the subset of times to pass to _check_input_st()
            _mask = _time_mask(inst.times, tmin, tmax, sfreq=inst.info["sfreq"])
            _times = inst.times[_mask].copy()
            _, default_nfft, _ = _check_input_st(_times, None)
            n_fft = method_kw.get("n_fft", default_nfft)
            *_, freqs = _compute_freqs_st(fmin, fmax, n_fft, inst.info["sfreq"])

        # use Evoked.comment or str(Epochs.event_id) as the default comment...
        if comment is None:
            comment = getattr(inst, "comment", ",".join(getattr(inst, "event_id", "")))
        # ...but don't overwrite if it's coming in with a comment already set
        if isinstance(inst, dict):
            inst.setdefault("comment", comment)
        else:
            self._comment = getattr(self, "_comment", comment)
        super().__init__(
            inst,
            method,
            freqs,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            decim=decim,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    def __getstate__(self):
        """Prepare AverageTFR object for serialization."""
        out = super().__getstate__()
        out.update(nave=self.nave, comment=self.comment)
        # NOTE: self._itc should never exist in the instance returned to the user; it
        # is temporarily present in the output from the tfr_array_* function, and is
        # split out into a separate AverageTFR object (and deleted from the object
        # holding power estimates) before those objects are passed back to the user.
        # The following lines are there because we make use of __getstate__ to achieve
        # that splitting of objects.
        if hasattr(self, "_itc"):
            out.update(itc=self._itc)
        return out

    def __setstate__(self, state):
        """Unpack AverageTFR from serialized format."""
        super().__setstate__(state)
        self._comment = state.get("comment", "")
        self._nave = state.get("nave", 1)

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, comment):
        self._comment = comment

    @property
    def nave(self):
        return self._nave

    @nave.setter
    def nave(self, nave):
        self._nave = nave

    def _get_instance_data(self, time_mask):
        # AverageTFRs can be constructed from Epochs data, so we triage shape here.
        # Evoked data get a fake singleton "epoch" axis prepended
        dim = slice(None) if _get_instance_type_string(self) == "Epochs" else np.newaxis
        data = self.inst.get_data(picks=self._picks)[dim, :, time_mask]
        self._nave = getattr(self.inst, "nave", data.shape[0])
        return data


@fill_doc
class AverageTFRArray(AverageTFR):
    """Data object for *precomputed* spectrotemporal representations of averaged data.

    Parameters
    ----------
    %(info_not_none)s
    %(data_tfr)s
    %(times)s
    %(freqs_tfr_array)s
    nave : int
        The number of averaged TFRs.
    %(comment_averagetfr_attr)s
    %(method_tfr_array)s

    Attributes
    ----------
    %(baseline_tfr_attr)s
    %(ch_names_tfr_attr)s
    %(comment_averagetfr_attr)s
    %(freqs_tfr_attr)s
    %(info_not_none)s
    %(method_tfr_attr)s
    %(nave_tfr_attr)s
    %(sfreq_tfr_attr)s
    %(shape_tfr_attr)s

    See Also
    --------
    AverageTFR
    EpochsTFRArray
    mne.Epochs.compute_tfr
    mne.Evoked.compute_tfr
    """

    def __init__(
        self, info, data, times, freqs, *, nave=None, comment=None, method=None
    ):
        state = dict(info=info, data=data, times=times, freqs=freqs)
        for name, optional in dict(nave=nave, comment=comment, method=method).items():
            if optional is not None:
                state[name] = optional
        self.__setstate__(state)


@fill_doc
class EpochsTFR(BaseTFR, GetEpochsMixin):
    """Data object for spectrotemporal representations of epoched data.

    .. important::
        The preferred means of creating EpochsTFR objects from :class:`~mne.Epochs`
        objects is via the instance method :meth:`~mne.Epochs.compute_tfr`.
        To create an EpochsTFR object from pre-computed data (i.e., a NumPy array) use
        :class:`~mne.time_frequency.EpochsTFRArray`.

    Parameters
    ----------
    inst : instance of Epochs
        The data from which to compute the time-frequency representation.
    %(freqs_tfr_epochs)s
    %(method_tfr_epochs)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(decim_tfr)s
    %(events_epochstfr)s

        .. deprecated:: 1.7
            Pass an instance of :class:`~mne.Epochs` as ``inst`` instead, or use
            :class:`~mne.time_frequency.EpochsTFRArray` which retains the old API.
    %(event_id_epochstfr)s

        .. deprecated:: 1.7
            Pass an instance of :class:`~mne.Epochs` as ``inst`` instead, or use
            :class:`~mne.time_frequency.EpochsTFRArray` which retains the old API.
    selection : array
        List of indices of selected events (not dropped or ignored etc.). For
        example, if the original event array had 4 events and the second event
        has been dropped, this attribute would be np.array([0, 2, 3]).

        .. deprecated:: 1.7
            Pass an instance of :class:`~mne.Epochs` as ``inst`` instead, or use
            :class:`~mne.time_frequency.EpochsTFRArray` which retains the old API.
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

        .. deprecated:: 1.7
            Pass an instance of :class:`~mne.Epochs` as ``inst`` instead, or use
            :class:`~mne.time_frequency.EpochsTFRArray` which retains the old API.
    %(metadata_epochstfr)s

        .. deprecated:: 1.7
            Pass an instance of :class:`~mne.Epochs` as ``inst`` instead, or use
            :class:`~mne.time_frequency.EpochsTFRArray` which retains the old API.
    %(n_jobs)s
    %(verbose)s
    %(method_kw_tfr)s

    Attributes
    ----------
    %(baseline_tfr_attr)s
    %(ch_names_tfr_attr)s
    %(comment_tfr_attr)s
    %(drop_log)s
    %(event_id_attr)s
    %(events_attr)s
    %(freqs_tfr_attr)s
    %(info_not_none)s
    %(metadata_attr)s
    %(method_tfr_attr)s
    %(selection_attr)s
    %(sfreq_tfr_attr)s
    %(shape_tfr_attr)s

    See Also
    --------
    mne.Epochs.compute_tfr
    RawTFR
    AverageTFR
    EpochsTFRArray

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        *,
        inst=None,
        freqs=None,
        method=None,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        decim=1,
        events=None,
        event_id=None,
        selection=None,
        drop_log=None,
        metadata=None,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        from ..epochs import BaseEpochs

        # dict is allowed for __setstate__ compatibility
        _validate_type(
            inst, (BaseEpochs, dict), "object passed to EpochsTFR constructor", "Epochs"
        )
        super().__init__(
            inst,
            method,
            freqs,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            decim=decim,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @fill_doc
    def __getitem__(self, item):
        """Subselect epochs from an EpochsTFR.

        Parameters
        ----------
        %(item)s
            Access options are the same as for :class:`~mne.Epochs` objects, see the
            docstring Notes section of :meth:`mne.Epochs.__getitem__` for explanation.

        Returns
        -------
        %(getitem_epochstfr_return)s
        """
        return super().__getitem__(item)

    def __getstate__(self):
        """Prepare EpochsTFR object for serialization."""
        out = super().__getstate__()
        out.update(
            metadata=self._metadata,
            drop_log=self.drop_log,
            event_id=self.event_id,
            events=self.events,
            selection=self.selection,
            raw_times=self._raw_times,
        )
        return out

    def __setstate__(self, state):
        """Unpack EpochsTFR from serialized format."""
        if state["data"].ndim != 4:
            raise ValueError(f"EpochsTFR data should be 4D, got {state['data'].ndim}.")
        super().__setstate__(state)
        self._metadata = state.get("metadata", None)
        n_epochs = self.shape[0]
        n_times = self.shape[-1]
        fake_samps = np.linspace(
            n_times, n_times * (n_epochs + 1), n_epochs, dtype=int, endpoint=False
        )
        fake_events = np.dstack(
            (fake_samps, np.zeros_like(fake_samps), np.ones_like(fake_samps))
        ).squeeze(axis=0)
        self.events = state.get("events", _ensure_events(fake_events))
        self.event_id = state.get("event_id", _check_event_id(None, self.events))
        self.drop_log = state.get("drop_log", tuple())
        self.selection = state.get("selection", np.arange(n_epochs))
        self._bad_dropped = True  # always true, need for `equalize_event_counts()`

    def __next__(self, return_event_id=False):
        """Iterate over EpochsTFR objects.

        NOTE: __iter__() and _stop_iter() are defined by the GetEpochs mixin.

        Parameters
        ----------
        return_event_id : bool
            If ``True``, return both the EpochsTFR data and its associated ``event_id``.

        Returns
        -------
        epoch : array of shape (n_channels, n_freqs, n_times)
            The single-epoch time-frequency data.
        event_id : int
            The integer event id associated with the epoch. Only returned if
            ``return_event_id`` is ``True``.
        """
        if self._current >= len(self._data):
            self._stop_iter()
        epoch = self._data[self._current]
        event_id = self.events[self._current][-1]
        self._current += 1
        if return_event_id:
            return epoch, event_id
        return epoch

    def _check_singleton(self):
        """Check if self contains only one Epoch, and return it as an AverageTFR."""
        if self.shape[0] > 1:
            calling_func = inspect.currentframe().f_back.f_code.co_name
            raise NotImplementedError(
                f"Cannot call {calling_func}() from EpochsTFR with multiple epochs; "
                "please subselect a single epoch before plotting."
            )
        return list(self.iter_evoked())[0]

    def _get_instance_data(self, time_mask):
        return self.inst.get_data(picks=self._picks)[:, :, time_mask]

    def _update_epoch_attributes(self):
        # adjust dims and shape
        if self.method != "stockwell":  # stockwell consumes epochs dimension
            self._dims = ("epoch",) + self._dims
            self._shape = (len(self.inst),) + self._shape
        # we need these for to_data_frame()
        self.event_id = self.inst.event_id.copy()
        self.events = self.inst.events.copy()
        self.selection = self.inst.selection.copy()
        # we need these for __getitem__()
        self.drop_log = deepcopy(self.inst.drop_log)
        self._metadata = self.inst.metadata
        # we need this for compatibility with equalize_event_counts()
        self._bad_dropped = True

    def average(self, method="mean", *, dim="epochs", copy=False):
        """Aggregate the EpochsTFR across epochs, frequencies, or times.

        Parameters
        ----------
        method : "mean" | "median" | callable
            How to aggregate the data across the given ``dim``. If callable,
            must take a :class:`NumPy array<numpy.ndarray>` of shape
            ``(n_epochs, n_channels, n_freqs, n_times)`` and return an array
            with one fewer dimensions (which dimension is collapsed depends on
            the value of ``dim``). Default is ``"mean"``.
        dim : "epochs" | "freqs" | "times"
            The dimension along which to combine the data.
        copy : bool
            Whether to return a copy of the modified instance, or modify in place.
            Ignored when ``dim="epochs"`` or ``"times"`` because those options return
            different types (:class:`~mne.time_frequency.AverageTFR` and
            :class:`~mne.time_frequency.EpochsSpectrum`, respectively).

        Returns
        -------
        tfr : instance of EpochsTFR | AverageTFR | EpochsSpectrum
            The aggregated TFR object.

        Notes
        -----
        Passing in ``np.median`` is considered unsafe for complex data; pass
        the string ``"median"`` instead to compute the *marginal* median
        (i.e. the median of the real and imaginary components separately).
        See discussion here:

        https://github.com/scipy/scipy/pull/12676#issuecomment-783370228
        """
        _check_option("dim", dim, ("epochs", "freqs", "times"))
        axis = self._dims.index(dim[:-1])  # self._dims entries aren't plural

        func = _check_combine(mode=method, axis=axis)
        data = func(self.data)

        n_epochs, n_channels, n_freqs, n_times = self.data.shape
        freqs, times = self.freqs, self.times
        if dim == "epochs":
            expected_shape = self._data.shape[1:]
        elif dim == "freqs":
            expected_shape = (n_epochs, n_channels, n_times)
            freqs = np.mean(self.freqs, keepdims=True)
        elif dim == "times":
            expected_shape = (n_epochs, n_channels, n_freqs)
            times = np.mean(self.times, keepdims=True)

        if data.shape != expected_shape:
            raise RuntimeError(
                "EpochsTFR.average() got a method that resulted in data of shape "
                f"{data.shape}, but it should be {expected_shape}."
            )
        state = self.__getstate__()
        # restore singleton freqs axis (not necessary for epochs/times: class changes)
        if dim == "freqs":
            data = np.expand_dims(data, axis=axis)
        else:
            state["dims"] = (*state["dims"][:axis], *state["dims"][axis + 1 :])
        state["data"] = data
        state["info"] = deepcopy(self.info)
        state["freqs"] = freqs
        state["times"] = times
        if dim == "epochs":
            state["inst_type_str"] = "Evoked"
            state["nave"] = n_epochs
            state["comment"] = f"{method} of {n_epochs} EpochsTFR{_pl(n_epochs)}"
            out = AverageTFR(inst=state)
            out._data_type = "Average Power"
            return out

        elif dim == "times":
            return EpochsSpectrum(
                state,
                method=None,
                fmin=None,
                fmax=None,
                tmin=None,
                tmax=None,
                picks=None,
                exclude=None,
                proj=None,
                remove_dc=None,
                n_jobs=None,
            )
        # ↓↓↓ these two are for dim == "freqs"
        elif copy:
            return EpochsTFR(inst=state, method=None, freqs=None)
        else:
            self._data = np.expand_dims(data, axis=axis)
            self._freqs = freqs
            return self

    @verbose
    def drop(self, indices, reason="USER", verbose=None):
        """Drop epochs based on indices or boolean mask.

        .. note:: The indices refer to the current set of undropped epochs
                  rather than the complete set of dropped and undropped epochs.
                  They are therefore not necessarily consistent with any
                  external indices (e.g., behavioral logs). To drop epochs
                  based on external criteria, do not use the ``preload=True``
                  flag when constructing an Epochs object, and call this
                  method before calling the :meth:`mne.Epochs.drop_bad` or
                  :meth:`mne.Epochs.load_data` methods.

        Parameters
        ----------
        indices : array of int or bool
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        reason : str
            Reason for dropping the epochs ('ECG', 'timeout', 'blink' etc).
            Default: 'USER'.
        %(verbose)s

        Returns
        -------
        epochs : instance of Epochs or EpochsTFR
            The epochs with indices dropped. Operates in-place.
        """
        from ..epochs import BaseEpochs

        BaseEpochs.drop(self, indices=indices, reason=reason, verbose=verbose)

        return self

    def iter_evoked(self, copy=False):
        """Iterate over EpochsTFR to yield a sequence of AverageTFR objects.

        The AverageTFR objects will each contain a single epoch (i.e., no averaging is
        performed). This method resets the EpochTFR instance's iteration state to the
        first epoch.

        Parameters
        ----------
        copy : bool
            Whether to yield copies of the data and measurement info, or views/pointers.
        """
        self.__iter__()
        state = self.__getstate__()
        state["inst_type_str"] = "Evoked"
        state["dims"] = state["dims"][1:]  # drop "epochs"

        while True:
            try:
                data, event_id = self.__next__(return_event_id=True)
            except StopIteration:
                break
            if copy:
                state["info"] = deepcopy(self.info)
                state["data"] = data.copy()
            else:
                state["data"] = data
            state["nave"] = 1
            yield AverageTFR(inst=state, method=None, freqs=None, comment=str(event_id))

    @verbose
    @copy_doc(BaseTFR.plot)
    def plot(
        self,
        picks=None,
        *,
        exclude=(),
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        baseline=None,
        mode="mean",
        dB=False,
        combine=None,
        layout=None,  # TODO deprecate; not used in orig implementation
        yscale="auto",
        vlim=(None, None),
        cnorm=None,
        cmap=None,
        colorbar=True,
        title=None,  # don't deprecate this one; has (useful) option title="auto"
        mask=None,
        mask_style=None,
        mask_cmap="Greys",
        mask_alpha=0.1,
        axes=None,
        show=True,
        verbose=None,
    ):
        singleton_epoch = self._check_singleton()
        return singleton_epoch.plot(
            picks=picks,
            exclude=exclude,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            baseline=baseline,
            mode=mode,
            dB=dB,
            combine=combine,
            layout=layout,
            yscale=yscale,
            vlim=vlim,
            cnorm=cnorm,
            cmap=cmap,
            colorbar=colorbar,
            title=title,
            mask=mask,
            mask_style=mask_style,
            mask_cmap=mask_cmap,
            mask_alpha=mask_alpha,
            axes=axes,
            show=show,
            verbose=verbose,
        )

    @verbose
    @copy_doc(BaseTFR.plot_topo)
    def plot_topo(
        self,
        picks=None,
        baseline=None,
        mode="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        vmin=None,  # TODO deprecate in favor of `vlim` (needs helper func refactor)
        vmax=None,
        layout=None,
        cmap=None,
        title=None,  # don't deprecate; topo titles aren't standard (color, size, just.)
        dB=False,
        colorbar=True,
        layout_scale=0.945,
        show=True,
        border="none",
        fig_facecolor="k",
        fig_background=None,
        font_color="w",
        yscale="auto",
        verbose=None,
    ):
        singleton_epoch = self._check_singleton()
        return singleton_epoch.plot_topo(
            picks=picks,
            baseline=baseline,
            mode=mode,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            vmin=vmin,
            vmax=vmax,
            layout=layout,
            cmap=cmap,
            title=title,
            dB=dB,
            colorbar=colorbar,
            layout_scale=layout_scale,
            show=show,
            border=border,
            fig_facecolor=fig_facecolor,
            fig_background=fig_background,
            font_color=font_color,
            yscale=yscale,
            verbose=verbose,
        )

    @verbose
    @copy_doc(BaseTFR.plot_joint)
    def plot_joint(
        self,
        *,
        timefreqs=None,
        picks=None,
        exclude=(),
        combine="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        baseline=None,
        mode="mean",
        dB=False,
        yscale="auto",
        vlim=(None, None),
        cnorm=None,
        cmap=None,
        colorbar=True,
        title=None,
        show=True,
        topomap_args=None,
        image_args=None,
        verbose=None,
    ):
        singleton_epoch = self._check_singleton()
        return singleton_epoch.plot_joint(
            timefreqs=timefreqs,
            picks=picks,
            exclude=exclude,
            combine=combine,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            baseline=baseline,
            mode=mode,
            dB=dB,
            yscale=yscale,
            vlim=vlim,
            cnorm=cnorm,
            cmap=cmap,
            colorbar=colorbar,
            title=title,
            show=show,
            topomap_args=topomap_args,
            image_args=image_args,
            verbose=verbose,
        )

    @copy_doc(BaseTFR.plot_topomap)
    def plot_topomap(
        self,
        tmin=None,
        tmax=None,
        fmin=0.0,
        fmax=np.inf,
        *,
        ch_type=None,
        baseline=None,
        mode="mean",
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=2,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%1.1e",
        units=None,
        axes=None,
        show=True,
    ):
        singleton_epoch = self._check_singleton()
        return singleton_epoch.plot_topomap(
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            ch_type=ch_type,
            baseline=baseline,
            mode=mode,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            units=units,
            axes=axes,
            show=show,
        )


@fill_doc
class EpochsTFRArray(EpochsTFR):
    """Data object for *precomputed* spectrotemporal representations of epoched data.

    Parameters
    ----------
    %(info_not_none)s
    %(data_tfr)s
    %(times)s
    %(freqs_tfr_array)s
    %(comment_tfr_attr)s
    %(method_tfr_array)s
    %(events_epochstfr)s
    %(event_id_epochstfr)s
    %(selection)s
    %(drop_log)s
    %(metadata_epochstfr)s

    Attributes
    ----------
    %(baseline_tfr_attr)s
    %(ch_names_tfr_attr)s
    %(comment_tfr_attr)s
    %(drop_log)s
    %(event_id_attr)s
    %(events_attr)s
    %(freqs_tfr_attr)s
    %(info_not_none)s
    %(metadata_attr)s
    %(method_tfr_attr)s
    %(selection_attr)s
    %(sfreq_tfr_attr)s
    %(shape_tfr_attr)s

    See Also
    --------
    AverageTFR
    mne.Epochs.compute_tfr
    mne.Evoked.compute_tfr
    """

    def __init__(
        self,
        info,
        data,
        times,
        freqs,
        *,
        comment=None,
        method=None,
        events=None,
        event_id=None,
        selection=None,
        drop_log=None,
        metadata=None,
    ):
        state = dict(info=info, data=data, times=times, freqs=freqs)
        optional = dict(
            comment=comment,
            method=method,
            events=events,
            event_id=event_id,
            selection=selection,
            drop_log=drop_log,
            metadata=metadata,
        )
        for name, value in optional.items():
            if value is not None:
                state[name] = value
        self.__setstate__(state)


@fill_doc
class RawTFR(BaseTFR):
    """Data object for spectrotemporal representations of continuous data.

    .. warning:: The preferred means of creating RawTFR objects from
                 :class:`~mne.io.Raw` objects is via the instance method
                 :meth:`~mne.io.Raw.compute_tfr`. Direct class instantiation
                 is not supported.

    Parameters
    ----------
    inst : instance of Raw
        The data from which to compute the time-frequency representation.
    %(method_tfr)s
    %(freqs_tfr)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(reject_by_annotation_tfr)s
    %(decim_tfr)s
    %(n_jobs)s
    %(verbose)s
    %(method_kw_tfr)s

    Attributes
    ----------
    ch_names : list
        The channel names.
    freqs : array
        Frequencies at which the amplitude, power, or fourier coefficients
        have been computed.
    %(info_not_none)s
    method : str
        The method used to compute the spectra (``'morlet'``, ``'multitaper'``
        or ``'stockwell'``).

    See Also
    --------
    mne.io.Raw.compute_tfr
    EpochsTFR
    AverageTFR

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        inst,
        method=None,
        freqs=None,
        *,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        reject_by_annotation=False,
        decim=1,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        from ..io import BaseRaw

        # dict is allowed for __setstate__ compatibility
        _validate_type(
            inst, (BaseRaw, dict), "object passed to RawTFR constructor", "Raw"
        )
        super().__init__(
            inst,
            method,
            freqs,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            reject_by_annotation=reject_by_annotation,
            decim=decim,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    def __getitem__(self, item):
        """Get RawTFR data.

        Parameters
        ----------
        item : int | slice | array-like
            Indexing is similar to a :class:`NumPy array<numpy.ndarray>`; see
            Notes.

        Returns
        -------
        %(getitem_tfr_return)s

        Notes
        -----
        The last axis is always time, the next-to-last axis is always
        frequency, and the first axis is always channel. If
        ``method='multitaper'`` and ``output='complex'`` then the second axis
        will be taper index.

        Integer-, list-, and slice-based indexing is possible:

        - ``raw_tfr[[0, 2]]`` gives the whole time-frequency plane for the
          first and third channels.
        - ``raw_tfr[..., :3, :]`` gives the first 3 frequency bins and all
          times for all channels (and tapers, if present).
        - ``raw_tfr[..., :100]`` gives the first 100 time samples in all
          frequency bins for all channels (and tapers).
        - ``raw_tfr[(4, 7)]`` is the same as ``raw_tfr[4, 7]``.

        .. note::

           Unlike :class:`~mne.io.Raw` objects (which returns a tuple of the
           requested data values and the corresponding times), accessing
           :class:`~mne.time_frequency.RawTFR` values via subscript does
           **not** return the corresponding frequency bin values. If you need
           them, use ``RawTFR.freqs[freq_indices]`` or
           ``RawTFR.get_data(..., return_freqs=True)``.
        """
        from ..io import BaseRaw

        self._parse_get_set_params = partial(BaseRaw._parse_get_set_params, self)
        return BaseRaw._getitem(self, item, return_times=False)

    def _get_instance_data(self, time_mask, reject_by_annotation):
        start, stop = np.where(time_mask)[0][[0, -1]]
        rba = "NaN" if reject_by_annotation else None
        data = self.inst.get_data(
            self._picks, start, stop + 1, reject_by_annotation=rba
        )
        # prepend a singleton "epochs" axis
        return data[np.newaxis]


@fill_doc
class RawTFRArray(RawTFR):
    """Data object for *precomputed* spectrotemporal representations of continuous data.

    Parameters
    ----------
    %(info_not_none)s
    %(data_tfr)s
    %(times)s
    %(freqs_tfr_array)s
    %(method_tfr_array)s

    Attributes
    ----------
    %(baseline_tfr_attr)s
    %(ch_names_tfr_attr)s
    %(freqs_tfr_attr)s
    %(info_not_none)s
    %(method_tfr_attr)s
    %(sfreq_tfr_attr)s
    %(shape_tfr_attr)s

    See Also
    --------
    RawTFR
    mne.io.Raw.compute_tfr
    EpochsTFRArray
    AverageTFRArray
    """

    def __init__(
        self,
        info,
        data,
        times,
        freqs,
        *,
        method=None,
    ):
        state = dict(info=info, data=data, times=times, freqs=freqs)
        if method is not None:
            state["method"] = method
        self.__setstate__(state)


def combine_tfr(all_tfr, weights="nave"):
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
        if weights not in ("nave", "equal"):
            raise ValueError('Weights must be a list of float, or "nave" or "equal"')
        if weights == "nave":
            weights = np.array([e.nave for e in all_tfr], float)
            weights /= weights.sum()
        else:  # == 'equal'
            weights = [1.0 / len(all_tfr)] * len(all_tfr)
    weights = np.array(weights, float)
    if weights.ndim != 1 or weights.size != len(all_tfr):
        raise ValueError("Weights must be the same size as all_tfr")

    ch_names = tfr.ch_names
    for t_ in all_tfr[1:]:
        assert t_.ch_names == ch_names, ValueError(
            f"{tfr} and {t_} do not contain the same channels"
        )
        assert np.max(np.abs(t_.times - tfr.times)) < 1e-7, ValueError(
            f"{tfr} and {t_} do not contain the same time instants"
        )

    # use union of bad channels
    bads = list(set(tfr.info["bads"]).union(*(t_.info["bads"] for t_ in all_tfr[1:])))
    tfr.info["bads"] = bads

    # XXX : should be refactored with combined_evoked function
    tfr.data = sum(w * t_.data for w, t_ in zip(weights, all_tfr))
    tfr.nave = max(int(1.0 / sum(w**2 / e.nave for w, e in zip(weights, all_tfr))), 1)
    return tfr


# Utils


# ↓↓↓↓↓↓↓↓↓↓↓ this is still used in _stockwell.py
def _get_data(inst, return_itc):
    """Get data from Epochs or Evoked instance as epochs x ch x time."""
    from ..epochs import BaseEpochs
    from ..evoked import Evoked

    if not isinstance(inst, BaseEpochs | Evoked):
        raise TypeError("inst must be Epochs or Evoked")
    if isinstance(inst, BaseEpochs):
        data = inst.get_data(copy=False)
    else:
        if return_itc:
            raise ValueError("return_itc must be False for evoked data")
        data = inst.data[np.newaxis].copy()
    return data


def _prepare_picks(info, data, picks, axis):
    """Prepare the picks."""
    picks = _picks_to_idx(info, picks, exclude="bads")
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


def _preproc_tfr(
    data,
    times,
    freqs,
    tmin,
    tmax,
    fmin,
    fmax,
    mode,
    baseline,
    vmin,
    vmax,
    dB,
    sfreq,
    copy=None,
):
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


def _ensure_slice(decim):
    """Aux function checking the decim parameter."""
    _validate_type(decim, ("int-like", slice), "decim")
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
    fname : path-like
        The file name, which should end with ``-tfr.h5``.
    tfr : RawTFR | EpochsTFR | AverageTFR | list of RawTFR | list of EpochsTFR | list of AverageTFR
        The (list of) TFR object(s) to save in one file. If ``tfr.comment`` is ``None``,
        a sequential numeric string name will be generated on the fly, based on the
        order in which the TFR objects are passed. This can be used to selectively load
        single TFR objects from the file later.
    %(overwrite)s
    %(verbose)s

    See Also
    --------
    read_tfrs

    Notes
    -----
    .. versionadded:: 0.9.0
    """  # noqa E501
    _, write_hdf5 = _import_h5io_funcs()
    out = []
    if not isinstance(tfr, list | tuple):
        tfr = [tfr]
    for ii, tfr_ in enumerate(tfr):
        comment = ii if getattr(tfr_, "comment", None) is None else tfr_.comment
        state = tfr_.__getstate__()
        if "metadata" in state:
            state["metadata"] = _prepare_write_metadata(state["metadata"])
        out.append((comment, state))
    write_hdf5(fname, out, overwrite=overwrite, title="mnepython", slash="replace")


@verbose
def read_tfrs(fname, condition=None, *, verbose=None):
    """Load a TFR object from disk.

    Parameters
    ----------
    fname : path-like
        Path to a TFR file in HDF5 format, which should end with ``-tfr.h5`` or
        ``-tfr.hdf5``.
    condition : int or str | list of int or str | None
        The condition to load. If ``None``, all conditions will be returned.
        Defaults to ``None``.
    %(verbose)s

    Returns
    -------
    tfr : RawTFR | EpochsTFR | AverageTFR | list of RawTFR | list of EpochsTFR | list of AverageTFR
        The loaded time-frequency object.

    See Also
    --------
    mne.time_frequency.RawTFR.save
    mne.time_frequency.EpochsTFR.save
    mne.time_frequency.AverageTFR.save
    write_tfrs

    Notes
    -----
    .. versionadded:: 0.9.0
    """  # noqa E501
    read_hdf5, _ = _import_h5io_funcs()
    fname = _check_fname(fname=fname, overwrite="read", must_exist=False)
    valid_fnames = tuple(
        f"{sep}tfr.{ext}" for sep in ("-", "_") for ext in ("h5", "hdf5")
    )
    check_fname(fname, "tfr", valid_fnames)
    logger.info(f"Reading {fname} ...")
    hdf5_dict = read_hdf5(fname, title="mnepython", slash="replace")
    # single TFR from TFR.save()
    if "inst_type_str" in hdf5_dict:
        if "epoch" in hdf5_dict["dims"]:
            Klass = EpochsTFR
        elif "nave" in hdf5_dict:
            Klass = AverageTFR
        else:
            Klass = RawTFR
        out = Klass(inst=hdf5_dict)
        if getattr(out, "metadata", None) is not None:
            out.metadata = _prepare_read_metadata(out.metadata)
        return out
    # maybe multiple TFRs from write_tfrs()
    return _read_multiple_tfrs(hdf5_dict, condition=condition, verbose=verbose)


@verbose
def _read_multiple_tfrs(tfr_data, condition=None, *, verbose=None):
    """Read (possibly multiple) TFR datasets from an h5 file written by write_tfrs()."""
    out = list()
    keys = list()
    # tfr_data is a list of (comment, tfr_dict) tuples
    for key, tfr in tfr_data:
        keys.append(str(key))  # auto-assigned keys are ints
        is_epochs = tfr["data"].ndim == 4
        is_average = "nave" in tfr
        if condition is not None:
            if not is_average:
                raise NotImplementedError(
                    "condition is only supported when reading AverageTFRs."
                )
            if key != condition:
                continue
        tfr = dict(tfr)
        tfr["info"] = Info(tfr["info"])
        tfr["info"]._check_consistency()
        if "metadata" in tfr:
            tfr["metadata"] = _prepare_read_metadata(tfr["metadata"])
        # additional keys needed for TFR __setstate__
        defaults = dict(baseline=None, data_type="Power Estimates")
        if is_epochs:
            Klass = EpochsTFR
            defaults.update(
                inst_type_str="Epochs", dims=("epoch", "channel", "freq", "time")
            )
        elif is_average:
            Klass = AverageTFR
            defaults.update(inst_type_str="Evoked", dims=("channel", "freq", "time"))
        else:
            Klass = RawTFR
            defaults.update(inst_type_str="Raw", dims=("channel", "freq", "time"))
        out.append(Klass(inst=defaults | tfr))
    if len(out) == 0:
        raise ValueError(
            f'Cannot find condition "{condition}" in this file. '
            f'The file contains conditions {", ".join(keys)}'
        )
    if len(out) == 1:
        out = out[0]
    return out


def _get_timefreqs(tfr, timefreqs):
    """Find and/or setup timefreqs for `tfr.plot_joint`."""
    # Input check
    timefreq_error_msg = (
        "Supplied `timefreqs` are somehow malformed. Please supply None, "
        "a list of tuple pairs, or a dict of such tuple pairs, not {}"
    )
    if isinstance(timefreqs, dict):
        for k, v in timefreqs.items():
            for item in (k, v):
                if len(item) != 2 or any(not _is_numeric(n) for n in item):
                    raise ValueError(timefreq_error_msg, item)
    elif timefreqs is not None:
        if not hasattr(timefreqs, "__len__"):
            raise ValueError(timefreq_error_msg.format(timefreqs))
        if len(timefreqs) == 2 and all(_is_numeric(v) for v in timefreqs):
            timefreqs = [tuple(timefreqs)]  # stick a pair of numbers in a list
        else:
            for item in timefreqs:
                if (
                    hasattr(item, "__len__")
                    and len(item) == 2
                    and all(_is_numeric(n) for n in item)
                ):
                    pass
                else:
                    raise ValueError(timefreq_error_msg.format(item))

    # If None, automatic identification of max peak
    else:
        order = max((1, tfr.data.shape[2] // 30))
        peaks_idx = argrelmax(tfr.data, order=order, axis=2)
        if peaks_idx[0].size == 0:
            _, p_t, p_f = np.unravel_index(tfr.data.argmax(), tfr.data.shape)
            timefreqs = [(tfr.times[p_t], tfr.freqs[p_f])]
        else:
            peaks = [tfr.data[0, f, t] for f, t in zip(peaks_idx[1], peaks_idx[2])]
            peakmax_idx = np.argmax(peaks)
            peakmax_time = tfr.times[peaks_idx[2][peakmax_idx]]
            peakmax_freq = tfr.freqs[peaks_idx[1][peakmax_idx]]

            timefreqs = [(peakmax_time, peakmax_freq)]

    timefreqs = {
        tuple(k): np.asarray(timefreqs[k])
        if isinstance(timefreqs, dict)
        else np.array([0, 0])
        for k in timefreqs
    }

    return timefreqs


def _check_tfr_complex(tfr, reason="source space estimation"):
    """Check that time-frequency epochs or average data is complex."""
    if not np.iscomplexobj(tfr.data):
        raise RuntimeError(f"Time-frequency data must be complex for {reason}")


def _merge_if_grads(data, info, ch_type, sphere, combine=None):
    if ch_type == "grad":
        grad_picks = _pair_grad_sensors(info, topomap_coords=False)
        pos = _find_topomap_coords(info, picks=grad_picks[::2], sphere=sphere)
        grad_method = combine if isinstance(combine, str) else "rms"
        data, _ = _merge_ch_data(data[grad_picks], ch_type, [], method=grad_method)
    else:
        pos, _ = _get_pos_outlines(info, picks=ch_type, sphere=sphere)
    return data, pos


@verbose
def _prep_data_for_plot(
    data,
    times,
    freqs,
    *,
    tmin=None,
    tmax=None,
    fmin=None,
    fmax=None,
    baseline=None,
    mode=None,
    dB=False,
    verbose=None,
):
    # baseline
    copy = baseline is not None
    data = rescale(data, times, baseline, mode, copy=copy, verbose=verbose)
    # crop times
    time_mask = np.nonzero(_time_mask(times, tmin, tmax))[0]
    times = times[time_mask]
    # crop freqs
    freq_mask = np.nonzero(_time_mask(freqs, fmin, fmax))[0]
    freqs = freqs[freq_mask]
    # crop data
    data = data[..., freq_mask, :][..., time_mask]
    # complex amplitude → real power; real-valued data is already power (or ITC)
    if np.iscomplexobj(data):
        data = (data * data.conj()).real
    if dB:
        data = 10 * np.log10(data)
    return data, times, freqs
