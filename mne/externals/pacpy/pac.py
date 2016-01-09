# -*- coding: utf-8 -*-
"""
Functions to calculate phase-amplitude coupling.
"""
from __future__ import division
import numpy as np
from scipy.signal import hilbert
from scipy.stats.mstats import zscore
from .filt import firf, morletf


def _x_sanity(lo=None, hi=None):
    if lo is not None:
        if np.any(np.isnan(lo)):
            raise ValueError("lo contains NaNs")

    if hi is not None:
        if np.any(np.isnan(hi)):
            raise ValueError("hi contains NaNs")

    if (hi is not None) and (lo is not None):
        if lo.size != hi.size:
            raise ValueError("lo and hi must be the same length")


def _range_sanity(f_lo=None, f_hi=None):
    if f_lo is not None:
        if len(f_lo) != 2:
            raise ValueError("f_lo must contain two elements")

        if f_lo[0] < 0:
            raise ValueError("Elements in f_lo must be > 0")

    if f_hi is not None:
        if len(f_hi) != 2:
            raise ValueError("f_hi must contain two elements")
        if f_hi[0] < 0:
            raise ValueError("Elements in f_hi must be > 0")


def plv(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the phase-locking value (PLV) method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function, False
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the phase time series of the low-bandpass
        of the amplitude of the high-bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import plv
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> plv(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.99863308613553081
    """

    # Arg check
    _x_sanity(lo, hi)

    # Filter setup
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)
        amp = np.abs(hilbert(hi))
        hi = filterfn(amp, f_lo, fs, **filter_kwargs)

        lo = np.angle(hilbert(lo))
        hi = np.angle(hilbert(hi))

    # Make arrays the same size
    lo, hi = _trim_edges(lo, hi)

    # Calculate PLV
    pac = np.abs(np.mean(np.exp(1j * (lo - hi))))

    return pac


def _trim_edges(lo, hi):
    """
    Remove extra edge artifact from the signal with the shorter filter
    so that its time series is identical to that of the filtered signal
    with a longer filter.
    """
    
    if len(lo) == len(hi):
        return lo, hi  # Die early if there's nothing to do.
    elif len(lo) < len(hi):
        Ndiff = len(hi) - len(lo)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')
        hi = hi[np.int(Ndiff / 2):np.int(-Ndiff / 2)]
    else:
        Ndiff = len(lo) - len(hi)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')                
        lo = lo[np.int(Ndiff / 2):np.int(-Ndiff / 2)]

    return lo, hi


def mi_tort(lo, hi, f_lo, f_hi, fs=1000, Nbins=20, filterfn=None,
            filter_kwargs=None):
    """
    Calculate PAC using the modulation index method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to ue as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering ranges
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    Nbins : int
        Number of bins to split up the low frequency oscillation cycle

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import mi_tort
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> mi_tort(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.34898478944110811
    """

    # Arg check
    _x_sanity(lo, hi)
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle'
            'must be an integer >1.')

    # Filter setup
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)

        hi = np.abs(hilbert(hi))
        lo = np.angle(hilbert(lo))

    # Make arrays the same size
    lo, hi = _trim_edges(lo, hi)

    # Convert the phase time series from radians to degrees
    phadeg = np.degrees(lo)

    # Calculate PAC
    binsize = 360 / Nbins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(hi[phaserange])

    p_j = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        p_j[b] = mean_amp[b] / sum(mean_amp)

    h = -np.sum(p_j * np.log10(p_j))
    h_max = np.log10(Nbins)
    pac = (h_max - h) / h_max

    return pac


def _ols(y, X):
    """Custom OLS (to minimize outside dependecies)"""

    dummy = np.repeat(1.0, X.shape[0])
    X = np.hstack([X, dummy[:, np.newaxis]])

    beta_hat, resid, _, _ = np.linalg.lstsq(X, y)
    y_hat = np.dot(X, beta_hat)

    return y_hat, beta_hat


def glm(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the generalized linear model (GLM) method

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_high : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import glm
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> glm(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.69090396896138917
    """

    # Arg check
    _x_sanity(lo, hi)

    # Filter series
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)

        hi = np.abs(hilbert(hi))
        lo = np.angle(hilbert(lo))

    # Make arrays the same size
    lo, hi = _trim_edges(lo, hi)

    # First prepare GLM
    y = hi
    X_pre = np.vstack((np.cos(lo), np.sin(lo)))
    X = X_pre.T
    y_hat, beta_hat = _ols(y, X)
    resid = y - y_hat

    # Calculate PAC from GLM residuals
    pac = 1 - np.sum(resid ** 2) / np.sum(
        (hi - np.mean(hi)) ** 2)

    return pac


def mi_canolty(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None,
               n_surr=100):
    """
    Calculate PAC using the modulation index (MI) method defined in Canolty,
    2006

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    n_surr : int
        Number of surrogate tests to run to calculate normalized MI

    Returns
    -------
    pac : scalar
      PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import mi_canolty
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> mi_canolty(lo, hi, (4,8), (80,150)) # Calculate PAC
    1.1605177063713188
    """

    # Arg check
    _x_sanity(lo, hi)

    # Filter series
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)

        hi = np.abs(hilbert(hi))
        lo = np.angle(hilbert(lo))

    # Make arrays the same size
    lo, hi = _trim_edges(lo, hi)

    # Calculate modulation index
    pac = np.abs(np.mean(hi * np.exp(1j * lo)))

    # Calculate surrogate MIs
    pacS = np.zeros(n_surr)
    np.random.seed(0)
    for s in range(n_surr):
        loS = np.roll(lo, np.random.randint(len(lo)))
        pacS[s] = np.abs(np.mean(hi * np.exp(1j * loS)))

    # Return z-score of observed PAC compared to null distribution
    return (pac - np.mean(pacS)) / np.std(pacS)


def ozkurt(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate PAC using the method defined in Ozkurt & Schnitzler, 2011

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'. 
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING! 
        - In expert mode the user needs to filter the data AND apply the 
        hilbert transform. 
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
      PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import ozkurt
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> ozkurt(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.48564417921240238
    """

    # Arg check
    _x_sanity(lo, hi)

    # Filter series
    if filterfn is None:
        filterfn = firf

    if filter_kwargs is None:
        filter_kwargs = {}

    # Filter then hilbert
    if filterfn is not False:
        _range_sanity(f_lo, f_hi)
        lo = filterfn(lo, f_lo, fs, **filter_kwargs)
        hi = filterfn(hi, f_hi, fs, **filter_kwargs)

        hi = np.abs(hilbert(hi))
        lo = np.angle(hilbert(lo))

    # Make arrays the same size
    lo, hi = _trim_edges(lo, hi)

    # Calculate PAC
    pac = np.abs(np.sum(hi * np.exp(1j * lo))) / \
        (np.sqrt(len(lo)) * np.sqrt(np.sum(hi**2)))
    return pac


def otc(x, f_hi, f_step, fs=1000,
        w=3, event_prc=95, t_modsig=None, t_buffer=.01):
    """
    Calculate the oscillation-triggered coupling measure of phase-amplitude
    coupling from Dvorak, 2014.

    Parameters
    ----------
    x : array-like, 1d
        The time series
    f_hi : (low, high), Hz
        The low frequency filtering range
    f_step : float, Hz
        The width of each frequency bin in the time-frequency representation
    fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the
        oscillation whose frequency is the center of the bandpass filter
    event_prc : float (in range 0-100)
        The percentile threshold of the power signal of an oscillation
        for an event to be declared
    t_modsig : (min, max)
        Time (seconds) around an event to extract to define the modulation
        signal
    t_buffer : float
        Minimum time (seconds) in between high frequency events

    Returns
    -------
    pac : float
        phase-amplitude coupling value
    tf : 2-dimensional array
        time-frequency representation of input signal
    a_events : array
        samples at which a high frequency event occurs
    mod_sig : array
        modulation signal (see Dvorak, 2014)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import otc
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pac, _, _, _ = otc(lo + hi, (80,150), 4) # Calculate PAC
    >>> print pac
    2.1324570402314196
    """

    # Arg check
    _x_sanity(x, None)
    _range_sanity(None, f_hi)
    # Set default time range for modulatory signal
    if t_modsig is None:
        t_modsig = (-1, 1)
    if f_step <= 0:
        raise ValueError('Frequency band width must be a positive number.')
    if t_modsig[0] > t_modsig[1]:
        raise ValueError('Invalid time range for modulation signal.')

    # Calculate the time-frequency representation
    f0s = np.arange(f_hi[0], f_hi[1], f_step)
    tf = _morletT(x, f0s, w=w, fs=fs)

    # Find the high frequency activity event times
    F = len(f0s)
    a_events = np.zeros(F, dtype=object)
    for f in range(F):
        a_events[f] = _peaktimes(
            zscore(np.abs(tf[f])), prc=event_prc, t_buffer=t_buffer)

    # Calculate the modulation signal
    samp_modsig = np.arange(t_modsig[0] * fs, t_modsig[1] * fs)
    samp_modsig = samp_modsig.astype(int)
    S = len(samp_modsig)
    mod_sig = np.zeros([F, S])

    # For each frequency in the time-frequency representation, calculate a
    # modulation signal
    for f in range(F):
        # Exclude high frequency events that are too close to the signal
        # boundaries to extract an entire modulation signal
        mask = np.ones(len(a_events[f]), dtype=bool)
        mask[a_events[f] <= samp_modsig[-1]] = False
        mask[a_events[f] >= (len(x) - samp_modsig[-1])] = False
        a_events[f] = a_events[f][mask]

        # Calculate the average LFP around each high frequency event
        E = len(a_events[f])
        for e in range(E):
            cur_ecog = x[a_events[f][e] + samp_modsig]
            mod_sig[f] = mod_sig[f] + cur_ecog / E

    # Calculate modulation strength, the range of the modulation signal
    mod_strength = np.zeros(F)
    for f in range(F):
        mod_strength = np.max(mod_sig[f]) - np.min(mod_sig[f])

    # Calculate PAC
    pac = np.max(mod_strength)

    return pac, tf, a_events, mod_sig


def _peaktimes(x, prc=95, t_buffer=.01, fs=1000):
    """
    Calculate event times for which the power signal x peaks

    Parameters
    ----------
    x : array
        Time series of power
    prc : float (in range 0-100)
        The percentile threshold of x for an event to be declares
    t_buffer : float
        Minimum time (seconds) in between events
    fs : float
        Sampling rate
    """
    if np.logical_or(prc < 0, prc >= 100):
        raise ValueError('Percentile threshold must be between 0 and 100.')

    samp_buffer = np.int(np.round(t_buffer * fs))
    hi = x > np.percentile(x, prc)
    event_intervals = _chunk_time(hi, samp_buffer=samp_buffer)
    E = np.int(np.size(event_intervals) / 2)
    events = np.zeros(E, dtype=object)

    for e in range(E):
        temp = x[np.arange(event_intervals[e][0], event_intervals[e][1] + 1)]
        events[e] = event_intervals[e][0] + np.argmax(temp)

    return events


def _chunk_time(x, samp_buffer=0):
    """
    Define continuous chunks of integers

    Parameters
    ----------
    x : array
        Array of integers
    samp_buffer : int
        Minimum number of samples between chunks

    Returns
    -------
    chunks : array (#chunks x 2)
        List of the sample bounds for each chunk
    """
    if samp_buffer < 0:
        raise ValueError(
            'Buffer between signal peaks must be a positive number')
    if samp_buffer != int(samp_buffer):
        raise ValueError('Number of samples must be an integer')

    if type(x[0]) == np.bool_:
        Xs = np.arange(len(x))
        x = Xs[x]
    X = len(x)

    cur_start = x[0]
    cur_samp = x[0]
    Nchunk = 0
    chunks = []
    for i in range(1, X):
        if x[i] > (cur_samp + samp_buffer + 1):
            if Nchunk == 0:
                chunks = [cur_start, cur_samp]
            else:
                chunks = np.vstack([chunks, [cur_start, cur_samp]])

            Nchunk = Nchunk + 1
            cur_start = x[i]

        cur_samp = x[i]

    # Add final row to chunk
    if Nchunk == 0:
        chunks = [[cur_start, cur_samp]]
    else:
        chunks = np.vstack([chunks, [cur_start, cur_samp]])

    return chunks


def _morletT(x, f0s, w=3, fs=1000, s=1):
    """
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets

    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the center of the 
        bandpass filter
    Fs : float
        Sampling rate
    s : float
        Scaling factor

    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """
    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F, T], dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], fs=fs, w=w, s=s)

    return mwt


def comodulogram(lo, hi, p_range, a_range, dp, da, fs=1000,
                 pac_method='mi_tort',
                 filterfn=None, filter_kwargs=None):
    """
    Calculate PAC for many small frequency bands

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    p_range : (low, high), Hz
        The low frequency filtering range
    a_range : (low, high), Hz
        The high frequency filtering range
    dp : float, Hz
        Width of the low frequency filtering range for each PAC calculation
    da : float, Hz
        Width of the high frequency filtering range for each PAC calculation
    fs : float
        The sampling rate (default = 1000Hz)
    pac_method : string
        Method to calculate PAC.
        'mi_tort' - See Tort, 2008
        'plv' - See Penny, 2008
        'glm' - See Penny, 2008
        'mi_canolty' - See Canolty, 2006
        'ozkurt' - See Ozkurt & Schnitzler, 2011
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    comod : array-like, 2d
        Matrix of phase-amplitude coupling values for each combination of the
        phase frequency bin and the amplitude frequency bin

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import comodulogram
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> comod = comodulogram(lo, hi, (5,25), (75,175), 10, 50) # Calculate PAC
    >>> print comod
    [[ 0.32708628  0.32188585]
     [ 0.3295994   0.32439953]]
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(p_range, a_range)
    if dp <= 0:
        raise ValueError('Width of lo frequqnecy range must be positive')
    if da <= 0:
        raise ValueError('Width of hi frequqnecy range must be positive')

    # method check
    method2fun = {'plv': plv, 'mi_tort': mi_tort, 'mi_canolty': mi_canolty,
                  'ozkurt': ozkurt, 'glm': glm}
    pac_fun = method2fun.get(pac_method, None)
    if pac_fun == None:
        raise ValueError('PAC method given is invalid.')

    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0], p_range[1], dp)
    f_amps = np.arange(a_range[0], a_range[1], da)
    P = len(f_phases)
    A = len(f_amps)

    # Calculate PAC for every combination of P and A
    comod = np.zeros((P, A))
    for p in range(P):
        f_lo = (f_phases[p], f_phases[p] + dp)

        for a in range(A):
            f_hi = (f_amps[a], f_amps[a] + da)

            comod[p, a] = pac_fun(lo, hi, f_lo, f_hi, fs=fs,
                                  filterfn=filterfn, filter_kwargs=filter_kwargs)

    return comod


def pa_series(lo, hi, f_lo, f_hi, fs=1000, filterfn=None, filter_kwargs=None):
    """
    Calculate the phase and amplitude time series

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    amp : array-like, 1d
        Time series of amplitude

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> print pha
    [ 1.57079633  1.60849544  1.64619455 ...,  1.45769899  1.4953981  1.53309721]
    """

    # Arg check
    _x_sanity(lo, hi)
    _range_sanity(f_lo, f_hi)

    # Filter setup
    if filterfn is None:
        filterfn = firf
        filter_kwargs = {}

    # Filter
    xlo = filterfn(lo, f_lo, fs, **filter_kwargs)
    xhi = filterfn(hi, f_hi, fs, **filter_kwargs)

    # Calculate phase time series and amplitude time series
    pha = np.angle(hilbert(xlo))
    amp = np.abs(hilbert(xhi))

    # Make arrays the same size
    pha, amp = _trim_edges(pha, amp)

    return pha, amp


def pa_dist(pha, amp, Nbins=10):
    """
    Calculate distribution of amplitude over a cycle of phases

    Parameters
    ----------
    pha : array
        Phase time series
    amp : array
        Amplitude time series
    Nbins : int
        Number of phase bins in the distribution,
        uniformly distributed between -pi and pi.

    Returns
    -------
    dist : array
        Average amplitude in each phase bins
    phase_bins : array
        The boundaries to each phase bin. Note the length is 1 + len(dist)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series, pa_dist
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> phase_bins, dist = pa_dist(pha, amp)
    >>> print dist
    [  7.21154110e-01   8.04347122e-01   4.49207087e-01   2.08747058e-02
       8.03854240e-05   3.45166617e-05   3.45607343e-05   3.51091029e-05
       7.73644631e-04   1.63514941e-01]
    """
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle must be an integer >1.')
    if len(pha) != len(amp):
        raise ValueError(
            'Phase and amplitude time series must be of same length.')

    phase_bins = np.linspace(-np.pi, np.pi, int(Nbins + 1))
    dist = np.zeros(int(Nbins))

    for b in range(int(Nbins)):
        t_phase = np.logical_and(pha >= phase_bins[b],
                                 pha < phase_bins[b + 1])
        dist[b] = np.mean(amp[t_phase])

    return phase_bins[:-1], dist
