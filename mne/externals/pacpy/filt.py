from __future__ import division
import numpy as np

from scipy.signal import filtfilt
from scipy.signal import firwin2, firwin
from scipy.signal import morlet


def firf(x, f_range, fs=1000, w=3):
    """
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = np.float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    return _remove_edge(x_filt, Ntaps)


def firfls(x, f_range, fs=1000, w=3, tw=.15):
    """
    Filter signal with an FIR filter
    *Like firls in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter
    tw : float
        Transition width of the filter in normalized frequency space

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    if np.logical_or(tw < 0, tw > 1):
        raise ValueError('Transition width must be between 0 and 1.')

    nyq = fs / 2
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Characterize desired filter
    f = [0, (1 - tw) * f_range[0] / nyq, f_range[0] / nyq,
         f_range[1] / nyq, (1 + tw) * f_range[1] / nyq, 1]
    m = [0, 0, 1, 1, 0, 0]
    if any(np.diff(f) < 0):
        raise RuntimeError(
            'Invalid FIR filter parameters.'
            'Please decrease the transition width parameter.')

    # Perform filtering
    taps = firwin2(Ntaps, f, m)
    x_filt = filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    return _remove_edge(x_filt, Ntaps)


def morletf(x, f0, fs=1000, w=3, s=1, M=None, norm='sss'):
    """
    NOTE: This function is not currently ready to be interfaced with pacpy
    This is because the frequency input is not a range, which is a big
    assumption in how the api is currently designed

    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase

    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of 
        cycles of the oscillation with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2

    Returns
    -------
    x_trans : array
        Complex time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    if M == None:
        M = 2 * s * w * fs / f0

    morlet_f = morlet(M, w=w, s=s)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f)) * 2
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    x_filtR = np.convolve(x, np.real(morlet_f), mode='same')
    x_filtI = np.convolve(x, np.imag(morlet_f), mode='same')

    # Remove edge artifacts
    #x_filtR = _remove_edge(x_filtR, M/2.)
    #x_filtI = _remove_edge(x_filtI, M/2.)

    return x_filtR + 1j * x_filtI


def _remove_edge(x, N):
    """
    Calculate the number of points to remove for edge artifacts

    x : array
        time series to remove edge artifacts from
    N : int
        length of filter
    """
    N = int(N)
    return x[N:-N]
