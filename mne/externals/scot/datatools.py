# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""
Summary
-------
Tools for basic data manipulation.
"""

import numpy as np


def cut_segments(rawdata, tr, start, stop):
    """ Cut continuous signal into segments.

    This function cuts segments from a continuous signal. Segments are stop - start samples long.

    Parameters
    ----------
    rawdata : array_like
        Input data of shape [`n`,`m`], with `n` samples and `m` signals.
    tr : list of int
        Trigger positions.
    start : int
        Window start (offset relative to trigger)
    stop : int
        Window end (offset relative to trigger)

    Returns
    -------
    x : ndarray
        Segments cut from `rawdata`. Individual segments are stacked along the third dimension.

    See also
    --------
    cat_trials : Concatenate segments

    Examples
    --------
    >>> data = np.random.randn(1000, 5)
    >>> tr = [250, 500, 750]
    >>> x = cut_segments(data, tr, 50, 100)
    >>> x.shape
    (50, 5, 3)
    """
    rawdata = np.atleast_2d(rawdata)
    tr = np.array(tr, dtype='int').ravel()
    win = range(start, stop)
    return np.dstack([rawdata[tr[t] + win, :] for t in range(len(tr))])


def cat_trials(x):
    """ Concatenate trials along time axis.

    Parameters
    ----------
    x : array_like
        Segmented input data of shape [`n`,`m`,`t`], with `n` time samples, `m` signals, and `t` trials.

    Returns
    -------
    out : ndarray
        Trials are concatenated along the first (time) axis. Shape of the output is [`n``t`,`m`].

    See also
    --------
    cut_segments : Cut segments from continuous data

    Examples
    --------
    >>> x = np.random.randn(150, 4, 6)
    >>> y = cat_trials(x)
    >>> y.shape
    (900, 4)
    """
    x = np.atleast_3d(x)
    t = x.shape[2]
    return np.squeeze(np.vstack(np.dsplit(x, t)), axis=2)


def dot_special(x, a):
    """ Trial-wise dot product.

    This function calculates the dot product of `x[:,:,i]` with `a` for each `i`.

    Parameters
    ----------
    x : array_like
        Segmented input data of shape [`n`,`m`,`t`], with `n` time samples, `m` signals, and `t` trials.
        The dot product is calculated for each trial.
    a : array_like
        Second argument

    Returns
    -------
    out : ndarray
        Returns the dot product of each trial.

    Examples
    --------
    >>> x = np.random.randn(150, 40, 6)
    >>> a = np.ones((40, 7))
    >>> y = dot_special(x, a)
    >>> y.shape
    (150, 7, 6)
    """
    x = np.atleast_3d(x)
    a = np.atleast_2d(a)
    return np.dstack([x[:, :, i].dot(a) for i in range(x.shape[2])])


def randomize_phase(data):
    """ Phase randomization.

    This function randomizes the input array's spectral phase along the first dimension.

    Parameters
    ----------
    data : array_like
        Input array

    Returns
    -------
    out : ndarray
        Array of same shape as `data`.

    Notes
    -----
    The algorithm randomizes the phase component of the input's complex fourier transform.

    Examples
    --------
    .. plot::
        :include-source:

        from pylab import *
        from scot.datatools import randomize_phase
        np.random.seed(1234)
        s = np.sin(np.linspace(0,10*np.pi,1000)).T
        x = np.vstack([s, np.sign(s)]).T
        y = randomize_phase(x)
        subplot(2,1,1)
        title('Phase randomization of sine wave and rectangular function')
        plot(x), axis([0,1000,-3,3])
        subplot(2,1,2)
        plot(y), axis([0,1000,-3,3])
        plt.show()
    """
    data = np.asarray(data)
    data_freq = np.fft.rfft(data, axis=0)
    data_freq = np.abs(data_freq) * np.exp(1j*np.random.random_sample(data_freq.shape)*2*np.pi)
    return np.fft.irfft(data_freq, data.shape[0], axis=0)