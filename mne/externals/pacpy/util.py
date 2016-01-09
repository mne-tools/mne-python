from __future__ import division
import numpy as np
import math


def fasthilbert(x, axis=-1):
    """
    Redefinition of scipy.signal.hilbert, which is very slow for some lengths
    of the signal x. This version zero-pads the signal to the next power of 2
    for speed.
    """
    x = np.array(x)
    N = x.shape[axis]
    N2 = 2**(int(math.log(len(x), 2)) + 1)
    Xf = np.fft.fft(x, N2, axis=axis)
    h = np.zeros(N2)
    h[0] = 1
    h[1:(N2 + 1) // 2] = 2

    x = np.fft.ifft(Xf * h, axis=axis)
    return x[:N]
