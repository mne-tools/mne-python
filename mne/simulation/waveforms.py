# author: ngayraud
#
# Created on Wed Feb 21 11:13:50 2018.

import numpy as np


def get_waveform(waveform):
    """Check waveform exists and return the callable.

    Returns a simple sinusoide if not found.

    Parameters
    ----------
    waveform : str | callable
        If str, mus be one of the known waveforms: 'sin', 'p300_target,
        'p300_nontarget'.
    """
    known_waveforms = {
        'sin': waveform_sin,
        'p300_target': waveform_p300_target,
        'p300_nontarget': waveform_p300_nontarget,
    }
    if isinstance(waveform, str) and waveform in known_waveforms.keys():
        return known_waveforms[waveform]
    elif hasattr(waveform, "__call__"):
        return waveform
    else:
        raise TypeError('Unrecognised type. Accepted inputs: str, callable. '
                        'List of accepted str: sin, p300_target, '
                        'p300_nontarget.')


def waveform_sin(times, amplitude=1e-8, freq=10., phase=0.):
    """Generate a sinusoide waveform.    .

    Returns a simple sinusoide

    Parameters
    ----------
    times : array
        Array of times
    amplitude : float
        amplitude of the sinusoide
    freq : float
        ordinary frequency of the sinusoide
    phase : float
        phase of the sinusoide
    """
    return amplitude * np.sin(2. * np.pi * freq * times + phase)


def waveform_p300_target(times, peak=0.3, amplitude=15.0):
    """Generate a p300 target waveform.

    Create a p300 target waveform.

    Parameters
    ----------
    times : array
        Array of times
    peak : float
        peak of the p300
    amplitude : array
        amplitude of the p300
    """
    return (1e-9 * amplitude * np.cos(14. * (times - peak)) *
            np.exp(-(times - peak + 0.04)**2 / 0.02) +
            1e-9 * amplitude / 6.0 * np.sin(22. * (times - peak)) *
            np.exp(-(times - peak + 0.24)**2 / 0.02))


def waveform_p300_nontarget(times, peak=0.3, amplitude=15.0):
    """Generate a p300 nontarget waveform.

    Create a p300 nontarget waveform.

    Parameters
    ----------
    times : array
        Array of times
    peak : float
        peak of the p300
    amplitude : array
        amplitude of the p300
    """
    return (1e-9 * (amplitude / 4.0) * np.cos(14. * (times - peak)) *
            np.exp(-(times - peak + 0.04)**2 / 0.05) +
            1e-9 * (amplitude / 6.0) * np.sin(22. * (times - peak)) *
            np.exp(-(times - peak + 0.24)**2 / 0.02))
