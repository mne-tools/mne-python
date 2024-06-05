"""
.. _ex-morlet-fwhm:

Morlet wavelet example
----------------------
This shows a simple example of the relationship between ``n_cycles`` and
the FWHM using :func:`mne.time_frequency.fwhm`.
"""

import matplotlib.pyplot as plt
import numpy as np

from mne.time_frequency import fwhm, morlet

sfreq, freq, n_cycles = 1000.0, 10, 7  # i.e., 700 ms
this_fwhm = fwhm(freq, n_cycles)
wavelet = morlet(sfreq=sfreq, freqs=freq, n_cycles=n_cycles)
M, w = len(wavelet), n_cycles  # convert to SciPy convention
s = w * sfreq / (2 * freq * np.pi)  # from SciPy docs

_, ax = plt.subplots(layout="constrained")
colors = dict(real="#66CCEE", imag="#EE6677")
t = np.arange(-M // 2 + 1, M // 2 + 1) / sfreq
for kind in ("real", "imag"):
    ax.plot(
        t,
        getattr(wavelet, kind),
        label=kind,
        color=colors[kind],
    )
ax.plot(t, np.abs(wavelet), label="abs", color="k", lw=1.0, zorder=6)
half_max = np.max(np.abs(wavelet)) / 2.0
ax.plot(
    [-this_fwhm / 2.0, this_fwhm / 2.0],
    [half_max, half_max],
    color="k",
    linestyle="-",
    label="FWHM",
    zorder=6,
)
ax.legend(loc="upper right")
ax.set(xlabel="Time (s)", ylabel="Amplitude")
