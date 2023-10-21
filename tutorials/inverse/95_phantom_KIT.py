"""
.. _tut-phantom-kit:

============================
KIT phantom dataset tutorial
============================

Here we read KIT data TODO EXPLAIN.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%
import mne

data_path = mne.datasets.phantom_kit.data_path()

raw = mne.io.read_raw_kit(data_path / "002_phantom_11Hz_100uA.con")
raw.crop(300).load_data()  # cut from ~800 to 300s for speed
raw.plot(duration=30, n_channels=50, scalings=dict(mag=5e-12))

# %%
# We can also look at the power spectral density to see the phantom oscillations.

spectrum = raw.copy().crop(0, 60).compute_psd(n_fft=10000)
fig = spectrum.plot()
fig.axes[0].set_xlim(0, 50)
