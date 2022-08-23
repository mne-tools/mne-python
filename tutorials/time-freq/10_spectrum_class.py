# -*- coding: utf-8 -*-
"""
.. _tut-spectrum-class:

==============================================================
The Spectrum and EpochsSpectrum classes: frequency-domain data
==============================================================

This tutorial shows how to create and visualize frequency-domain
representations of your data, starting from continuous :class:`~mne.io.Raw`,
discontinuous :class:`~mne.Epochs`, or averaged :class:`~mne.Evoked` data.

As usual we'll start by importing the modules we need, and loading our
`sample`_ dataset:
"""

# %%

import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' / 'sample' /
                        'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=60)

# %%
# All three sensor-space containers (:class:`~mne.io.Raw`,
# :class:`~mne.Epochs`, and :class:`~mne.Evoked`) have a
# :meth:`~mne.io.Raw.compute_psd` method with the same options.

raw.compute_psd()

# %%
# By default, the spectral estimation method will be the
# :footcite:p:`Welch1967` method for continuous data, and the multitaper
# method :footcite:`Slepian1978` for epoched or averaged data. This default can
# be overridden by passing ``method='welch'`` or ``method='multitaper'`` to the
# :meth:`~mne.io.Raw.compute_psd` method.
#
# There are many other options available as well; for example we can compute a
# spectrum from a given span of times, for a chosen frequency range, and for a
# subset of the available channels:

raw.compute_psd(method='multitaper', tmin=10, tmax=20, fmin=5, fmax=30,
                picks='eeg')

# %%
# You can also pass some parameters to the underlying spectral estimation
# function, such as the FFT window length and overlap for the Welch method; see
# the docstrings of :class:`mne.time_frequency.Spectrum` (esp. its
# ``method_kw`` parameter) and the spectral estimation functions
# :func:`~mne.time_frequency.psd_array_welch` and
# :func:`~mne.time_frequency.psd_array_multitaper` for details.
#
# For epoched data, the class of the spectral estimate will be
# :class:`mne.time_frequency.EpochsSpectrum` instead of
# :class:`mne.time_frequency.Spectrum`, but most of the API is the same for the
# two classes. For example, both have a
# :meth:`~mne.time_frequency.EpochsSpectrum.get_data` method with an option to
# return the bin frequencies:

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
epo_spectrum = epochs.compute_psd()
psds, freqs = epo_spectrum.get_data(return_freqs=True)
print(psds.shape)
print(freqs.shape)

# %%
# Additionally, both :class:`~mne.time_frequency.Spectrum` and
# :class:`~mne.time_frequency.EpochsSpectrum` have ``__getitem__`` methods,
# meaning their data can be accessed by square-bracket indexing. For
# :class:`mne.time_frequency.Spectrum` objects (computed from
# :class:`~mne.io.Raw` or :class:`~mne.Evoked` data), the indexing works
# similar to a :class:`NumPy array<numpy.ndarray>`:

evoked = epochs['auditory'].average()
evk_spectrum = evoked.compute_psd()
evk_spectrum[:4, :3]  # the first 3 frequency bins for the first 4 channels

# %%
# .. hint::
#    If the original :class:`~mne.Epochs` object had a metadata dataframe
#    attached, the derived :class:`~mne.time_frequency.EpochsSpectrum` will
#    inherit that metadata and will hence also support subselecting epochs via
#    :ref:`Pandas query strings <pandas:indexing.query>`.
#
# In contrast, the :class:`~mne.time_frequency.EpochsSpectrum` has indexing
# similar to :class:`~mne.Epochs` objects: you can use string values to select
# spectral estimates for specific epochs based on their condition names;
# selection via :term:`hierarchical event descriptors` (HEDs) is also possible:

epo_spectrum['visual']  # gets both "visual/left" and "visual/right" epochs
