====================================
Spectral and Time-frequency Analysis
====================================


Source Space
^^^^^^^^^^^^

Currently, MNE-Python provides a set of functions
allowing to compute spectral analyses in the source space.
Many these functions return :func:`mne.SourceEstimate` objects or collections thereof.

.. note::
    The :func:`mne.SourceEstimate` object was initially designed for classical time-domain analyses.
    In this context, the time axis can actually refer to frequencies. This might be improved
    in the future.


The following functions are based on minimum norm estimates (MNE).

- :func:`mne.minimum_norm.compute_source_psd_epochs` returns single-trial power spectral density (PSD) esitmates using multi-tapers.
  Here, the time axis actually refers to frequencies, even if labeled as time.

- :func:`mne.minimum_norm.compute_source_psd` returns power spectral density (PSD) esitmates from continuous data usign FFT.
  Here, the time axis actually refers to frequencies, even if labeled as time.

- :func:`mne.minimum_norm.source_band_induced_power` returns a collection of time-domain :func:`mne.SourceEstimate` for each
  frequency band, based on Morlet-Wavelets.

- :func:`mne.minimum_norm.source_induced_power` returns power and inter-trial-coherence (ITC) as raw numpy arrays, based on Morlet-Wavelets.

Alternatively, the source power spectral density can also be estimated using the DICS beamformer,
see :func:`mne.beamformer.dics_source_power`.
 