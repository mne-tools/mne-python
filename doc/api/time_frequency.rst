
Time-Frequency
==============

:py:mod:`mne.time_frequency`:

.. automodule:: mne.time_frequency
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency

.. autosummary::
   :toctree: ../generated/

   AverageTFR
   AverageTFRArray
   BaseTFR
   EpochsTFR
   EpochsTFRArray
   RawTFR
   RawTFRArray
   CrossSpectralDensity
   Spectrum
   SpectrumArray
   EpochsSpectrum
   EpochsSpectrumArray

Functions that operate on mne-python objects:

.. autosummary::
   :toctree: ../generated/

   combine_spectrum
   combine_tfr
   csd_tfr
   csd_fourier
   csd_multitaper
   csd_morlet
   pick_channels_csd
   read_csd
   fit_iir_model_raw
   tfr_morlet
   tfr_multitaper
   tfr_stockwell
   read_tfrs
   write_tfrs
   read_spectrum

Functions that operate on ``np.ndarray`` objects:

.. autosummary::
   :toctree: ../generated/

   csd_array_fourier
   csd_array_multitaper
   csd_array_morlet
   dpss_windows
   fwhm
   morlet
   stft
   istft
   stftfreq
   psd_array_multitaper
   psd_array_welch
   tfr_array_morlet
   tfr_array_multitaper
   tfr_array_stockwell


:py:mod:`mne.time_frequency.tfr`:

.. automodule:: mne.time_frequency.tfr
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency.tfr

.. autosummary::
   :toctree: ../generated/

   cwt
   morlet
