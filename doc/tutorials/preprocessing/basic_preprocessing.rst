Basic preprocessing
===================
Down-sampling
-------------
When performing experiments where timing is critical, a signal with a high
sampling rate is desired. However, having a signal with a much higher sampling
rate than necessary needlessly consumes memory and slows down computations
operating on the data. To avoid that, you can down-sample your time series.

Since down-sampling reduces the timing precision of events, we recommend first
extracting epochs and down-sampling the Epochs object::

    >>> # Down-sampling to a new sampling frequency of 100 Hz
    >>> epochs_downsampled = epochs.resample(100, copy=True)

.. figure:: ../_images/sphx_glr_plot_resample_001.png
  :scale: 50%
  :align: center

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_time_frequency_plot_compute_raw_data_spectrum.py`

Bandpass filtering
------------------
Bandpass filtering is a classical step in preprocessing; it filters the data
with a bandpass filter to select the frequency range of interest::

    >>> # bandpass filtering to the range [10, 50] Hz
    >>> raw.filter(10, 50)


Removing power-line noise
-------------------------
Power-line noise is a noise created by the electrical network.
It is composed of sharp peaks at 50Hz (or 60Hz depending on your geographical location).
Some peaks may also be present at the harmonic frequencies, i.e. the integer multiples of
the power-line frequency, e.g. 100Hz, 150Hz, ... (or 120Hz, 180Hz, ...).

Removing power-line noise can be done with a Notch filter, directly on the Raw object,
specifying an array of frequency to be cut off::

    >>> raw.notch_filter(np.arange(60, 241, 60), picks=picks)

.. figure:: ../../_images/sphx_glr_plot_compute_raw_data_spectrum_002.png
   :target: ../../auto_examples/time_frequency/plot_compute_raw_data_spectrum.html
   :scale: 50%
   :align: center

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_time_frequency_plot_compute_raw_data_spectrum.py`

.. topic:: See also:

    * :ref:`sphx_glr_auto_examples_preprocessing_plot_rereference_eeg.py`
    * :ref:`sphx_glr_auto_examples_preprocessing_plot_maxwell_filter.py`
    * :ref:`sphx_glr_auto_examples_preprocessing_plot_shift_evoked.py`
