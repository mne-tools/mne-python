"""
Filtering and Resampling
========================

"""

# Down-sampling
# -------------
# When performing experiments where timing is critical, a signal with a high
# sampling rate is desired. However, having a signal with a much higher sampling
# rate than necessary needlessly consumes memory and slows down computations
# operating on the data. To avoid that, you can down-sample your time series.
#
# Since down-sampling reduces the timing precision of events, we recommend first
# extracting epochs and down-sampling the Epochs object::
#
#     >>> # Down-sampling to a new sampling frequency of 100 Hz
#     >>> epochs_downsampled = epochs.resample(100, copy=True)
#
# .. figure:: ../../../../_images/sphx_glr_plot_resample_001.png
#     :target: ../../auto_examples/time_frequency/plot_compute_raw_data_spectrum.html
#     :scale: 50%
#     :align: center
#
# .. topic:: Examples:
#
#     * :ref:`sphx_glr_auto_examples_time_frequency_plot_compute_raw_data_spectrum.py`
#
# Bandpass filtering
# ------------------
# Bandpass filtering is a classical step in preprocessing; it filters the data
# with a bandpass filter to select the frequency range of interest::
#
#     >>> # bandpass filtering to the range [10, 50] Hz
#     >>> raw.filter(10, 50)
