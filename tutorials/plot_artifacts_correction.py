"""
====================
Artifacts Correction
====================

"""

###############################################################################
# Removing power-line noise
# -------------------------
#
# Power-line noise is a noise created by the electrical network.
# It is composed of sharp peaks at 50Hz (or 60Hz depending on your geographical location).
# Some peaks may also be present at the harmonic frequencies, i.e. the integer multiples of
# the power-line frequency, e.g. 100Hz, 150Hz, ... (or 120Hz, 180Hz, ...).
#
# Removing power-line noise can be done with a Notch filter, directly on the Raw object,
# specifying an array of frequency to be cut off::
#
#     >>> raw.notch_filter(np.arange(60, 241, 60), picks=picks)
#
# .. figure:: ../../../../_images/sphx_glr_plot_compute_raw_data_spectrum_002.png
#     :target: ../../auto_examples/time_frequency/plot_compute_raw_data_spectrum.html
#     :scale: 50%
#     :align: center
#
# .. topic:: Examples:
#
#     * :ref:`sphx_glr_auto_examples_time_frequency_plot_compute_raw_data_spectrum.py`
#
# .. topic:: See also:
#
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_rereference_eeg.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_maxwell_filter.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_shift_evoked.py`
