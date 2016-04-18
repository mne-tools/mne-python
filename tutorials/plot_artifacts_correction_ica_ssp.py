"""

.. _tut_artifacts_correct_ica_ssp:

Artifact Correction with ICA and SSP
====================================

ICA finds directions in the feature space
corresponding to projections with high non-Gaussianity. We thus obtain
a decomposition into independent components, and the artifact’s contribution is
localized in only a small number of components.
These components have to be correctly identified and removed.

If EOG or ECG recordings are available, they can be used in ICA to automatically
select the corresponding artifact components from the decomposition. To do so,
you have to first build an Epoch object around blink or heartbeat event::

"""

# Create an Epoch object from heartbeat events
# ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)

##############################################################################
# Independent Component Analysis (ICA)
# ------------------------------------

# Then you need to create an ICA estimator and fit it on the data::
#
#     >>> # Initialize the ICA estimator and fit it on the data
#     >>> ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
#     >>> ica.fit(raw, picks=picks)
#
# Then you have to call the :func:`ICA.find_bads_ecg` function. It will
# compute a score that quantify how much the Epoch is correlated to each independent
# component, and select the components with the best scores. Then you just have to
# exclude the components with func:`ICA.exclude`::
#
#     >>> # Find components highy correlated with heartbeat events
#     >>> ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
#     >>> # Here we remove only 3 components
#     >>> ica.exclude += ecg_inds[:3]
#
# .. figure:: ../../../../_images/sphx_glr_plot_ica_from_raw_005.png
#     :target: ../../auto_tutorials/plot_ica_from_raw.html
#     :scale: 50%
#
# .. topic:: Examples:
#
#     * :ref:`sphx_glr_auto_tutorials_plot_ica_from_raw.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_find_ecg_artifacts.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_find_eog_artifacts.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_eog_artifact_histogram.py`
#
#
# If EOG or ECG recordings are not available, you can visually select the artifact
# components. This can be done on the independent components time series or on the independent
# components spatial distribution, looking for
# the characteristic shapes of blinks and heartbeats artifacts::
#
#     >>> #
#     >>>
#
# You can also use this manual identification on one subject in order to automatically
# find corresponding components in other subject, using
# :func:`CORRMAP <mne.preprocessing.ica.corrmap>`
# (See :ref:`sphx_glr_auto_examples_preprocessing_plot_corrmap_detection.py` for
# a detailed example of using CORRMAP).
#
# .. topic:: Examples:
#
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_corrmap_detection.py`
#     * :ref:`sphx_glr_auto_examples_preprocessing_plot_run_ica.py`
#
# ICA-based artifact rejection is done using the :class:`mne.preprocessing.ICA`
# class, see the :ref:`ica` section in the manual for more
# information on ICA's concepts.

##############################################################################
# Signal-Space Projection (SSP)
# -----------------------------
#
# Instead of using ICA, you can also use Signal-Space Projection (SSP) to extract artifacts.
# SSP-based rejection is done using the
# :func:`compute_proj_ecg <mne.preprocessing.compute_proj_ecg>` and
# :func:`compute_proj_eog <mne.preprocessing.compute_proj_eog>` methods,
# see :ref:`ssp` section in the manual for more information.
# The commands look like::

#     >>> ecg_proj, ecg_event = mne.preprocessing.compute_proj_ecg(raw)
