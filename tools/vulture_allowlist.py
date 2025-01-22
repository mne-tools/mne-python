"""Vulture allowlist.

Python names that we want Vulture to ignore need to be added to this file, see:

https://github.com/jendrikseipp/vulture/blob/main/README.md#whitelists
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

numba_conditional
options_3d
invisible_fig
brain_gc
windows_like_datetime
garbage_collect
renderer_notebook
qt_windows_closed
download_is_error
exitstatus
startdir
pg_backend
recwarn
pytestmark
nbexec
disabled_event_channels
ch_subset_adjacency
few_surfaces
want_orig_dist
eeglab_montage
invisible_fig
captions_new
comments_new
items_new
f4
set_channel_types_eyetrack
_use_test_3d_backend
verbose_debug
metadata_routing

# Decoding
_._more_tags
_.multi_class
_.preserves_dtype
deep

# Backward compat or rarely used
RawFIF
estimate_head_mri_t
plot_epochs_psd_topomap
plot_epochs_psd
plot_psd_topomap
plot_raw_psd_topo
plot_psd_topo
read_ctf_comp
read_bad_channels
set_cache_dir
spatial_dist_adjacency
set_cuda_device
eegbci.standardize
_.plot_topo_image
_._get_tags
_.mahalanobis
exc_type
exc_value
_.name_html

# Unused but for compat or CIs
fit_params_  # search_light
format_epilog  # false alarm for opt parser
_._fit_transform  # in getattr
_.plot_3d  # not tested for all classes
_.error_norm  # cov
_download_all_example_data  # CIs
_cleanup_agg
_notebook_vtk_works
_.drop_inds_

# mne/io/ant/tests/test_ant.py
andy_101
na_271

# mne/io/snirf/tests/test_snirf.py
_.dataTimeSeries
_.sourceIndex
_.detectorIndex
_.wavelengthIndex
_.dataType
_.dataTypeIndex
_.dataTypeLabel
_.dataTypeLabel
_.SubjectID
_.MeasurementDate
_.MeasurementTime
_.LengthUnit
_.TimeUnit
_.FrequencyUnit
_.wavelengths
_.sourcePos3D
_.detectorPos3D

# numerics.py
_.noise_variance_
_.n_features_

# Brain, Coreg, PyVista
_._Iren
_.active_scalars_name
_.active_vectors_name
_._plotter
_.set_fmax
_.set_fmid
_.set_fmin
_.EnterEvent
_.MouseMoveEvent
_.LeaveEvent
_.SetEventInformation
_.CharEvent
_.KeyPressEvent
_.KeyReleaseEvent
_PyVistaRenderer
_TimeInteraction
set_3d_options
_._has_lpa_data
_._has_nasion_data
_._has_rpa_data
_._nearest_transformed_high_res_mri_idx_rpa
_._nearest_transformed_high_res_mri_idx_nasion
_._nearest_transformed_high_res_mri_idx_lpa

# Figures (prevent GC for example)
_.decim_data
_.button_help
_.button_proj
_.mne_animation
_.RS
_.showNormal
_.showFullScreen
_.isFullScreen
_._span_selector
ypress
scroll
keypress
azim
_loc
eventson
_.argtypes
_.restype
_.labelpad
_.fake_keypress

# Used in ignored files
_qt_raise_window
_qt_disable_paint
_qt_get_stylesheet
