:orphan:

.. _api_reference:

====================
Python API Reference
====================

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-Python, grouped thematically by analysis
stage. Functions and classes that are not
below a module heading are found in the :py:mod:`mne` namespace.

MNE-Python also provides multiple command-line scripts that can be called
directly from a terminal, see :ref:`python_commands`.

.. contents::
   :local:
   :depth: 2


Most-used classes
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   io.Raw
   Epochs
   Evoked
   Info

Reading raw data
================

:py:mod:`mne.io`:

.. currentmodule:: mne.io

.. automodule:: mne.io
   :no-members:
   :no-inherited-members:

.. autosummary::
  :toctree: generated/
  :template: function.rst

  anonymize_info
  read_events_eeglab
  read_raw_artemis123
  read_raw_bti
  read_raw_cnt
  read_raw_ctf
  read_raw_edf
  read_raw_kit
  read_raw_nicolet
  read_raw_eeglab
  read_raw_brainvision
  read_raw_egi
  read_raw_fif

Base class:

.. autosummary::
   :toctree: generated
   :template: class.rst

   BaseRaw

:py:mod:`mne.io.kit`:

.. currentmodule:: mne.io.kit

.. automodule:: mne.io.kit
   :no-members:
   :no-inherited-members:

.. autosummary::
  :toctree: generated/
  :template: function.rst

   read_mrk

File I/O
========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated
   :template: function.rst

   decimate_surface
   get_head_surf
   get_meg_helmet_surf
   get_volume_labels_from_aseg
   get_volume_labels_from_src
   parse_config
   read_labels_from_annot
   read_bem_solution
   read_bem_surfaces
   read_cov
   read_dipole
   read_epochs
   read_epochs_kit
   read_epochs_eeglab
   read_events
   read_evokeds
   read_forward_solution
   read_label
   read_morph_map
   read_proj
   read_reject_parameters
   read_selection
   read_source_estimate
   read_source_spaces
   read_surface
   read_trans
   read_tri
   save_stc_as_volume
   write_labels_to_annot
   write_bem_solution
   write_bem_surfaces
   write_cov
   write_events
   write_evokeds
   write_forward_solution
   write_label
   write_proj
   write_source_spaces
   write_surface
   write_trans
   io.read_info
   io.show_fiff

Base class:

.. autosummary::
   :toctree: generated
   :template: class.rst

   BaseEpochs

Creating data objects from arrays
=================================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EvokedArray
   EpochsArray
   io.RawArray

.. currentmodule:: mne

.. autosummary::
  :toctree: generated/
  :template: function.rst

  create_info


Datasets
========

.. currentmodule:: mne.datasets

:py:mod:`mne.datasets`:

.. automodule:: mne.datasets
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   brainstorm.bst_auditory.data_path
   brainstorm.bst_resting.data_path
   brainstorm.bst_raw.data_path
   eegbci.load_data
   megsim.data_path
   megsim.load_data
   mtrf.data_path
   fetch_hcp_mmp_parcellation
   multimodal.data_path
   sample.data_path
   somato.data_path
   spm_face.data_path
   visual_92_categories.data_path


Visualization
=============

.. currentmodule:: mne.viz

:py:mod:`mne.viz`:

.. automodule:: mne.viz
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ClickableImage

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_background_image
   compare_fiff
   circular_layout
   mne_analyze_colormap
   plot_bem
   plot_connectivity_circle
   plot_cov
   plot_dipole_amplitudes
   plot_dipole_locations
   plot_drop_log
   plot_epochs
   plot_events
   plot_evoked
   plot_evoked_image
   plot_evoked_topo
   plot_evoked_topomap
   plot_evoked_joint
   plot_evoked_field
   plot_evoked_white
   plot_filter
   plot_head_positions
   plot_ideal_filter
   plot_compare_evokeds
   plot_ica_sources
   plot_ica_components
   plot_ica_properties
   plot_ica_scores
   plot_ica_overlay
   plot_epochs_image
   plot_layout
   plot_montage
   plot_projs_topomap
   plot_raw
   plot_raw_psd
   plot_sensors
   plot_snr_estimate
   plot_source_estimates
   plot_sparse_source_estimates
   plot_tfr_topomap
   plot_topo_image_epochs
   plot_topomap
   plot_trans
   snapshot_brain_montage


Preprocessing
=============

Projections:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Projection

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_epochs
   compute_proj_evoked
   compute_proj_raw
   read_proj
   write_proj

:py:mod:`mne.channels`:

.. currentmodule:: mne.channels

.. automodule:: mne.channels
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Layout
   Montage
   DigMontage

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fix_mag_coil_types
   read_montage
   read_dig_montage
   read_layout
   find_layout
   make_eeg_layout
   make_grid_layout
   read_ch_connectivity
   equalize_channels
   rename_channels
   generate_2d_layout

:py:mod:`mne.preprocessing`:

.. currentmodule:: mne.preprocessing

.. automodule:: mne.preprocessing
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ICA
   Xdawn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_ecg
   compute_proj_eog
   create_ecg_epochs
   create_eog_epochs
   find_ecg_events
   find_eog_events
   fix_stim_artifact
   ica_find_ecg_events
   ica_find_eog_events
   infomax
   maxwell_filter
   read_ica
   run_ica
   corrmap

EEG referencing:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_reference_channels
   set_bipolar_reference
   set_eeg_reference

:py:mod:`mne.filter`:

.. currentmodule:: mne.filter

.. automodule:: mne.filter
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   construct_iir_filter
   create_filter
   estimate_ringing_samples
   filter_data
   notch_filter
   resample

:py:mod:`mne.chpi`

.. currentmodule:: mne.chpi

.. automodule:: mne.chpi
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   filter_chpi
   head_pos_to_trans_rot_t
   read_head_pos
   write_head_pos

:py:mod:`mne.transforms`

.. currentmodule:: mne.transforms

.. automodule:: mne.transforms
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Transform

.. autosummary::
   :toctree: generated/
   :template: function.rst

   quat_to_rot
   rot_to_quat

Events
======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Annotations
   AcqParserFIF

.. autosummary::
   :toctree: generated/
   :template: function.rst

   concatenate_events
   find_events
   find_stim_steps
   make_fixed_length_events
   merge_events
   parse_config
   pick_events
   read_events
   write_events
   concatenate_epochs

:py:mod:`mne.event`:

.. currentmodule:: mne.event

.. autosummary::
  :toctree: generated/
  :template: function.rst

   define_target_events

:py:mod:`mne.epochs`:

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_channels_epochs
   average_movements
   combine_event_ids
   equalize_epoch_counts


Sensor Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   combine_evoked
   concatenate_raws
   equalize_channels
   grand_average
   pick_channels
   pick_channels_cov
   pick_channels_forward
   pick_channels_regexp
   pick_types
   pick_types_forward
   pick_info
   read_epochs
   read_reject_parameters
   read_selection
   rename_channels


Covariance computation
======================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Covariance

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_covariance
   compute_raw_covariance
   cov.regularize
   make_ad_hoc_cov
   read_cov
   write_cov


MRI Processing
==============

.. currentmodule:: mne

Step by step instructions for using :func:`gui.coregistration`:

 - `Coregistration for subjects with structural MRI
   <http://www.slideshare.net/mne-python/mnepython-coregistration>`_
 - `Scaling a template MRI for subjects for which no MRI is available
   <http://www.slideshare.net/mne-python/mnepython-scale-mri>`_

.. autosummary::
   :toctree: generated/
   :template: function.rst

   gui.coregistration
   gui.fiducials
   create_default_subject
   scale_mri
   scale_bem
   scale_labels
   scale_source_space


Forward Modeling
================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Forward
   SourceSpaces

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_source_space_distances
   apply_forward
   apply_forward_raw
   average_forward_solutions
   convert_forward_solution
   forward.restrict_forward_to_label
   forward.restrict_forward_to_stc
   make_bem_model
   make_bem_solution
   make_forward_dipole
   make_forward_solution
   make_field_map
   make_sphere_model
   morph_source_spaces
   read_bem_surfaces
   read_forward_solution
   read_trans
   read_source_spaces
   read_surface
   sensitivity_map
   setup_source_space
   setup_volume_source_space
   surface.complete_surface_info
   write_bem_surfaces
   write_trans

:py:mod:`mne.bem`:

.. automodule:: mne.bem
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.bem

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ConductorModel

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fit_sphere_to_headshape
   get_fitting_dig
   make_watershed_bem
   make_flash_bem
   convert_flash_mris


Inverse Solutions
=================

:py:mod:`mne.minimum_norm`:

.. automodule:: mne.minimum_norm
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.minimum_norm

.. autosummary::
   :toctree: generated/
   :template: class.rst

   InverseOperator

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_inverse
   apply_inverse_epochs
   apply_inverse_raw
   compute_source_psd
   compute_source_psd_epochs
   compute_rank_inverse
   estimate_snr
   make_inverse_operator
   read_inverse_operator
   source_band_induced_power
   source_induced_power
   write_inverse_operator
   point_spread_function
   cross_talk_function

:py:mod:`mne.inverse_sparse`:

.. automodule:: mne.inverse_sparse
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.inverse_sparse

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mixed_norm
   tf_mixed_norm
   gamma_map

:py:mod:`mne.beamformer`:

.. automodule:: mne.beamformer
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.beamformer

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lcmv
   lcmv_epochs
   lcmv_raw
   dics
   dics_epochs
   dics_source_power
   rap_music
   tf_dics
   tf_lcmv

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Dipole
   DipoleFixed

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fit_dipole

:py:mod:`mne.dipole`:

.. automodule:: mne.dipole
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.dipole

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_phantom_dipoles


Source Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BiHemiLabel
   Label
   MixedSourceEstimate
   SourceEstimate
   VolSourceEstimate

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_morph_matrix
   extract_label_time_course
   grade_to_tris
   grade_to_vertices
   grow_labels
   label_sign_flip
   morph_data
   morph_data_precomputed
   read_labels_from_annot
   read_dipole
   read_label
   read_source_estimate
   save_stc_as_volume
   split_label
   stc_to_label
   transform_surface_to
   vertex_to_mni
   write_labels_to_annot
   write_label


Time-Frequency
==============

:py:mod:`mne.time_frequency`:

.. automodule:: mne.time_frequency
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency

.. autosummary::
   :toctree: generated/
   :template: class.rst

   AverageTFR
   EpochsTFR

Functions that operate on mne-python objects:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   csd_epochs
   psd_welch
   psd_multitaper
   fit_iir_model_raw
   tfr_morlet
   tfr_multitaper
   tfr_stockwell
   tfr_array_morlet
   tfr_array_multitaper
   tfr_array_stockwell
   read_tfrs
   write_tfrs

Functions that operate on ``np.ndarray`` objects:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   csd_array
   dpss_windows
   morlet
   stft
   istft
   stftfreq
   psd_array_multitaper
   psd_array_welch


:py:mod:`mne.time_frequency.tfr`:

.. automodule:: mne.time_frequency.tfr
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency.tfr

.. autosummary::
   :toctree: generated/
   :template: function.rst

   cwt
   morlet


Connectivity Estimation
=======================

:py:mod:`mne.connectivity`:

.. automodule:: mne.connectivity
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.connectivity

.. autosummary::
   :toctree: generated/
   :template: function.rst

   seed_target_indices
   spectral_connectivity
   phase_slope_index


Statistics
==========

:py:mod:`mne.stats`:

.. automodule:: mne.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.stats

.. autosummary::
   :toctree: generated/
   :template: function.rst

   bonferroni_correction
   fdr_correction
   permutation_cluster_test
   permutation_cluster_1samp_test
   permutation_t_test
   spatio_temporal_cluster_test
   spatio_temporal_cluster_1samp_test
   ttest_1samp_no_p
   linear_regression
   linear_regression_raw
   f_oneway
   f_mway_rm
   f_threshold_mway_rm
   summarize_clusters_stc

Functions to compute neighbor/adjacency matrices for cluster-level statistics:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   spatial_dist_connectivity
   spatial_src_connectivity
   spatial_tris_connectivity
   spatial_inter_hemi_connectivity
   spatio_temporal_src_connectivity
   spatio_temporal_tris_connectivity
   spatio_temporal_dist_connectivity


Simulation
==========

:py:mod:`mne.simulation`:

.. automodule:: mne.simulation
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.simulation

.. autosummary::
   :toctree: generated/
   :template: function.rst

   simulate_evoked
   simulate_raw
   simulate_stc
   simulate_sparse_stc
   select_source_in_label


.. _api_decoding:

Decoding
========

:py:mod:`mne.decoding`:

.. automodule:: mne.decoding
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   CSP
   EMS
   FilterEstimator
   LinearModel
   PSDEstimator
   Scaler
   TemporalFilter
   TimeFrequency
   UnsupervisedSpatialFilter
   Vectorizer
   ReceptiveField
   TimeDelayingRidge
   SlidingEstimator
   GeneralizingEstimator

Functions that assist with decoding and model fitting:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_ems
   cross_val_multiscore
   get_coef


Realtime
========

:py:mod:`mne.realtime`:

.. automodule:: mne.realtime
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RtEpochs
   RtClient
   MockRtClient
   FieldTripClient
   StimServer
   StimClient


MNE-Report
==========

:py:mod:`mne.report`:

.. currentmodule:: mne.report

.. automodule:: mne.report
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Report


Logging and Configuration
=========================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_config_path
   get_config
   open_docs
   set_log_level
   set_log_file
   set_config
   sys_info
   verbose

:py:mod:`mne.cuda`:

.. currentmodule:: mne.cuda

.. automodule:: mne.cuda
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   init_cuda
