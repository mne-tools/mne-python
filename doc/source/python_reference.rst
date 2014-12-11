=========
Reference
=========

.. automodule:: mne
   :no-members:
   :no-inherited-members:

This is the classes and functions reference of mne-python. Functions are
grouped thematically by analysis stage. In addition, all File I/O functions
are collected in a separate section. Functions and classes that are not below
a module heading are found in the :py:mod:`mne` namespace.


.. toctree::
   :maxdepth: 2

   python_reference


Classes
=======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   io.Raw
   io.RawFIFF
   Epochs
   Evoked
   SourceEstimate
   VolSourceEstimate
   MixedSourceEstimate
   Covariance
   Label
   BiHemiLabel
   preprocessing.ICA
   decoding.CSP
   decoding.Scaler
   decoding.ConcatenateChannels
   decoding.FilterEstimator
   decoding.PSDEstimator
   realtime.RtEpochs
   realtime.RtClient
   realtime.MockRtClient
   realtime.StimServer
   realtime.StimClient
   report.Report

Logging and Configuration
=========================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_config_path
   get_config
   set_log_level
   set_log_file
   set_config

:py:mod:`mne.cuda`:

.. automodule:: mne.cuda
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.cuda

.. autosummary::
   :toctree: generated/
   :template: function.rst

   init_cuda

Reading raw data
================

:py:mod:`mne.io`:

.. currentmodule:: mne.io

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Raw

Functions:

.. autosummary::
  :toctree: generated/
  :template: function.rst

  read_raw_bti
  read_raw_edf
  read_raw_kit
  read_raw_brainvision
  read_raw_egi

.. currentmodule:: mne.io.kit

:py:mod:`mne.io.kit`:

.. autosummary::
  :toctree: generated/
  :template: function.rst

   read_mrk

File I/O
========

.. currentmodule:: mne

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   decimate_surface
   get_head_surf
   get_meg_helmet_surf
   parse_config
   read_labels_from_annot
   read_bem_solution
   read_bem_surfaces
   read_cov
   read_dip
   read_epochs
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
   save_stc_as_volume
   write_labels_to_annot
   write_bem_surface
   write_cov
   write_events
   write_evokeds
   write_forward_solution
   write_label
   write_proj
   write_source_spaces
   write_surface
   write_trans


Creating data objects from arrays
=================================

Classes:

.. currentmodule:: mne

:py:mod:`mne`:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EvokedArray
   EpochsArray

.. currentmodule:: mne.io

:py:mod:`mne.io`:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RawArray

Functions:

.. currentmodule:: mne

:py:mod:`mne`:

.. autosummary::
  :toctree: generated/
  :template: function.rst

  create_info


Sample datasets
===============

:py:mod:`mne.datasets.sample`:

.. automodule:: mne.datasets.sample
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.sample

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data_path

:py:mod:`mne.datasets.spm_face`:

.. automodule:: mne.datasets.spm_face
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.spm_face

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data_path

:py:mod:`mne.datasets.megsim`:

.. automodule:: mne.datasets.megsim
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.megsim

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data_path
   load_data


Visualization
=============

:py:mod:`mne.viz`:

.. automodule:: mne.viz
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.viz

.. autosummary::
   :toctree: generated/
   :template: function.rst

   circular_layout
   mne_analyze_colormap
   plot_connectivity_circle
   plot_cov
   plot_drop_log
   plot_evoked
   plot_evoked_image
   plot_evoked_topomap
   plot_evoked_field
   plot_ica_sources
   plot_ica_components
   plot_ica_scores
   plot_ica_overlay
   plot_image_epochs
   plot_projs_topomap
   plot_raw
   plot_raw_psds
   plot_events
   plot_source_estimates
   plot_sparse_source_estimates
   plot_tfr_topomap
   plot_topo
   plot_topo_image_epochs
   plot_topomap
   compare_fiff

.. currentmodule:: mne.io

.. autosummary::
   :toctree: generated/
   :template: function.rst

   show_fiff

Preprocessing
=============

Projections:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_epochs
   compute_proj_evoked
   compute_proj_raw
   read_proj
   write_proj

Manipulate channels and set sensors locations for processing and plotting:

.. currentmodule:: mne.channels

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_montage
   apply_montage
   read_layout
   find_layout
   make_eeg_layout
   make_grid_layout
   read_ch_connectivity
   equalize_channels
   rename_channels

:py:mod:`mne.preprocessing`:

.. automodule:: mne.preprocessing
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.preprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_ecg
   compute_proj_eog
   create_ecg_epochs
   create_eog_epochs
   find_ecg_events
   find_eog_events
   find_outlier_adaptive
   ica_find_ecg_events
   ica_find_eog_events
   read_ica
   run_ica
   infomax

:py:mod:`mne.filter`:

.. automodule:: mne.filter
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.filter

.. autosummary::
   :toctree: generated/
   :template: function.rst

   band_pass_filter
   construct_iir_filter
   high_pass_filter
   low_pass_filter


Events
======

.. currentmodule:: mne

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

.. currentmodule:: mne.event

.. autosummary::
  :toctree: generated/
  :template: function.rst

   define_target_events

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   combine_event_ids
   equalize_epoch_counts
   add_channels_epochs

Sensor Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   concatenate_raws
   equalize_channels
   get_chpi_positions
   pick_channels
   pick_channels_cov
   pick_channels_forward
   pick_channels_regexp
   pick_types
   pick_types_evoked
   pick_types_forward
   read_ch_connectivity
   read_epochs
   read_reject_parameters
   read_selection
   rename_channels


Covariance
==========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_covariance
   compute_raw_data_covariance
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

:py:mod:`mne`:

.. currentmodule:: mne

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_source_space_distances
   apply_forward
   apply_forward_raw
   average_forward_solutions
   convert_forward_solution
   do_forward_solution
   make_forward_solution
   make_field_map
   read_bem_surfaces
   read_forward_solution
   read_trans
   read_source_spaces
   read_surface
   sensitivity_map
   setup_source_space
   setup_volume_source_space
   write_bem_surface
   write_trans

.. currentmodule:: mne.forward

.. autosummary::
   :toctree: generated/
   :template: function.rst

   restrict_forward_to_label
   restrict_forward_to_stc

:py:mod:`mne.source_space`:

.. automodule:: mne.source_space
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.source_space

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SourceSpaces

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_volume_labels_from_aseg

Inverse Solutions
=================

:py:mod:`mne.minimum_norm`:

.. automodule:: mne.minimum_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.minimum_norm

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   InverseOperator

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_inverse
   apply_inverse_epochs
   apply_inverse_raw
   compute_rank_inverse
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


Source Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   collect_transforms
   compute_morph_matrix
   extract_label_time_course
   grade_to_tris
   grade_to_vertices
   grow_labels
   label_sign_flip
   morph_data
   morph_data_precomputed
   read_annot
   read_dip
   read_label
   read_source_estimate
   save_stc_as_volume
   split_label
   stc_to_label
   transform_coordinates
   transform_surface_to
   vertex_to_mni
   write_annot
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
   :template: function.rst

   ar_raw
   compute_raw_psd
   compute_epochs_psd
   iir_filter_raw
   morlet
   tfr_morlet
   tfr_multitaper
   single_trial_power
   yule_walker
   ar_raw
   iir_filter_raw
   stft
   istft
   stftfreq
   tfr_stockwell
   write_tfrs
   read_tfrs


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

Functions to compute connectivity (adjacency) matrices for cluster-level statistics

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   spatial_dist_connectivity
   spatial_src_connectivity
   spatial_tris_connectivity
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

   generate_evoked
   generate_sparse_stc
   select_source_in_label

Decoding
========

:py:mod:`mne.decoding`:

.. automodule:: mne.decoding
   :no-members:
   :no-inherited-members:

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Scaler
   ConcatenateChannels
   PSDEstimator
   FilterEstimator
   CSP

Realtime
========

:py:mod:`mne.realtime`:

.. automodule:: mne.realtime
   :no-members:
   :no-inherited-members:

Classes:

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

.. automodule:: mne.report
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.report

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Report
