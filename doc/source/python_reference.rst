=========
Reference
=========

This is the classes and functions reference of mne-python. Functions are 
grouped thematically. In addition, all File I/O functions are collected in 
a separate section.


.. automodule:: mne
   :no-members:
   :no-inherited-members:


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mne.BiHemiLabel
   mne.Covariance
   mne.Epochs
   mne.fiff.Evoked
   mne.fiff.Raw
   mne.Label
   mne.preprocessing.ICA
   mne.SourceEstimate


Connectivity Analysis
=====================

.. automodule:: mne.connectivity
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.connectivity

.. autosummary::
   :toctree: generated/
   :template: function.rst

   seed_target_indices
   spectral_connectivity


Data Simulation
===============

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

   
Events
======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   concatenate_events
   find_events
   make_fixed_length_events
   merge_events
   parse_config
   pick_events
   read_events
   write_events

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   combine_event_ids
   equalize_epoch_counts



File I/O
========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mne.fiff.Evoked
   mne.fiff.Raw


.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_bem_surfaces
   read_cov
   read_dip
   read_epochs
   read_events
   read_forward_solution
   read_label
   read_proj
   read_reject_parameters
   read_selection
   read_source_estimate
   read_source_spaces
   read_surface
   read_trans
   save_stc_as_volume
   write_bem_surface
   write_cov
   write_events
   write_label
   write_proj
   write_source_spaces
   write_surface
   write_trans


Forward Modeling
================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_forward
   apply_forward_raw
   compute_covariance
   compute_raw_data_covariance
   read_bem_surfaces
   read_cov
   read_forward_solution
   read_trans
   read_source_spaces
   read_surface
   write_bem_surface
   write_cov
   write_trans


Inverse Solutions
=================

.. automodule:: mne.minimum_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.minimum_norm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_inverse
   apply_inverse_epochs
   apply_inverse_raw
   make_inverse_operator
   read_inverse_operator
   source_band_induced_power
   source_induced_power
   write_inverse_operator

.. automodule:: mne.mixed_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.mixed_norm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mixed_norm
   tf_mixed_norm

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


Plotting
========

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
   plot_evoked
   plot_ica_panel
   plot_image_epochs
   plot_sparse_source_estimates
   plot_topo
   plot_topo_image_epochs
   plot_topo_phase_lock
   plot_topo_power


Preprocessing
=============

.. automodule:: mne.preprocessing
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.preprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_ecg
   compute_proj_eog
   find_ecg_events
   find_eog_events
   ica_find_ecg_events
   ica_find_eog_events
   read_ica

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

.. automodule:: mne.epochs
 :no-members:
 :no-inherited-members:


Projections
===========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_proj_epochs
   compute_proj_evoked
   compute_proj_raw
   read_proj
   transform_coordinates
   write_proj


Sensor Space
============

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fiff.pick_channels
   fiff.pick_channels_cov
   fiff.pick_channels_forward
   fiff.pick_channels_regexp
   fiff.pick_types
   fiff.pick_types_evoked
   fiff.pick_types_forward

   read_epochs
   read_reject_parameters
   read_selection


Source Space
============

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_morph_matrix
   extract_label_time_course
   grade_to_tris
   grade_to_vertices
   grow_labels
   labels_from_parc
   label_sign_flip
   morph_data
   morph_data_precomputed
   read_dip
   read_label
   read_source_estimate
   save_stc_as_volume
   spatial_dist_connectivity
   spatial_src_connectivity
   spatial_tris_connectivity
   spatio_temporal_src_connectivity
   spatio_temporal_tris_connectivity
   spatio_temporal_dist_connectivity
   stc_to_label
   vertex_to_mni
   write_label


Statistics
==========

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
   spatio_temporal_cluster_1samp_test


Time-Frequency Analysis
=======================

.. automodule:: mne.time_frequency
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.time_frequency

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ar_raw
   compute_raw_psd
   iir_filter_raw
   induced_power
   morlet
   single_trial_power
   yule_walker
   ar_raw
   iir_filter_raw
   stft
   istft


Verbosity
=========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   set_log_level
   set_log_file
   verbose
