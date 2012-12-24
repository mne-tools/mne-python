=========
Reference
=========

This is the classes and functions reference of mne-python.

.. automodule:: mne
   :no-members:
   :no-inherited-members:


Classes
=======

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mne.fiff.Raw
   mne.fiff.Evoked
   mne.Epochs
   mne.Covariance
   mne.SourceEstimate
   mne.Label
   mne.BiHemiLabel
   mne.preprocessing.ICA


Connectivity Analysis
=====================

.. automodule:: mne.connectivity
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.connectivity

.. autosummary::
   :toctree: generated/
   :template: function.rst

   spectral_connectivity
   seed_target_indices


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
   select_source_in_label
   generate_sparse_stc

   
Events
======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   find_events
   merge_events
   pick_events
   make_fixed_length_events
   concatenate_events
   parse_config

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

   mne.fiff.Raw
   mne.fiff.Evoked
   mne.Covariance


.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_cov
   write_cov
   read_events
   write_events
   read_forward_solution
   read_label
   write_label
   read_bem_surfaces
   write_bem_surface
   read_surface
   read_source_spaces
   read_epochs
   read_reject_parameters
   read_source_estimate
   save_stc_as_volume
   read_trans
   write_trans
   read_proj
   write_proj
   read_selection
   read_dip


Forward Modeling
================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_covariance
   compute_raw_data_covariance
   apply_forward
   apply_forward_raw


Inverse Solutions
=================

.. automodule:: mne.minimum_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.minimum_norm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_inverse_operator
   write_inverse_operator
   make_inverse_operator
   apply_inverse
   apply_inverse_raw
   apply_inverse_epochs
   source_band_induced_power
   source_induced_power

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

   plot_evoked
   plot_cov
   plot_topo
   plot_sparse_source_estimates
   plot_ica_panel
   plot_topo_power
   plot_topo_phase_lock
   plot_image_epochs
   plot_topo_image_epochs
   circular_layout
   plot_connectivity_circle
   mne_analyze_colormap


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
   find_eog_events
   find_ecg_events
   read_ica
   ica_find_eog_events
   ica_find_ecg_events

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

   transform_coordinates
   compute_proj_epochs
   compute_proj_evoked
   compute_proj_raw


Sensor Space
============

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   fiff.pick_types
   fiff.pick_channels
   fiff.pick_types_evoked
   fiff.pick_channels_regexp
   fiff.pick_channels_forward
   fiff.pick_types_forward
   fiff.pick_channels_cov


Source Space
============

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   extract_label_time_course
   labels_from_parc
   label_sign_flip
   stc_to_label
   grow_labels

   vertex_to_mni

   morph_data
   morph_data_precomputed
   compute_morph_matrix

   grade_to_tris
   grade_to_vertices
   spatial_src_connectivity
   spatial_tris_connectivity
   spatial_dist_connectivity
   spatio_temporal_src_connectivity
   spatio_temporal_tris_connectivity
   spatio_temporal_dist_connectivity
   

Statistics
==========

.. automodule:: mne.stats
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.stats

.. autosummary::
   :toctree: generated/
   :template: function.rst

   permutation_t_test
   permutation_cluster_test
   permutation_cluster_1samp_test
   spatio_temporal_cluster_1samp_test
   fdr_correction
   bonferroni_correction


Time-Frequency Analysis
=======================

.. automodule:: mne.time_frequency
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.time_frequency

.. autosummary::
   :toctree: generated/
   :template: function.rst

   induced_power
   single_trial_power
   compute_raw_psd
   morlet
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
