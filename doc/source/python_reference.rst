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


Classes
=======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   BiHemiLabel
   Covariance
   Epochs
   fiff.Evoked
   fiff.Raw
   Label
   preprocessing.ICA
   SourceEstimate


Logging and Configuration
=========================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   set_log_level
   set_log_file

:py:mod:`mne.utils`:

.. automodule:: mne.utils
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   get_config_path
   get_config
   set_config


File I/O
========

.. currentmodule:: mne

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fiff.Evoked
   fiff.Raw

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   parse_config
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

:py:mod:`mne.datasets.sample`:

.. automodule:: mne.datasets.sample
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.sample

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
   find_ecg_events
   find_eog_events
   ica_find_ecg_events
   ica_find_eog_events
   read_ica

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
   make_fixed_length_events
   merge_events
   parse_config
   pick_events
   read_events
   write_events

:py:mod:`mne.epochs`:

.. automodule:: mne.epochs
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   combine_event_ids
   equalize_epoch_counts


Sensor Space Data
=================

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


Forward Modeling
================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_forward
   apply_forward_raw
   read_bem_surfaces
   read_forward_solution
   read_trans
   read_source_spaces
   read_surface
   write_bem_surface
   write_trans


Inverse Solutions
=================

:py:mod:`mne.minimum_norm`:

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
   compute_rank_inverse
   make_inverse_operator
   read_inverse_operator
   source_band_induced_power
   source_induced_power
   write_inverse_operator

:py:mod:`mne.mixed_norm`:

.. automodule:: mne.mixed_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.mixed_norm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mixed_norm
   tf_mixed_norm

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


Source Space Data
=================

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
   stc_to_label
   transform_coordinates
   vertex_to_mni
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
   iir_filter_raw
   induced_power
   morlet
   single_trial_power
   yule_walker
   ar_raw
   iir_filter_raw
   stft
   istft


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
   spatio_temporal_cluster_1samp_test

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
