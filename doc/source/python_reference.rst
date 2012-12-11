=========
Reference
=========

This is the classes and functions reference of mne-python.

.. automodule:: mne
   :no-members:
   :no-inherited-members:

Classes reference
=================

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

Functions reference
===================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_cov
   write_cov
   compute_covariance
   compute_raw_data_covariance
   read_events
   write_events
   find_events
   merge_events
   pick_events
   make_fixed_length_events
   concatenate_events
   read_forward_solution
   apply_forward
   apply_forward_raw
   label_time_courses
   read_label
   label_sign_flip
   write_label
   stc_to_label
   grow_labels
   read_bem_surfaces
   read_surface
   write_bem_surface
   read_source_spaces
   vertex_to_mni
   read_epochs
   equalize_epoch_counts
   read_stc
   write_stc
   read_w
   write_w
   read_source_estimate
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
   save_stc_as_volume
   parse_config
   read_reject_parameters
   transform_coordinates
   read_trans
   write_trans
   read_proj
   write_proj
   compute_proj_epochs
   compute_proj_evoked
   compute_proj_raw
   read_selection
   read_dip
   set_log_level
   set_log_file
   verbose
   fiff.pick_types
   fiff.pick_channels
   fiff.pick_types_evoked
   fiff.pick_channels_regexp
   fiff.pick_channels_forward
   fiff.pick_types_forward
   fiff.pick_channels_cov

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
   mne_analyze_colormap

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

.. automodule:: mne.connectivity
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.connectivity

.. autosummary::
   :toctree: generated/
   :template: function.rst

   spectral_connectivity
   seed_target_indices

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
