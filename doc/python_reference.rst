:orphan:

.. _api_reference:

====================
Python API Reference
====================

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-Python, grouped thematically by analysis
stage. Functions and classes that are not
below a module heading are found in the ``mne`` namespace.

MNE-Python also provides multiple command-line scripts that can be called
directly from a terminal, see :ref:`python_commands`.

.. contents::
   :local:
   :depth: 2


:py:mod:`mne`:

.. automodule:: mne
   :no-members:
   :no-inherited-members:

Most-used classes
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

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

   anonymize_info
   read_raw_artemis123
   read_raw_bti
   read_raw_cnt
   read_raw_ctf
   read_raw_curry
   read_raw_edf
   read_raw_bdf
   read_raw_gdf
   read_raw_kit
   read_raw_nicolet
   read_raw_nirx
   read_raw_eeglab
   read_raw_brainvision
   read_raw_egi
   read_raw_fif
   read_raw_eximia
   read_raw_fieldtrip

Base class:

.. autosummary::
   :toctree: generated

   BaseRaw

:py:mod:`mne.io.kit`:

.. currentmodule:: mne.io.kit

.. automodule:: mne.io.kit
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   read_mrk

File I/O
========

.. currentmodule:: mne

.. autosummary::
   :toctree: generated

   channel_type
   channel_indices_by_type
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
   read_epochs_fieldtrip
   read_events
   read_evokeds
   read_evoked_fieldtrip
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
   what
   io.read_info
   io.show_fiff

Base class:

.. autosummary::
   :toctree: generated

   BaseEpochs

Creating data objects from arrays
=================================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   EvokedArray
   EpochsArray
   io.RawArray
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

   brainstorm.bst_auditory.data_path
   brainstorm.bst_resting.data_path
   brainstorm.bst_raw.data_path
   eegbci.load_data
   eegbci.standardize
   fetch_aparc_sub_parcellation
   fetch_fsaverage
   fetch_hcp_mmp_parcellation
   fnirs_motor.data_path
   hf_sef.data_path
   kiloword.data_path
   limo.load_data
   misc.data_path
   mtrf.data_path
   multimodal.data_path
   opm.data_path
   sleep_physionet.age.fetch_data
   sleep_physionet.temazepam.fetch_data
   sample.data_path
   somato.data_path
   spm_face.data_path
   visual_92_categories.data_path
   phantom_4dbti.data_path


Visualization
=============

.. currentmodule:: mne.viz

:py:mod:`mne.viz`:

.. automodule:: mne.viz
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   ClickableImage
   add_background_image
   compare_fiff
   circular_layout
   mne_analyze_colormap
   plot_bem
   plot_brain_colorbar
   plot_connectivity_circle
   plot_cov
   plot_csd
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
   plot_sensors_connectivity
   plot_snr_estimate
   plot_source_estimates
   link_brains
   plot_volume_source_estimates
   plot_vector_source_estimates
   plot_sparse_source_estimates
   plot_tfr_topomap
   plot_topo_image_epochs
   plot_topomap
   plot_alignment
   snapshot_brain_montage
   plot_arrowmap
   set_3d_backend
   get_3d_backend
   use_3d_backend
   set_3d_view
   set_3d_title
   create_3d_figure


Preprocessing
=============

Projections:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   Projection
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

   Layout
   DigMontage
   fix_mag_coil_types
   read_polhemus_fastscan
   get_builtin_montages
   make_dig_montage
   read_dig_polhemus_isotrak
   read_dig_captrack
   read_dig_dat
   read_dig_egi
   read_dig_fif
   read_dig_hpts
   make_standard_montage
   read_custom_montage
   compute_dev_head_t
   read_layout
   find_layout
   make_eeg_layout
   make_grid_layout
   find_ch_connectivity
   read_ch_connectivity
   equalize_channels
   rename_channels
   generate_2d_layout
   make_1020_channel_selections

:py:mod:`mne.preprocessing`:

.. currentmodule:: mne.preprocessing

.. automodule:: mne.preprocessing
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   ICA
   Xdawn
   annotate_movement
   compute_average_dev_head_t
   compute_current_source_density
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
   mark_flat
   maxwell_filter
   oversampled_temporal_projection
   peak_finder
   read_ica
   run_ica
   corrmap
   read_ica_eeglab

:py:mod:`mne.preprocessing.nirs`:

.. currentmodule:: mne.preprocessing.nirs

.. automodule:: mne.preprocessing.nirs
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   optical_density
   beer_lambert_law
   source_detector_distances
   short_channels
   scalp_coupling_index

EEG referencing:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

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

   compute_chpi_amplitudes
   compute_chpi_locs
   compute_head_pos
   extract_chpi_locs_ctf
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

   Transform
   quat_to_rot
   rot_to_quat
   read_ras_mni_t

Events
======

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   Annotations
   AcqParserFIF
   concatenate_events
   find_events
   find_stim_steps
   make_fixed_length_events
   make_fixed_length_epochs
   merge_events
   parse_config
   pick_events
   read_annotations
   read_events
   write_events
   concatenate_epochs
   events_from_annotations
   annotations_from_events

:py:mod:`mne.event`:

.. automodule:: mne.event
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.event

.. autosummary::
   :toctree: generated/

   define_target_events
   shift_time_events

:py:mod:`mne.epochs`:

.. automodule:: mne.epochs
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/

   add_channels_epochs
   average_movements
   combine_event_ids
   equalize_epoch_counts


Sensor Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

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

   Covariance
   compute_covariance
   compute_raw_covariance
   cov.compute_whitener
   cov.prepare_noise_cov
   cov.regularize
   compute_rank
   make_ad_hoc_cov
   read_cov
   write_cov


MRI Processing
==============

.. currentmodule:: mne

Step by step instructions for using :func:`gui.coregistration`:

 - `Coregistration for subjects with structural MRI
   <https://www.slideshare.net/mne-python/mnepython-coregistration>`_
 - `Scaling a template MRI for subjects for which no MRI is available
   <https://www.slideshare.net/mne-python/mnepython-scale-mri>`_

.. autosummary::
   :toctree: generated/

   coreg.get_mni_fiducials
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

   Forward
   SourceSpaces
   add_source_space_distances
   apply_forward
   apply_forward_raw
   average_forward_solutions
   convert_forward_solution
   decimate_surface
   dig_mri_distances
   forward.compute_depth_prior
   forward.compute_orient_prior
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
   use_coil_def
   write_bem_surfaces
   write_trans

:py:mod:`mne.bem`:

.. automodule:: mne.bem
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.bem

.. autosummary::
   :toctree: generated/

   ConductorModel
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

   InverseOperator
   apply_inverse
   apply_inverse_epochs
   apply_inverse_raw
   compute_source_psd
   compute_source_psd_epochs
   compute_rank_inverse
   estimate_snr
   make_inverse_operator
   prepare_inverse_operator
   read_inverse_operator
   source_band_induced_power
   source_induced_power
   write_inverse_operator
   make_resolution_matrix
   resolution_metrics
   get_cross_talk
   get_point_spread

:py:mod:`mne.inverse_sparse`:

.. automodule:: mne.inverse_sparse
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.inverse_sparse

.. autosummary::
   :toctree: generated/

   mixed_norm
   tf_mixed_norm
   gamma_map
   make_stc_from_dipoles

:py:mod:`mne.beamformer`:

.. automodule:: mne.beamformer
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.beamformer

.. autosummary::
   :toctree: generated/

   Beamformer
   read_beamformer
   make_lcmv
   apply_lcmv
   apply_lcmv_epochs
   apply_lcmv_raw
   apply_lcmv_cov
   make_dics
   apply_dics
   apply_dics_csd
   apply_dics_epochs
   rap_music
   tf_dics
   tf_lcmv

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   Dipole
   DipoleFixed
   fit_dipole

:py:mod:`mne.dipole`:

.. automodule:: mne.dipole
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.dipole

.. autosummary::
   :toctree: generated/

   get_phantom_dipoles


Source Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   BiHemiLabel
   Label
   MixedSourceEstimate
   SourceEstimate
   VectorSourceEstimate
   VolSourceEstimate
   VolVectorSourceEstimate
   SourceMorph
   compute_source_morph
   head_to_mni
   head_to_mri
   extract_label_time_course
   grade_to_tris
   grade_to_vertices
   label.select_sources
   grow_labels
   label_sign_flip
   labels_to_stc
   morph_labels
   random_parcellation
   read_labels_from_annot
   read_dipole
   read_label
   read_source_estimate
   read_source_morph
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

   AverageTFR
   EpochsTFR
   CrossSpectralDensity

Functions that operate on mne-python objects:

.. autosummary::
   :toctree: generated/

   csd_fourier
   csd_multitaper
   csd_morlet
   pick_channels_csd
   read_csd
   fit_iir_model_raw
   psd_welch
   psd_multitaper
   tfr_morlet
   tfr_multitaper
   tfr_stockwell
   read_tfrs
   write_tfrs

Functions that operate on ``np.ndarray`` objects:

.. autosummary::
   :toctree: generated/

   csd_array_fourier
   csd_array_multitaper
   csd_array_morlet
   dpss_windows
   morlet
   stft
   istft
   stftfreq
   psd_array_multitaper
   psd_array_welch
   tfr_array_morlet
   tfr_array_multitaper
   tfr_array_stockwell


:py:mod:`mne.time_frequency.tfr`:

.. automodule:: mne.time_frequency.tfr
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency.tfr

.. autosummary::
   :toctree: generated/

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

   degree
   envelope_correlation
   phase_slope_index
   seed_target_indices
   spectral_connectivity


.. _api_reference_statistics:

Statistics
==========

:py:mod:`mne.stats`:

.. automodule:: mne.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.stats

Parametric statistics (see :mod:`scipy.stats` and :mod:`statsmodels` for more
options):

.. autosummary::
   :toctree: generated/

   ttest_1samp_no_p
   f_oneway
   f_mway_rm
   f_threshold_mway_rm
   linear_regression
   linear_regression_raw

Mass-univariate multiple comparison correction:

.. autosummary::
   :toctree: generated/

   bonferroni_correction
   fdr_correction

Non-parametric (clustering) resampling methods:

.. autosummary::
   :toctree: generated/

   permutation_cluster_test
   permutation_cluster_1samp_test
   permutation_t_test
   spatio_temporal_cluster_test
   spatio_temporal_cluster_1samp_test
   summarize_clusters_stc
   bootstrap_confidence_interval

Compute ``connectivity`` matrices for cluster-level statistics:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   channels.find_ch_connectivity
   channels.read_ch_connectivity
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

   add_chpi
   add_ecg
   add_eog
   add_noise
   simulate_evoked
   simulate_raw
   simulate_stc
   simulate_sparse_stc
   select_source_in_label
   SourceSimulator

.. _api_decoding:

Decoding
========

:py:mod:`mne.decoding`:

.. automodule:: mne.decoding
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

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
   SPoC

Functions that assist with decoding and model fitting:

.. autosummary::
   :toctree: generated/

   compute_ems
   cross_val_multiscore
   get_coef


Realtime
========

Realtime functionality has moved to the standalone module :mod:`mne_realtime`.

MNE-Report
==========

:py:mod:`mne`:

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   Report
   open_report


Logging and Configuration
=========================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/

   get_config_path
   get_config
   open_docs
   set_log_level
   set_log_file
   set_config
   sys_info
   verbose

:py:mod:`mne.utils`:

.. currentmodule:: mne.utils

.. automodule:: mne.utils
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   deprecated
   warn

:py:mod:`mne.cuda`:

.. currentmodule:: mne.cuda

.. automodule:: mne.cuda
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   get_cuda_memory
   init_cuda
   set_cuda_device
