.. _api_reference:

=============
API Reference
=============

This is the classes and functions reference of mne-python. Functions are
grouped thematically by analysis stage. Functions and classes that are not
below a module heading are found in the :py:mod:`mne` namespace.


.. contents::
   :local:
   :depth: 2


Classes
=======

.. currentmodule:: mne
.. autosummary::
   :toctree: generated/
   :template: class.rst

   io.fiff.Raw
   io.fiff.raw.RawFIF
   epochs.Epochs
   evoked.Evoked
   source_space.SourceSpaces
   source_estimate.SourceEstimate
   source_estimate.VolSourceEstimate
   source_estimate.MixedSourceEstimate
   cov.Covariance
   dipole.Dipole
   label.Label
   label.BiHemiLabel
   transforms.Transform
   preprocessing.ica.ICA
   decoding.csp.CSP
   decoding.transformer.Scaler
   decoding.transformer.FilterEstimator
   decoding.transformer.PSDEstimator
   decoding.time_gen.GeneralizationAcrossTime
   decoding.time_gen.TimeDecoding
   realtime.epochs.RtEpochs
   realtime.client.RtClient
   realtime.mockclient.MockRtClient
   realtime.stim_server_client.StimServer
   realtime.stim_server_client.StimClient
   report.Report

Logging and Configuration
=========================

.. currentmodule:: mne.utils

.. automodule:: mne.utils
   :no-members:
   :no-inherited-members:

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

   fiff.Raw

Functions:

.. automodule:: mne.io
   :no-members:
   :no-inherited-members:

.. autosummary::
  :toctree: generated/
  :template: function.rst

  bti.bti.read_raw_bti
  edf.edf.read_raw_edf
  kit.kit.read_raw_kit
  brainvision.brainvision.read_raw_brainvision
  egi.egi.read_raw_egi
  fiff.raw.read_raw_fif

.. currentmodule:: mne.io.kit

:py:mod:`mne.io.kit`:

.. autosummary::
  :toctree: generated/
  :template: function.rst

   coreg.read_mrk

File I/O
========

.. currentmodule:: mne

Functions:

.. autosummary::
   :toctree: generated
   :template: function.rst

   surface.decimate_surface
   surface.get_head_surf
   surface.get_meg_helmet_surf
   source_space.get_volume_labels_from_aseg
   misc.parse_config
   label.read_labels_from_annot
   bem.read_bem_solution
   bem.read_bem_surfaces
   cov.read_cov
   dipole.read_dipole
   epochs.read_epochs
   io.kit.read_epochs_kit
   event.read_events
   evoked.read_evokeds
   forward.forward.read_forward_solution
   label.read_label
   surface.read_morph_map
   proj.read_proj
   misc.read_reject_parameters
   selection.read_selection
   source_estimate.read_source_estimate
   source_space.read_source_spaces
   surface.read_surface
   transforms.read_trans
   source_estimate.save_stc_as_volume
   label.write_labels_to_annot
   bem.write_bem_solution
   bem.write_bem_surfaces
   cov.write_cov
   event.write_events
   evoked.write_evokeds
   forward.forward.write_forward_solution
   label.write_label
   proj.write_proj
   source_space.write_source_spaces
   surface.write_surface
   transforms.write_trans


Creating data objects from arrays
=================================

Classes:

.. currentmodule:: mne.evoked

:py:mod:`mne`:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   EvokedArray
.. currentmodule:: mne.epochs
.. autosummary::
   :toctree: generated/
   :template: class.rst

   EpochsArray

.. currentmodule:: mne.io.array.array

:py:mod:`mne.io`:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RawArray

Functions:

.. currentmodule:: mne.io.meas_info

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

.. currentmodule:: mne.datasets.sample.sample

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data_path

:py:mod:`mne.datasets.spm_face`:

.. automodule:: mne.datasets.spm_face
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.spm_face.spm_data

.. autosummary::
   :toctree: generated/
   :template: function.rst

   data_path

:py:mod:`mne.datasets.brainstorm`:

.. automodule:: mne.datasets.brainstorm
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.brainstorm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   bst_auditory.data_path
   bst_resting.data_path
   bst_raw.data_path

:py:mod:`mne.datasets.megsim`:

.. automodule:: mne.datasets.megsim
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.datasets.megsim.megsim

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

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   utils.ClickableImage

Functions:

.. currentmodule:: mne.viz
.. autosummary::
   :toctree: generated/
   :template: function.rst

   circle.circular_layout
   circle.plot_connectivity_circle
   utils.mne_analyze_colormap
   misc.plot_cov
   misc.plot_dipole_amplitudes
   _3d.plot_dipole_locations
   epochs.plot_drop_log
   epochs.plot_epochs
   misc.plot_events
   evoked.plot_evoked
   evoked.plot_evoked_image
   evoked.plot_evoked_topo
   topomap.plot_evoked_topomap
   _3d.plot_evoked_field
   evoked.plot_evoked_white
   ica.plot_ica_sources
   topomap.plot_ica_components
   ica.plot_ica_scores
   ica.plot_ica_overlay
   epochs.plot_epochs_image
   montage.plot_montage
   topomap.plot_projs_topomap
   raw.plot_raw
   raw.plot_raw_psd
   evoked.plot_snr_estimate
   _3d.plot_source_estimates
   _3d.plot_sparse_source_estimates
   topomap.plot_tfr_topomap
   topo.plot_topo_image_epochs
   topomap.plot_topomap
   utils.compare_fiff
   utils.add_background_image

.. currentmodule:: mne.io

:py:mod:`mne.io`:

.. automodule:: mne.io
   :no-members:
   :no-inherited-members:
.. autosummary::
   :toctree: generated/
   :template: function.rst

   open.show_fiff

Preprocessing
=============

Projections:

.. currentmodule:: mne.proj

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

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   layout.Layout
   montage.Montage
   montage.DigMontage

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   channels.fix_mag_coil_types
   montage.read_montage
   montage.read_dig_montage
   layout.read_layout
   layout.find_layout
   layout.make_eeg_layout
   layout.make_grid_layout
   channels.read_ch_connectivity
   channels.equalize_channels
   channels.rename_channels
   layout.generate_2d_layout

:py:mod:`mne.preprocessing`:

.. automodule:: mne.preprocessing
 :no-members:
 :no-inherited-members:

.. currentmodule:: mne.preprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ssp.compute_proj_ecg
   ssp.compute_proj_eog
   ecg.create_ecg_epochs
   eog.create_eog_epochs
   ecg.find_ecg_events
   eog.find_eog_events
   ica.ica_find_ecg_events
   ica.ica_find_eog_events
   maxwell.maxwell_filter
   ica.read_ica
   ica.run_ica

EEG referencing:

.. currentmodule:: mne.io

.. autosummary::
   :toctree: generated/
   :template: function.rst

   reference.add_reference_channels
   reference.set_bipolar_reference
   reference.set_eeg_reference

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

   misc.parse_config
.. currentmodule:: mne.event

.. autosummary::
   :toctree: generated/
   :template: function.rst

   concatenate_events
   find_events
   find_stim_steps
   make_fixed_length_events
   merge_events
   pick_events
   read_events
   write_events
   define_target_events

.. currentmodule:: mne.epochs

.. autosummary::
   :toctree: generated/
   :template: function.rst

   combine_event_ids
   equalize_epoch_counts
   add_channels_epochs
   concatenate_epochs

Sensor Space Data
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   evoked.combine_evoked
   io.base.concatenate_raws
   channels.channels.equalize_channels
   evoked.grand_average
   chpi.get_chpi_positions
   io.pick.pick_channels
   io.pick.pick_channels_cov
   io.pick.pick_channels_forward
   io.pick.pick_channels_regexp
   io.pick.pick_types
   io.pick.pick_types_forward
   epochs.read_epochs
   misc.read_reject_parameters
   selection.read_selection
   channels.channels.rename_channels


Covariance
==========

.. currentmodule:: mne.cov

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_covariance
   compute_raw_covariance
   make_ad_hoc_cov
   read_cov
   write_cov
   regularize


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
   coreg.create_default_subject
   coreg.scale_mri
   coreg.scale_bem
   coreg.scale_labels
   coreg.scale_source_space


Forward Modeling
================

:py:mod:`mne`:

.. currentmodule:: mne

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   _make_forward.make_forward_solution
   _field_interpolation.make_field_map
   transforms.read_trans
   surface.read_surface
   proj.sensitivity_map
   transforms.write_trans

:py:mod:`mne.bem`:

.. currentmodule:: mne.bem

.. autosummary::
   :toctree: generated/
   :template: function.rst

   write_bem_surfaces
   read_bem_surfaces
   make_bem_model
   make_bem_solution
   make_sphere_model
   make_watershed_bem
   make_flash_bem
   convert_flash_mris

:py:mod:`mne.forward.forward`:

.. currentmodule:: mne.forward.forward

.. autosummary::
   :toctree: generated/
   :template: function.rst

   apply_forward
   apply_forward_raw
   average_forward_solutions
   convert_forward_solution
   do_forward_solution
   read_forward_solution
   restrict_forward_to_label
   restrict_forward_to_stc

:py:mod:`mne.source_space`:

.. currentmodule:: mne.source_space

.. autosummary::
   :toctree: generated/
   :template: function.rst

   add_source_space_distances
   morph_source_spaces
   read_source_spaces
   setup_source_space
   setup_volume_source_space

Inverse Solutions
=================

:py:mod:`mne.minimum_norm`:

.. automodule:: mne.minimum_norm
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.minimum_norm.inverse

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
   estimate_snr
   make_inverse_operator
   read_inverse_operator
   write_inverse_operator
.. currentmodule:: mne.minimum_norm.time_frequency

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_source_psd
   compute_source_psd_epochs
   source_band_induced_power
   source_induced_power
.. currentmodule:: mne.minimum_norm.psf_ctf

.. autosummary::
   :toctree: generated/
   :template: function.rst

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

   mxne_inverse.mixed_norm
   mxne_inverse.tf_mixed_norm
   _gamma_map.gamma_map

:py:mod:`mne.beamformer`:

.. automodule:: mne.beamformer
  :no-members:
  :no-inherited-members:

.. currentmodule:: mne.beamformer._lcmv

.. autosummary::
   :toctree: generated/
   :template: function.rst

   lcmv
   lcmv_epochs
   lcmv_raw
.. currentmodule:: mne.beamformer._dics

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dics
   dics_epochs
   dics_source_power
.. currentmodule:: mne.beamformer._rap_music

.. autosummary::
   :toctree: generated/
   :template: function.rst

   rap_music

:py:mod:`mne`:

.. currentmodule:: mne

Functions:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dipole.fit_dipole


Source Space Data
=================

.. currentmodule:: mne.source_estimate

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compute_morph_matrix
   extract_label_time_course
   grade_to_tris
   grade_to_vertices
   morph_data
   morph_data_precomputed
   read_source_estimate
   save_stc_as_volume
.. currentmodule:: mne.label

.. autosummary::
   :toctree: generated/
   :template: function.rst

   grow_labels
   label_sign_flip
   read_labels_from_annot
   read_label
   split_label
   stc_to_label
   write_labels_to_annot
   write_label
.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dipole.read_dipole
   transforms.transform_surface_to
   source_space.vertex_to_mni

Time-Frequency
==============

:py:mod:`mne.time_frequency`:

.. automodule:: mne.time_frequency
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.time_frequency

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tfr.AverageTFR

Functions that operate on mne-python objects:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   csd.compute_epochs_csd
   psd.compute_epochs_psd
   psd.compute_raw_psd
   ar.fit_iir_model_raw
   tfr.tfr_morlet
   tfr.tfr_multitaper
   _stockwell.tfr_stockwell
   tfr.read_tfrs
   tfr.write_tfrs

Functions that operate on ``np.ndarray`` objects:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tfr.cwt_morlet
   multitaper.dpss_windows
   tfr.morlet
   multitaper.multitaper_psd
   tfr.single_trial_power
   stft.stft
   stft.istft
   stft.stftfreq


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

   utils.seed_target_indices
   spectral.spectral_connectivity
   effective.phase_slope_index


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

   multi_comp.bonferroni_correction
   multi_comp.fdr_correction
   cluster_level.permutation_cluster_test
   cluster_level.permutation_cluster_1samp_test
   permutations.permutation_t_test
   cluster_level.spatio_temporal_cluster_test
   cluster_level.spatio_temporal_cluster_1samp_test
   cluster_level.ttest_1samp_no_p
   regression.linear_regression
   regression.linear_regression_raw
   parametric.f_mway_rm

Functions to compute connectivity (adjacency) matrices for cluster-level statistics

.. currentmodule:: mne.source_estimate

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

   evoked.simulate_evoked
   raw.simulate_raw
   source.simulate_stc
   source.simulate_sparse_stc
   source.select_source_in_label

Decoding
========

:py:mod:`mne.decoding`:

.. automodule:: mne.decoding.transformer
   :no-members:
   :no-inherited-members:

Classes:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Scaler
   EpochsVectorizer
   PSDEstimator
   FilterEstimator
.. currentmodule:: mne.decoding.csp
.. autosummary::
   :toctree: generated/
   :template: class.rst

   CSP
.. currentmodule:: mne.decoding.time_gen
.. autosummary::
   :toctree: generated/
   :template: class.rst

   GeneralizationAcrossTime

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

   epochs.RtEpochs
   client.RtClient
   mockclient.MockRtClient
   fieldtrip_client.FieldTripClient
   stim_server_client.StimServer
   stim_server_client.StimClient

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
