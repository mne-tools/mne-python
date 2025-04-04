
Preprocessing
=============

Projections:

.. currentmodule:: mne

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/class_no_inherited_members.rst

   Projection

.. autosummary::
   :toctree: ../generated/

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
   :toctree: ../generated/

   Layout
   DigMontage
   compute_native_head_t
   fix_mag_coil_types
   read_polhemus_fastscan
   get_builtin_montages
   make_dig_montage
   read_dig_polhemus_isotrak
   read_dig_captrak
   read_dig_dat
   read_dig_egi
   read_dig_fif
   read_dig_hpts
   read_dig_localite
   make_standard_montage
   read_custom_montage
   transform_to_head
   compute_dev_head_t
   read_layout
   find_layout
   make_eeg_layout
   make_grid_layout
   find_ch_adjacency
   get_builtin_ch_adjacencies
   read_ch_adjacency
   equalize_channels
   unify_bad_channels
   rename_channels
   generate_2d_layout
   make_1020_channel_selections
   combine_channels

:py:mod:`mne.preprocessing`:

.. currentmodule:: mne.preprocessing

.. automodule:: mne.preprocessing
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

   ICA
   Xdawn
   EOGRegression
   annotate_amplitude
   annotate_break
   annotate_movement
   annotate_muscle_zscore
   annotate_nan
   compute_average_dev_head_t
   compute_current_source_density
   compute_bridged_electrodes
   compute_fine_calibration
   compute_maxwell_basis
   compute_proj_ecg
   compute_proj_eog
   compute_proj_hfc
   cortical_signal_suppression
   create_ecg_epochs
   create_eog_epochs
   find_bad_channels_lof
   find_bad_channels_maxwell
   find_ecg_events
   find_eog_events
   fix_stim_artifact
   ica_find_ecg_events
   ica_find_eog_events
   infomax
   interpolate_bridged_electrodes
   equalize_bads
   maxwell_filter
   maxwell_filter_prepare_emptyroom
   oversampled_temporal_projection
   peak_finder
   read_ica
   read_eog_regression
   realign_raw
   regress_artifact
   corrmap
   read_ica_eeglab
   read_fine_calibration
   write_fine_calibration
   apply_pca_obs

:py:mod:`mne.preprocessing.nirs`:

.. currentmodule:: mne.preprocessing.nirs

.. automodule:: mne.preprocessing.nirs
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

   optical_density
   beer_lambert_law
   source_detector_distances
   short_channels
   scalp_coupling_index
   temporal_derivative_distribution_repair

:py:mod:`mne.preprocessing.ieeg`:

.. currentmodule:: mne.preprocessing.ieeg

.. automodule:: mne.preprocessing.ieeg
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

   project_sensors_onto_brain
   make_montage_volume
   warp_montage

:py:mod:`mne.preprocessing.eyetracking`:

.. currentmodule:: mne.preprocessing.eyetracking

.. automodule:: mne.preprocessing.eyetracking
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

   Calibration
   read_eyelink_calibration
   set_channel_types_eyetrack
   convert_units
   get_screen_visual_angle
   interpolate_blinks

EEG referencing:

.. currentmodule:: mne

.. autosummary::
   :toctree: ../generated/

   add_reference_channels
   set_bipolar_reference
   set_eeg_reference

:py:mod:`mne.filter`:

.. currentmodule:: mne.filter

.. automodule:: mne.filter
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

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
   :toctree: ../generated/

   compute_chpi_amplitudes
   compute_chpi_snr
   compute_chpi_locs
   compute_head_pos
   extract_chpi_locs_ctf
   extract_chpi_locs_kit
   filter_chpi
   get_active_chpi
   get_chpi_info
   head_pos_to_trans_rot_t
   read_head_pos
   write_head_pos

:py:mod:`mne.transforms`

.. currentmodule:: mne.transforms

.. automodule:: mne.transforms
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: ../generated/

   Transform
   quat_to_rot
   rot_to_quat
   read_ras_mni_t
