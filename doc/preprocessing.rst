
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
   make_standard_montage
   read_custom_montage
   compute_dev_head_t
   read_layout
   find_layout
   make_eeg_layout
   make_grid_layout
   find_ch_adjacency
   read_ch_adjacency
   equalize_channels
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
   :toctree: generated/

   ICA
   Xdawn
   annotate_flat
   annotate_movement
   annotate_muscle_zscore
   compute_average_dev_head_t
   compute_current_source_density
   compute_fine_calibration
   compute_maxwell_basis
   compute_proj_ecg
   compute_proj_eog
   create_ecg_epochs
   create_eog_epochs
   find_bad_channels_maxwell
   find_ecg_events
   find_eog_events
   fix_stim_artifact
   ica_find_ecg_events
   ica_find_eog_events
   infomax
   equalize_bads
   maxwell_filter
   oversampled_temporal_projection
   peak_finder
   read_ica
   realign_raw
   regress_artifact
   corrmap
   read_ica_eeglab
   read_fine_calibration
   write_fine_calibration

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
   temporal_derivative_distribution_repair

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
   extract_chpi_locs_kit
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
