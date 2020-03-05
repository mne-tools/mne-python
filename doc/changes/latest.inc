.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Changelog" for new features
   - "Bug" for bug fixes
   - "API" for backward-incompatible changes

.. _current:

Current (0.20.dev0)
-------------------

Changelog
~~~~~~~~~

- Improved :func:`mne.viz.plot_epochs` to label epoch counts starting from 0 `Sophie Herbst`_

- Add :func:`minimum_norm.apply_inverse_cov` to compute static power by applying inverse solutions to a data covariance matrix by `Denis Engemann`_,`Luke Bloy`_, and `Eric Larson`_

- Add :func:`mne.minimum_norm.resolution_metrics` to compute various resolution metrics for inverse solutions, by `Olaf Hauk`_

- Add current source density :func:`mne.preprocessing.compute_current_source_density` to compute the surface Laplacian in order to reduce volume conduction in data by `Alex Rockhill`_

- Add :func:`mne.chpi.extract_chpi_locs_ctf` to extract cHPI coil locations from CTF data  by `Luke Bloy`_

- Add :func:`mne.chpi.compute_chpi_amplitudes`, :func:`mne.chpi.compute_chpi_locs`, and :func:`mne.chpi.compute_head_pos` to compute head positions from cHPI coil locations by `Eric Larson`_ and `Luke Bloy`_

- Add :func:`mne.io.Raw.set_meas_date` by `Eric Larson`_

- Add ``copy`` parameter to :meth:`mne.Epochs.iter_evoked` by `Alex Gramfort`_

- Add command :ref:`gen_mne_setup_source_space` to quickly set up bilateral hemisphere surface-based source space with subsampling by `Victor Ferat`_.

- Add command :ref:`gen_mne_sys_info` to print system information by `Eric Larson`_

- Add function :func:`mne.make_fixed_length_epochs` to segment raw into fixed length epochs by `Mohammad Daneshzand`_

- Add support for computing patch information only in surface source space creation with ``add_dist='patch'`` in :func:`mne.setup_source_space` and ``dist_limit=0`` in :func:`mne.add_source_space_distances` by `Eric Larson`_

- Add :class:`mne.Label.restrict` to restrict a label to vertices within a source space by `Eric Larson`_

- Add support for passing a destination source space ``src_to`` in :func:`mne.compute_source_morph` to ensure morphing for multiple subjects results in consistent STCs, by `Eric Larson`_

- Add support for plotting fNIRS channels in :func:`mne.viz.plot_alignment` and :func:`mne.Evoked.animate_topomap` by `Eric Larson`_ and `Robert Luke`_

- Add command line tool :ref:`gen_mne_anonymize` for anonymizing raw fiff files by `Luke Bloy`_

- Add support to :func:`mne.io.anonymize_info` to allow time offset to be applied to dates by `Luke Bloy`_

- Add support for computing resolution matrix to get point spread functions (PSF) and cross-talk functions (CTF) in :func:`mne.minimum_norm.make_inverse_resolution_matrix`, :func:`mne.beamformer.make_lcmv_resolution_matrix`, :func:`mne.minimum_norm.get_cross_talk`, :func:`mne.minimum_norm.get_point_spread` by `Olaf Hauk`_

- Add :func:`mne.preprocessing.read_ica_eeglab` to read EEGLAB ICA decompositions by `Christian O'Reilly`_

- Add keyboard functionality to interactive colorbar plotting TFRs by `Stefan Repplinger`_

- Add the ability to ``return_event_id`` in :func:`mne.read_events` for use with MNE-C produced ``-annot.fif`` files, by `Eric Larson`_

- Add option to :func:`mne.connectivity.spectral_connectivity` to compute corrected imaginary PLV by `Adonay Nunes`_

- Add :func:`mne.SourceEstimate.estimate_snr` to estimate source-space SNR, by `Kaisu Lankinen`_ and `Padma Sundaram`_

- Add option to specify the coordinate frame in :func:`mne.channels.read_custom_montage` by `Eric Larson`_

- Add option to use ``match_case=False`` to :meth:`mne.io.Raw.set_montage` and related functions by `Eric Larson`_

- Add reader for NIRx data in :func:`mne.io.read_raw_nirx` by `Robert Luke`_

- Add function to convert NIRS data to optical density :func:`mne.preprocessing.nirs.optical_density` by `Robert Luke`_

- Add function to convert NIRS data to haemoglobin concentration :func:`mne.preprocessing.nirs.beer_lambert_law` by `Robert Luke`_

- Add functions to calculate spatial information of NIRS channels :func:`mne.preprocessing.nirs.source_detector_distances` and :func:`mne.preprocessing.nirs.short_channels` by `Robert Luke`_

- Add reader for ``*.dat`` electrode position files :func:`mne.channels.read_dig_dat` by `Christian Brodbeck`_

- Improved :func:`mne.viz.plot_events` to always show event counts by `Eric Larson`_

- Improved :ref:`limo-dataset` usage and :ref:`example <ex-limo-data>` for usage of :func:`mne.stats.linear_regression` by `Jose Alanis`_

- Add support for ``reduce_rank=True`` for vector beamformers by `Eric Larson`_

- Add ``method='sphere'`` support for using Freesurfer spherical inflation in :func:`mne.decimate_surface` to facilitate mesh downsampling for use with :func:`mne.make_bem_model` by `Eric Larson`_

- Speed up :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` calculations by vectorizing linear algebra calls by `Dmitrii Altukhov`_ and `Eric Larson`_

- Speed up :func:`mne.make_forward_solution` using Numba, by `Eric Larson`_

- For KIT systems without built-in layout, :func:`mne.channels.find_layout` now falls back on an automatically generated layout, by `Christian Brodbeck`_

- :meth:`mne.Epochs.plot` now takes a ``epochs_colors`` parameter to color specific epoch segments by `Mainak Jas`_

- Add command :ref:`gen_mne_setup_forward_model` to quickly create a BEM model for a subject by `Victor Ferat`_.

- Add command :ref:`gen_mne_prepare_bem_model` to quickly create a BEM solution using the linear collocation approach by `Victor Ferat`_.

- Add support for reading ``.bdip`` Xfit binary dipole fit files in :func:`mne.read_dipole` by `Eric Larson`_

- Allow returning vector source estimates from sparse inverse solvers through ``pick_ori='vector'`` by `Christian Brodbeck`_

- Add NIRS support to :func:`mne.viz.plot_topomap` by `Robert Luke`_

- Add the ability to :func:`mne.channels.equalize_channels` to also re-order the channels and also operate on instances of :class:`mne.Info`, :class:`mne.Forward`, :class:`mne.Covariance` and :class:`mne.time_frequency.CrossSpectralDensity` by `Marijn van Vliet`_

- Allow `mne.channels.read_custom_montage` to handle fiducial points for BESA spherical (``.elp``) files by `Richard Höchenberger`_

- Add ``rank`` argument to :func:`mne.fit_dipole` by `Eric Larson`_

- Add function to convert events to annotations :func:`mne.annotations_from_events` by `Nicolas Barascud`_

- Add function to calculate scalp coupling index for fNIRS data :func:`mne.preprocessing.nirs.scalp_coupling_index` by `Robert Luke`_

- Add ``item`` argument to :meth:`mne.Epochs.get_data` for faster access to NumPy data arrays compared to :meth:`mne.Epochs.__getitem__` for frequent access on large :class:`mne.Epochs` objects, by `Eric Larson`_

- More accurate coordinate system for Easycap montages in :func:`mne.channels.make_standard_montage` by `Christian Brodbeck`_

- Add ``border`` argument to :func:`mne.viz.plot_topomap`. ``border`` controls the value of the edge points to which topomap values are extrapolated. ``border='mean'`` sets these points value to the average of their neighbours. By `Mikołaj Magnuski`_

- Add function :func:`mne.viz.link_brains` to link time properties of multiple brain objects interactively by `Guillaume Favelier`_

- Add function :func:`mne.preprocessing.annotate_movement` to annotate periods with head motion and :func:`mne.preprocessing.compute_average_dev_head_t` to re-estimate the device to head transform with average head position during segments with acceptable head movement. by `Adonay Nunes`_

- Make selected channels more distinguishable in :meth:`mne.Epochs.plot_sensors` when using ``kind='select'`` by `Mikołaj Magnuski`_

- Allow retrieval of the number of Infomax ICA iterations via the new ``return_n_iter`` keyword argument of :func:`mne.preprocessing.infomax` by `Richard Höchenberger`_

- Expose the number of ICA iterations during the fitting procedure via the ``n_iter_`` attribute of :class:`mne.preprocessing.ICA` by `Richard Höchenberger`_

Bug
~~~

- Allow :func:`mne.channels.read_dig_hpts` to accept point categories (eg. EEG) to be specified in upper case by `Alex Gramfort`_.

- Fix a bug in :meth:`mne.MixedSourceEstimate.plot_surface` that prevented plotting in latest PySurfer by `Christian O'Reilly`_.

- Fix ``method='eLORETA'`` in :func:`mne.minimum_norm.apply_inverse`, by `Eric Larson`_:

  1. regularization, which was off by roughly a factor of a square root
  2. units, which are now nAm instead of being arbitrary
  3. explained variance and residuals, which were previously unavailable
  4. ``force_equal`` in the case of loose orientations, which previously
     defaulted to ``True`` now defaults to ``False``.
     Reapplying orientation priors at each iteration of the algorithm
     sufficiently preserves a smooth transition between loose=0. to loose=1.
     without compromising localization accuracy in testing.

- Fix bug in ``method='eLORETA'`` for :func:`mne.minimum_norm.apply_inverse` (and variants) to allow restricting source estimation to a label by `Luke Bloy`_

- Fix bug in :func:`mne.compute_covariance` and :func:`mne.compute_raw_covariance` where biased normalization (based on degrees of freedom) was used and ``cov.nfree`` was not set properly by `Eric Larson`_

- Fix :func:`mne.VectorSourceEstimate.normal` to account for cortical patch statistics using ``use_cps=True`` by `Eric Larson`_

- Fix ``pick_ori='normal'`` for :func:`mne.minimum_norm.apply_inverse` when the inverse was computed with ``loose=1.`` and the forward solution was not in surface orientation, by `Eric Larson`_

- Fix missing xlabel for :func:`mne.io.Raw.plot_psd` and similar functions when passing a subset of axes from a figure by `Alex Gramfort`_

- Fix wrong DataFrame index when ``index=None`` in methods :meth:`mne.io.Raw.to_data_frame`, :meth:`mne.Epochs.to_data_frame`, :meth:`mne.Evoked.to_data_frame`, and :meth:`mne.SourceEstimate.to_data_frame`, by `Daniel McCloy`_.

- Fix incorrect scaling of cluster temporal extent in :func:`mne.stats.summarize_clusters_stc` by `Daniel McCloy`_.

- Fix :func:`mne.time_frequency.read_tfrs` to properly read :class:`~mne.Info` transforms by `Eric Larson`_

- Fix :func:`mne.viz.plot_sensors` to always plot in head coordinate frame by `Daniel McCloy`_.

- Fix :func:`mne.io.read_raw_cnt` for version 1 of format in the presence of annotations by `Alex Gramfort`_.

- Fix :class:`mne.decoding.GeneralizingEstimator` and related classes to support multi-class sklearn scorers such as ``'roc_auc_ovo'`` and ``'roc_auc_ovo_weighted'`` by `Eric Larson`_

- Fix :meth:`mne.io.read_raw_ctf` to set measurement date from CTF ds files by `Luke Bloy`_.

- Fix :meth:`mne.read_epochs_eeglab` that ignored channel locations by `Alex Gramfort`_.

- Fix :meth:`mne.io.read_raw_brainvision` when channel names have spaces by `Sebastian Major`_.

- Fix :meth:`mne.io.Raw.anonymize` to correctly reset ``raw.annotations.orig_time`` by `Luke Bloy`_.

- Fix :meth:`mne.io.Raw.anonymize` to correctly avoid shifting ``raw.annotations.onset`` relative to ``raw.first_samp`` by `Eric Larson`_

- Fix :meth:`mne.io.Raw.set_channel_types` and :meth:`mne.io.Raw.rename_channels` and related methods to return the instance instead of ``None`` by `Eric Larson`_

- :meth:`mne.Epochs.iter_evoked` now does not return a copy of info when ``copy=False`` (default parameter) by `Alex Gramfort`_

- The attribute :class:`mne.Annotations.orig_time <mne.Annotations>` is now read-only, and is a :class:`~python:datetime.datetime` object (or None) rather than float, by `Eric Larson`_

- Fix bug where MRI distances were not properly initialized in :ref:`gen_mne_coreg` by `Eric Larson`_

- Fix ``xscale='log'`` in :meth:`mne.io.Raw.plot_psd` and related functions by `Alex Gramfort`_

- Unify behavior of ``raw.annotations.append(...)`` when ``raw.info['meas_date']`` is None to make onsets absolute relative to ``first_samp`` as they are when ``raw.info['meas_date']`` is not None; i.e., you might need to do ``raw.annotations.append(old_time + raw.first_time)``, by `Eric Larson`_

- The :class:`info['meas_date'] <mne.Info>` entry is now a :class:`~python:datetime.datetime` object (or None) rather than float, by `Eric Larson`_

- Fix :meth:`mne.time_frequency.AverageTFR.save` and related functions to deal with :class:`~python:datetime.datetime` objects properly by `Eric Larson`_

- Fix date reading before Unix time zero (1970 Jan 1) on Windows by `Alex Rockhill`_.

- Fix bug running subprocesses (e.g., in :func:`mne.bem.make_watershed_bem`) in Jupyter notebooks, by `Eric Larson`_

- Fix :meth:`mne.io.read_raw_artemis123` to add HPI locations to digitization points, by `Luke Bloy`_

- Fix :meth:`mne.Epochs.shift_time` and :meth:`mne.Evoked.shift_time` to return the modified :class:`~mne.Epochs` or :class:`~mne.Evoked` instance (instead of ``None``) by `Daniel McCloy`_.

- Fix :func:`mne.read_annotations` for text files with zero or one annotations, by `Adonay Nunes`_

- Fix bug in :class:`~mne.preprocessing.ICA` where requesting extended infomax via ``fit_params={'extended': True}`` was overridden, by `Daniel McCloy`_.

- Fix bug in :meth:`mne.Epochs.interpolate_bads` where the ``origin`` was not used during MEG or EEG data interpolation by `Eric Larson`_. Old behavior can be achieved using ``origin=(0., 0., 0.)`` for EEG and ``origin=(0., 0., 0.04)`` for MEG, and the new default is ``origin='auto'``, which uses a head-digitization-based fit.

- Fix bug in :func:`mne.write_evokeds` where ``evoked.nave`` was not saved properly when multiple :class:`~mne.Evoked` instances were written to a single file, by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.mark_flat` where acquisition skips were not handled properly, by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_bem` where some sources were not plotted by `Jean-Remi King`_ and `Eric Larson`_

- Fix TAL channel parsing (annotations) for EDF-D files by `Clemens Brunner`_

- Fix bug with :func:`mne.viz.plot_dipole_locations` when plotting in head coordinates by `Eric Larson`_

- Fix handling in cases where no components are found in :func:`mne.preprocessing.corrmap` by `Eric Larson`_

- Fix handling of repeated events in :class:`mne.Epochs` by `Fahimeh Mamashli`_ and `Alex Gramfort`_

- Fix many bugs with plotting sensors overlaid on a head outline. All plotting is now done in head coordinates and scaled by ``head_radius``, which defaults to 0.095, by `Eric Larson`_

- Fix :func:`mne.io.anonymize_info` to allow shifting dates of service and to match anticipated changes in mne-cpp by `Luke Bloy`_

- Fix reading of cardinals in .htps files (identifier are int not strings) by `Alex Gramfort`_

- Fix IO of TFRs when event_id contain a / in one of the keys by `Alex Gramfort`_

- Fix plotting of TFRs with linear y-scale (misaligned frequency bins), by `Stefan Repplinger`_ and `Eric Larson`_

- Fix ``info['sfreq']`` when decimating in :func:`mne.time_frequency.tfr_multitaper` and :func:`mne.time_frequency.tfr_morlet` and make sure an error is raised when exceed Nyquist frequency by `Adonay Nunes`_

- Fix bug in EDF(+) loading, filter values ignored by insufficient regex, by `Demetres Kostas`_

- Fix missing scaling of tolerance parameter in :func:`mne.inverse_sparse.tf_mixed_norm` and :func:`mne.inverse_sparse.mixed_norm`, by `Mathurin Massias`_

- Fix the automatic scaling of the glyphs in :func:`mne.viz.plot_vector_source_estimates` by using 10% of the brain width, by `Guillaume Favelier`_

- Fix handling of missing values (NaNs) in :func:`mne.time_frequency.psd_welch` by `Clemens Brunner`_

- Fix :class:`mne.decoding.LinearModel` to support the refitted estimator of ``GridSearchCV`` in ``sklearn`` by `Chun-Hui Li`_

- Fix :func:`mne.viz.plot_topomap` when there are bad channels in the provided :class:`mne.Info` instance by `Eric Larson`_

- Fix check for subject definition existing in source space in :func:`mne.compute_source_morph` and related functions by `Eric Larson`_

- Fix creation of new figures in :meth:`mne.Epochs.plot_image` and :func:`mne.viz.plot_epochs_image` by `Eric Larson`_

- Fix bug in :class:`~mne.preprocessing.Xdawn` where filters where selected along the incorrect axis, by `Henrich Kolkhorst`_

- Fix bug in :func:`mne.bem.make_watershed_bem` where some surfaces were saved incorrectly in the working directory by `Yu-Han Luo`_

- Fix support for multiple TAL (annotations) channels in BDF reader by `Clemens Brunner`_

- Fix missing `brainmask` file type in :func:`mne.bem.make_watershed_bem` by changing its default value to `ws.mgz` by `Fu-Te Wong`_

- Fix bug in :func:`mne.channels.make_standard_montage` which would return `easycap-M1` even when requesting `easycap-M10` by `Christian Brodbeck`_

- Fix the way planar gradiometers are combined in :func:`mne.viz.plot_tfr_topomap` and :meth:`mne.Epochs.plot_psd_topomap` by `Geoff Brookshire`_

- Improve error message when trying to load FieldTrip data from a cell array by `Thomas Hartmann`_

- Fix bug in :func:`mne.read_evoked_fieldtrip` causing it to crash when channels were present in the provided Info object but were not present in the data by `Thomas Hartmann`_

- Fix placement of extrapolation points in :meth:`mne.Evoked.plot_topomap` and related functions when exactly three channels were used by `Mikołaj Magnuski`_.

- Fix bug in reading annotations in :func:`read_annotations`, which would not accept ";" character by `Adam Li`_

- Include ``fit_params`` when saving an :class:`~mne.preprocessing.ICA` instance to disk by `Richard Höchenberger`_

- Update old url link in :func:`mne.datasets.eegbci.load_data` to ``EEGMI_URL = 'https://physionet.org/files/eegmmidb/1.0.0/'`` by `Ramiro Gatti`_

API
~~~

- :meth:`mne.Epochs.plot` now accepts an ``event_id`` parameter (useful in tandem with ``event_colors`` for specifying event colors by name) by `Daniel McCloy`_.

- New time conversion options for methods :meth:`mne.io.Raw.to_data_frame`, :meth:`mne.Epochs.to_data_frame`, :meth:`mne.Evoked.to_data_frame`, and :meth:`mne.SourceEstimate.to_data_frame`, by `Daniel McCloy`_.

- :meth:`mne.Epochs.shift_time` and :meth:`mne.Evoked.shift_time` now allow shifting times by arbitrary amounts (previously only by integer multiples of the sampling period), by `Daniel McCloy`_ and `Eric Larson`_.

- The ``head_pos`` argument of :func:`mne.Evoked.plot_topomap` and related functions has been deprecated in favor of ``head_radius``, by `Eric Larson`_.

- The ``layout`` argument to topomap-related functions such as :meth:`mne.Evoked.plot_topomap` and :func:`mne.viz.plot_tfr_topomap` has been deprecated in favor of channel-position based flattening based on the ``info`` and ``sphere`` argument, by `Eric Larson`_.

- The APIs of :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` are now more similar to :func:`mne.viz.plot_projs_topomap` by `Daniel McCloy`_.

- The function :func:`mne.setup_volume_source_space` has a ``sphere_units`` argument that defaults to ``'mm'`` in 0.20 but will change to ``'m'`` in 0.21, set it to avoid a warning by `Eric Larson`_.

- :func:`mne.viz.plot_projs_topomap` and the related methods :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` now accept parameter ``vlim`` to control the colormap, with keyword ``'joint'`` computing the colormap jointly across all projectors of a given channel type, by `Daniel McCloy`_.

- New methods :meth:`mne.io.Raw.get_channel_types`, :meth:`mne.Epochs.get_channel_types`, :meth:`mne.Evoked.get_channel_types` by `Daniel McCloy`_.

- Deprecate ``mne.minimum_norm.point_spread_function`` and ``mne.minimum_norm.cross_talk_function`` by `Alex Gramfort`_

- Deprecate ``montage`` parameter of :func:`mne.create_info` in favor of :meth:`mne.io.Raw.set_montage`, :meth:`mne.Info.set_montage`, and related functions by `Eric Larson`_

- :func:`mne.channels.equalize_channels` no longer operates in-place by default, but instead makes copies of the objects when necessary (see ``copy`` parameter) by `Marijn van Vliet`_

- :func:`mne.channels.equalize_channels` now uses the first object in the list as a template for channel ordering by `Marijn van Vliet`_

- :func:`mne.channels.equalize_channels` now also re-orders the channels to match, in addition to dropping channels that are not shared by all objects by `Marijn van Vliet`_

- :meth:`mne.io.Raw.pick_channels`, :meth:`mne.Epochs.pick_channels` and :meth:`mne.Evoked.pick_channels` now have an ``ordered`` parameter to enforce the ordering of the picked channels by `Marijn van Vliet`_

- New methods :meth:`mne.Forward.pick_channels`, :meth:`mne.Covariance.pick_channels`, :meth:`mne.Info.pick_channels`, :meth:`mne.time_frequency.CrossSpectralDensity.pick_channels` by `Marijn van Vliet`_

- New attributes ``mne.Forward.ch_names`` and ``mne.Info.ch_names`` by `Marijn van Vliet`_

- In :class:`~mne.preprocessing.Xdawn`, the components are stored in the rows of attributes ``filters_`` and ``patterns_`` to be consistent with :class:`~mne.decoding.CSP` and :class:`~mne.preprocessing.ICA` by `Henrich Kolkhorst`_

- Drop ``unit`` keyword argument from :func:`mne.channels.read_custom_montage`, as it was unused by `Richard Höchenberger`_

- Deprecate :func:`mne.preprocessing.run_ica`, use :meth:`mne.preprocessing.ICA.detect_artifacts` instead by `Richard Höchenberger`_
