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

- Add :func:`mne.minimum_norm.resolution_metrics` to compute various resolution metrics for inverse solutions, by `Olaf Hauk`_

- Add current source density :func:`mne.preprocessing.compute_current_source_density` to compute the surface Laplacian in order to reduce volume conduction in data by `Alex Rockhill`_

- Add :func:`mne.chpi.calculate_head_pos_ctf` by `Luke Bloy`_

- Add :func:`mne.io.Raw.set_meas_date` by `Eric Larson`_

- Add command :ref:`gen_mne_setup_source_space` to quickly set up bilateral hemisphere surface-based source space with subsampling by `Victor Ferat`_.

- Add command :ref:`gen_mne_sys_info` to print system information by `Eric Larson`_

- Add function :func:`mne.make_fixed_length_epochs` to segment raw into fixed length epochs by `Mohammad Daneshzand`_

- Add support for passing a destination source space ``src_to`` in :func:`mne.compute_source_morph` to ensure morphing for multiple subjects results in consistent STCs, by `Eric Larson`_

- Add support for plotting fNIRS channels in :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add command line tool :ref:`gen_mne_anonymize` for anonymizing raw fiff files by `Luke Bloy`_

- Add support to :func:`mne.io.anonymize_info` to allow time offset to be applied to dates by `Luke Bloy`_

- Add support for computing resolution matrix to get point spread functions (PSF) and cross-talk functions (CTF) in :func:`mne.minimum_norm.make_resolution_matrix`, :func:`mne.minimum_norm.get_cross_talk`, :func:`mne.minimum_norm.get_point_spread` by `Olaf Hauk`_

- Add keyboard functionality to interactive colorbar plotting TFRs by `Stefan Repplinger`_

- Add the ability to ``return_event_id`` in :func:`mne.read_events` for use with MNE-C produced ``-annot.fif`` files, by `Eric Larson`_

- Add option to :func:`mne.connectivity.spectral_connectivity` to compute corrected imaginary PLV by `Adonay Nunes`_

- Add :func:`mne.SourceEstimate.estimate_snr` to estimate source-space SNR, by `Kaisu Lankinen`_ and `Padma Sundaram`_

- Add option to specify the coordinate frame in :func:`mne.channels.read_custom_montage` by `Eric Larson`_

- Add reader for NIRx data in :func:`mne.io.read_raw_nirx` by `Robert Luke`_

- Add function to convert NIRS data to optical density :func:`mne.preprocessing.nirs.optical_density` by `Robert Luke`_

- Add function to convert NIRS data to haemoglobin concentration :func:`mne.preprocessing.nirs.beer_lambert_law` by `Robert Luke`_

- Add functions to calculate spatial information of NIRS channels :func:`mne.preprocessing.nirs.source_detector_distances` and :func:`mne.preprocessing.nirs.short_channels` by `Robert Luke`_

- Add reader for ``*.dat`` electrode position files :func:`mne.channels.read_dig_dat` by `Christian Brodbeck`_

- Improved :ref:`limo-dataset` usage and :ref:`example <ex-limo-data>` for usage of :func:`mne.stats.linear_regression` by `Jose Alanis`_

- Speed up :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` calculations by vectorizing linear algebra calls by `Dmitrii Altukhov`_ and `Eric Larson`_

- For KIT systems without built-in layout, :func:`mne.channels.find_layout` now falls back on an automatically generated layout, by `Christian Brodbeck`_

- :meth:`mne.Epochs.plot` now takes a ``epochs_colors`` parameter to color specific epoch segments by `Mainak Jas`_

- Add command :ref:`gen_mne_setup_forward_model` to quickly create a BEM model for a subject by `Victor Ferat`_.

- Add command :ref:`gen_mne_prepare_bem_model` to quickly create a BEM solution using the linear collocation approach by `Victor Ferat`_.

- Allow returning vector source estimates from sparse inverse solvers through ``pick_ori='vector'`` by `Christian Brodbeck`_

- Add NIRS support to :func:`mne.viz.plot_topomap` by `Robert Luke`_

- Add the ability to :func:`mne.channels.equalize_channels` to also re-order the channels and also operate on instances of :class:`mne.Info`, :class:`mne.Forward`, :class:`mne.Covariance` and :class:`mne.time_frequency.CrossSpectralDensity` by `Marijn van Vliet`_

Bug
~~~

- Fix incorrect scaling of cluster temporal extent :func:`mne.stats.summarize_clusters_stc` by `Daniel McCloy`_.

- Fix :meth:`mne.io.read_raw_ctf` to set measurement date from CTF ds files by `Luke Bloy`_.

- Fix :meth:`mne.io.Raw.anonymize` to correctly reset ``raw.annotations.orig_time`` by `Luke Bloy`_.

- Fix :meth:`mne.io.Raw.anonymize` to correctly avoid shifting ``raw.annotations.onset`` relative to ``raw.first_samp`` by `Eric Larson`_

- The attribute :class:`mne.Annotations.orig_time <mne.Annotations>` is now read-only, and is a :class:`~python:datetime.datetime` object (or None) rather than float, by `Eric Larson`_

- Unify behavior of ``raw.annotations.append(...)`` when ``raw.info['meas_date']`` is None to make onsets absolute relative to ``first_samp`` as they are when ``raw.info['meas_date']`` is not None; i.e., you might need to do ``raw.annotations.append(old_time + raw.first_time)``, by `Eric Larson`_

- The :class:`info['meas_date'] <mne.Info>` entry is now a :class:`~python:datetime.datetime` object (or None) rather than float, by `Eric Larson`_

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

API
~~~

- :meth:`mne.Epochs.shift_time` and :meth:`mne.Evoked.shift_time` now allow shifting times by arbitrary amounts (previously only by integer multiples of the sampling period), by `Daniel McCloy`_ and `Eric Larson`_.

- The APIs of :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` are now more similar to :func:`mne.viz.plot_projs_topomap` by `Daniel McCloy`_.

- :func:`mne.viz.plot_projs_topomap` now accepts both :class:`~mne.channels.Layout` and :class:`~mne.Info` (previously one or the other), and will ignore :class:`~mne.Info` if :class:`~mne.channels.Layout` is provided `Daniel McCloy`_.

- :func:`mne.viz.plot_projs_topomap` and the related methods :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` now accept parameter ``vlim`` to control the colormap, with keyword ``'joint'`` computing the colormap jointly across all projectors of a given channel type, by `Daniel McCloy`_.

- New methods :meth:`mne.io.Raw.get_channel_types`, :meth:`mne.Epochs.get_channel_types`, :meth:`mne.Evoked.get_channel_types` by `Daniel McCloy`_.

- Deprecate ``mne.minimum_norm.point_spread_function`` and ``mne.minimum_norm.cross_talk_function`` by `Alex Gramfort`_

- :func:`mne.channels.equalize_channels` no longer operates in-place by default, but instead makes copies of the objects when necessary (see ``copy`` parameter) by `Marijn van Vliet`_

- :func:`mne.channels.equalize_channels` now uses the first object in the list as a template for channel ordering by `Marijn van Vliet`_

- :func:`mne.channels.equalize_channels` now also re-orders the channels to match, in addition to dropping channels that are not shared by all objects by `Marijn van Vliet`_

- :meth:`mne.io.Raw.pick_channels`, :meth:`mne.Epochs.pick_channels` and :meth:`mne.Evoked.pick_channels` now have an ``ordered`` parameter to enforce the ordering of the picked channels by `Marijn van Vliet`_

- New methods :meth:`mne.Forward.pick_channels`, :meth:`mne.Covariance.pick_channels`, :meth:`mne.Info.pick_channels`, :meth:`mne.time_frequency.CrossSpectralDensity.pick_channels` by `Marijn van Vliet`_

- New attributes ``mne.Forward.ch_names`` and ``mne.Info.ch_names`` by `Marijn van Vliet`_
