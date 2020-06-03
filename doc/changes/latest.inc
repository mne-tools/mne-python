.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Changelog" for new features
   - "Bug" for bug fixes
   - "API" for backward-incompatible changes

.. _current:

Current (0.21.dev0)
-------------------

Changelog
~~~~~~~~~

- Allow picking channels in raw instances (e.g., :meth:`mne.io.Raw.pick_types`) without preloading data, by `Eric Larson`_

- :meth:`mne.preprocessing.ICA.plot_sources` now plots annotation markers similar to :meth:`mne.io.Raw.plot` by `Luke Bloy`_

- Add support for signals in mV for :func:`mne.io.read_raw_brainvision` by `Clemens Brunner`_

- :meth:`mne.Epochs.plot_psd_topomap` and :func:`mne.viz.plot_epochs_psd_topomap` now allow joint colorbar limits across subplots, by `Daniel McCloy`_.

- Add :class:`mne.MixedVectorSourceEstimate` for vector source estimates for mixed source spaces, by `Eric Larson`_

- Add :meth:`mne.MixedSourceEstimate.surface` and :meth:`mne.MixedSourceEstimate.volume` methods to allow surface and volume extraction by `Eric Larson`_

- Add support to :func:`mne.extract_label_time_course` for vector-valued and volumetric source estimates by `Eric Larson`_

- Add method :meth:`mne.VolSourceEstimate.in_label` by `Eric Larson`_

- Add ``tol_kind`` option to :func:`mne.compute_rank` by `Eric Larson`_

- Allow resampling raw data with :func:`mne.io.Raw.resample` without preloading data, by `Eric Larson`_

- Allow using ``pick_ori='vector'`` with a fixed-orientation inverse to facilitate visualization with :func:`mne.viz.plot_vector_source_estimates` by `Eric Larson`_

- :func:`mne.viz.plot_dipole_locations` and :meth:`mne.Dipole.plot_locations` gained a ``title`` argument to specify a custom figure title in ``orthoview`` mode by `Richard Höchenberger`_

- Add temporal derivative distribution repair (TDDR) :func:`mne.preprocessing.nirs.temporal_derivative_distribution_repair` with shortened alias ``mne.preprocessing.nirs.tddr`` by `Robert Luke`_

- Add :func:`mne.read_freesurfer_lut` to make it easier to work with volume atlases by `Eric Larson`_

- Add functionality to interpolate bad NIRS channels by `Robert Luke`_

- Added ability of :func:`mne.io.read_raw_nirx` to open data by passing path to header file `Robert Luke`_

- Add :meth:`mne.channels.DigMontage.rename_channels` to allow renaming montage channels by `Eric Larson`_

- Add support to in :meth:`mne.io.Raw.plot` for passing ``clipping`` as a float to clip to a proportion of the dedicated channel range by `Eric Larson`_

- Add function :func:`mne.preprocessing.annotate_muscle_zscore` to annotate periods with muscle artifacts. by `Adonay Nunes`_

- Add :func:`mne.stats.ttest_ind_no_p` to mirror :func:`mne.stats.ttest_1samp_no_p` with hat correction by `Eric Larson`_

- Speed up raw data reading without preload in :func:`mne.io.read_raw_nirx` by `Eric Larson`_

- Support for saving movies of source time courses (STCs) with ``brain.save_movie`` method and from graphical user interface by `Guillaume Favelier`_

- Add ``mri`` and ``show_orientation`` arguments to :func:`mne.viz.plot_bem` by `Eric Larson`_

- Add "on_missing='raise'" to :meth:`mne.io.Raw.set_montage` and related functions to allow ignoring of missing electrode coordinates by `Adam Li`_

- Add ``plot`` option to :meth:`mne.viz.plot_filter` allowing selection of which filter properties are plotted and added option for user to supply ``axes`` by `Robert Luke`_

- Add estimation method legend to :func:`mne.viz.plot_snr_estimate` by `Eric Larson`_

- Add ``sources`` and ``detectors`` options for fNIRS use of :meth:`mne.viz.plot_alignment` allowing plotting of optode locations in addition to channel midpoint ``channels`` and ``path`` between fNIRS optodes by `Kyle Mathewson`_

- Add ECoG misc EDF dataset to the :ref:`tut_working_with_ecog` tutorial to show snapshots of time-frequency activity by `Adam Li`_

- BIDS conformity: When saving FIF files to disk and the files are split into parts, the ``split_naming='bids'`` parameter now uses a "_split-%d" naming instead of the previous "_part-%d", by `Stefan Appelhoff`_

- Add support for whitening and multiple channel types in :func:`mne.beamformer.make_dics` by `Marijn van Vliet`_

- Add support for loading complex numbers from mat files by `Thomas Hartmann`_

- Add generic reader function :func:`mne.io.read_raw` that loads files based on their extensions (it wraps the underlying specific ``read_raw_xxx`` functions) by `Clemens Brunner`_

- Add ``'auto'`` option to :meth:`mne.preprocessing.ICA.find_bads_ecg` to automatically determine the threshold for CTPS method by `Yu-Han Luo`_

Bug
~~~

- Fix bug with :func:`mne.preprocessing.ICA.find_bads_eog` when more than one EOG components are present by `Christian O'Reilly`_

- Fix bug with :func:`mne.io.Raw.set_meas_date` to support setting ``meas_date`` to ``None``, by `Luke Bloy`_

- Fix bug with :func:`mne.setup_volume_source_space` when ``volume_label`` was supplied where voxels slightly (in a worst case, about 37% times ``pos`` in distance) outside the voxel-grid-based bounds of regions were errantly included, by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.annotate_movement` where bad data segments, specified in ``raw.annotations``, would be handled incorrectly by `Luke Bloy`_

- Fix bug with :func:`mne.compute_source_morph` when more than one volume source space was present (e.g., when using labels) where only the first label would be interpolated when ``mri_resolution=True`` by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.compute_source_psd_epochs` and :func:`mne.minimum_norm.source_band_induced_power` raised errors when ``method='eLORETA'`` by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.apply_inverse` where the explained variance did not work for complex data by `Eric Larson`_

- Fix to enable interactive plotting with no colorbar with :func:`mne.viz.plot_evoked_topomap` by `Daniel McCloy`_

- Fix plotting with :func:`mne.viz.plot_evoked_topomap` to pre-existing axes by `Daniel McCloy`_

- The default plotting mode for :func:`mne.io.Raw.plot` and :ref:`mne browse_raw` has been changed to ``clipping=3.`` to facilitate data analysis with large deflections, by `Eric Larson`_

- PSD plots will now show non-data channels (e.g., ``misc``) if those channels are explicitly passed to ``picks``, by `Daniel McCloy`_.

- Fix handling of NaN when using TFCE in clustering functions such as :func:`mne.stats.spatio_temporal_cluster_1samp_test` by `Eric Larson`_

- Fix handling of signs when using TFCE by `Eric Larson`_

- The :class:`mne.MixedSourceEstimate` class has been clarified to contain two cortical surface source spaces, plus at least one other source space. Creating source estimates in other orderings is not supported, by `Eric Larson`_

- Fix bug where :class:`VolSourceEstimate.vertices <mne.VolSourceEstimate>` was an instance of :class:`~numpy.ndarray` instead of :class:`python:list` of one :class:`~numpy.ndarray`, by `Eric Larson`_

- Fix default to be ``foreground=None`` in :func:`mne.viz.plot_source_estimates` to use white or black text based on the background color by `Eric Larson`_

- Fix bug with :func:`mne.io.Raw.plot` where toggling all projectors did not actually take effect by `Eric Larson`_

- Fix bug with :func:`mne.read_epochs` when loading data in complex format with ``preload=False`` by `Eric Larson`_

- Fix bug with :meth:`mne.Epochs.save` where the file splitting calculations did not account for the sizes of non-data writes by `Eric Larson`_

- Fix bug with :class:`mne.Report` where the BEM section could not be toggled by `Eric Larson`_

- Fix bug when using :meth:`mne.Epochs.crop` to exclude the baseline period would break :func:`mne.Epochs.save` / :func:`mne.read_epochs` round-trip by `Eric Larson`_

- Fix bug with :func:`mne.viz.plot_bem` and :class:`mne.Report` when plotting BEM contours when MRIs are not in standard FreeSurfer orientation by `Eric Larson`_

- Fix bugs with :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` where:

  - Noise normalization factors ``weight_norm='unit-noise-gain'`` and ``weight_norm='nai'`` were computed incorrectly
  - ``pick_ori='max-power'`` computed the max-power orientation incorrectly
  - ``pick_ori='normal'`` did not compute power or noise normalization factors correctly
  - :func:`mne.beamformer.apply_lcmv_cov` did not apply whitening and projections properly

- Fix :ref:`mne setup_forward_model` to have it actually compute the BEM solution in addition to creating the BEM model by `Eric Larson`_

- Fix bug with :func:`mne.io.read_raw_edf` where null bytes were not properly handled, causing an error when opening a file by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.nirs.scalp_coupling_index` where filter transition was incorrectly assigned by `Robert Luke`_

- Fix bug with :func:`mne.make_forward_dipole` where :func:`mne.write_forward_solution` could not be used by `Eric Larson`_

- Fix bug with :meth:`mne.io.Raw.plot` when ``scalings='auto'`` where bad data would prevent channel plotting by `Eric Larson`_

- Default ``border`` and ``extrapolate`` arguments for :func:`mne.Evoked.plot_topomap` and related functions were changed from ``0.`` to ``'mean'`` and ``'box'`` to ``'auto'``, respectively, to help more accurately reflect sensor geometries and boundary conditions. ``extrapolate='auto'`` uses ``extrapolate='local'`` for MEG data and ``extrapolate='head'`` otherwise, by `Eric Larson`_

- Fix bug that prevents ``n_jobs`` from being a NumPy integer type, by `Daniel McCloy`_.

- Fix bug with :func:`mne.io.read_raw_gdf` where birthdays were not parsed properly, leading to an error by `Svea Marie Meyer`_

- Fix bug with :func:`mne.io.read_raw_edf` where recording ID was not read properly for non-ASCII characters by `Lx37`_

- Fix bug with :func:`mne.get_volume_labels_from_aseg` where the returned labels were alphabetical instead of reflecting their volumetric ID-based order by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.find_bad_channels_maxwell` where good data of exactly ``step`` duration would lead to an error by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.find_bad_channels_maxwell` where indices were not handled properly when MEG channels were not first in the raw instance, and logging messages incorrectly reported the interval used by `Eric Larson`_

- Make :func:`mne.set_config` accept path-like input values by `Richard Höchenberger`_

- Fix bug with :func:`mne.write_labels_to_annot` and :func:`mne.datasets.fetch_hcp_mmp_parcellation` where label name strings were not properly terminated, leading to problems loading in FreeSurfer by `Eric Larson`_

- Fix bug with :func:`mne.beamformer.make_dics` where complex conjugates were not applied properly by `Britta Westner`_ and `Eric Larson`_

- Fix bug with :func:`mne.bem.make_watershed_bem` where the RAS coordinates of watershed bem surfaces were not updated correctly from the volume file by `Yu-Han Luo`_

- Fix bug in FieldTrip reader functions when channels are missing in the ``info`` object by `Thomas Hartmann`_

- Throw proper error when trying to import FieldTrip Epochs data with non-uniform time for trials by `Thomas Hartmann`_

- Throw proper error when trying to import FieldTrip data saved by an old, incompatible version by `Thomas Hartmann`_

- Fix bug in :func:`mne.read_epochs_fieldtrip` when importing data without a ``trialinfo`` field by `Thomas Hartmann`_

- Fix bug in :meth:`mne.preprocessing.ICA.plot_properties` where time series plot doesn't start at the proper tmin by `Teon Brooks`_

API
~~~

- Python 3.5 is no longer supported, Python 3.6+ is required, by `Eric Larson`_

- The ``normalize_fwd`` argument of :func:`mne.beamformer.make_dics` has been deprecated in favor of ``depth``, by `Eric Larson`_

- Add ``n_cols`` parameter to :meth:`mne.preprocessing.ICA.plot_scores` to allow plotting scores in multiple columns, by `Luke Bloy`_

- In :func:`mne.stats.permutation_cluster_test` and :func:`mne.stats.permutation_cluster_1samp_test` the default parameter value ``out_type='mask'`` has changed to ``None``, which in 0.21 means ``'mask'`` but will change to mean ``'indices'`` in the next version, by `Daniel McCloy`_

- ``vmin`` and ``vmax`` parameters are deprecated in :meth:`mne.Epochs.plot_psd_topomap` and :func:`mne.viz.plot_epochs_psd_topomap`; use new ``vlim`` parameter instead, by `Daniel McCloy`_.

- The method ``stc_mixed.plot_surface`` for a :class:`mne.MixedSourceEstimate` has been deprecated in favor of :meth:`stc.surface().plot(...) <mne.MixedSourceEstimate.surface>` by `Eric Larson`_

- Add ``use_dev_head_trans`` parameter to :func:`mne.preprocessing.annotate_movement` to allow choosing the device to head transform is used to define the fixed cHPI coordinates By `Luke Bloy`_

- The function ``mne.channels.read_dig_captrack`` will be deprecated in version 0.22 in favor of :func:`mne.channels.read_dig_captrak` to correct the spelling error: "captraCK" -> "captraK", by `Stefan Appelhoff`_

- The ``threshold`` argument in :meth:`mne.preprocessing.ICA.find_bads_ecg` defaults to ``None`` in version 0.21 but will change to ``'auto'`` in 0.22 by `Yu-Han Luo`_
