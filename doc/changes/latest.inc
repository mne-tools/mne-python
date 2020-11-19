.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. _current:

Current (0.22.dev0)
-------------------

.. |Eduard Ort| replace:: **Eduard Ort**

.. |Aniket Pradhan| replace:: **Aniket Pradhan**

.. |Tod Flak| replace:: **Tod Flak**

.. |Victoria Peterson| replace:: **Victoria Peterson**


Enhancements
~~~~~~~~~~~~
- Add :class:`mne.decoding.SSD` for spatial filtering with spatio-spectral-decomposition (:gh:`7070` **by new contributor** |Victoria Peterson|_ and `Denis Engemann`_)

- Add options to use labels in :func:`mne.minimum_norm.get_point_spread` and :func:`mne.minimum_norm.get_cross_talk` (:gh:`8275` by `Olaf Hauk`_)

- Update ``surfaces`` argument in :func:`mne.viz.plot_alignment` to allow dict for transparency values, and set default for sEEG data to have transparency (:gh:`8445` by `Keith Doelling`_)

- Add ``n_pca_components`` argument to :func:`mne.viz.plot_ica_overlay` (:gh:`8351` by `Eric Larson`_)

- Add :func:`mne.stc_near_sensors` to facilitate plotting ECoG data (:gh:`8190` by `Eric Larson`_)

- Add ``proj`` argument to :func:`mne.make_fixed_length_epochs` (:gh:`8351` by `Eric Larson`_)

- Add :func:`mne.preprocessing.realign_raw` to realign simultaneous raw recordings in the presence of clock drift (:gh:`8539` by `Eric Larson`_)

- Reduce memory usage of volume source spaces (:gh:`8379` by `Eric Larson`_)

- Speed up heavy use of :meth:`mne.SourceMorph.apply` for volumetric source spaces by use of the method :meth:`mne.SourceMorph.compute_vol_morph_mat` (:gh:`8366` by `Eric Larson`_)

- Add support for non-uniform ``zooms`` (e.g., when using a surrogate MRI via :func:`mne.scale_mri`) in volumetric morphing  (:gh:`8642` by `Eric Larson`_)

- In :func:`mne.compute_source_morph` ``zooms`` are no longer required to match the spacing of ``src_to``, which is useful to ensure the morphing is accurate when the ``src_to`` spacing is large (e.g., 1 cm) (:gh:`8642` by `Eric Larson`_)

- Add volumetric source space support to :func:`mne.labels_to_stc` (:gh:`8447` by `Eric Larson`_)

- Speed up :class:`mne.decoding.TimeDelayingRidge` with edge correction using Numba (:gh:`8323` by `Eric Larson`_)

- Add :meth:`mne.Epochs.reset_drop_log_selection` to facilitate writing epochs with many ignored entries in their drop log (:gh:`8449` by `Eric Larson`_)

- Add sEEG source visualization using :func:`mne.stc_near_sensors` and sEEG working tutorial (:gh:`8402` by `Eric Larson`_ and `Adam Li`_)

- Add :meth:`mne.channels.DigMontage.get_positions`, which will return a dictionary of channel positions, coordinate frame and fiducial locations (:gh:`8460` by `Adam Li`_)

- Add support for writing digitization points in a coordinate frame other than head in :meth:`mne.channels.DigMontage.save` (:gh:`8532` by `Eric Larson`_)

- Add ``picks`` parameter to :func:`mne.preprocessing.fix_stim_artifact` to specify which channel needs to be fixed (:gh:`8482` by `Alex Gramfort`_)

- Further improved documentation building instructions and execution on Windows (:gh:`8502` by `kalenkovich`_ and `Eric Larson`_)

- Add option to disable TQDM entirely with ``MNE_TQDM='off'`` (:gh:`8515` by `Eric Larson`_)

- `mne.preprocessing.ICA.plot_sources` now displays an `mne.preprocessing.ICA.plot_properties` window when right-clicking on component names on the y-axis (:gh:`8381` by `Daniel McCloy`_)


Bugs
~~~~
- Fix :func:`mne.io.read_raw_curry` to deal with Curry datasets that have channels that are listed in the labels file, but which are absent from the saved data file (e.g. 'Ref' channel).  Also now populates info['meas_date'] if possible (:gh:`8400` **by new contributor** |Tod Flak|_)

- Fix bug with mne.io.egi.tests/test_egi.py where it mandatorily downloaded testing data when it was not necessary (:gh:`8474` **by new contributor** |Aniket Pradhan|_)

- Fix bug with reading split files that have dashes in the filename (:gh:`8339` **by new contributor** |Eduard Ort|_)

- Fix bug with `~mne.viz.plot_epochs_image` when ``order`` is supplied and multiple conditions are plotted (:gh:`8377` by `Daniel McCloy`_ )

- Fix bug with :func:`mne.viz.plot_source_estimates` when using the PyVista backend where singleton time points were not handled properly (:gh:`8285` by `Eric Larson`_)

- Fix bug when passing ``axes`` to plotting functions, :func:`matplotlib.pyplot.tight_layout` will not be called when the figure was created using a constrained layout (:gh:`8344` by `Eric Larson`_)

- Fix bug with compensated CTF data when picking channels without preload (:gh:`8318` by `Eric Larson`_)

- Fix bug when merging fNIRS channels in :func:`mne.viz.plot_evoked_topomap` and related functions (:gh:`8306` by `Robert Luke`_)

- Fix bug where events could overflow when writing to FIF (:gh:`8448` by `Eric Larson`_)

- :func:`mne.io.read_raw_edf` now supports EDF files with invalid recording dates (:gh:`8283` by `Clemens Brunner`_)

- Fix bug with :func:`mne.io.Raw.save` when using ``split_naming='bids'`` where non-split files would still be named ``name_split-01_meg.fif`` instead of the requested ``name_meg.fif`` (:gh:`8464` by `Alex Gramfort`_ and `Eric Larson`_)

- Fix bug with :class:`mne.preprocessing.ICA` where ``n_pca_components`` as a :class:`python:float` would give the number of components that explained less than or equal to the given variance. It now gives greater than the given number for better usability and consistency with :class:`sklearn.decomposition.PCA`. Generally this will mean that one more component will be included (:gh:`8326` by `Eric Larson`_)

- Fix bug with :class:`mne.preprocessing.ICA` where projections were not tracked properly (:gh:`8343` by `Eric Larson`_)

- Fix bug with :func:`mne.preprocessing.read_ica_eeglab` where full-rank data were not handled properly (:gh:`8326` by `Eric Larson`_)

- Fix bug with :ref:`somato-dataset` where the BEM was not included (:gh:`8317` by `Eric Larson`_)

- Fix bug with coordinate frames when performing volumetric morphs via :func:`mne.compute_source_morph` and :meth:`mne.SourceMorph.apply` that could lead to ~5 mm bias (:gh:`8642` by `Eric Larson`_)

- Fix missing documentation of :func:`mne.io.read_raw_nihon` in :ref:`tut-imorting-eeg-data` (:gh`8320` by `Adam Li`_)

- Fix bug with :func:`mne.add_reference_channels` when :func:`mne.io.Raw.reorder_channels` or related methods are used afterward (:gh:`8303`, :gh:`#8484` by `Eric Larson`_)

- Fix bug where the ``verbose`` arguments to :meth:`mne.Evoked.apply_baseline` and :meth:`mne.Epochs.apply_baseline` were not keyword-only (:gh:`8349` by `Eric Larson`_)

- ``ICA.max_pca_components`` will not be altered by calling `~mne.preprocessing.ICA.fit` anymore. Instead, the new attribute ``ICA.max_pca_components_`` will be set (:gh:`8321` by `Richard Höchenberger`_)

- Fix bug that `~mne.viz.plot_ica_overlay` would sometimes not create red traces (:gh:`8341` by `Richard Höchenberger`_)

- Fix bug with :class:`~mne.preprocessing.ICA` where ``n_components=None, n_pca_components=None`` could lead to unstable unmixing matrix inversion by making ``n_components=None`` also use the lesser of ``n_components=0.999999`` and ``n_components=n_pca_components`` (:gh:`8351` by `Eric Larson`_)

- The ``ica.n_pca_components`` property is no longer be updated during :meth:`mne.preprocessing.ICA.fit`, instead ``ica.n_components_`` will be added to the instance (:gh:`8351` by `Eric Larson`_)

- Pass ``rank`` everyhwere in forward preparation for source imaging. This bug affected sparse solvers when using maxfilter data  (:gh:`8368` by `Alex Gramfort`_)

- Fix bug in :func:`mne.viz.plot_alignment` where ECoG and sEEG channels were not plotted and fNIRS channels were always plotted in the head coordinate frame (:gh:`8393` by `Eric Larson`_)

- Fix bug in :func:`mne.set_bipolar_reference` where ``ch_info`` could contain invalid channel information keys (:gh:`8416` by `Eric Larson`_)

- When reading BrainVision raw data, the channel units and types were sometimes not inferred correctly (:gh:`8434` by `Richard Höchenberger`_)

- Attempting to remove baseline correction from preloaded `~mne.Epochs` will now raise an exception (:gh:`8435` by `Richard Höchenberger`_)

- :meth:`mne.Report.parse_folder` will now correctly handle split FIFF files (:gh:`8486`, :gh:`8491` by `Richard Höchenberger`_)

- Fix bug where BrainVision channel names, event types, and event descriptions containing commas were incorrectly parsed (:gh:`8492` by `Stefan Appelhoff`_)

- Fix bug in :func:`mne.preprocessing.compute_fine_calibration` where the magnetometer calibration coefficients were computed incorrectly (:gh:`8522` by `Eric Larson`_)

- Fix bug in :func:`mne.io.read_raw_eeglab` where empty event durations led to an error (:gh:`8384` by `Mikołaj Magnuski`_)

- Fix reading GDF files with excluded channels in :func:`mne.io.read_raw_gdf` (:gh:`8520` by `Clemens Brunner`_)

API changes
~~~~~~~~~~~

- Added argument ``colors`` to `mne.grow_labels` (:gh:`8519` by `Olaf Hauk`_)

- Added `mne.SourceEstimate.apply_baseline` method for baseline-correction of source estimates (:gh:`8452` by `Olaf Hauk`_)

- Minimum required versions were increased for core dependencies NumPy (1.15.4), SciPy (1.1.0), and Matplotlib (3.0) and for the optional dependencies scikit-learn (0.20.2) and pandas (0.23.4) (:gh:`8374` by `Eric Larson`_)

- The parameter ``on_split_missing`` has been added to :func:`mne.io.read_raw_fif` and its default will change from ``'warn'`` to ``'raise'`` in 0.23, by (:gh:`8357` `Eric Larson`_)

- The ``max_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_components`` during initialization and ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` instead (:gh:`8351` by `Eric Larson`_)

- The ``n_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` (:gh:`8356` by `Eric Larson`_)

- The ``trans`` argument of :func:`mne.extract_label_time_course` is deprecated and will be removed in 0.23 as it is no longer necessary (:gh:`8389` by `Eric Larson`_)

- New `mne.viz.Brain.set_time` method to set the displayed time in seconds (:gh:`8415` by `Daniel McCloy`_)

- Update the ``backend`` parameter of :func:`mne.viz.plot_source_estimates` to integrate ``pyvista`` (:gh:`8395` by `Guillaume Favelier`_)

- Add ``group_by`` parameter to `mne.viz.plot_epochs` and `mne.Epochs.plot` to allow displaying channel data by sensor position (:gh:`8381` by `Daniel McCloy`_)

- Add ``proj='reconstruct'`` to :meth:`mne.Evoked.plot` and related functions to apply projectors and then undo the signal bias using field mapping by `Eric Larson`_

- When picking a subset of channels, or when dropping channels from `~mne.io.Raw`, `~mne.Epochs`, or `~mne.Evoked`, projectors that can only be applied to the removed channels will now be dropped automatically by `Richard Höchenberger`_

- :class:`mne.Report` now can add topomaps of SSP projectors to the generated report. This behavior can be toggled via the new ``projs`` argument by `Richard Höchenberger`_

- Add function :func:`mne.channels.combine_channels` to combine channels from Raw, Epochs, or Evoked according to ROIs (combinations including mean, median, or standard deviation; can also use a callable) by `Johann Benerradi`_

- Improved documentation building instructions and execution on Windows by `Eric Larson`_, `kalenkovich`_ and `Martin Schulz`_

- When passing a list of `~mne.Evoked` objects to `~mne.viz.plot_compare_evokeds`, each evoked's ``.comment`` attribute will be used to label the trace. If ``.comment`` is empty, a 1-based index is assigned as the label by  `Richard Höchenberger`_

- Speed up :func:`mne.stats.summarize_clusters_stc` using Numba by `Yu-Han Luo`_

- Add ``reject_by_annotation=True`` to :func:`mne.make_fixed_length_epochs` and :meth:`mne.preprocessing.ICA.plot_properties` to reject bad data segments based on annotation by `Yu-Han Luo`_

- Add reader for optical imaging data recorded using ISS Imgagent I/II hardware and BOXY recording software in :func:`mne.io.read_raw_boxy` by `Kyle Mathewson`_ and `Jonathan Kuziek`_

- `~mne.Report.parse_folder` now accepts a path-like folder name (it used to work with strings only) by `Alex Gramfort`_

Bug
~~~

- Fix bug for writing and reading complex evoked data modifying :func:`mne.write_evokeds` and :func:`mne.read_evokeds` by `Lau Møller Andersen`_

- Fix bug by adding error message when trying to save complex stc data in a non.-h5 format :meth:`mne.VolSourceEstimate.save` by `Lau Møller Andersen`_

- Fix bug with :func:`mne.preprocessing.ICA.find_bads_eog` when more than one EOG components are present by `Christian O'Reilly`_

- Fix bug to permit :meth:`stc.project('nn', src) <mne.VectorSourceEstimate.project>` to be applied after ``stc`` was restricted to an :class:`mne.Label` by `Luke Bloy`_

- Fix bug with :func:`mne.io.Raw.set_meas_date` to support setting ``meas_date`` to ``None``, by `Luke Bloy`_

- Fix bug with :func:`mne.setup_volume_source_space` when ``volume_label`` was supplied where voxels slightly (in a worst case, about 37% times ``pos`` in distance) outside the voxel-grid-based bounds of regions were errantly included, by `Eric Larson`_

- Fix bug with :func:`mne.io.read_raw_ctf` when reference magnetometers have the compensation grade marked by `Eric Larson`_

- Fix bug with `mne.SourceSpaces.export_volume` with ``use_lut=False`` where no values were written by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.annotate_movement` where bad data segments, specified in ``raw.annotations``, would be handled incorrectly by `Luke Bloy`_

- Fix bug with :func:`mne.compute_source_morph` when more than one volume source space was present (e.g., when using labels) where only the first label would be interpolated when ``mri_resolution=True`` by `Eric Larson`_

- Fix bug with :func:`mne.compute_source_morph` when morphing to a volume source space when ``src_to`` is used and the destination subject is not ``fsaverage`` by `Eric Larson`_

- Fix bug with :func:`mne.compute_source_morph` where outermost voxels in the destination source space could be errantly omitted by `Eric Larson`_

- Fix bug with :func:`mne.compute_source_morph` where complex data was cast to real when doing a volumetric morph by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.compute_source_psd_epochs` and :func:`mne.minimum_norm.source_band_induced_power` raised errors when ``method='eLORETA'`` by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.apply_inverse` where the explained variance did not work for complex data by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.compute_current_source_density` where values were not properly computed; maps should now be more focal, by `Alex Rockhill`_ and `Eric Larson`_

- Fix bug with :func:`mne.combine_evoked` where equal-weighted averages were wrongly computed as equal-weighted sums, by `Daniel McCloy`_

- Fix bug with setting HTML classes when using :meth:`mne.Report.add_bem_to_section` by `Eric Larson`_

- Fix bug with convex-hull based MEG helmet creation where the face area could be covered up, by `Eric Larson`_

- Fix to enable interactive plotting with no colorbar with :func:`mne.viz.plot_evoked_topomap` by `Daniel McCloy`_

- Fix plotting with :func:`mne.viz.plot_evoked_topomap` to pre-existing axes by `Daniel McCloy`_

- Fix bug with :func:`mne.viz.plot_vector_source_estimates` using the PyVista backend with ``time_viewer=True`` when updating the arrow colormaps by `Eric Larson`_

- The default plotting mode for :func:`mne.io.Raw.plot` and :ref:`mne browse_raw` has been changed to ``clipping=3.`` to facilitate data analysis with large deflections, by `Eric Larson`_

- PSD plots will now show non-data channels (e.g., ``misc``) if those channels are explicitly passed to ``picks``, by `Daniel McCloy`_.

- Fix bug with :func:`mne.time_frequency.read_tfrs` where ``info['meas_date']`` was not parsed correctly, by `Eric Larson`_

- Fix bug with :func:`mne.time_frequency.tfr_array_stockwell` where inputs were not properly validated by `Eric Larson`_

- Fix handling of NaN when using TFCE in clustering functions such as :func:`mne.stats.spatio_temporal_cluster_1samp_test` by `Eric Larson`_

- Fix handling of signs when using TFCE by `Eric Larson`_

- The :class:`mne.MixedSourceEstimate` class has been clarified to contain two cortical surface source spaces, plus at least one other source space. Creating source estimates in other orderings is not supported, by `Eric Larson`_

- Fix bug where :class:`VolSourceEstimate.vertices <mne.VolSourceEstimate>` was an instance of :class:`~numpy.ndarray` instead of :class:`python:list` of one :class:`~numpy.ndarray`, by `Eric Larson`_

- Fix default to be ``foreground=None`` in :func:`mne.viz.plot_source_estimates` to use white or black text based on the background color by `Eric Larson`_

- Fix bug with writing EGI and CTF `mne.Info` to H5 format, e.g., with `mne.time_frequency.AverageTFR.save` by `Eric Larson`_

- Fix bug with :func:`mne.io.Raw.plot` where toggling all projectors did not actually take effect by `Eric Larson`_

- Fix bug with :func:`mne.read_epochs` when loading data in complex format with ``preload=False`` by `Eric Larson`_

- Fix bug with :meth:`mne.Epochs.save` where the file splitting calculations did not account for the sizes of non-data writes by `Eric Larson`_

- Fix bug with :class:`mne.Epochs` when metadata was not subselected properly when ``event_repeated='drop'`` by `Eric Larson`_

- Fix bug with :class:`mne.Epochs` where ``epochs.drop_log`` was a list of list of str rather than an immutable tuple of tuple of str (not meant to be changed by the user) by `Eric Larson`_

- Fix bug with :class:`mne.Report` where the BEM section could not be toggled by `Eric Larson`_

- Fix bug when using :meth:`mne.Epochs.crop` to exclude the baseline period would break :func:`mne.Epochs.save` / :func:`mne.read_epochs` round-trip by `Eric Larson`_

- Fix bug with `mne.Epochs.subtract_evoked` where using decimated epochs would lead to an error by `Eric Larson`_

- Fix bug with :func:`mne.viz.plot_bem` and :class:`mne.Report` when plotting BEM contours when MRIs are not in standard FreeSurfer orientation by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.make_inverse_operator` where it would warn even when an explicit ``rank`` was used by `Eric Larson`_

- Fix bugs with :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` where:

  - Noise normalization factors ``weight_norm='unit-noise-gain'`` and ``weight_norm='nai'`` were computed incorrectly
  - ``pick_ori='max-power'`` computed the max-power orientation incorrectly
  - ``pick_ori='normal'`` did not compute power or noise normalization factors correctly
  - :func:`mne.beamformer.apply_lcmv_cov` did not apply whitening and projections properly

- Fix :ref:`mne setup_forward_model` to have it actually compute the BEM solution in addition to creating the BEM model by `Eric Larson`_

- Fix bug with :func:`mne.io.read_raw_edf` where null bytes were not properly handled, causing an error when opening a file by `Eric Larson`_

- Fix bug with :func:`mne.Report` where unicode characters were not rendered properly (encoding for HTML was not set) by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.nirs.scalp_coupling_index` where filter transition was incorrectly assigned by `Robert Luke`_

- Fix bug with :func:`mne.make_forward_dipole` where :func:`mne.write_forward_solution` could not be used by `Eric Larson`_

- Fix bug with :meth:`mne.io.Raw.plot` when ``scalings='auto'`` where bad data would prevent channel plotting by `Eric Larson`_

- Default ``border`` and ``extrapolate`` arguments for :func:`mne.Evoked.plot_topomap` and related functions were changed from ``0.`` to ``'mean'`` and ``'box'`` to ``'auto'``, respectively, to help more accurately reflect sensor geometries and boundary conditions. ``extrapolate='auto'`` uses ``extrapolate='local'`` for MEG data and ``extrapolate='head'`` otherwise, by `Eric Larson`_

- Fix bug that prevents ``n_jobs`` from being a NumPy integer type, by `Daniel McCloy`_.

- Fix bug with :func:`mne.epochs.average_movements` where epoch weights were computed using all basis vectors instead of the internal basis only by `Eric Larson`_

- Fix bug with :func:`mne.io.read_raw_gdf` where birthdays were not parsed properly, leading to an error by `Svea Marie Meyer`_

- Fix bug with :func:`mne.io.read_raw_edf` where recording ID was not read properly for non-ASCII characters by `Lx37`_

- Fix bug with :func:`mne.get_volume_labels_from_aseg` where the returned labels were alphabetical instead of reflecting their volumetric ID-based order by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.find_bad_channels_maxwell` where good data of exactly ``step`` duration would lead to an error by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.find_bad_channels_maxwell` where indices were not handled properly when MEG channels were not first in the raw instance, and logging messages incorrectly reported the interval used by `Eric Larson`_

- Make :func:`mne.set_config` accept path-like input values by `Richard Höchenberger`_

- Fix bug with :func:`mne.write_labels_to_annot` and :func:`mne.datasets.fetch_hcp_mmp_parcellation` where label name strings were not properly terminated, leading to problems loading in FreeSurfer by `Eric Larson`_

- Fix bug with :func:`mne.beamformer.make_dics` where complex conjugates were not applied properly by `Britta Westner`_ and `Eric Larson`_

- Fix bug with :func:`mne.bem.make_watershed_bem` where the RAS coordinates of watershed bem surfaces were not updated correctly from the volume file by `Yu-Han Luo`_

- Fix bug with :meth:`mne.io.Raw.get_channel_types` and related methods where the ordering of ``picks`` was not preserved, by `Eric Larson`_

- Fix bug with :meth:`mne.io.Raw.plot_psd` with ``average=False`` and multiple channel types where channel locations were not shown properly by `Eric Larson`_

- Fix bug in FieldTrip reader functions when channels are missing in the ``info`` object by `Thomas Hartmann`_

- Throw proper error when trying to import FieldTrip Epochs data with non-uniform time for trials by `Thomas Hartmann`_

- Throw proper error when trying to import FieldTrip data saved by an old, incompatible version by `Thomas Hartmann`_

- Fix bug in :func:`mne.read_epochs_fieldtrip` when importing data without a ``trialinfo`` field by `Thomas Hartmann`_

- Fix bug in :meth:`mne.preprocessing.ICA.plot_properties` where time series plot doesn't start at the proper tmin by `Teon Brooks`_

- Fix bug with :meth:`mne.preprocessing.ICA.plot_properties` where a :class:`mne.io.Raw` object with annotations would lead to an error by `Yu-Han Luo`_

API
~~~

- Python 3.5 is no longer supported, Python 3.6+ is required, by `Eric Larson`_

- ``adjacency`` has replaced ``connectivity`` in the names of:

  1. Arguments to clustering functions, such as `mne.stats.permutation_cluster_test`, and
  2. Function names for defining adjacency, such as `mne.spatio_temporal_src_adjacency` replacing ``mne.spatio_temporal_src_connectivity``.

  "connectivity" is now reserved for discussions of functional and effective connectivity of the brain, and "adjacency" for source or sensor neighbor definitions for cluster-based analyses, by `Eric Larson`_.

- The default for the ``standardize_names`` argument of :func:`mne.io.read_raw_kit` will change from ``True`` to ``False`` in 0.22, by `Eric Larson`_

- The ``normalize_fwd`` argument of :func:`mne.beamformer.make_dics` has been deprecated in favor of ``depth``, by `Eric Larson`_

- Add ``n_cols`` parameter to :meth:`mne.preprocessing.ICA.plot_scores` to allow plotting scores in multiple columns, by `Luke Bloy`_

- In :func:`mne.stats.permutation_cluster_test` and :func:`mne.stats.permutation_cluster_1samp_test` the default parameter value ``out_type='mask'`` has changed to ``None``, which in 0.21 means ``'mask'`` but will change to mean ``'indices'`` in the next version, by `Daniel McCloy`_

- The default window size set by ``filter_length`` when ``method='spectrum_fit'`` in :meth:`mne.io.Raw.notch_filter` will change from ``None`` (use whole file) to ``'10s'`` in 0.22, by `Eric Larson`_

- ``vmin`` and ``vmax`` parameters are deprecated in :meth:`mne.Epochs.plot_psd_topomap` and :func:`mne.viz.plot_epochs_psd_topomap`; use new ``vlim`` parameter instead, by `Daniel McCloy`_.

- The method ``stc_mixed.plot_surface`` for a :class:`mne.MixedSourceEstimate` has been deprecated in favor of :meth:`stc.surface().plot(...) <mne.MixedSourceEstimate.surface>` by `Eric Larson`_

- The method ``stc.normal`` for :class:`mne.VectorSourceEstimate` has been deprecated in favor of :meth:`stc.project('nn', src) <mne.VectorSourceEstimate.project>` by `Eric Larson`_

- Add ``use_dev_head_trans`` parameter to :func:`mne.preprocessing.annotate_movement` to allow choosing the device to head transform is used to define the fixed cHPI coordinates by `Luke Bloy`_

- The function ``mne.channels.read_dig_captrack`` will be deprecated in version 0.22 in favor of :func:`mne.channels.read_dig_captrak` to correct the spelling error: "captraCK" -> "captraK", by `Stefan Appelhoff`_

- The ``threshold`` argument in :meth:`mne.preprocessing.ICA.find_bads_ecg` defaults to ``None`` in version 0.21 but will change to ``'auto'`` in 0.22 by `Yu-Han Luo`_

- The default argument ``meg=True`` in :func:`mne.pick_types` will change to ``meg=False`` in version 0.22 by `Clemens Brunner`_

- Parameter ``event_colors`` in `mne.viz.plot_epochs` and `mne.Epochs.plot` is deprecated, replaced by ``event_color`` which is consistent with `mne.viz.plot_raw` and provides greater flexibility (:gh:`8381` by `Daniel McCloy`_)
