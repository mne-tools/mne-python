:orphan:

.. include:: links.inc
.. _whats_new:

What's new
==========
.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Changelog" for new features
   - "Bug" for bug fixes
   - "API" for backward-incompatible changes

.. currentmodule:: mne

.. _current:

Current
-------

Changelog
~~~~~~~~~

Bug
~~~

- Fix bug in handling of :class:`mne.Evoked` types that were not produced by MNE-Python (e.g., alternating average) by `Eric Larson`_

API
~~~

.. _changes_0_18:

Version 0.18
------------

Changelog
~~~~~~~~~

- Add ``event_id='auto'`` in :func:`mne.events_from_annotations` to accommodate Brainvision markers by `Jona Sassenhagen`_, `Joan Massich`_ and `Eric Larson`_

- Add example on how to simulate raw data using subject anatomy, by `Ivana Kojcic`_,`Eric Larson`_,`Samuel Deslauriers-Gauthier`_ and`Kostiantyn Maksymenko`_

- :func:`mne.beamformer.apply_lcmv_cov` returns static source power after supplying a data covariance matrix to the beamformer filter by `Britta Westner`_ and `Marijn van Vliet`_

- Add ``butterfly`` and ``order`` arguments to :func:`mne.viz.plot_epochs` and offer separated traces for non-meg data (seeg, eeg, ecog) in butterfly view by `Stefan Repplinger`_ and `Eric Larson`_

- :meth:`mne.Epochs.get_data` now takes a ``picks`` parameter by `Jona Sassenhagen`_

- :func:`~mne.viz.plot_compare_evokeds` will generate topo plots if ``axes='topo'`` by `Jona Sassenhagen`_

- ``mne.viz.iter_topography`` can yield an additional axis, e.g., for plotting legends by `Jona Sassenhagen`_ and `Daniel McCloy`_

- Default plot title reflects channel type when ``picks`` is a channel type in :func:`~mne.viz.plot_compare_evokeds` by `Daniel McCloy`_

- Color scale limits in :func:`~mne.viz.plot_topo_image_epochs` are now computed separately per channel type in combined mag/grad plots, by `Daniel McCloy`_

- :func:`mne.simulation.simulate_stc` now allows for label overlaps by `Nathalie Gayraud`_, and `Ivana Kojcic`_

- Add ``long_format`` option to the pandas dataframe exporters, e.g :meth:`mne.Epochs.to_data_frame` by `Denis Engemann`_

- Add example on how to load standard montage :ref:`plot_montage` by `Joan Massich`_

- Add new tutorial on :ref:`tut-eeg-fsaverage-source-modeling` by `Alex Gramfort`_, and `Joan Massich`_

- Add :meth:`mne.Epochs.apply_hilbert` and :meth:`mne.Evoked.apply_hilbert` by `Eric Larson`_

- Add convenience ``fsaverage`` subject dataset fetcher / updater :func:`mne.datasets.fetch_fsaverage` by `Eric Larson`_

- Add ``fmin`` and ``fmax`` argument to :meth:`mne.time_frequency.AverageTFR.crop` and to :meth:`mne.time_frequency.EpochsTFR.crop` to crop TFR objects along frequency axis by `Dirk Gütlin`_

- Add support to :func:`mne.read_annotations` to read CNT formats by `Joan Massich`_

- Add ``reject`` parameter to :meth:`mne.preprocessing.ICA.plot_properties` to visualize rejected epochs by `Antoine Gauthier`_

- Add support for picking channels using channel name and type strings to functions with ``picks`` arguments, along with a convenience :meth:`mne.io.Raw.pick`, :meth:`mne.Epochs.pick`, and :meth:`mne.Evoked.pick` method, by `Eric Larson`_

- Add new tutorial on :ref:`tut-sleep-stage-classif` by `Alex Gramfort`_, `Stanislas Chambon`_ and `Joan Massich`_

- Add data fetchers for polysomnography (PSG) recordings from Physionet (:func:`mne.datasets.sleep_physionet.age.fetch_data` and :func:`mne.datasets.sleep_physionet.temazepam.fetch_data`) by `Alex Gramfort`_ and `Joan Massich`_

- Add envelope correlation code in :func:`mne.connectivity.envelope_correlation` by `Denis Engemann`_, `Sheraz Khan`_, and `Eric Larson`_

- Add option to toggle all projectors in :meth:`mne.io.Raw.plot` and related functions by `Eric Larson`_

- Add support for indexing, slicing, and iterating :class:`mne.Annotations` by `Joan Massich`_

- :meth:`mne.io.Raw.plot` now uses the lesser of ``n_channels`` and ``raw.ch_names``, by `Joan Massich`_

- Add support for FIR filtering in :meth:`mne.io.Raw.plot` and :ref:`gen_mne_browse_raw` by passing ``filtorder=0`` or ``--filtorder 0``, respectively, by `Eric Larson`_

- Add ``chunk_duration`` parameter to :func:`mne.events_from_annotations` to allow multiple events from a single annotation by `Joan Massich`_

- Add :class:`mne.simulation.SourceSimulator` class to simplify simulating SourceEstimates, by `Samuel Deslauriers-Gauthier`_, `Kostiantyn Maksymenko`_, `Nathalie Gayraud`_, `Ivana Kojcic`_, `Alex Gramfort`_, and `Eric Larson`_

- :func:`mne.io.read_raw_edf` now detects analog stim channels labeled ``'STATUS'`` and sets them as stim channel. :func:`mne.io.read_raw_edf` no longer synthesize TAL annotations into stim channel but stores them in ``raw.annotations`` when reading by `Joan Massich`_

- Add `mne.simulation.add_noise` for ad-hoc noise addition to `io.Raw`, `Epochs`, and `Evoked` instances, by `Eric Larson`_

- Add ``drop_refs=True`` parameter to :func:`set_bipolar_reference` to drop/keep anode and cathode channels after applying the reference by `Cristóbal Moënne-Loccoz`_.

- Add processing of reference MEG channels to :class:`mne.preprocessing.ICA` by `Jeff Hanna`_

- Add use of :func:`scipy.signal.windows.dpss` for faster multitaper window computations in PSD functions by `Eric Larson`_

- Add :func:`mne.morph_labels` to facilitate morphing label sets obtained from parcellations, by `Eric Larson`_

- Add :func:`mne.labels_to_stc` to facilitate working with label data, by `Eric Larson`_

- Add :func:`mne.label.select_sources` to simplify the selection of sources within a label, by `Samuel Deslauriers-Gauthier`_

- Add support for using :class:`mne.Info` in :func:`mne.simulation.simulate_raw` instead of :class:`mne.io.Raw` by `Eric Larson`_

- Add support for passing an iterable and stim channel values using ``stc`` parameter of :func:`mne.simulation.simulate_raw` by `Eric Larson`_

- Add ``overlap`` argument to :func:`mne.make_fixed_length_events` by `Eric Larson`_

- Add approximate distance-based ``spacing`` source space decimation algorithm to :func:`mne.setup_source_space` by `Eric Larson`_

- Add 448-labels subdivided aparc cortical parcellation by `Denis Engemann`_ and `Sheraz Khan`_

- Add option to improve rendering in :ref:`gen_mne_coreg` for modern graphics cards by `Eric Larson`_

- Add :func:`mne.preprocessing.mark_flat` to automate marking of flat channels and segments of raw data by `Eric Larson`_

- Add support for CUDA-based correlation computations and progress bars in :class:`mne.decoding.ReceptiveField` by `Eric Larson`_

- Add support for file-like objects in :func:`mne.io.read_raw_fif` as long as preloading is used by `Eric Larson`_

- Add keyboard shortcuts to nativate volume source estimates in time using (shift+)left/right arrow keys by `Mainak Jas`_

- Add option to SSP preprocessing functions (e.g., :func:`mne.preprocessing.compute_proj_eog` and :func:`mne.compute_proj_epochs`) to process MEG channels jointly with ``meg='combined'`` by `Eric Larson`_

- Add Epoch selection and metadata functionality to :class:`mne.time_frequency.EpochsTFR` using new mixin class by `Keith Doelling`_

- Add ``reject_by_annotation`` argument to :func:`mne.preprocessing.find_ecg_events` by `Eric Larson`_

- Add ``pca`` argument to return the rank-reduced whitener in :func:`mne.cov.compute_whitener` by `Eric Larson`_

- Add ``extrapolate`` argument to :func:`mne.viz.plot_topomap` for better control of extrapolation points placement by `Mikołaj Magnuski`_

- Add ``channel_wise`` argument to :func:`mne.io.Raw.apply_function` to allow applying a function on multiple channels at once by `Hubert Banville`_

- Add option ``copy='auto'`` to control data copying in :class:`mne.io.RawArray` by `Eric Larson`_

- The ``mri`` parameter in :func:`mne.setup_volume_source_space` is now automatically set to ``T1.mgz`` if ``subject`` is provided. This allows to get a :class:`mne.SourceSpaces` of kind ``volume`` more automatically. By `Alex Gramfort`_

- Add better ``__repr__`` for constants, and :class:`info['dig'] <mne.Info>` entries via :class:`mne.io.DigPoint` by `Eric Larson`_

- Allow string argument in :meth:`mne.io.Raw.drop_channels` to remove a single channel by `Clemens Brunner`_

- Add additional depth weighting options for inverse solvers (e.g., :func:`mne.inverse_sparse.gamma_map` and :func:`mne.inverse_sparse.mixed_norm`) by `Eric Larson`_

- Add depth weighting to LCMV beamformers via ``depth`` argument in :func:`mne.beamformer.make_lcmv` by `Eric Larson`_

- Allow toggling of DC removal in :meth:`mne.io.Raw.plot` by pressing the 'd' key by `Clemens Brunner`_

- Improved clicking in :meth:`mne.io.Raw.plot` (left click on trace toggles bad, left click on background sets green line, right click anywhere removes green line) by `Clemens Brunner`_

- Add ``mne.realtime.LSLClient`` for realtime data acquisition with LSL streams of data by `Teon Brooks`_ and `Mainak Jas`_

- Add partial support for PyVista as a 3D backend using :func:`mne.viz.use_3d_backend` by `Guillaume Favelier`_

- Add option ``ids = None`` in :func:`mne.event.shift_time_events` for considering all events by `Nikolas Chalas`_ and `Joan Massich`_

- Add ``mne.realtime.MockLSLStream`` to simulate an LSL stream for testing and examples by `Teon Brooks`_

Bug
~~~

- Fix annotations in split fif files :func:`mne.io.read_raw_fif` by `Joan Massich`_

- Fix :meth:`mne.Epochs.plot` with ``scalings='auto'`` to properly compute channel-wise scalings by `Stefan Repplinger`_

- Fix :func:`mne.gui.coregistration` and :ref:`mne coreg <gen_mne_coreg>` crashing with segmentation fault when switching subjects by `Eric Larson`_

- Fix :func:`mne.io.read_raw_brainvision` to accommodate vmrk files which do not have any annotations by `Alexander Kovrig`_

- Fix :meth:`mne.io.Raw.plot` and :meth:`mne.Epochs.plot` to auto-scale ``misc`` channel types by default by `Eric Larson`_

- Fix filtering functions (e.g., :meth:`mne.io.Raw.filter`) to properly take into account the two elements in ``n_pad`` parameter by `Bruno Nicenboim`_

- Fix `feature_names` parameter change after fitting in :class:`mne.decoding.ReceptiveField` by `Jean-Remi King`_

- Fix index error in :func:`mne.io.read_raw_cnt` when creating stim_channel manually by `Joan Massich`_

- Fix bug with ``weight_norm='unit-gain'`` in :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` by `Britta Westner`_

- Fix 32bits annotations in :func:`mne.io.read_raw_cnt` by `Joan Massich`_

- Fix :func:`mne.events_from_annotations` to ignore ``'BAD_'` and ``'EDGE_'`` annotations by default using a new default ``regexp`` by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.mark_flat` where ``raw.first_samp`` was not taken into account by `kalenkovich`_

- Fix date parsing in :func:`mne.io.read_raw_cnt` by `Joan Massich`_

- Fix topological checks and error messages for BEM surfaces in :func:`mne.make_bem_model` by `Eric Larson`_

- Fix default HTML language of :class:`mne.Report` to be ``"en-us"`` instead of ``"fr"`` and allow setting via ``report.lang`` property by `Eric Larson`_

- Fix bug where loading epochs with ``preload=True`` and subsequently using :meth:`mne.Epochs.drop_bad` with new ``reject`` or ``flat`` entries leads to improper data (and ``epochs.selection``) since v0.16.0 by `Eric Larson`_.
  If your code uses ``Epochs(..., preload=True).drop_bad(reject=..., flat=...)``, we recommend regenerating these data.

- Fix :ref:`gen_mne_flash_bem` to properly utilize ``flash30`` images when conversion from DICOM images is used, and to properly deal with non-standard acquisition affines by `Eric Larson`_

- Fix :meth:`mne.io.Raw.set_annotations` with ``annotations=None`` to create an empty annotations object with ``orig_time`` that matches the :class:`mne.io.Raw` instance by `Eric Larson`_

- Fix :func:`mne.io.read_raw_edf` returning all the annotations with the same name in GDF files by `Joan Massich`_

- Fix boundaries during plotting of raw data with :func:`mne.io.Raw.plot` and :ref:`gen_mne_browse_raw` on scaled displays (e.g., macOS with HiDPI/Retina screens) by `Clemens Brunner`_

- Fix bug where filtering was not performed with ``lowpass`` or ``highpass`` in :meth:`mne.io.Raw.plot` and :ref:`gen_mne_browse_raw` by `Eric Larson`_

- Fix :func:`mne.simulation.simulate_evoked` that was failing to simulate the noise with heterogeneous sensor types due to poor conditioning of the noise covariance and make sure the projections from the noise covariance are taken into account `Alex Gramfort`_

- Fix checking of ``data`` dimensionality in :class:`mne.SourceEstimate` and related constructors by `Eric Larson`_

- Fix :meth:`mne.io.Raw.append` annotations miss-alignment  by `Joan Massich`_

- Fix hash bug in the ``mne.io.edf`` module when installing on Windows by `Eric Larson`_

- Fix :func:`mne.io.read_raw_edf` reading duplicate channel names by `Larry Eisenman`_

- Fix :func:`set_bipolar_reference` in the case of generating all bipolar combinations and also in the case of repeated channels in both lists (anode and cathode) by `Cristóbal Moënne-Loccoz`_

- Fix missing code for computing the median when ``method='median'`` in :meth:`mne.Epochs.average` by `Cristóbal Moënne-Loccoz`_

- Fix CTF helmet plotting in :func:`mne.viz.plot_evoked_field` by `Eric Larson`_

- Fix saving of rejection parameters in :meth:`mne.Epochs.save` by `Eric Larson`_

- Fix orientations returned by :func:`mne.dipole.get_phantom_dipoles` (half were flipped 180 degrees) by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_compare_evokeds` when ``evoked.times[0] >= 0`` would cause a problem with ``vlines='auto'`` mode by `Eric Larson`_

- Fix path bugs in :func:`mne.bem.make_flash_bem` and :ref:`gen_mne_flash_bem` by `Eric Larson`_

- Fix :meth:`mne.time_frequency.AverageTFR.plot_joint` mishandling bad channels, by `David Haslacher`_ and `Jona Sassenhagen`_

- Fix :func:`mne.beamformer.make_lcmv` failing when full rank data is used (i.e., when no projection is done) with ``reg=0.``, by `Eric Larson`_

- Fix issue with bad channels ignored in :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.make_dics` by `Alex Gramfort`_

- Fix :func:`mne.compute_proj_raw` when ``duration != None`` not to apply existing proj and to avoid using duplicate raw data samples by `Eric Larson`_

- Fix ``reject_by_annotation`` not being passed internally by :func:`mne.preprocessing.create_ecg_epochs` and :ref:`mne clean_eog_ecg <gen_mne_clean_eog_ecg>` to :func:`mne.preprocessing.find_ecg_events` by `Eric Larson`_

- Fix :func:`mne.io.read_raw_edf` failing when EDF header fields (such as patient name) contained special characters, by `Clemens Brunner`_

- Fix :func:`mne.io.read_raw_eeglab` incorrectly parsing event durations by `Clemens Brunner`_

- Fix :func:`mne.io.read_raw_egi` when cropping non-preloaded EGI MFF data by `Alex Gramfort`_

- Fix :meth:`mne.io.Raw.interpolate_bads` for interpolating CTF MEG channels when reference sensors are present by `jeythekey`_

- Fix a bug in :meth:`mne.io.Raw.resample`, where resampling events could result in indices > n_times-1, by `jeythekey`_

- Fix :meth:`mne.preprocessing.ICA.score_sources` to use the ``sfreq`` of the raw data to filter rather than the ``sfreq`` when the ICA was fit, by `jeythekey`_

- Fix a bug in :class:`mne.preprocessing.ICA`, where manually setting the attribute ``ICA.exclude`` to an np.array resulted in the removal of random components when later also providing the ``exclude`` argument to any ``apply...``-method, by `jeythekey`_

- Ascending changed to descending sorting of scores for integer ``..._criterion`` arguments in :func:`mne.preprocessing.ICA.detect_artifacts` and :func:`mne.preprocessing.run_ica`, as it used to  be documented; the docstring in these functions was corrected for float ``..._criterion`` arguments, by `jeythekey`_

API
~~~

- Deprecate ``cov, iir_params, blink, ecg, chpi, random_state`` and support for :class:`mne.io.Raw` instance inputs in :func:`mne.simulation.simulate_raw`; use :func:`mne.simulation.add_noise`, :func:`mne.simulation.add_ecg`, :func:`mne.simulation.add_eog`, and :func:`mne.simulation.add_chpi` by `Eric Larson`_

- Add ``overwrite`` parameter in :func:`mne.Epochs.save` by `Katarina Slama`_

- Add ``stim_channel`` parameter in :func:`mne.io.read_raw_cnt` to toggle stim channel synthesis by `Joan Massich`_

- Python 2 is no longer supported; MNE-Python now requires Python 3.5+, by `Eric Larson`_

- A new class :class:`mne.VolVectorSourceEstimate` is returned by :func:`mne.minimum_norm.apply_inverse` (and related functions) when a volume source space and ``pick_ori='vector'`` is used, by `Eric Larson`_

- Converting a forward solution with a volume source space to fixed orientation using :func:`mne.convert_forward_solution` now raises an error, by `Eric Larson`_

- ``raw.estimate_rank`` has been deprecated and will be removed in 0.19 in favor of :func:`mne.compute_rank`  by `Eric Larson`_

- :class:`Annotations` are now kept sorted (by onset time) during instantiation and :meth:`~Annotations.append` operations, by `Eric Larson`_

- Deprecate :func:`mne.io.find_edf_events` by `Joan Massich`_

- Deprecate ``limit_depth_chs`` in :func:`mne.minimum_norm.make_inverse_operator` in favor of ``depth=dict(limit_depth_chs=...)`` by `Eric Larson`_

- Reading BDF and GDF files with :func:`mne.io.read_raw_edf` is deprecated and replaced by :func:`mne.io.read_raw_bdf` and :func:`mne.io.read_raw_gdf`, by `Clemens Brunner`_

- :func:`mne.forward.compute_depth_prior` has been reworked to operate directly on :class:`Forward` instance as ``forward`` rather than a representation scattered across the parameters ``G, is_fixed_ori, patch_info``, by `Eric Larson`_

- Deprecate ``method='extended-infomax'`` in :class:`mne.preprocessing.ICA`; Extended Infomax can now be computed with ``method='infomax'`` and ``fit_params=dict(extended=True)`` by `Clemens Brunner`_

- Fix support for supplying ``extrapolate`` via :meth:`ica.plot_properties(..., topomap_args=dict(extrapolate=...)) <mne.preprocessing.ICA.plot_properties>` by `Sebastian Castano`_

- The peak finder that was formerly accessible via ``from mne.preprocessing.peak_finder import peak_finder`` should now be imported directly from the enclosing namespace as ``from mne.preprocessing import peak_finder`` by `Eric Larson`_

- Deprecate ``mne.realtime`` module to make a standalone module `mne-realtime` that will live outside of this package by `Teon Brooks`_

.. _changes_0_17:

Version 0.17
------------

Changelog
~~~~~~~~~

- Add new tutorial for :class:`mne.Annotations` and ``events`` by `Joan Massich`_ and  `Alex Gramfort`_

- Add support for saving :class:`mne.Annotations` as CSV and TXT files by `Joan Massich`_ and `Alex Gramfort`_

- Add :meth:`mne.Epochs.shift_time` that shifts the time axis of :class:`mne.Epochs` by `Thomas Hartmann`_

- Add :func:`mne.viz.plot_arrowmap` computes arrowmaps using Hosaka-Cohen transformation from magnetometer or gradiometer data, these arrows represents an estimation of the current flow underneath the MEG sensors by `Sheraz Khan`_

- Add :func:`mne.io.read_raw_fieldtrip`, :func:`mne.read_epochs_fieldtrip` and :func:`mne.read_evoked_fieldtrip` to import FieldTrip data. By `Thomas Hartmann`_ and `Dirk Gütlin`_.

- Add ``rank`` parameter to :func:`mne.compute_covariance`, :func:`mne.cov.regularize` and related functions to preserve data rank and speed up computation using low-rank computations during regularization by `Eric Larson`_ and `Denis Engemann`_

- Add new function :func:`mne.read_annotations` that can read annotations in EEGLAB, BrainVision, EDF and Brainstorm formats by `Joan Massich`_ and `Alex Gramfort`_.

- Add capability to read and save Epochs containing complex data (e.g. after Hilbert-transform) using :meth:`mne.Epochs.save` and :func:`mne.read_epochs`, by `Stefan Repplinger`_, `Eric Larson`_ and `Alex Gramfort`_

- Add optically pumped magnetometer dataset and example by `Rasmus Zetter`_ and `Eric Larson`_

- Add ``orgin`` parameter to :meth:`mne.io.Raw.time_as_index` to allow ``times`` to be relative to this ``origin`` by `Joan Massich`_

- Add ``title`` argument to :meth:`mne.SourceEstimate.plot` by `Eric Larson`_

- :func:`mne.io.Raw.set_annotations` now changes ``orig_time`` to match ``meas_date`` and shift ``self.annotations.onset`` accordingly. Previous behavior is deprecated and would be removed in 0.18. Work by `Joan Massich`_

- Add :func:`mne.compute_source_morph` which creates a :class:`mne.SourceMorph` object to unify morphing any type of source estimates (surface or volume) from one subject to another for group studies. It is now possible to do group studies when working on the volume with MNE. Work by `Tommy Clausner`_ during GSOC 2018 with the help of `Alex Gramfort`_ and `Eric Larson`_.

- Add ability to pass threshold for EOG to :func:`mne.preprocessing.find_eog_events` and :func:`mne.preprocessing.create_eog_epochs` by `Peter Molfese`_

- Add possibility to save :class:`mne.VolSourceEstimate` and :class:`mne.MixedSourceEstimate` to HDF5 format (file extension .h5) with :meth:`mne.VolSourceEstimate.save` and :meth:`mne.MixedSourceEstimate.save` by `Alex Gramfort`_

- Add `replace` parameter to :meth:`mne.io.Raw.add_events` to allow adding events while removing the old ones on the stim channel by `Alex Gramfort`_

- Add ability to pass ``axes`` to ``ts_args`` and ``topomap_args`` of :meth:`mne.viz.plot_evoked_joint` by `Jona Sassenhagen`_

- Add ability to pass a precomputed forward solution to :func:`mne.simulation.simulate_raw` by `Eric Larson`_

- Add ability to read and write beamformers with :func:`mne.beamformer.read_beamformer` and :class:`mne.beamformer.Beamformer.save` by `Eric Larson`_

- Add resting-state source power spectral estimation example :ref:`sphx_glr_auto_examples_datasets_plot_opm_rest_data.py` by `Eric Larson`_, `Denis Engemann`_, and `Luke Bloy`_

- Add :func:`mne.channels.make_1020_channel_selections` to group 10/20-named EEG channels by location, by `Jona Sassenhagen`_

- Add helmet for Artemis123 for :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add support for reading MATLAB ``v7.3+`` files in :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` via `pymatreader`_ by `Steven Gutstein`_, `Eric Larson`_, and `Thomas Hartmann`_

- Add support for raw PSD plots in :meth:`mne.Report.parse_folder` via ``raw_psd`` argument of :class:`mne.Report` by `Eric Larson`_

- Add `trig_shift_by_type` parameter in :func:`mne.io.read_raw_brainvision` to allow to specify offsets for arbitrary marker types by `Henrich Kolkhorst`_

- Add progress bar support to :class:`mne.decoding.SlidingEstimator` and :class:`mne.decoding.GeneralizingEstimator` by `Eric Larson`_

- Add interactive visualization of volume source estimates using :func:`mne.viz.plot_volume_source_estimates` by `Mainak Jas`_

- Add :func:`mne.head_to_mri` to convert positions from head coordinates to MRI RAS coordinates, by `Joan Massich`_ and `Alex Gramfort`_

- Add improved CTF helmet for :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add handling in :func:`mne.combine_evoked` and :func:`mne.grand_average` for input with the same channels in different orders, if required, by `Jona Sassenhagen`_

- Add `split_naming` parameter to the `Raw.save` method to allow for BIDS-compatible raw file name construction by `Teon Brooks`_

- Add ``origin`` parameter to :meth:`mne.Evoked.interpolate_bads` and related methods by `Eric Larson`_

- Add automated MEG helmet shape approximation to :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add capability to save a :class:`mne.Report` to an HDF5 file to :meth:`mne.Report.save` by `Marijn van Vliet`_

- Add :func:`mne.open_report` to read back a :class:`mne.Report` object that was saved to an HDF5 file by `Marijn van Vliet`_

- Add multi-taper estimation to :func:`mne.minimum_norm.compute_source_psd` by `Eric Larson`_

- Add support for custom, e.g. robust, averaging methods in :meth:`mne.Epochs.average` by `Jona Sassenhagen`_

- Add support for Neuromag 122 system by `Alex Gramfort`_

- Add function ``mne.io.read_annotations_brainvision`` for reading directly Brainvision marker files by `Alex Gramfort`_

- Add :meth:`mne.Report.remove` method to remove existing figures from a report, by `Marijn van Vliet`_

- Add sign to output of max-power orientation for :func:`mne.beamformer.make_dics` by `Eric Larson`_

- Add support for ``pick_ori='max-power'`` when ``weight_norm=None`` in :func:`mne.beamformer.make_lcmv` by `Marijn van Vliet`_

- Add support for ``weight_norm='nai'`` for all ``pick_ori`` options in :func:`mne.beamformer.make_lcmv` by `Marijn van Vliet`_

- Add support for ``weight_norm='nai'`` to :func:`mne.beamformer.make_dics` by `Marijn van Vliet`_

- Add parameter ``rank=None`` to :func:`mne.beamformer.make_dics` by `Marijn van Vliet`_

- Add parameter ``rank='full'`` to :func:`mne.beamformer.make_lcmv`, which can be set to ``None`` to auto-compute the rank of the covariance matrix before regularization by `Marijn van Vliet`_

- Handle different time vectors in topography plots using :func:`mne.viz.plot_evoked_topo` by `Jussi Nurminen`_

- Speed up :func:`mne.inverse_sparse.mixed_norm` if the :solver: parameter is set to `bcd` using :func:`scipy.linalg.get_blas_funcs` by `Quentin Bertrand`_

Bug
~~~

- Fix bug with scaling of data in ``mne.cov._compute_covariance_auto`` that was affecting the :class:`mne.decoding.SPoC` estimator by `David Sabbagh`_

- Fix :func:`mne.io.Raw.plot_projs_topomap` by `Joan Massich`_

- Fix bug in :func:`mne.minimum_norm.compute_source_psd` where the ``stc.times`` output was scaled by 1000, by `Eric Larson`_

- Fix default values for ``'diagonal_fixed'`` estimation method of :func:`mne.compute_covariance` to be ``0.1`` for all channel types, as in :func:`mne.cov.regularize` by `Eric Larson`_

- Fix reading edf file annotations by `Joan Massich`_

- Fix bug with reading events from BrainVision files by `Stefan Appelhoff`_

- Fix bug where :func:`mne.io.read_raw_eeglab` would warn when the stim channel is populated with an array of zeros by `Joan Massich`_

- Fix 2nd column of events in BrainVision to no longer store duration but rather be contained by ``raw.annotations`` by `Alex Gramfort`_

- Fix checking of the correctness of the ``prepared=True`` argument in :func:`mne.minimum_norm.apply_inverse` and related functions by `Eric Larson`_

- Fix bug of not showing ERD's in baseline rescaled tfr topomaps if grads are combined by `Erkka Heinila`_

- Fix bug with FIF I/O where strings were written in UTF-8 format instead of Latin-1 by `Eric Larson`_

- Fix bug with reading measurement dates from BrainVision files by `Stefan Appelhoff`_

- Fix bug with `mne.fit_dipole` where the residual was returned as ``ndarray`` instead of :class:`mne.Evoked` instance, by `Eric Larson`_

- Fix bug with ``mne flash_bem`` when ``flash30`` is not used by `Eric Larson`_

- Fix bug with :func:`mne.stats.permutation_cluster_test` and :func:`mne.stats.spatio_temporal_cluster_test` where ``threshold=None`` was not calculated properly for a f-oneway test by `Daniel McCloy`_ and `Eric Larson`_

- Fix bug with channel names in ``mgh70`` montage in :func:`mne.channels.read_montage` by `Eric Larson`_

- Fix duplication of ``info['hpi_meas']`` and ``info['hpi_results']`` by `Sara Sommariva`_

- Fix bug in :func:`mne.io.read_raw_edf` when reading large files on Windows by `Marcin Koculak`_

- Fix check in :func:`mne.viz.plot_sensors` for invalid channel locations by `Eric Larson`_

- Fix bug in :func:`mne.io.read_raw_edf` where GDF files had ``info['highpass']`` and ``info['lowpass']`` set to NaN and ``info['meas_date']`` set incorrectly, by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.ICA.apply` to handle arrays as `exclude` property by `Joan Massich`_

- Fix bug in ``method='eLORETA'`` for :func:`mne.minimum_norm.apply_inverse` when using a sphere model and saved ``inv`` by `Eric Larson`_

- Fix bug in :class:`mne.io.Raw` where warnings were emitted when objects were deleted by `Eric Larson`_

- Fix vector data support for :class:`mne.VolSourceEstimate` by `Christian Brodbeck`_

- Fix bug with IIR filtering axis in :func:`mne.filter.filter_data` by `Eric Larson`_

- Fix bug with non-boxcar windows in :meth:`mne.io.Raw.resample` and :func:`mne.filter.resample` by `Eric Larson`_

- Fix bug in :func:`mne.minimum_norm.apply_inverse` where applying an MEG-only inverse would raise an error about needing an average EEG reference by `Eric Larson`_

- Fix bug in ``inst.apply_proj()`` where an average EEG reference was always added by `Eric Larson`_

- Fix bug in :func:`mne.time_frequency.tfr_morlet`, :func:`mne.time_frequency.tfr_multitaper`, and :func:`mne.time_frequency.tfr_stockwell` where not all data channels were picked by `Eric Larson`_

- Fix bug in :meth:`mne.preprocessing.ICA.plot_overlay` and :func:`mne.make_field_map` for CTF data with compensation by `Eric Larson`_

- Fix bug in :func:`mne.create_info` passing ``int`` as ``ch_names`` on Windows by `Eric Larson`_

- Fix bug in ``mne.realtime.RtEpochs`` where events during the buildup of the buffer were not correctly processed when incoming data buffers are smaller than the epochs by `Henrich Kolkhorst`_

- Fix bug in :func:`mne.io.read_raw_brainvision` where 1-indexed BrainVision events were not being converted into 0-indexed mne events by `Steven Bethard`_

- Fix bug in :func:`mne.viz.plot_snr_estimate` and :func:`mne.minimum_norm.estimate_snr` where the inverse rank was not properly utilized (especially affecting SSS'ed MEG data) by `Eric Larson`_

- Fix error when saving stc as nifti image when using volume source space formed by more than one label by `Alex Gramfort`_

- Fix error when interpolating MEG channels with compensation using reference channels (like for CTF data) by `Alex Gramfort`_

- Fix bug in :func:`mne.make_sphere_model` where EEG sphere model coefficients were not optimized properly by `Eric Larson`_

- Fix bug in :func:`mne.io.read_raw_ctf` to read bad channels and segments from CTF ds files by `Luke Bloy`_

- Fix problem with :meth:`mne.io.Raw.add_channels` where ``raw.info['bads']`` was replicated by `Eric Larson`_

- Fix bug with :class:`mne.Epochs` where an error was thrown when resizing data (e.g., during :meth:`mne.Epochs.drop_bad`) by `Eric Larson`_

- Fix naming of ``raw.info['buffer_size_sec']`` to be ``raw.buffer_size_sec`` as it is a writing parameter rather than a measurement parameter by `Eric Larson`_

- Fix EGI-MFF parser not to require ``dateutil`` package by `Eric Larson`_

- Fix error when running LCMV on MEG channels with compensation using reference channels (like for CTF data) by `Alex Gramfort`_

- Fix the use of :func:`sklearn.model_selection.cross_val_predict` with :class:`mne.decoding.SlidingEstimator` by `Alex Gramfort`_

- Fix event sample number increase when combining many Epochs objects with :func:`mne.concatenate_epochs` with  by `Jasper van den Bosch`_

- Fix title of custom slider images to :class:`mne.Report` by `Marijn van Vliet`_

- Fix missing initialization of ``self._current`` in :class:`mne.Epochs` by `Henrich Kolkhorst`_

- Fix processing of data with bad segments and acquisition skips with new ``skip_by_annotation`` parameter in :func:`mne.preprocessing.maxwell_filter` by `Eric Larson`_

- Fix symlinking to use relative paths in ``mne flash_bem` and ``mne watershed_bem`` by `Eric Larson`_

- Fix error in mne coreg when saving with scaled MRI if fiducials haven't been saved by `Ezequiel Mikulan`_

- Fix normalization error in :func:`mne.beamformer.make_lcmv` when ``pick_ori='normal', weight_norm='unit_noise_gain'`` by `Marijn van Vliet`_

- Fix MNE-C installation instructions by `buildqa`_

- Fix computation of max-power orientation in :func:`mne.beamformer.make_dics` when ``pick_ori='max-power', weight_norm='unit_noise_gain'`` by `Marijn van Vliet`_

API
~~~

- Deprecated separate reading of annotations and synthesis of STI014 channels in readers by `Joan Massich`_:

  - Deprecated ``mne.io.read_annotations_eeglab``
  - Deprecated ``annot`` and ``annotmap`` parameters in :meth:`~mne.io.read_raw_edf`
  - Deprecated ``stim_channel`` parameters in :func:`~mne.io.read_raw_edf`, :func:`~mne.io.read_raw_brainvision`, and :func:`~mne.io.read_raw_eeglab`

  Annotations are now added to ``raw`` instances directly upon reading as :attr:`raw.annotations <mne.io.Raw.annotations>`.
  They can also be read separately with :func:`mne.read_annotations` for EEGLAB, BrainVision, EDF, and Brainstorm formats.
  Use :func:`mne.events_from_annotations(raw.annotations) <mne.events_from_annotations>`
  to convert these to events instead of the old way (using STI014 channel synthesis followed by :func:`mne.find_events(raw) <mne.find_events>`).

  In 0.17 (this release)
    Use ``read_raw_...(stim_channel=False)`` to disable warnings (and stim channel synthesis), but other arguments for ``stim_channel`` will still be supported.

  In 0.18
    The only supported option will be ``read_raw_...(stim_channel=False)``, and all stim-channel-synthesis arguments will be removed. At this point, ``stim_channel`` should be removed from scripts for future compatibility, but ``stim_channel=False`` will still be acceptable for backward compatibility.

  In 0.19
    The ``stim_channel`` keyword arguments will be removed from ``read_raw_...`` functions.

- Calling :meth:``mne.io.pick.pick_info`` removing channels that are needed by compensation matrices (``info['comps']``) no longer raises ``RuntimeException`` but instead logs an info level message. By `Luke Bloy`_

- :meth:`mne.Epochs.save` now has the parameter `fmt` to specify the desired format (precision) saving epoched data, by `Stefan Repplinger`_, `Eric Larson`_ and `Alex Gramfort`_

- Deprecated ``mne.SourceEstimate.morph_precomputed``, ``mne.SourceEstimate.morph``, ``mne.compute_morph_matrix``, ``mne.morph_data_precomputed`` and ``mne.morph_data`` in favor of :func:`mne.compute_source_morph`, by `Tommy Clausner`_

- Prepare transition to Python 3. This release will be the last release compatible with Python 2. The next version will be Python 3 only.

- CUDA support now relies on CuPy_ instead of ``PyCUDA`` and ``scikits-cuda``. It can be installed using ``conda install cupy``. By `Eric Larson`_

- Functions requiring a color cycle will now default to Matplotlib rcParams colors, by `Stefan Appelhoff`_

- :meth:`mne.Evoked.plot_image` has gained the ability to ``show_names``, and if a selection is provided to ``group_by``, ``axes`` can now receive a `dict`, by `Jona Sassenhagen`_

- Calling :meth:`mne.Epochs.decimate` with ``decim=1`` no longer copies the data by `Henrich Kolkhorst`_

- Removed blocking (waiting for new epochs) in ``mne.realtime.RtEpochs.get_data()`` by `Henrich Kolkhorst`_

- Warning messages are now only emitted as :func:`warnings.warn_explicit` rather than also being emitted as ``logging`` messages (unless a logging file is being used) to avoid duplicate warning messages, by `Eric Larson`_

- Deprecated save_stc_as_volume function in favor of :meth:`mne.VolSourceEstimate.as_volume` and :meth:`mne.VolSourceEstimate.save_as_volume` by `Alex Gramfort`_

- `src.kind` now equals to `'mixed'` (and not `'combined'`) for a mixed source space (made of surfaces and volume grids) by `Alex Gramfort`_

- Deprecation of :attr:`mne.io.Raw.annotations` property in favor of :meth:`mne.io.Raw.set_annotations` by `Joan Massich`_

- The default value of ``stop_receive_thread`` in ``mne.realtime.RtEpochs.stop`` has been changed to ``True`` by `Henrich Kolkhorst`_

- Using the :meth:`mne.io.Raw.add_channels` on an instance with memmapped data will now resize the memmap file to append the new channels on Windows and Linux, by `Eric Larson`_

- :attr:`mne.io.Raw.annotations` when missing is set to an empty :class:`mne.Annotations` rather than ``None`` by `Joan Massich`_ and `Alex Gramfort`_

- Mismatches in CTF compensation grade are now checked in inverse computation by `Eric Larson`_


Authors
~~~~~~~

People who contributed to this release  (in alphabetical order):

* 	Alexandre Gramfort
* 	Antoine Gauthier
* 	Britta Westner
* 	Christian Brodbeck
* 	Clemens Brunner
* 	Daniel McCloy
* 	David Sabbagh
* 	Denis A. Engemann
* 	Eric Larson
* 	Ezequiel Mikulan
* 	Henrich Kolkhorst
* 	Hubert Banville
* 	Jasper J.F. van den Bosch
* 	Jen Evans
* 	Joan Massich
* 	Johan van der Meer
* 	Jona Sassenhagen
* 	Kambiz Tavabi
* 	Lorenz Esch
* 	Luke Bloy
* 	Mainak Jas
* 	Manu Sutela
* 	Marcin Koculak
* 	Marijn van Vliet
* 	Mikolaj Magnuski
* 	Peter J. Molfese
* 	Sam Perry
* 	Sara Sommariva
* 	Sergey Antopolskiy
* 	Sheraz Khan
* 	Stefan Appelhoff
* 	Stefan Repplinger
* 	Steven Bethard
* 	Teekuningas
* 	Teon Brooks
* 	Thomas Hartmann
* 	Thomas Jochmann
* 	Tom Dupré la Tour
* 	Tristan Stenner
* 	buildqa
* 	jeythekey

.. _changes_0_16:

Version 0.16
------------

Changelog
~~~~~~~~~

- Add possibility to pass dict of floats as argument to :func:`mne.make_ad_hoc_cov` by `Nathalie Gayraud`_

- Add support for metadata in :class:`mne.Epochs` by `Chris Holdgraf`_, `Alex Gramfort`_, `Jona Sassenhagen`_, and `Eric Larson`_

- Add support for plotting a dense head in :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Allow plotting in user-created mayavi Scene in :func:`mne.viz.plot_alignment` by `Daniel McCloy`_

- Reduce memory consumption and do not require data to be loaded in :meth:`mne.Epochs.apply_baseline` by `Eric Larson`_

- Add option ``render_bem`` to :meth:`mne.Report.parse_folder` by `Eric Larson`_

- Add to :func:`mne.viz.plot_alignment` plotting of coordinate frame axes via ``show_axes`` and terrain-style interaction via ``interaction``, by `Eric Larson`_

- Add option ``initial_event`` to :func:`mne.find_events` by `Clemens Brunner`_

- Left and right arrow keys now scroll by 25% of the visible data, whereas Shift+left/right scroll by a whole page in :meth:`mne.io.Raw.plot` by `Clemens Brunner`_

- Add support for gantry tilt angle determination from Elekta FIF file header by `Chris Bailey`_

- Add possibility to concatenate :class:`mne.Annotations` objects with ``+`` or ``+=`` operators by `Clemens Brunner`_

- Add support for MaxShield raw files in :class:`mne.Report` by `Eric Larson`_

- Add ability to plot whitened data in :meth:`mne.io.Raw.plot`, :meth:`mne.Epochs.plot`, :meth:`mne.Evoked.plot`, and :meth:`mne.Evoked.plot_topo` by `Eric Larson`_

- Workaround for reading EGI MFF files with physiological signals that also present a bug from the EGI system in :func:`mne.io.read_raw_egi` by `Federico Raimondo`_

- Add support for reading subject height and weight in ``info['subject_info']`` by `Eric Larson`_

- Improve online filtering of raw data when plotting with :meth:`mne.io.Raw.plot` to filter in segments in accordance with the default ``skip_by_annotation=('edge', 'bad_acq_skip')`` of :meth:`mne.io.Raw.filter` to avoid edge ringing by `Eric Larson`_

- Add support for multiple head position files, plotting of sensors, and control of plotting color and axes in :func:`mne.viz.plot_head_positions` by `Eric Larson`_

- Add ability to read and write :class:`Annotations` separate from :class:`mne.io.Raw` instances via :meth:`Annotations.save` and :func:`read_annotations` by `Eric Larson`_

- Add option to unset a montage by passing `None` to :meth:`mne.io.Raw.set_montage` by `Clemens Brunner`_

- Add sensor denoising via :func:`mne.preprocessing.oversampled_temporal_projection` by `Eric Larson`_

- Add ``mne.io.pick.get_channel_types`` which returns all available channel types in MNE by `Clemens Brunner`_

- Use standard PCA instead of randomized PCA whitening prior to ICA to increase reproducibility by `Clemens Brunner`_

- Plot sEEG electrodes in :func:`mne.viz.plot_alignment` by `Alex Gramfort`_

- Add support for any data type like sEEG or ECoG in covariance related functions (estimation, whitening and plotting) by `Alex Gramfort`_ and `Eric Larson`_

- Add function ``mne.io.read_annotations_eeglab`` to allow loading annotations from EEGLAB files, by `Alex Gramfort`_

- :meth:`mne.io.Raw.set_montage` now accepts a string as its ``montage`` argument; this will set a builtin montage, by `Clemens Brunner`_

- Add 4D BTi phantom dataset :func:`mne.datasets.phantom_4dbti.data_path`, by `Alex Gramfort`_

- Changed the background color to grey in :func:`mne.viz.plot_alignment` to make helmet more visible, by `Alex Gramfort`_

- Add :meth:`mne.io.Raw.reorder_channels`, :meth:`mne.Evoked.reorder_channels`, etc. to reorder channels, by `Eric Larson`_

- Add to ``mne coreg`` and :func:`mne.gui.coregistration` by `Eric Larson`_:

  - Improved visibility of points inside the head
  - Projection of EEG electrodes
  - Orientation of extra points toward the surface
  - Scaling points by distance to the head surface
  - Display of HPI points
  - ICP fitting with convergence criteria
  - Faster display updates
  - Scaling of ``mri/*.mgz`` files
  - Scaling of ``mri/trainsforms/talairach.xfm`` files for conversion to MNI space

- Add ability to exclude components interactively by clicking on their labels in :meth:`mne.preprocessing.ICA.plot_components` by `Mikołaj Magnuski`_

- Add reader for manual annotations of raw data produced by Brainstorm by `Anne-Sophie Dubarry`_

- Add eLORETA noise normalization for minimum-norm solvers by `Eric Larson`_

- Tighter duality gap computation in ``mne.inverse_sparse.tf_mxne_optim`` and new parametrization with ``alpha`` and  ``l1_ratio`` instead of ``alpha_space`` and ``alpha_time`` by `Mathurin Massias`_ and `Daniel Strohmeier`_

- Add ``dgap_freq`` parameter in ``mne.inverse_sparse.mxne_optim`` solvers to control the frequency of duality gap computation by `Mathurin Massias`_ and `Daniel Strohmeier`_

- Add support for reading Eximia files by `Eric Larson`_ and `Federico Raimondo`_

- Add the Picard algorithm to perform ICA for :class:`mne.preprocessing.ICA`, by `Pierre Ablin`_ and `Alex Gramfort`_

- Add ability to supply a mask to the plot in :func:`mne.viz.plot_evoked_image` by `Jona Sassenhagen`_

- Add ``connectivity=False`` to cluster-based statistical functions to perform non-clustering stats by `Eric Larson`_

- Add :func:`mne.time_frequency.csd_morlet` and :func:`mne.time_frequency.csd_array_morlet` to estimate cross-spectral density using Morlet wavelets, by `Marijn van Vliet`_

- Add multidictionary time-frequency support to :func:`mne.inverse_sparse.tf_mixed_norm` by `Mathurin Massias`_ and `Daniel Strohmeier`_

- Add new DICS implementation as :func:`mne.beamformer.make_dics`, :func:`mne.beamformer.apply_dics`, :func:`mne.beamformer.apply_dics_csd` and :func:`mne.beamformer.apply_dics_epochs`, by `Marijn van Vliet`_ and `Susanna Aro`_

Bug
~~~

- Fix bug in EEG interpolation code to do nothing if there is no channel to interpolate by `Mainak Jas`_

- Fix bug in ``mne.preprocessing.peak_finder`` to output datatype consistently and added input check for empty vectors by `Tommy Clausner`_

- Fix bug in :func:`mne.io.read_raw_brainvision` to use the correct conversion for filters from time constant to frequency by `Stefan Appelhoff`_

- Fix bug with events when saving split files using :meth:`mne.Epochs.save` by `Eric Larson`_

- Fix bug in :class:`mne.decoding.SlidingEstimator` and :class:`mne.decoding.GeneralizingEstimator` to allow :func:`mne.decoding.cross_val_multiscore` to automatically detect whether the `base_estimator` is a classifier and use a `StratifiedKFold` instead of a `KFold` when `cv` is not specified, by `Jean-Remi King`_

- Fix bug in :func:`mne.set_eeg_reference` to remove an average reference projector when setting the reference to ``[]`` (i.e. do not change the existing reference) by `Clemens Brunner`_

- Fix bug in threshold-free cluster enhancement parameter validation (:func:`mne.stats.permutation_cluster_1samp_test` and :func:`mne.stats.permutation_cluster_test`) by `Clemens Brunner`_

- Fix bug in :meth:`mne.io.Raw.plot` to correctly display event types when annotations are present by `Clemens Brunner`_

- Fix bug in :func:`mne.stats.spatio_temporal_cluster_test` default value for ``threshold`` is now calculated based on the array sizes in ``X``, by `Eric Larson`_

- Fix bug in :func:`mne.simulation.simulate_raw` with ``use_cps=True`` where CPS was not actually used by `Eric Larson`_

- Fix bug in :func:`mne.simulation.simulate_raw` where 1- and 3-layer BEMs were not properly transformed using ``trans`` by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_alignment` where the head surface file ``-head.fif`` was not used even though present by `Chris Bailey`_

- Fix bug when writing compressed sparse column matrices (e.g., Maxwell filtering cross-talk matrices) by `Marijn van Vliet`_ and `Eric Larson`_

- Fix bug in :meth:`mne.io.Raw.plot_psd` to correctly deal with ``reject_by_annotation=False`` by `Clemens Brunner`_

- Fix bug in :func:`mne.make_fixed_length_events` when hitting corner case problems rounding to sample numbers by `Eric Larson`_

- Fix bug in :class:`mne.Epochs` when passing events as list with ``event_id=None``  by `Alex Gramfort`_

- Fix bug in :meth:`mne.Report.add_figs_to_section` when passing :class:`numpy.ndarray` by `Eric Larson`_

- Fix bug in CSS class setting in `mne.Report` BEM section by `Eric Larson`_

- Fix bug in :class:`Annotations` where annotations that extend to the end of a recording were not extended properly by `Eric Larson`_

- Fix bug in :meth:`mne.io.Raw.filter` to properly raw data with acquisition skips in separate segments by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.maxwell_filter` where homogeneous fields were not removed for CTF systems by `Eric Larson`_

- Fix computation of average quaternions in :func:`mne.preprocessing.maxwell_filter` by `Eric Larson`_

- Fix bug in writing ``raw.annotations`` where empty annotations could not be written to disk, by `Eric Larson`_

- Fix support for writing FIF files with acquisition skips by using empty buffers rather than writing zeros by `Eric Larson`_

- Fix bug in the ``mne make_scalp_surfaces`` command where ``--force`` (to bypass topology check failures) was ignored by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.maxwell_filter` when providing ``origin`` in ``'meg'`` coordinate frame for recordings with a MEG to head transform (i.e., non empty-room recordings) by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_cov` that ignored ``colorbar`` argument by `Nathalie Gayraud`_

- Fix bug when picking CTF channels that could cause data saved to disk to be unreadable by `Eric Larson`_

- Fix bug when reading event latencies (in samples) from eeglab files didn't translate indices to 0-based python indexing by `Mikołaj Magnuski`_

- Fix consistency between :class:`mne.Epochs` and :func:`mne.stats.linear_regression_raw` in converting between samples and times (:func:`mne.stats.linear_regression_raw` now rounds, instead of truncating) by `Phillip Alday`_

- Fix bug in ``mne coreg`` where sphere surfaces were scaled by `Eric Larson`_

- Fix bug in :meth:`mne.Evoked.plot_topomap` when using ``proj='interactive'`` mode by `Eric Larson`_

- Fix bug when passing ``show_sensors=1`` to :func:`mne.viz.plot_compare_evokeds` resulted in sensors legend placed in lower right of the figure (position 4 in matplotlib), not upper right by `Mikołaj Magnuski`_

- Fix handling of annotations when cropping and concatenating raw data by `Alex Gramfort`_ and `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.create_ecg_epochs` where ``keep_ecg=False`` was ignored by `Eric Larson`_

- Fix bug in :meth:`mne.io.Raw.plot_psd` when ``picks is not None`` and ``picks`` spans more than one channel type by `Eric Larson`_

- Fix bug in :class:`mne.make_forward_solution` when passing data with compensation channels (e.g. CTF) that contain bad channels by `Alex Gramfort`_

- Fix bug in :meth:`mne.SourceEstimate.get_peak` and :meth:`mne.VolSourceEstimate.get_peak` when there is only a single time point by `Marijn van Vliet`_

- Fix bug in :func:`mne.io.read_raw_edf` when reading BDF files stimulus channels are now not scaled anymore by `Clemens Brunner`_

API
~~~

- Channels with unknown locations are now assigned position ``[np.nan, np.nan, np.nan]`` instead of ``[0., 0., 0.]``, by `Eric Larson`_

- Removed unused ``image_mask`` argument from :func:`mne.viz.plot_topomap` by `Eric Larson`_

- Unknown measurement dates are now stored as ``info['meas_date'] = None`` rather than using the current date. ``None`` is also now used when anonymizing data and when determining the machine ID for writing files, by `Mainak Jas`_ and `Eric Larson`_

- :meth:`mne.Evoked.plot` will now append the number of epochs averaged for the evoked data in the first plot title, by `Eric Larson`_

- Changed the line width in :func:`mne.viz.plot_bem` from 2.0 to 1.0 for better visibility of underlying structures, by `Eric Larson`_

- Changed the behavior of :meth:`mne.io.Raw.pick_channels` and similar methods to be consistent with :func:`mne.pick_channels` to treat channel list as a set (ignoring order) -- if reordering is necessary use ``inst.reorder_channels``, by `Eric Larson`_

- Changed the labeling of some plotting functions to use more standard capitalization and units, e.g. "Time (s)" instead of "time [sec]" by `Eric Larson`_

- ``mne.time_frequency.csd_epochs`` has been refactored into :func:`mne.time_frequency.csd_fourier` and :func:`mne.time_frequency.csd_multitaper`, by `Marijn van Vliet`_

- ``mne.time_frequency.csd_array`` has been refactored into :func:`mne.time_frequency.csd_array_fourier` and :func:`mne.time_frequency.csd_array_multitaper`, by `Marijn van Vliet`_

- Added ``clean_names=False`` parameter to :func:`mne.io.read_raw_ctf` for control over cleaning of main channel names and compensation channel names from CTF suffixes by `Oleh Kozynets`_

- The functions ``lcmv``, ``lcmv_epochs``, and ``lcmv_raw`` are now deprecated in favor of :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.apply_lcmv`, :func:`mne.beamformer.apply_lcmv_epochs`, and :func:`mne.beamformer.apply_lcmv_raw`, by `Britta Westner`_

- The functions ``mne.beamformer.dics``, ``mne.beamformer.dics_epochs`` and ``mne.beamformer.dics_source_power`` are now deprecated in favor of :func:`mne.beamformer.make_dics`, :func:`mne.beamformer.apply_dics`, and :func:`mne.beamformer.apply_dics_csd`, by `Marijn van Vliet`_


Authors
~~~~~~~

People who contributed to this release  (in alphabetical order):

* Alejandro Weinstein
* Alexandre Gramfort
* Annalisa Pascarella
* Anne-Sophie Dubarry
* Britta Westner
* Chris Bailey
* Chris Holdgraf
* Christian Brodbeck
* Claire Braboszcz
* Clemens Brunner
* Daniel McCloy
* Denis A. Engemann
* Desislava Petkova
* Dominik Krzemiński
* Eric Larson
* Erik Hornberger
* Fede Raimondo
* Henrich Kolkhorst
* Jean-Remi King
* Jen Evans
* Joan Massich
* Jon Houck
* Jona Sassenhagen
* Juergen Dammers
* Jussi Nurminen
* Kambiz Tavabi
* Katrin Leinweber
* Kostiantyn Maksymenko
* Larry Eisenman
* Luke Bloy
* Mainak Jas
* Marijn van Vliet
* Mathurin Massias
* Mikolaj Magnuski
* Nathalie Gayraud
* Oleh Kozynets
* Phillip Alday
* Pierre Ablin
* Stefan Appelhoff
* Stefan Repplinger
* Tommy Clausner
* Yaroslav Halchenko

.. _changes_0_15:

Version 0.15
------------

Changelog
~~~~~~~~~

- :meth:`mne.channels.Layout.plot` and :func:`mne.viz.plot_layout` now allows plotting a subset of channels with ``picks`` argument by `Jaakko Leppakangas`_

- Add .bvef extension (BrainVision Electrodes File) to :func:`mne.channels.read_montage` by `Jean-Baptiste Schiratti`_

- Add :func:`mne.decoding.cross_val_multiscore` to allow scoring of multiple tasks, typically used with :class:`mne.decoding.SlidingEstimator`, by `Jean-Remi King`_

- Add :class:`mne.decoding.ReceptiveField` module for modeling electrode response to input features by `Chris Holdgraf`_

- Add :class:`mne.decoding.TimeDelayingRidge` class, used by default by :class:`mne.decoding.ReceptiveField`, to speed up auto- and cross-correlation computations and enable Laplacian regularization by `Ross Maddox`_ and `Eric Larson`_

- Add new :mod:`mne.datasets.mtrf <mne.datasets.mtrf.data_path>` dataset by `Chris Holdgraf`_

- Add example of time-frequency decoding with CSP by `Laura Gwilliams`_

- Add :class:`mne.decoding.SPoC` to fit and apply spatial filters based on continuous target variables, by `Jean-Remi King`_ and `Alexandre Barachant`_

- Add Fieldtrip's electromyogram dataset, by `Alexandre Barachant`_

- Add ``reject_by_annotation`` option to :func:`mne.preprocessing.find_eog_events` (which is also utilised by :func:`mne.preprocessing.create_eog_epochs`) to omit data that is annotated as bad by `Jaakko Leppakangas`_

- Add example for fast screening of event-related dynamics in frequency bands by `Denis Engemann`_

- Add :meth:`mne.time_frequency.EpochsTFR.save` by `Jaakko Leppakangas`_

- Add butterfly mode (toggled with 'b' key) to :meth:`mne.io.Raw.plot` by `Jaakko Leppakangas`_

- Add ``axes`` parameter to plot_topo functions by `Jaakko Leppakangas`_

- Add options to change time windowing in :func:`mne.chpi.filter_chpi` by `Eric Larson`_

- :meth:`mne.channels.Montage.plot`, :meth:`mne.channels.DigMontage.plot`, and :func:`mne.viz.plot_montage` now allow plotting channel locations as a topomap by `Clemens Brunner`_

- Add ``background_color`` parameter to :meth:`mne.Evoked.plot_topo` and :func:`mne.viz.plot_evoked_topo` and improve axes rendering as done in :func:`mne.viz.plot_compare_evokeds` by `Alex Gramfort`_

- Add support for GDF files in :func:`mne.io.read_raw_edf` by `Nicolas Barascud`_

- Add :func:`mne.io.find_edf_events` for getting the events as they are in the EDF/GDF header by `Jaakko Leppakangas`_

- Speed up :meth:`mne.io.Raw.plot` and :meth:`mne.Epochs.plot` using (automatic) decimation based on low-passing with ``decim='auto'`` parameter by `Eric Larson`_ and `Jaakko Leppakangas`_

- Add ``mne.inverse_sparse.mxne_optim.dgap_l21l1`` for computing the duality gap for TF-MxNE as the new stopping criterion by `Daniel Strohmeier`_

- Add option to return a list of :class:`Dipole` objects in sparse source imaging methods by `Daniel Strohmeier`_

- Add :func:`mne.inverse_sparse.make_stc_from_dipoles` to generate stc objects from lists of dipoles by `Daniel Strohmeier`_

- Add :func:`mne.channels.find_ch_connectivity` that tries to infer the correct connectivity template using channel info. If no template is found, it computes the connectivity matrix using :class:`Delaunay <scipy.spatial.Delaunay>` triangulation of the 2d projected channel positions by `Jaakko Leppakangas`_

- Add IO support for EGI MFF format by `Jaakko Leppakangas`_  and `ramonapariciog`_

- Add option to use matplotlib backend when plotting with :func:`mne.viz.plot_source_estimates` by `Jaakko Leppakangas`_

- Add :meth:`mne.channels.Montage.get_pos2d` to get the 2D positions of channels in a montage by `Clemens Brunner`_

- Add MGH 60- and 70-channel standard montages to :func:`mne.channels.read_montage` by `Eric Larson`_

- Add option for embedding SVG instead of PNG in HTML for :class:`mne.Report` by `Eric Larson`_

- Add confidence intervals, number of free parameters, and χ² to :func:`mne.fit_dipole` and :func:`mne.read_dipole` by `Eric Larson`_

- :attr:`mne.SourceEstimate.data` is now writable, writing to it will also update :attr:`mne.SourceEstimate.times` by `Marijn van Vliet`_

- :meth:`mne.io.Raw.plot` and :meth:`mne.Epochs.plot` now use anti-aliasing to draw signals by `Clemens Brunner`_

- Allow using saved ``DigMontage`` to import digitization to :func:`mne.gui.coregistration` by `Jaakko Leppakangas`_

- Add function :func:`mne.channels.get_builtin_montages` to list all built-in montages by `Clemens Brunner`_

- :class:`mne.decoding.SlidingEstimator` and :class:`mne.decoding.GeneralizingEstimator` now accept ``**fit_params`` at fitting by `Jean-Remi King`_

- Add :class:`mne.VectorSourceEstimate` class which enables working with both source power and dipole orientations by `Marijn van Vliet`_

- Add option ``pick_ori='vector'`` to :func:`mne.minimum_norm.apply_inverse` to produce :class:`mne.VectorSourceEstimate` by `Marijn van Vliet`_

- Add support for :class:`numpy.random.RandomState` argument to ``seed`` in :mod:`statistical clustering functions <mne.stats>` and better documentation of exact 1-sample tests by `Eric Larson`_

- Extend :func:`mne.viz.plot_epochs_image`/:meth:`mne.Epochs.plot_image` with regards to grouping by or aggregating over channels. See the new example at `examples/visualization/plot_roi_erpimage_by_rt.py` by `Jona Sassenhagen`_

- Add bootstrapped confidence intervals to :func:`mne.viz.plot_compare_evokeds` by `Jona Sassenhagen`_ and `Denis Engemann`_

- Add example on how to plot ERDS maps (also known as ERSP) by `Clemens Brunner`_

- Add support for volume source spaces to :func:`spatial_src_connectivity` and :func:`spatio_temporal_src_connectivity` by `Alex Gramfort`_

- Plotting raw data (:func:`mne.viz.plot_raw` or :meth:`mne.io.Raw.plot`) with events now includes event numbers (if there are not more than 50 events on a page) by `Clemens Brunner`_

- Add filtering functions :meth:`mne.Epochs.filter` and :meth:`mne.Evoked.filter`, as well as ``pad`` argument to :meth:`mne.io.Raw.filter` by `Eric Larson`_

- Add high frequency somatosensory MEG dataset by `Jussi Nurminen`_

- Add reduced set of labels for HCPMMP-1.0 parcellation in :func:`mne.datasets.fetch_hcp_mmp_parcellation` by `Eric Larson`_

- Enable morphing between hemispheres with ``mne.compute_morph_matrix`` by `Christian Brodbeck`_

- Add ``return_residual`` to :func:`mne.minimum_norm.apply_inverse` by `Eric Larson`_

- Add ``return_drop_log`` to :func:`mne.preprocessing.compute_proj_eog` and :func:`mne.preprocessing.compute_proj_ecg` by `Eric Larson`_

- Add time cursor and category/amplitude status message into the single-channel evoked plot by `Jussi Nurminen`_

BUG
~~~

- Fixed a bug when creating spherical volumetric grid source spaces in :func:`setup_volume_source_space` by improving the minimum-distance computations, which in general will decrease the number of used source space points by `Eric Larson`_

- Fix bug in :meth:`mne.io.read_raw_brainvision` read .vhdr files with ANSI codepage by `Okba Bekhelifi`_ and `Alex Gramfort`_

- Fix unit scaling when reading in EGI digitization files using :func:`mne.channels.read_dig_montage` by `Matt Boggess`_

- Fix ``picks`` default in :meth:`mne.io.Raw.filter` to include ``ref_meg`` channels by default by `Eric Larson`_

- Fix :class:`mne.decoding.CSP` order of spatial filter in ``patterns_`` by `Alexandre Barachant`_

- :meth:`mne.concatenate_epochs` now maintains the relative position of events during concatenation by `Alexandre Barachant`_

- Fix bug in script `mne make_scalp_surfaces` by `Denis Engemann`_ (this bug prevented creation of high-resolution meshes when they were absent in the first place.)

- Fix writing of raw files with empty set of annotations by `Jaakko Leppakangas`_

- Fix bug in :meth:`mne.preprocessing.ICA.plot_properties` where merging gradiometers would fail by `Jaakko Leppakangas`_

- Fix :func:`mne.viz.plot_sensors` to maintain proper aspect ratio by `Eric Larson`_

- Fix :func:`mne.viz.plot_topomap` to allow 0 contours by `Jaakko Leppakangas`_

- Fix :class:`mne.preprocessing.ICA` source-picking to increase threshold for rank estimation to 1e-14 by `Jesper Duemose Nielsen`_

- Fix :func:`mne.set_bipolar_reference` to support duplicates in anodes by `Jean-Baptiste Schiratti`_ and `Alex Gramfort`_

- Fix visuals of :func:`mne.viz.plot_evoked` and a bug where ylim changes when using interactive topomap plotting by `Jaakko Leppakangas`_

- Fix :meth:`mne.Evoked.plot_topomap` when using the ``mask`` argument with paired gradiometers by `Eric Larson`_

- Fix bug in :meth:`mne.Label.fill` where an empty label raised an error, by `Eric Larson`_

- Fix :func:`mne.io.read_raw_ctf` to also include the samples in the last block by `Jaakko Leppakangas`_

- Fix :meth:`mne.preprocessing.ICA.save` to close file before attempting to delete it when write fails by `Jesper Duemose Nielsen`_

- Fix :func:`mne.simulation.simulate_evoked` to use nave parameter instead of snr, by `Yousra Bekhti`_

- Fix :func:`mne.read_bem_surfaces` for BEM files missing normals by `Christian Brodbeck`_

- Fix :func:`mne.transform_surface_to` to actually copy when ``copy=True`` by `Eric Larson`_

- Fix :func:`mne.io.read_raw_brainvision` to read vectorized data correctly by `Jaakko Leppakangas`_ and `Phillip Alday`_

- Fix :func:`mne.connectivity.spectral_connectivity` so that if ``n_jobs > 1`` it does not ignore last ``n_epochs % n_jobs`` epochs by `Mikołaj Magnuski`_

- Fix :func:`mne.io.read_raw_edf` to infer sampling rate correctly when reading EDF+ files where tal-channel has a higher sampling frequency by `Jaakko Leppakangas`_

- Fix default value of ``kind='topomap'`` in :meth:`mne.channels.Montage.plot` to be consistent with :func:`mne.viz.plot_montage` by `Clemens Brunner`_

- Fix bug in :meth:`to_data_frame <mne.io.Raw.to_data_frame>` where non-consecutive picks would make the function crash by `Jaakko Leppakangas`_

- Fix channel picking and drop in :class:`mne.time_frequency.EpochsTFR` by `Lukáš Hejtmánek`_

- Fix :func:`mne.SourceEstimate.transform` to properly update :attr:`mne.SourceEstimate.times` by `Marijn van Vliet`_

- Fix :func:`mne.viz.plot_evoked_joint` to allow custom titles without appending information about the channels by `Jaakko Leppakangas`_

- Fix writing a forward solution after being processed by :func:`mne.forward.restrict_forward_to_label` or :func:`mne.forward.restrict_forward_to_stc` by `Marijn van Vliet`_

- Fix bug in :func:`mne.viz.plot_compare_evokeds` where ``truncate_yaxis`` was ignored (default is now ``False``), by `Jona Sassenhagen`_

- Fix bug in :func:`mne.viz.plot_evoked` where all xlabels were removed when using ``spatial_colors=True``, by `Jesper Duemose Nielsen`_

- Fix field mapping :func:`mne.make_field_map` and MEG bad channel interpolation functions (e.g., :meth:`mne.Evoked.interpolate_bads`) to choose a better number of components during pseudoinversion when few channels are available, by `Eric Larson`_

- Fix bug in :func:`mne.io.read_raw_brainvision`, changed default to read coordinate information if available and added test, by `Jesper Duemose Nielsen`_

- Fix bug in :meth:`mne.SourceEstimate.to_original_src` where morphing failed if two vertices map to the same target vertex, by `Marijn van Vliet`_

- Fix :class:`mne.preprocessing.Xdawn` to give verbose error messages about rank deficiency and handle transforming :class:`mne.Evoked`, by `Eric Larson`_

- Fix bug in DC and Nyquist frequency multitaper PSD computations, e.g. in :func:`mne.time_frequency.psd_multitaper`, by `Eric Larson`_

- Fix default padding type for :meth:`mne.Epochs.resample` and :meth:`mne.Evoked.resample` to be ``'edge'`` by default, by `Eric Larson`_

- Fix :func:`mne.inverse_sparse.mixed_norm`, :func:`mne.inverse_sparse.tf_mixed_norm` and :func:`mne.inverse_sparse.gamma_map` to work with volume source space and sphere head models in MEG by `Alex Gramfort`_ and `Yousra Bekhti`_

- Fix :meth:`mne.Evoked.as_type` channel renaming to append ``'_v'`` instead of ``'_virtual'`` to channel names to comply with shorter naming (15 char) requirements, by `Eric Larson`_

- Fix treatment of CTF HPI coils as fiducial points in :func:`mne.gui.coregistration` by `Eric Larson`_

- Fix resampling of events along with raw in :func:`mne.io.Raw` to now take into consideration the value of ``first_samp`` by `Chris Bailey`_

- Fix labels of PSD plots in :func:`mne.viz.plot_raw_psd` by `Alejandro Weinstein`_

- Fix depth weighting of sparse solvers (:func:`mne.inverse_sparse.mixed_norm`, :func:`mne.inverse_sparse.tf_mixed_norm` and :func:`mne.inverse_sparse.gamma_map`) with free orientation source spaces to improve orientation estimation by `Alex Gramfort`_ and `Yousra Bekhti`_

- Fix the threshold in :func:`mne.beamformer.rap_music` to properly estimate the rank by `Yousra Bekhti`_

- Fix treatment of vector inverse in :func:`mne.minimum_norm.apply_inverse_epochs` by `Emily Stephen`_

- Fix :func:`mne.find_events` when passing a list as stim_channel parameter by `Alex Gramfort`_

- Fix parallel processing when computing covariance with shrinkage estimators by `Denis Engemann`_

API
~~~
- Add ``skip_by_annotation`` to :meth:`mne.io.Raw.filter` to process data concatenated with e.g. :func:`mne.concatenate_raws` separately. This parameter will default to the old behavior (treating all data as a single block) in 0.15 but will change to ``skip_by_annotation='edge'``, which will separately filter the concatenated chunks separately, in 0.16. This should help prevent potential problems with filter-induced ringing in concatenated files, by `Eric Larson`_

- ICA channel names have now been reformatted to start from zero, e.g. ``"ICA000"``, to match indexing schemes in :class:`mne.preprocessing.ICA` and related functions, by `Stefan Repplinger`_ and `Eric Larson`_

- Add :func:`mne.beamformer.make_lcmv` and :func:`mne.beamformer.apply_lcmv`, :func:`mne.beamformer.apply_lcmv_epochs`, and :func:`mne.beamformer.apply_lcmv_raw` to enable the separate computation and application of LCMV beamformer weights by `Britta Westner`_, `Alex Gramfort`_, and `Denis Engemann`_.

- Add ``weight_norm`` parameter to enable both unit-noise-gain beamformer and neural activity index (weight normalization) and make whitening optional by allowing ``noise_cov=None`` in ``mne.beamformer.lcmv``, ``mne.beamformer.lcmv_epochs``, and ``mne.beamformer.lcmv_raw``, by `Britta Westner`_, `Alex Gramfort`_, and `Denis Engemann`_.

- Add new filtering mode ``fir_design='firwin'`` (default in the next 0.16 release) that gets improved attenuation using fewer samples compared to ``fir_design='firwin2'`` (default in the current 0.15 release) by `Eric Larson`_

- Make the goodness of fit (GOF) of the dipoles returned by :func:`mne.beamformer.rap_music` consistent with the GOF of dipoles returned by :func:`mne.fit_dipole` by `Alex Gramfort`_.

- :class:`mne.decoding.SlidingEstimator` will now replace ``mne.decoding.TimeDecoding`` to make it generic and fully compatible with scikit-learn, by `Jean-Remi King`_ and `Alex Gramfort`_

- :class:`mne.decoding.GeneralizingEstimator` will now replace ``mne.decoding.GeneralizationAcrossTime`` to make it generic and fully compatible with scikit-learn, by `Jean-Remi King`_ and `Alex Gramfort`_

- ``mne.viz.decoding.plot_gat_times``, ``mne.viz.decoding.plot_gat_matrix`` are now deprecated. Use matplotlib instead as shown in the examples, by `Jean-Remi King`_ and `Alex Gramfort`_

- Add ``norm_trace`` parameter to control single-epoch covariance normalization in :class:mne.decoding.CSP, by `Jean-Remi King`_

- Allow passing a list of channel names as ``show_names`` in function  :func:`mne.viz.plot_sensors` and methods :meth:`mne.Evoked.plot_sensors`, :meth:`mne.Epochs.plot_sensors` and :meth:`mne.io.Raw.plot_sensors` to show only a subset of channel names by `Jaakko Leppakangas`_

- Make function ``mne.io.eeglab.read_events_eeglab`` public to allow loading overlapping events from EEGLAB files, by `Jona Sassenhagen`_.

- :func:`mne.find_events` ``mask_type`` parameter will change from ``'not_and'`` to ``'and'`` in 0.16.

- Instead of raising an error, duplicate channel names in the data file are now appended with a running number by `Jaakko Leppakangas`_

- :func:`mne.io.read_raw_edf` has now ``'auto'`` option for ``stim_channel`` (default in version 0.16) that automatically detects if EDF annotations or GDF events exist in the header and constructs the stim channel based on these events by `Jaakko Leppakangas`_

- :meth:`mne.io.Raw.plot_psd` now rejects data annotated bad by default. Turn off with ``reject_by_annotation=False``, by `Eric Larson`_

- :func:`mne.set_eeg_reference` and the related methods (e.g., :meth:`mne.io.Raw.set_eeg_reference`) have a new argument ``projection``, which if set to False directly applies an average reference instead of adding an SSP projector, by `Clemens Brunner`_

- Deprecate ``plot_trans`` in favor of :func:`mne.viz.plot_alignment` and add ``bem`` parameter for plotting conductor model by `Jaakko Leppakangas`_

- :func:`mne.beamformer.tf_lcmv` now has a ``raw`` parameter to accommodate epochs objects that already have data loaded with ``preload=True``, with :meth:`mne.Epochs.load_data`, or that are read from disk, by `Eric Larson`_

- :func:`mne.time_frequency.psd_welch` and :func:`mne.time_frequency.psd_array_welch` now use a Hamming window (instead of a Hann window) by `Clemens Brunner`_

- ``picks`` parameter in ``mne.beamformer.lcmv``, ``mne.beamformer.lcmv_epochs``, ``mne.beamformer.lcmv_raw``, :func:`mne.beamformer.tf_lcmv` and :func:`mne.beamformer.rap_music` is now deprecated and will be removed in 0.16, by `Britta Westner`_, `Alex Gramfort`_, and `Denis Engemann`_.

- The keyword argument ``frequencies`` has been deprecated in favor of ``freqs`` in various time-frequency functions, e.g. :func:`mne.time_frequency.tfr_array_morlet`, by `Eric Larson`_

- Add ``patterns=False`` parameter in :class:`mne.decoding.ReceptiveField`. Turn on to compute inverse model coefficients, by `Nicolas Barascud`_

- The ``scale``, ``scale_time``, and ``unit`` parameters have been deprecated in favor of ``scalings``, ``scalings_time``, and ``units`` in :func:`mne.viz.plot_evoked_topomap` and related functions, by `Eric Larson`_

- ``loose`` parameter in inverse solvers has now a default value ``'auto'`` depending if the source space is a surface, volume, or discrete type by `Alex Gramfort`_ and `Yousra Bekhti`_

- The behavior of ``'mean_flip'`` label-flipping in :meth:`mne.extract_label_time_course` and related functions has been changed such that the flip, instead of having arbitrary sign, maximally aligns in the positive direction of the normals of the label, by `Eric Larson`_

- Deprecate force_fixed and surf_ori in :func:`mne.read_forward_solution` by `Daniel Strohmeier`_

- :func:`mne.convert_forward_solution` has a new argument ``use_cps``, which controls whether information on cortical patch statistics is applied while generating surface-oriented forward solutions with free and fixed orientation by `Daniel Strohmeier`_

- :func:`mne.write_forward_solution` writes a forward solution as a forward solution with free orientation in X/Y/Z RAS coordinates if it is derived from a forward solution with free orientation and as a forward solution with fixed orientation in surface-based local coordinates otherwise by `Daniel Strohmeier`_

- ``loose=None`` in inverse solvers is deprecated, use explicitly ``loose=0`` for fixed constraint and ``loose=1.0`` for free orientations by `Eric Larson`_

- Zero-channel-value in PSD calculation in :func:`mne.viz.plot_raw_psd` has been relaxed from error to warning by `Alejandro Weinstein`_

- Expose "rank" parameter in :func:`mne.viz.plot_evoked_white` to correct rank estimates on the spot during visualization by `Denis Engemann`_, `Eric Larson`_, `Alex Gramfort`_.

- Show channel name under mouse cursor on topography plots by `Jussi Nurminen`_

- Return maximum response amplitude from :meth:`mne.Evoked.get_peak`

Authors
~~~~~~~

People who contributed to this release  (in alphabetical order):

* akshay0724
* Alejandro Weinstein
* Alexander Rudiuk
* Alexandre Barachant
* Alexandre Gramfort
* Andrew Dykstra
* Britta Westner
* Chris Bailey
* Chris Holdgraf
* Christian Brodbeck
* Christopher Holdgraf
* Clemens Brunner
* Cristóbal Moënne-Loccoz
* Daniel McCloy
* Daniel Strohmeier
* Denis A. Engemann
* Emily P. Stephen
* Eric Larson
* Fede Raimondo
* Jaakko Leppakangas
* Jean-Baptiste Schiratti
* Jean-Remi King
* Jesper Duemose Nielsen
* Joan Massich
* Jon Houck
* Jona Sassenhagen
* Jussi Nurminen
* Laetitia Grabot
* Laura Gwilliams
* Luke Bloy
* Lukáš Hejtmánek
* Mainak Jas
* Marijn van Vliet
* Mathurin Massias
* Matt Boggess
* Mikolaj Magnuski
* Nicolas Barascud
* Nicole Proulx
* Phillip Alday
* Ramonapariciog Apariciogarcia
* Robin Tibor Schirrmeister
* Rodrigo Hübner
* S. M. Gutstein
* Simon Kern
* Teon Brooks
* Yousra Bekhti

.. _changes_0_14:

Version 0.14
------------

Changelog
~~~~~~~~~

- Add example of time-frequency decoding with CSP by `Laura Gwilliams`_

- Automatically create a legend in :func:`mne.viz.plot_evoked_topo` by `Jussi Nurminen`_

- Add I/O support for Artemis123 infant/toddler MEG data by `Luke Bloy`_

- Add filter plotting functions :func:`mne.viz.plot_filter` and :func:`mne.viz.plot_ideal_filter` as well as filter creation function :func:`mne.filter.create_filter` by `Eric Larson`_

- Add HCP-MMP1.0 parcellation dataset downloader by `Eric Larson`_

- Add option to project EEG electrodes onto the scalp in ``mne.viz.plot_trans`` by `Eric Larson`_

- Add option to plot individual sensors in :meth:`mne.io.Raw.plot_psd` by `Alex Gramfort`_ and `Eric Larson`_

- Add option to plot ECoG electrodes in ``mne.viz.plot_trans`` by `Eric Larson`_

- Add convenient default values to :meth:`mne.io.Raw.apply_hilbert` and :meth:`mne.io.Raw.apply_function` by `Denis Engemann`_

- Remove MNE-C requirement for :ref:`mne make_scalp_surfaces <gen_mne_make_scalp_surfaces>` by `Eric Larson`_

- Add support for FastTrack Polhemus ``.mat`` file outputs in ``hsp`` argument of :func:`mne.channels.read_dig_montage` by `Eric Larson`_

- Add option to convert 3d electrode plots to a snapshot with 2d electrode positions with :func:`mne.viz.snapshot_brain_montage` by `Chris Holdgraf`_

- Add skull surface plotting option to ``mne.viz.plot_trans`` by `Jaakko Leppakangas`_

- Add minimum-phase filtering option in :meth:`mne.io.Raw.filter` by `Eric Larson`_

- Add support for reading ASCII BrainVision files in :func:`mne.io.read_raw_brainvision` by `Eric Larson`_

- Add method of ICA objects for retrieving the component maps :meth:`mne.preprocessing.ICA.get_components` by `Jona Sassenhagen`_

- Add option to plot events in :func:`mne.viz.plot_epochs` by `Jaakko Leppakangas`_

- Add dipole definitions for older phantom at Otaniemi in :func:`mne.dipole.get_phantom_dipoles` by `Eric Larson`_

- Add spatial colors option for :func:`mne.viz.plot_raw_psd` by `Jaakko Leppakangas`_

- Add functions like :func:`get_volume_labels_from_src` to handle mixed source spaces by `Annalisa Pascarella`_

- Add convenience function for opening MNE documentation :func:`open_docs` by `Eric Larson`_

- Add option in :meth:`mne.io.Raw.plot` to display the time axis relative to ``raw.first_samp`` by `Mainak Jas`_

- Add new :mod:`mne.datasets.visual_92_categories <mne.datasets.visual_92_categories.data_path>` dataset by `Jaakko Leppakangas`_

- Add option in :func:`mne.io.read_raw_edf` to allow channel exclusion by `Jaakko Leppakangas`_

- Allow integer event codes in :func:`mne.read_epochs_eeglab` by `Jaakko Leppakangas`_

- Add ability to match channel names in a case insensitive manner when applying a :class:`mne.channels.Montage` by `Marijn van Vliet`_

- Add ``yscale`` keyword argument to :meth:`mne.time_frequency.AverageTFR.plot` that allows specifying whether to present the frequency axis in linear (``'linear'``) or log (``'log'``) scale. The default value is ``'auto'`` which detects whether frequencies are log-spaced and sets yscale to log. Added by `Mikołaj Magnuski`_

- Add :ref:`Representational Similarity Analysis (RSA) <ex-rsa-noplot>` example on :mod:`mne.datasets.visual_92_categories.data_path` dataset by `Jaakko Leppakangas`_, `Jean-Remi King`_ and `Alex Gramfort`_

- Add support for NeuroScan files with event type 3 in :func:`mne.io.read_raw_cnt` by `Marijn van Vliet`_

- Add interactive annotation mode to :meth:`mne.io.Raw.plot` (accessed by pressing 'a') by `Jaakko Leppakangas`_

- Add support for deleting all projectors or a list of indices in :meth:`mne.io.Raw.del_proj` by `Eric Larson`_

- Add source space plotting with :meth:`mne.SourceSpaces.plot` using ``mne.viz.plot_trans`` by `Eric Larson`_

- Add :func:`mne.decoding.get_coef` to retrieve and inverse the coefficients of a linear model - typically a spatial filter or pattern, by `Jean-Remi King`_

- Add support for reading in EGI MFF digitization coordinate files in :func:`mne.channels.read_dig_montage` by `Matt Boggess`_

- Add ``n_per_seg`` keyword argument to :func:`mne.time_frequency.psd_welch` and :func:`mne.time_frequency.psd_array_welch` that allows to control segment length independently of ``n_fft`` and use zero-padding when ``n_fft > n_per_seg`` by `Mikołaj Magnuski`_

- Add annotation aware data getter :meth:`mne.io.Raw.get_data` by `Jaakko Leppakangas`_

- Add support of dipole location visualization with MRI slice overlay with matplotlib to :func:`mne.viz.plot_dipole_locations` via mode='orthoview' parameter by `Jaakko Leppakangas`_ and `Alex Gramfort`_

- Add plotting of head positions as a function of time in :func:`mne.viz.plot_head_positions` by `Eric Larson`_

- Add ``real_filter`` option to ``mne.beamformer.dics``, ``mne.beamformer.dics_source_power``, ``mne.beamformer.tf_dics`` and ``mne.beamformer.dics_epochs`` by `Eric Larson`_, `Alex Gramfort`_ and `Andrea Brovelli`_.

- Add a demo script showing how to use a custom inverse solver with MNE by `Alex Gramfort`_

- Functions :func:`mne.preprocessing.create_ecg_epochs`, :func:`mne.preprocessing.create_eog_epochs`, :func:`mne.compute_raw_covariance` and ICA methods :meth:`mne.preprocessing.ICA.score_sources`, :meth:`mne.preprocessing.ICA.find_bads_ecg`, :meth:`mne.preprocessing.ICA.find_bads_eog` are now annotation aware by `Jaakko Leppakangas`_

- Allow using ``spatial_colors`` for non-standard layouts by creating custom layouts from channel locations and add ``to_sphere`` keyword to :func:`mne.viz.plot_sensors` to allow plotting sensors that are not on the head surface by `Jaakko Leppakangas`_

- Concatenating raws with :func:`mne.concatenate_raws` now creates boundary annotations automatically by `Jaakko Leppakangas`_

- :func:`mne.viz.plot_projs_topomap` now supports plotting EEG topomaps by passing in :class:`mne.Info` by `Eric Larson`_

BUG
~~~

- Fix bug with DICS and LCMV (e.g., ``mne.beamformer.lcmv``, ``mne.beamformer.dics``) where regularization was done improperly. The default ``reg=0.01`` has been changed to ``reg=0.05``, by `Andrea Brovelli`_, `Alex Gramfort`_, and `Eric Larson`_

- Fix callback function call in ``mne.viz.topo._plot_topo_onpick`` by `Erkka Heinila`_

- Fix reading multi-file CTF recordings in :func:`mne.io.read_raw_ctf` by `Niklas Wilming`_

- Fix computation of AR coefficients across channels in :func:`mne.time_frequency.fit_iir_model_raw` by `Eric Larson`_

- Fix maxfilter channel names extra space bug in :func:`mne.preprocessing.maxwell_filter` by `Sheraz Khan`_

- :func:`mne.channels.find_layout` now leaves out the excluded channels by `Jaakko Leppakangas`_

- Array data constructors :class:`mne.io.RawArray` and :class:`EvokedArray` now make a copy of the info structure by `Jaakko Leppakangas`_

- Fix bug with finding layouts in :func:`mne.viz.plot_projs_topomap` by `Eric Larson`_

- Fix bug :func:`mne.io.anonymize_info` when Info does not contain 'file_id' or 'meas_id' fields by `Jean-Remi King`_

- Fix colormap selection in :func:`mne.viz.plot_evoked_topomap` when using positive vmin with negative data by `Jaakko Leppakangas`_

- Fix channel name comparison in :func:`mne.channels.read_montage` so that if ``ch_names`` is provided, the returned montage will have channel names in the same letter case by `Jaakko Leppakangas`_

- Fix :meth:`inst.set_montage(montage) <mne.io.Raw.set_montage>` to only set ``inst.info['dev_head_t']`` if ``dev_head_t=True`` in :func:`mne.channels.read_dig_montage` by `Eric Larson`_

- Fix handling of events in ``mne.realtime.RtEpochs`` when the triggers were split between two buffers resulting in missing and/or duplicate epochs by `Mainak Jas`_ and `Antti Rantala`_

- Fix bug with automatic decimation in :func:`mne.io.read_raw_kit` by `Keith Doelling`_

- Fix bug with :func:`setup_volume_source_space` where arguments ``subject`` and ``subjects_dir`` were ignored by `Jaakko Leppakangas`_

- Fix sanity check for incompatible ``threshold`` and ``tail`` values in clustering functions like :func:`mne.stats.spatio_temporal_cluster_1samp_test` by `Eric Larson`_

- Fix ``_bad_dropped`` not being set when loading eeglab epoched files via :func:`mne.read_epochs_eeglab` which resulted in :func:`len` not working by `Mikołaj Magnuski`_

- Fix a bug in :meth:`mne.time_frequency.AverageTFR.plot` when plotting without a colorbar by `Jaakko Leppakangas`_

- Fix ``_filenames`` attribute in creation of :class:`mne.io.RawArray` with :meth:`mne.preprocessing.ICA.get_sources` by `Paul Pasler`_

- Fix contour levels in :func:`mne.viz.plot_evoked_topomap` to be uniform across topomaps by `Jaakko Leppakangas`_

- Fix bug in :func:`mne.preprocessing.maxwell_filter` where fine calibration indices were mismatched leading to an ``AssertionError`` by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.fix_stim_artifact` where non-data channels were interpolated by `Eric Larson`_

- :class:`mne.decoding.Scaler` now scales each channel independently using data from all time points (epochs and times) instead of scaling all channels for each time point. It also now accepts parameter ``scalings`` to determine the data scaling method (default is ``None`` to use static channel-type-based scaling), by `Asish Panda`_, `Jean-Remi King`_, and `Eric Larson`_

- Raise error if the cv parameter of ``mne.decoding.GeneralizationAcrossTime`` and ``mne.decoding.TimeDecoding`` is not a partition and the predict_mode is "cross-validation" by `Jean-Remi King`_

- Fix bug in :func:`mne.io.read_raw_edf` when ``preload=False`` and channels have different sampling rates by `Jaakko Leppakangas`_

- Fix :func:`mne.read_labels_from_annot` to set ``label.values[:]=1`` rather than 0 for consistency with the :class:`Label` class by `Jon Houck`_

- Fix plotting non-uniform freqs (for example log-spaced) in :meth:`mne.time_frequency.AverageTFR.plot` by `Mikołaj Magnuski`_

- Fix :func:`mne.minimum_norm.compute_source_psd` when used with ``pick_ori=None`` by `Annalisa Pascarella`_ and `Alex Gramfort`_

- Fix bug in :class:`mne.Annotations` where concatenating two raws where ``orig_time`` of the second run is ``None`` by `Jaakko Leppakangas`_

- Fix reading channel location from eeglab ``.set`` files when some of the channels do not provide this information. Previously all channel locations were ignored in such case, now they are read - unless a montage is provided by the user in which case only channel names are read from set file. By `Mikołaj Magnuski`_

- Fix reading eeglab ``.set`` files when ``.chanlocs`` structure does not contain ``X``, ``Y`` or ``Z`` fields by `Mikołaj Magnuski`_

- Fix bug with :func:`mne.simulation.simulate_raw` when ``interp != 'zero'`` by `Eric Larson`_

- Fix :func:`mne.fit_dipole` to handle sphere model rank deficiency properly by `Alex Gramfort`_

- Raise error in :func:`mne.concatenate_epochs` when concatenated epochs have conflicting event_id by `Mikołaj Magnuski`_

- Fix handling of ``n_components=None`` in :class:`mne.preprocessing.ICA` by `Richard Höchenberger`_

- Fix reading of fiducials correctly from CTF data in :func:`mne.io.read_raw_ctf` by `Jaakko Leppakangas`_

- Fix :func:`mne.beamformer.rap_music` to return dipoles with amplitudes in Am instead of nAm by `Jaakko Leppakangas`_

- Fix computation of duality gap in ``mne.inverse_sparse.mxne_optim.dgap_l21`` by `Mathurin Massias`_

API
~~~

- The filtering functions ``band_pass_filter``, ``band_stop_filter``, ``low_pass_filter``, and ``high_pass_filter`` have been deprecated in favor of :func:`mne.filter.filter_data` by `Eric Larson`_

- :class:`EvokedArray` now has default value ``tmin=0.`` by `Jaakko Leppakangas`_

- The ``ch_type`` argument for ``mne.viz.plot_trans`` has been deprecated, use ``eeg_sensors`` and ``meg_sensors`` instead, by `Eric Larson`_

- The default ``tmax=60.`` in :meth:`mne.io.Raw.plot_psd` will change to ``tmax=np.inf`` in 0.15, by `Eric Larson`_

- Base classes :class:`mne.io.BaseRaw` and :class:`mne.BaseEpochs` are now public to allow easier typechecking, by `Daniel McCloy`_

- :func:`mne.io.read_raw_edf` now combines triggers from multiple tal channels to 'STI 014' by `Jaakko Leppakangas`_

- The measurement info :class:`Info` no longer contains a potentially misleading ``info['filename']`` entry. Use class properties like :attr:`mne.io.Raw.filenames` or :attr:`mne.Epochs.filename` instead by `Eric Larson`_

- Default fiducial name change from 'nz' to 'nasion' in :func:`mne.channels.read_montage`, so that it is the same for both :class: `mne.channels.Montage` and :class: `mne.channels.DigMontage` by `Leonardo Barbosa`_

- MNE's additional files for the ``fsaverage`` head/brain model are now included in MNE-Python, and the now superfluous ``mne_root`` parameter to  :func:`create_default_subject` has been deprecated by `Christian Brodbeck`_

- An ``overwrite=False`` default parameter has been added to :func:`write_source_spaces` to protect against accidental overwrites, by `Eric Larson`_

- The :class:`mne.decoding.LinearModel` class will no longer support `plot_filters` and `plot_patterns`, use :class:`mne.EvokedArray` with :func:`mne.decoding.get_coef` instead, by `Jean-Remi King`_

- Made functions :func:`mne.time_frequency.tfr_array_multitaper`, :func:`mne.time_frequency.tfr_array_morlet`, :func:`mne.time_frequency.tfr_array_stockwell`, :func:`mne.time_frequency.psd_array_multitaper` and :func:`mne.time_frequency.psd_array_welch` public to allow computing TFRs and PSDs on numpy arrays by `Jaakko Leppakangas`_

- :meth:`mne.preprocessing.ICA.fit` now rejects data annotated bad by default. Turn off with ``reject_by_annotation=False``, by `Jaakko Leppakangas`_

- :func:`mne.io.read_raw_egi` now names channels with pattern 'E<idx>'. This behavior can be changed with parameter ``channel_naming`` by `Jaakko Leppakangas`_

- the `name`` parameter in :class:`mne.Epochs` is deprecated, by `Jaakko Leppakangas`_

Authors
~~~~~~~

People who contributed to this release  (in alphabetical order):

* Alexander Rudiuk
* Alexandre Gramfort
* Annalisa Pascarella
* Antti Rantala
* Asish Panda
* Burkhard Maess
* Chris Holdgraf
* Christian Brodbeck
* Cristóbal Moënne-Loccoz
* Daniel McCloy
* Denis A. Engemann
* Eric Larson
* Erkka Heinila
* Hermann Sonntag
* Jaakko Leppakangas
* Jakub Kaczmarzyk
* Jean-Remi King
* Jon Houck
* Jona Sassenhagen
* Jussi Nurminen
* Keith Doelling
* Leonardo S. Barbosa
* Lorenz Esch
* Lorenzo Alfine
* Luke Bloy
* Mainak Jas
* Marijn van Vliet
* Matt Boggess
* Matteo Visconti
* Mikolaj Magnuski
* Niklas Wilming
* Paul Pasler
* Richard Höchenberger
* Sheraz Khan
* Stefan Repplinger
* Teon Brooks
* Yaroslav Halchenko

.. _changes_0_13:

Version 0.13
------------

Changelog
~~~~~~~~~

- Add new class :class:`AcqParserFIF` to parse Elekta/Neuromag MEG acquisition info, allowing e.g. collecting epochs according to acquisition-defined averaging categories by `Jussi Nurminen`_

- Adds automatic determination of FIR filter parameters ``filter_length``, ``l_trans_bandwidth``, and ``h_trans_bandwidth`` and adds ``phase`` argument in e.g. in :meth:`mne.io.Raw.filter` by `Eric Larson`_

- Adds faster ``n_fft='auto'`` option to :meth:`mne.io.Raw.apply_hilbert` by `Eric Larson`_

- Adds new function ``mne.time_frequency.csd_array`` to compute the cross-spectral density of multivariate signals stored in an array, by `Nick Foti`_

- Add order params 'selection' and 'position' for :func:`mne.viz.plot_raw` to allow plotting of specific brain regions by `Jaakko Leppakangas`_

- Added the ability to decimate :class:`mne.Evoked` objects with :func:`mne.Evoked.decimate` by `Eric Larson`_

- Add generic array-filtering function :func:`mne.filter.filter_data` by `Eric Larson`_

- ``mne.viz.plot_trans`` now also shows head position indicators by `Christian Brodbeck`_

- Add label center of mass function :func:`mne.Label.center_of_mass` by `Eric Larson`_

- Added :func:`mne.viz.plot_ica_properties` that allows plotting of independent component properties similar to ``pop_prop`` in EEGLAB. Also :class:`mne.preprocessing.ICA` has :func:`mne.preprocessing.ICA.plot_properties` method now. Added by `Mikołaj Magnuski`_

- Add second-order sections (instead of ``(b, a)`` form) IIR filtering for reduced numerical error by `Eric Larson`_

- Add interactive colormap option to image plotting functions by `Jaakko Leppakangas`_

- Add support for the University of Maryland KIT system by `Christian Brodbeck`_

- Add support for \*.elp and \*.hsp files to the KIT2FIFF converter and :func:`mne.channels.read_dig_montage` by `Teon Brooks`_ and `Christian Brodbeck`_

- Add option to preview events in the KIT2FIFF GUI by `Christian Brodbeck`_

- Add approximation of size of :class:`io.Raw`, :class:`Epochs`, and :class:`Evoked` in :func:`repr` by `Eric Larson`_

- Add possibility to select a subset of sensors by lasso selector to :func:`mne.viz.plot_sensors` and :func:`mne.viz.plot_raw` when using order='selection' or order='position' by `Jaakko Leppakangas`_

- Add the option to plot brain surfaces and source spaces to :func:`viz.plot_bem` by `Christian Brodbeck`_

- Add the ``--filterchpi`` option to :ref:`mne browse_raw <gen_mne_browse_raw>`, by `Felix Raimundo`_

- Add the ``--no-decimate`` option to :ref:`mne make_scalp_surfaces <gen_mne_make_scalp_surfaces>` to skip the high-resolution surface decimation step, by `Eric Larson`_

- Add new class :class:`mne.decoding.EMS` to transform epochs with the event-matched spatial filters and add 'cv' parameter to :func:`mne.decoding.compute_ems`, by `Jean-Remi King`_

- Added :class:`mne.time_frequency.EpochsTFR` and average parameter in :func:`mne.time_frequency.tfr_morlet` and :func:`mne.time_frequency.tfr_multitaper` to compute time-frequency transforms on single trial epochs without averaging, by `Jean-Remi King`_ and `Alex Gramfort`_

- Added :class:`mne.decoding.TimeFrequency` to transform signals in scikit-learn pipelines, by `Jean-Remi King`_

- Added :class:`mne.decoding.UnsupervisedSpatialFilter` providing interface for scikit-learn decomposition algorithms to be used with MNE data, by `Jean-Remi King`_ and `Asish Panda`_

- Added support for multiclass decoding in :class:`mne.decoding.CSP`, by `Jean-Remi King`_ and `Alexandre Barachant`_

- Components obtained from :class:`mne.preprocessing.ICA` are now sorted by explained variance, by `Mikołaj Magnuski`_

- Adding an EEG reference channel using :func:`mne.add_reference_channels` will now use its digitized location from the FIFF file, if present, by `Chris Bailey`_

- Added interactivity to :func:`mne.preprocessing.ICA.plot_components` - passing an instance of :class:`io.Raw` or :class:`Epochs` in ``inst`` argument allows to open component properties by clicking on component topomaps, by `Mikołaj Magnuski`_

- Adds new function :func:`mne.viz.plot_compare_evokeds` to show multiple evoked time courses at a single location, or the mean over a ROI, or the GFP, automatically averaging and calculating a CI if multiple subjects are given, by `Jona Sassenhagen`_

- Added `transform_into` parameter into :class:`mne.decoding.CSP` to retrieve the average power of each source or the time course of each source, by `Jean-Remi King`_

- Added support for reading MaxShield (IAS) evoked data (e.g., from the acquisition machine) in :func:`mne.read_evokeds` by `Eric Larson`_

- Added support for functional near-infrared spectroscopy (fNIRS) channels by `Jaakko Leppakangas`_

- Added :attr:`mne.io.Raw.acqparser` convenience attribute for :class:`mne.AcqParserFIF` by `Eric Larson`_

- Added example of Representational Similarity Analysis, by `Jean-Remi King`_

BUG
~~~

- Fixed a bug where selecting epochs using hierarchical event IDs (HIDs) was *and*-like instead of *or*-like. When doing e.g. ``epochs[('Auditory', 'Left')]``, previously all trials that contain ``'Auditory'`` *and* ``'Left'`` (like ``'Auditory/Left'``) would be selected, but now any conditions matching ``'Auditory'`` *or* ``'Left'`` will be selected (like ``'Auditory/Left'``, ``'Auditory/Right'``, and ``'Visual/Left'``). This is now consistent with how epoch selection was done without HID tags, e.g. ``epochs[['a', 'b']]`` would select all epochs of type ``'a'`` and type ``'b'``. By `Eric Larson`_

- Fixed Infomax/Extended Infomax when the user provides an initial weights matrix by `Jair Montoya Martinez`_

- Fixed the default raw FIF writing buffer size to be 1 second instead of 10 seconds by `Eric Larson`_

- Fixed channel selection order when MEG channels do not come first in :func:`mne.preprocessing.maxwell_filter` by `Eric Larson`_

- Fixed color ranges to correspond to the colorbar when plotting several time instances with :func:`mne.viz.plot_evoked_topomap` by `Jaakko Leppakangas`_

- Added units to :func:`mne.io.read_raw_brainvision` for reading non-data channels and enable default behavior of inferring channel type by unit by `Jaakko Leppakangas`_ and `Pablo-Arias`_

- Fixed minor bugs with :func:`mne.Epochs.resample` and :func:`mne.Epochs.decimate` by `Eric Larson`_

- Fixed a bug where duplicate vertices were not strictly checked by :func:`mne.simulation.simulate_stc` by `Eric Larson`_

- Fixed a bug where some FIF files could not be read with :func:`mne.io.show_fiff` by `Christian Brodbeck`_ and `Eric Larson`_

- Fixed a bug where ``merge_grads=True`` causes :func:`mne.viz.plot_evoked_topo` to fail when plotting a list of evokeds by `Jaakko Leppakangas`_

- Fixed a bug when setting multiple bipolar references with :func:`set_bipolar_reference` by `Marijn van Vliet`_.

- Fixed image scaling in :func:`mne.viz.plot_epochs_image` when plotting more than one channel by `Jaakko Leppakangas`_

- Fixed :class:`mne.preprocessing.Xdawn` to fit shuffled epochs by `Jean-Remi King`_

- Fixed a bug with channel order determination that could lead to an ``AssertionError`` when using :class:`mne.Covariance` matrices by `Eric Larson`_

- Fixed the check for CTF gradient compensation in :func:`mne.preprocessing.maxwell_filter` by `Eric Larson`_

- Fixed the import of EDF files with encoding characters in :func:`mne.io.read_raw_edf` by `Guillaume Dumas`_

- Fixed :class:`mne.Epochs` to ensure that detrend parameter is not a boolean by `Jean-Remi King`_

- Fixed bug with ``mne.realtime.FieldTripClient.get_data_as_epoch`` when ``picks=None`` which crashed the function by `Mainak Jas`_

- Fixed reading of units in ``.elc`` montage files (from ``UnitsPosition`` field) so that :class:`mne.channels.Montage` objects are now returned with the ``pos`` attribute correctly in meters, by `Chris Mullins`_

- Fixed reading of BrainVision files by `Phillip Alday`_:

- Greater support for BVA files, especially older ones: alternate text coding schemes with fallback to Latin-1 as well as units in column headers

- Use online software filter information when present

- Fix comparisons of filter settings for determining "strictest"/"weakest" filter

- Weakest filter is now used for heterogeneous channel filter settings, leading to more consistent behavior with filtering methods applied to a subset of channels (e.g. ``Raw.filter`` with ``picks != None``).

- Fixed plotting and timing of :class:`Annotations` and restricted addition of annotations outside data range to prevent problems with cropping and concatenating data by `Jaakko Leppakangas`_

- Fixed ICA plotting functions to refer to IC index instead of component number by `Andreas Hojlund`_ and `Jaakko Leppakangas`_

- Fixed bug with ``picks`` when interpolating MEG channels by `Mainak Jas`_.

- Fixed bug in padding of Stockwell transform for signal of length a power of 2 by `Johannes Niediek`_

API
~~~

- The ``add_eeg_ref`` argument in core functions like :func:`mne.io.read_raw_fif` and :class:`mne.Epochs` has been deprecated in favor of using :func:`mne.set_eeg_reference` and equivalent instance methods like :meth:`raw.set_eeg_reference() <mne.io.Raw.set_eeg_reference>`. In functions like :func:`mne.io.read_raw_fif` where the default in 0.13 and older versions is ``add_eeg_ref=True``, the default will change to ``add_eeg_ref=False`` in 0.14, and the argument will be removed in 0.15.

- Multiple aspects of FIR filtering in MNE-Python has been refactored:

  1. New recommended defaults for ``l_trans_bandwidth='auto'``, ``h_trans_bandwidth='auto'``, and ``filter_length='auto'``. This should generally reduce filter artifacts at the expense of slight decrease in effective filter stop-band attenuation. For details see :ref:`tut_filtering_in_python`. The default values of ``l_trans_bandwidth=h_trans_bandwidth=0.5`` and ``filter_length='10s'`` will change to ``'auto'`` in 0.14.

  2. The ``filter_length=None`` option (i.e. use ``len(x)``) has been deprecated.

  3. An improved ``phase='zero'`` zero-phase FIR filtering has been added. Instead of running the designed filter forward and backward, the filter is applied once and we compensate for the linear phase of the filter. The previous ``phase='zero-double'`` default will change to ``phase='zero'`` in 0.14.

  4. A warning is provided when the filter is longer than the signal of interest, as this is unlikely to produce desired results.

  5. Previously, if the filter was as long or longer than the signal of interest, direct FFT-based computations were used. Now a single code path (overlap-add filtering) is used for all FIR filters. This could cause minor changes in how short signals are filtered.

- Support for Python 2.6 has been dropped, and the minimum supported dependencies are NumPy_ 1.8, SciPy_ 0.12, and Matplotlib_ 1.3 by `Eric Larson`_

- When CTF gradient compensation is applied to raw data, it is no longer reverted on save of :meth:`mne.io.Raw.save` by `Eric Larson`_

- Adds ``mne.time_frequency.csd_epochs`` to replace ``mne.time_frequency.csd_compute_epochs`` for naming consistency. ``mne.time_frequency.csd_compute_epochs`` is now deprecated and will be removed in mne 0.14, by `Nick Foti`_

- Weighted addition and subtraction of :class:`Evoked` as ``ev1 + ev2`` and ``ev1 - ev2`` have been deprecated, use explicit :func:`mne.combine_evoked(..., weights='nave') <mne.combine_evoked>` instead by `Eric Larson`_

- Deprecated support for passing a list of filenames to :class:`mne.io.Raw` constructor, use :func:`mne.io.read_raw_fif` and :func:`mne.concatenate_raws` instead by `Eric Larson`_

- Added options for setting data and date formats manually in :func:`mne.io.read_raw_cnt` by `Jaakko Leppakangas`_

- Now channels with units of 'C', 'µS', 'uS', 'ARU' and 'S' will be turned to misc by default in :func:`mne.io.read_raw_brainvision` by `Jaakko Leppakangas`_

- Add :func:`mne.io.anonymize_info` function to anonymize measurements and add methods to :class:`mne.io.Raw`, :class:`mne.Epochs` and :class:`mne.Evoked`, by `Jean-Remi King`_

- Now it is possible to plot only a subselection of channels in :func:`mne.viz.plot_raw` by using an array for order parameter by `Jaakko Leppakangas`_

- EOG channels can now be included when calling :func:`mne.preprocessing.ICA.fit` and a proper error is raised when trying to include unsupported channels by `Alexander Rudiuk`_

- :func:`mne.concatenate_epochs` and :func:`mne.compute_covariance` now check to see if all :class:`Epochs` instances have the same MEG-to-Head transformation, and errors by default if they do not by `Eric Larson`_

- Added option to pass a list of axes to :func:`mne.viz.plot_epochs_image` by `Mikołaj Magnuski`_

- Constructing IIR filters in :func:`mne.filter.construct_iir_filter` defaults to ``output='ba'`` in 0.13 but this will be changed to ``output='sos'`` by `Eric Larson`_

- Add ``zorder`` parameter to :func:`mne.Evoked.plot` and derived functions to sort allow sorting channels by e.g. standard deviation, by `Jona Sassenhagen`_

- The ``baseline`` parameter of :func:`mne.Epochs.apply_baseline` is set by default (None, 0), by `Felix Raimundo`_

- Adds :func:`mne.Evoked.apply_baseline` to be consistent with :func:`mne.Epochs.apply_baseline`, by `Felix Raimundo`_

- Deprecated the ``baseline`` parameter in :class:`mne.Evoked`, by `Felix Raimundo`_

- The API of :meth:`mne.SourceEstimate.plot` and :func:`mne.viz.plot_source_estimates` has been updated to reflect current PySurfer 0.6 API. The ``config_opts`` parameter is now deprecated and will be removed in mne 0.14, and the default representation for time will change from ``ms`` to ``s`` in mne 0.14. By `Christian Brodbeck`_

- The default dataset location has been changed from ``examples/`` in the MNE-Python root directory to ``~/mne_data`` in the user's home directory, by `Eric Larson`_

- A new option ``set_env`` has been added to :func:`mne.set_config` that defaults to ``False`` in 0.13 but will change to ``True`` in 0.14, by `Eric Larson`_

- The ``compensation`` parameter in :func:`mne.io.read_raw_fif` has been deprecated in favor of the method :meth:`mne.io.Raw.apply_gradient_compensation` by `Eric Larson`_

- ``mne.decoding.EpochsVectorizer`` has been deprecated in favor of :class:`mne.decoding.Vectorizer` by `Asish Panda`_

- The `epochs_data` parameter has been deprecated in :class:`mne.decoding.CSP`, in favour of the ``X`` parameter to comply to scikit-learn API, by `Jean-Remi King`_

- Deprecated ``mne.time_frequency.cwt_morlet`` and ``mne.time_frequency.single_trial_power`` in favour of :func:`mne.time_frequency.tfr_morlet` with parameter average=False, by `Jean-Remi King`_ and `Alex Gramfort`_

- Add argument ``mask_type`` to func:`mne.read_events` and func:`mne.find_events` to support MNE-C style of trigger masking by `Teon Brooks`_ and `Eric Larson`_

- Extended Infomax is now the new default in :func:`mne.preprocessing.infomax` (``extended=True``), by `Clemens Brunner`_

- :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` now take additional argument ``uint16_codec`` that allows to define the encoding of character arrays in set file. This helps in rare cases when reading a set file fails with ``TypeError: buffer is too small for requested array``. By `Mikołaj Magnuski`_

- Added :class:`mne.decoding.TemporalFilter` to filter data in scikit-learn pipelines, by `Asish Panda`_

- :func:`mne.preprocessing.create_ecg_epochs` now includes all the channels when ``picks=None`` by `Jaakko Leppakangas`_

- :func:`mne.set_eeg_reference` now allows moving from a custom to an average EEG reference by `Marijn van Vliet`_

Authors
~~~~~~~

The committer list for this release is the following (sorted by alphabetical order):

* Alexander Rudiuk
* Alexandre Barachant
* Alexandre Gramfort
* Asish Panda
* Camilo Lamus
* Chris Holdgraf
* Christian Brodbeck
* Christopher J. Bailey
* Christopher Mullins
* Clemens Brunner
* Denis A. Engemann
* Eric Larson
* Federico Raimondo
* Félix Raimundo
* Guillaume Dumas
* Jaakko Leppakangas
* Jair Montoya
* Jean-Remi King
* Johannes Niediek
* Jona Sassenhagen
* Jussi Nurminen
* Keith Doelling
* Mainak Jas
* Marijn van Vliet
* Michael Krause
* Mikolaj Magnuski
* Nick Foti
* Phillip Alday
* Simon-Shlomo Poil
* Teon Brooks
* Yaroslav Halchenko

.. _changes_0_12:

Version 0.12
------------

Changelog
~~~~~~~~~

- Add ``overlay_times`` parameter to :func:`mne.viz.plot_epochs_image` to be able to display for example reaction times on top of the images, by `Alex Gramfort`_

- Animation for evoked topomap in :func:`mne.Evoked.animate_topomap` by `Jaakko Leppakangas`_

- Make :func:`mne.channels.find_layout` more robust for KIT systems in the presence of bad or missing channels by `Jaakko Leppakangas`_

- Add raw movement compensation to :func:`mne.preprocessing.maxwell_filter` by `Eric Larson`_

- Add :class:`mne.Annotations` for for annotating segments of raw data by `Jaakko Leppakangas`_

- Add reading of .fif file montages by `Eric Larson`_

- Add system config utility :func:`mne.sys_info` by `Eric Larson`_

- Automatic cross-validation and scoring metrics in ``mne.decoding.GeneralizationAcrossTime``, by `Jean-Remi King`_

- ``mne.decoding.GeneralizationAcrossTime`` accepts non-deterministic cross-validations, by `Jean-Remi King`_

- Add plotting RMS of gradiometer pairs in :func:`mne.viz.plot_evoked_topo` by `Jaakko Leppakangas`_

- Add regularization methods to :func:`mne.compute_raw_covariance` by `Eric Larson`_.

- Add command ``mne show_info`` to quickly show the measurement info from a .fif file from the terminal by `Alex Gramfort`_.

- Add creating forward operator for dipole object :func:`mne.make_forward_dipole` by `Chris Bailey`_

- Add reading and estimation of fixed-position dipole time courses (similar to Elekta ``xfit``) using :func:`mne.read_dipole` and :func:`mne.fit_dipole` by `Eric Larson`_.

- Accept ``mne.decoding.GeneralizationAcrossTime``'s ``scorer`` parameter to be a string that refers to a scikit-learn_ metric scorer by `Asish Panda`_.

- Add method :func:`mne.Epochs.plot_image` calling :func:`mne.viz.plot_epochs_image` for better usability by `Asish Panda`_.

- Add :func:`mne.io.read_raw_cnt` for reading Neuroscan CNT files by `Jaakko Leppakangas`_

- Add ``decim`` parameter to ``mne.time_frequency.cwt_morlet``, by `Jean-Remi King`_

- Add method :func:`mne.Epochs.plot_topo_image` by `Jaakko Leppakangas`_

- Add the ability to read events when importing raw EEGLAB files, by `Jona Sassenhagen`_.

- Add function :func:`mne.viz.plot_sensors` and methods :func:`mne.Epochs.plot_sensors`, :func:`mne.io.Raw.plot_sensors` and :func:`mne.Evoked.plot_sensors` for plotting sensor positions and :func:`mne.viz.plot_layout` and :func:`mne.channels.Layout.plot` for plotting layouts by `Jaakko Leppakangas`_

- Add epoch rejection based on annotated segments by `Jaakko Leppakangas`_

- Add option to use new-style MEG channel names in :func:`mne.read_selection` by `Eric Larson`_

- Add option for ``proj`` in :class:`mne.EpochsArray` by `Eric Larson`_

- Enable the usage of :func:`mne.viz.plot_topomap` with an :class:`mne.Info` instance for location information, by `Jona Sassenhagen`_.

- Add support for electrocorticography (ECoG) channel type by `Eric Larson`_

- Add option for ``first_samp`` in :func:`mne.make_fixed_length_events` by `Jon Houck`_

- Add ability to auto-scale channel types for :func:`mne.viz.plot_raw` and :func:`mne.viz.plot_epochs` and corresponding object plotting methods by `Chris Holdgraf`_

BUG
~~~

- ``mne.time_frequency.compute_raw_psd``, ``mne.time_frequency.compute_epochs_psd``, :func:`mne.time_frequency.psd_multitaper`, and :func:`mne.time_frequency.psd_welch` no longer remove rows/columns of the SSP matrix before applying SSP projectors when picks are provided by `Chris Holdgraf`_.

- :func:`mne.Epochs.plot_psd` no longer calls a Welch PSD, and instead uses a Multitaper method which is more appropriate for epochs. Flags for this function are passed to :func:`mne.time_frequency.psd_multitaper` by `Chris Holdgraf`_

- Time-cropping functions (e.g., :func:`mne.Epochs.crop`, :func:`mne.Evoked.crop`, :func:`mne.io.Raw.crop`, :func:`mne.SourceEstimate.crop`) made consistent with behavior of ``tmin`` and ``tmax`` of :class:`mne.Epochs`, where nearest sample is kept. For example, for MGH data acquired with ``sfreq=600.614990234``, constructing ``Epochs(..., tmin=-1, tmax=1)`` has bounds ``+/-1.00064103``, and now ``epochs.crop(-1, 1)`` will also have these bounds (previously they would have been ``+/-0.99897607``). Time cropping functions also no longer use relative tolerances when determining the boundaries. These changes have minor effects on functions that use cropping under the hood, such as :func:`mne.compute_covariance` and :func:`mne.connectivity.spectral_connectivity`. Changes by `Jaakko Leppakangas`_ and `Eric Larson`_

- Fix EEG spherical spline interpolation code to account for average reference by `Mainak Jas`_

- MEG projectors are removed after Maxwell filtering by `Eric Larson`_

- Fix ``mne.decoding.TimeDecoding`` to allow specifying ``clf`` by `Jean-Remi King`_

- Fix bug with units (uV) in 'Brain Vision Data Exchange Header File Version 1.0' by `Federico Raimondo`_

- Fix bug where :func:`mne.preprocessing.maxwell_filter` ``destination`` parameter did not properly set device-to-head transform by `Eric Larson`_

- Fix bug in rank calculation of ``mne.utils.estimate_rank``, :func:`mne.io.Raw.estimate_rank`, and covariance functions where the tolerance was set to slightly too small a value, new 'auto' mode uses values from ``scipy.linalg.orth`` by `Eric Larson`_.

- Fix bug when specifying irregular ``train_times['slices']`` in ``mne.decoding.GeneralizationAcrossTime``, by `Jean-Remi King`_

- Fix colorbar range on norm data by `Jaakko Leppakangas`_

- Fix bug in :func:`mne.preprocessing.run_ica`, which used the ``ecg_criterion`` parameter for the EOG criterion instead of ``eog_criterion`` by `Christian Brodbeck`_

- Fix normals in CTF data reader by `Eric Larson`_

- Fix bug in :func:`mne.io.read_raw_ctf`, when omitting samples at the end by `Jaakko Leppakangas`_

- Fix ``info['lowpass']`` value for downsampled raw data by `Eric Larson`_

- Remove measurement date from :class:`mne.Info` in :func:`mne.io.Raw.anonymize` by `Eric Larson`_

- Fix bug that caused synthetic ecg channel creation even if channel was specified for ECG peak detection in :func:`mne.preprocessing.create_ecg_epochs` by `Jaakko Leppakangas`_

- Fix bug with vmin and vmax when None is passed in :func:`mne.viz.plot_topo_image_epochs` by `Jaakko Leppakangas`_

- Fix bug with :func:`mne.label_sign_flip` (and :func:`mne.extract_label_time_course`) by `Natalie Klein`_ and `Eric Larson`_

- Add copy parameter in :func:`mne.Epochs.apply_baseline` and :func:`mne.io.Raw.filter` methods by `Jona Sassenhagen`_ and `Alex Gramfort`_

- Fix bug in :func:`mne.merge_events` when using ``replace_events=False`` by `Alex Gramfort`_

- Fix bug in :class:`mne.Evoked` type setting in :func:`mne.stats.linear_regression_raw` by `Eric Larson`_

- Fix bug in :class: `mne.io.edf.RawEDF` highpass filter setting to take max highpass to match warning message by `Teon Brooks`_

- Fix bugs with coordinane frame adjustments in ``mne.viz.plot_trans`` by `Eric Larson`_

- Fix bug in colormap selection in :func:`mne.Evoked.plot_projs_topomap` by `Jaakko Leppakangas`_

- Fix bug in source normal adjustment that occurred when 1) patch information is available (e.g., when distances have been calculated) and 2) points are excluded from the source space (by inner skull distance) by `Eric Larson`_

- Fix bug when merging info that has a field with list of dicts by `Jaakko Leppakangas`_

- The BTi/4D reader now considers user defined channel labels instead of the hard-ware names, however only for channels other than MEG. By `Denis Engemann`_ and `Alex Gramfort`_.

- The BTi reader :func:`mne.io.read_raw_bti` can now read 2500 system data, by `Eric Larson`_

- Fix bug in :func:`mne.compute_raw_covariance` where rejection by non-data channels (e.g. EOG) was not done properly by `Eric Larson`_.

- Change default scoring method of ``mne.decoding.GeneralizationAcrossTime`` and ``mne.decoding.TimeDecoding`` to estimate the scores within the cross-validation as in scikit-learn_ as opposed to across all cross-validated ``y_pred``. The method can be changed with the ``score_mode`` parameter by `Jean-Remi King`_

- Fix bug in :func:`mne.io.Raw.save` where, in rare cases, automatically split files could end up writing an extra empty file that wouldn't be read properly by `Eric Larson`_

- Fix :class:``mne.realtime.StimServer`` by removing superfluous argument ``ip`` used while initializing the object by `Mainak Jas`_.

- Fix removal of projectors in :func:`mne.preprocessing.maxwell_filter` in ``st_only=True`` mode by `Eric Larson`_

API
~~~

- The default `picks=None` in :func:`mne.viz.plot_epochs_image` now only plots the first 5 channels, not all channels, by `Jona Sassenhagen`_

- The ``mesh_color`` parameter in :func:`mne.viz.plot_dipole_locations` has been removed (use `brain_color` instead), by `Marijn van Vliet`_

- Deprecated functions ``mne.time_frequency.compute_raw_psd`` and ``mne.time_frequency.compute_epochs_psd``, replaced by :func:`mne.time_frequency.psd_welch` by `Chris Holdgraf`_

- Deprecated function ``mne.time_frequency.multitaper_psd`` and replaced by :func:`mne.time_frequency.psd_multitaper` by `Chris Holdgraf`_

- The ``y_pred`` attribute in ``mne.decoding.GeneralizationAcrossTime`` and ``mne.decoding.TimeDecoding`` is now a numpy array, by `Jean-Remi King`_

- The :func:`mne.bem.fit_sphere_to_headshape` function now default to ``dig_kinds='auto'`` which will use extra digitization points, falling back to extra plus eeg digitization points if there not enough extra points are available.

- The :func:`mne.bem.fit_sphere_to_headshape` now has a ``units`` argument that should be set explicitly. This will default to ``units='mm'`` in 0.12 for backward compatibility but change to ``units='m'`` in 0.13.

- Added default parameters in Epochs class namely ``event_id=None``, ``tmin=-0.2`` and ``tmax=0.5``.

- To unify and extend the behavior of :func:`mne.compute_raw_covariance` relative to :func:`mne.compute_covariance`, the default parameter ``tstep=0.2`` now discards any epochs at the end of the :class:`mne.io.Raw` instance that are not the full ``tstep`` duration. This will slightly change the computation of :func:`mne.compute_raw_covariance`, but should only potentially have a big impact if the :class:`mne.io.Raw` instance is short relative to ``tstep`` and the last, too short (now discarded) epoch contained data inconsistent with the epochs that preceded it.

- The default ``picks=None`` in :func:`mne.io.Raw.filter` now picks eeg, meg, seeg, and ecog channels, by `Jean-Remi King`_ and `Eric Larson`_

- EOG, ECG and EMG channels are now plotted by default (if present in data) when using :func:`mne.viz.plot_evoked` by `Marijn van Vliet`_

- Replace pseudoinverse-based solver with much faster Cholesky solver in :func:`mne.stats.linear_regression_raw`, by `Jona Sassenhagen`_.

- CTF data reader now reads EEG locations from .pos file as HPI points by `Jaakko Leppakangas`_

- Subselecting channels can now emit a warning if many channels have been subselected from projection vectors. We recommend only computing projection vertors for and applying projectors to channels that will be used in the final analysis. However, after picking a subset of channels, projection vectors can be renormalized with :func:`mne.Info.normalize_proj` if necessary to avoid warnings about subselection. Changes by `Eric Larson`_ and `Alex Gramfort`_.

- Rename and deprecate ``mne.Epochs.drop_bad_epochs`` to :func:`mne.Epochs.drop_bad`, and `mne.Epochs.drop_epochs`` to :func:`mne.Epochs.drop` by `Alex Gramfort`_.

- The C wrapper ``mne.do_forward_solution`` has been deprecated in favor of the native Python version :func:`mne.make_forward_solution` by `Eric Larson`_

- The ``events`` parameter of :func:`mne.EpochsArray` is set by default to chronological time-samples and event values to 1, by `Jean-Remi King`_

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

* 348	Eric Larson
* 347	Jaakko Leppakangas
* 157	Alexandre Gramfort
* 139	Jona Sassenhagen
* 67	Jean-Remi King
* 32	Chris Holdgraf
* 31	Denis A. Engemann
* 30	Mainak Jas
* 16	Christopher J. Bailey
* 13	Marijn van Vliet
* 10	Mark Wronkiewicz
* 9	Teon Brooks
* 9	kaichogami
* 8	Clément Moutard
* 5	Camilo Lamus
* 5	mmagnuski
* 4	Christian Brodbeck
* 4	Daniel McCloy
* 4	Yousra Bekhti
* 3	Fede Raimondo
* 1	Jussi Nurminen
* 1	MartinBaBer
* 1	Mikolaj Magnuski
* 1	Natalie Klein
* 1	Niklas Wilming
* 1	Richard Höchenberger
* 1	Sagun Pai
* 1	Sourav Singh
* 1	Tom Dupré la Tour
* 1	jona-sassenhagen@
* 1	kambysese
* 1	pbnsilva
* 1	sviter
* 1	zuxfoucault

.. _changes_0_11:

Version 0.11
------------

Changelog
~~~~~~~~~

- Maxwell filtering (SSS) implemented in :func:`mne.preprocessing.maxwell_filter` by `Mark Wronkiewicz`_ as part of Google Summer of Code, with help from `Samu Taulu`_, `Jukka Nenonen`_, and `Jussi Nurminen`_. Our implementation includes support for:

  - Fine calibration

  - Cross-talk correction

  - Temporal SSS (tSSS)

  - Head position translation

  - Internal component regularization

- Compensation for movements using Maxwell filtering on epoched data in :func:`mne.epochs.average_movements` by `Eric Larson`_ and `Samu Taulu`_

- Add reader for Nicolet files in :func:`mne.io.read_raw_nicolet` by `Jaakko Leppakangas`_

- Add FIFF persistence for ICA labels by `Denis Engemann`_

- Display ICA labels in :func:`mne.viz.plot_ica_scores` and :func:`mne.viz.plot_ica_sources` (for evoked objects) by `Denis Engemann`_

- Plot spatially color coded lines in :func:`mne.Evoked.plot` by `Jona Sassenhagen`_ and `Jaakko Leppakangas`_

- Add reader for CTF data in :func:`mne.io.read_raw_ctf` by `Eric Larson`_

- Add support for Brainvision v2 in :func:`mne.io.read_raw_brainvision` by `Teon Brooks`_

- Improve speed of generalization across time ``mne.decoding.GeneralizationAcrossTime`` decoding up to a factor of seven by `Jean-Remi King`_ and `Federico Raimondo`_ and `Denis Engemann`_.

- Add the explained variance for each principal component, ``explained_var``, key to the :class:`mne.Projection` by `Teon Brooks`_

- Added methods ``mne.Epochs.add_eeg_average_proj``, ``mne.io.Raw.add_eeg_average_proj``, and ``mne.Evoked.add_eeg_average_proj`` to add an average EEG reference.

- Add reader for EEGLAB data in :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` by `Mainak Jas`_

BUG
~~~

- Fix bug that prevented homogeneous bem surfaces to be displayed in HTML reports by `Denis Engemann`_

- Added safeguards against ``None`` and negative values in reject and flat parameters in :class:`mne.Epochs` by `Eric Larson`_

- Fix train and test time window-length in ``mne.decoding.GeneralizationAcrossTime`` by `Jean-Remi King`_

- Added lower bound in :func:`mne.stats.linear_regression` on p-values ``p_val`` (and resulting ``mlog10_p_val``) using double floating point arithmetic limits by `Eric Larson`_

- Fix channel name pick in :func:`mne.Evoked.get_peak` method by `Alex Gramfort`_

- Fix drop percentages to take into account ``ignore`` option in :func:`mne.viz.plot_drop_log` and :func:`mne.Epochs.plot_drop_log` by `Eric Larson`_.

- :class:`mne.EpochsArray` no longer has an average EEG reference silently added (but not applied to the data) by default. Use ``mne.EpochsArray.add_eeg_ref`` to properly add one.

- Fix :func:`mne.io.read_raw_ctf` to read ``n_samp_tot`` instead of ``n_samp`` by `Jaakko Leppakangas`_

API
~~~

- :func:`mne.io.read_raw_brainvision` now has ``event_id`` argument to assign non-standard trigger events to a trigger value by `Teon Brooks`_

- :func:`mne.read_epochs` now has ``add_eeg_ref=False`` by default, since average EEG reference can be added before writing or after reading using the method ``mne.Epochs.add_eeg_ref``.

- :class:`mne.EpochsArray` no longer has an average EEG reference silently added (but not applied to the data) by default. Use ``mne.EpochsArray.add_eeg_average_proj`` to properly add one.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

* 171  Eric Larson
* 117  Jaakko Leppakangas
*  58  Jona Sassenhagen
*  52  Mainak Jas
*  46  Alexandre Gramfort
*  33  Denis A. Engemann
*  28  Teon Brooks
*  24  Clemens Brunner
*  23  Christian Brodbeck
*  15  Mark Wronkiewicz
*  10  Jean-Remi King
*   5  Marijn van Vliet
*   3  Fede Raimondo
*   2  Alexander Rudiuk
*   2  emilyps14
*   2  lennyvarghese
*   1  Marian Dovgialo

.. _changes_0_10:

Version 0.10
------------

Changelog
~~~~~~~~~

- Add support for generalized M-way repeated measures ANOVA for fully balanced designs with :func:`mne.stats.f_mway_rm` by `Denis Engemann`_

- Add epochs browser to interactively view and manipulate epochs with :func:`mne.viz.plot_epochs` by `Jaakko Leppakangas`_

- Speed up TF-MxNE inverse solver with block coordinate descent by `Daniel Strohmeier`_ and `Yousra Bekhti`_

- Speed up zero-phase overlap-add (default) filtering by a factor of up to 2 using linearity by `Ross Maddox`_ and `Eric Larson`_

- Add support for scaling and adjusting the number of channels/time per view by `Jaakko Leppakangas`_

- Add support to toggle the show/hide state of all sections with a single keypress ('t') in :class:`mne.Report` by `Mainak Jas`_

- Add support for BEM model creation :func:`mne.make_bem_model` by `Eric Larson`_

- Add support for BEM solution computation :func:`mne.make_bem_solution` by `Eric Larson`_

- Add ICA plotters for raw and epoch components by `Jaakko Leppakangas`_

- Add new object ``mne.decoding.TimeDecoding`` for decoding sensors' evoked response across time by `Jean-Remi King`_

- Add command ``mne freeview_bem_surfaces`` to quickly check BEM surfaces with Freeview by `Alex Gramfort`_.

- Add support for splitting epochs into multiple files in :func:`mne.Epochs.save` by `Mainak Jas`_ and `Alex Gramfort`_

- Add support for jointly resampling a raw object and event matrix to avoid issues with resampling status channels by `Marijn van Vliet`_

- Add new method :class:`mne.preprocessing.Xdawn` for denoising and decoding of ERP/ERF by `Alexandre Barachant`_

- Add support for plotting patterns/filters in :class:`mne.decoding.CSP` and :class:`mne.decoding.LinearModel` by `Romain Trachel`_

- Add new object :class:`mne.decoding.LinearModel` for decoding M/EEG data and interpreting coefficients of linear models with patterns attribute by `Romain Trachel`_ and `Alex Gramfort`_

- Add support to append new channels to an object from a list of other objects by `Chris Holdgraf`_

- Add interactive plotting of topomap from time-frequency representation by `Jaakko Leppakangas`_

- Add ``plot_topo`` method to ``Evoked`` object by `Jaakko Leppakangas`_

- Add fetcher :mod:`mne.datasets.brainstorm <mne.datasets>` for datasets used by Brainstorm in their tutorials by `Mainak Jas`_

- Add interactive plotting of single trials by right clicking on channel name in epochs browser by `Jaakko Leppakangas`_

- New logos and logo generation script by `Daniel McCloy`_

- Add ability to plot topomap with a "skirt" (channels outside of the head circle) by `Marijn van Vliet`_

- Add multiple options to ICA infomax and extended infomax algorithms (number of subgaussian components, computation of bias, iteration status printing), enabling equivalent computations to those performed by EEGLAB by `Jair Montoya Martinez`_

- Add :func:`mne.Epochs.apply_baseline` method to ``Epochs`` objects by `Teon Brooks`_

- Add ``preload`` argument to :func:`mne.read_epochs` to enable on-demand reads from disk by `Eric Larson`_

- Big rewrite of simulation module by `Yousra Bekhti`_, `Mark Wronkiewicz`_, `Eric Larson`_ and `Alex Gramfort`_. Allows to simulate raw with artifacts (ECG, EOG) and evoked data, exploiting the forward solution. See :func:`mne.simulation.simulate_raw`, :func:`mne.simulation.simulate_evoked` and :func:`mne.simulation.simulate_sparse_stc`

- Add :func:`mne.Epochs.load_data` method to :class:`mne.Epochs` by `Teon Brooks`_

- Add support for drawing topomaps by selecting an area in :func:`mne.Evoked.plot` by `Jaakko Leppakangas`_

- Add support for finding peaks in evoked data in :func:`mne.Evoked.plot_topomap` by `Jona Sassenhagen`_ and `Jaakko Leppakangas`_

- Add source space morphing in :func:`morph_source_spaces` and :func:`SourceEstimate.to_original_src` by `Eric Larson`_ and `Denis Engemann`_

- Adapt ``corrmap`` function (Viola et al. 2009) to semi-automatically detect similar ICs across data sets by `Jona Sassenhagen`_ and `Denis Engemann`_ and `Eric Larson`_

- Clarify docstring for :class:`mne.preprocessing.ICA` by `jeythekey`_

- New ``mne flash_bem`` command to compute BEM surfaces from Flash MRI images by `Lorenzo Desantis`_, `Alex Gramfort`_ and `Eric Larson`_. See :func:`mne.bem.make_flash_bem`.

- New gfp parameter in :func:`mne.Evoked.plot` method to display Global Field Power (GFP) by `Eric Larson`_.

- Add :meth:`mne.Report.add_slider_to_section` methods to :class:`mne.Report` by `Teon Brooks`_

BUG
~~~

- Fix ``mne.io.add_reference_channels`` not setting ``info[nchan]`` correctly by `Federico Raimondo`_

- Fix ``mne.stats.bonferroni_correction`` reject mask output to use corrected p-values by `Denis Engemann`_

- Fix FFT filter artifacts when using short windows in overlap-add by `Eric Larson`_

- Fix picking channels from forward operator could return a channel ordering different from ``info['chs']`` by `Chris Bailey`_

- Fix dropping of events after downsampling stim channels by `Marijn van Vliet`_

- Fix scaling in :func:``mne.viz.utils._setup_vmin_vmax`` by `Jaakko Leppakangas`_

- Fix order of component selection in :class:`mne.decoding.CSP` by `Clemens Brunner`_

API
~~~

- Rename and deprecate ``mne.viz.plot_topo`` for ``mne.viz.plot_evoked_topo`` by `Jaakko Leppakangas`_

- Deprecated :class: `mne.decoding.transformer.ConcatenateChannels` and replaced by :class: `mne.decoding.transformer.EpochsVectorizer` by `Romain Trachel`_

- Deprecated `lws` and renamed `ledoit_wolf` for the ``reg`` argument in :class:`mne.decoding.CSP` by `Romain Trachel`_

- Redesigned and rewrote :meth:`mne.Epochs.plot` (no backwards compatibility) during the GSOC 2015 by `Jaakko Leppakangas`_, `Mainak Jas`_, `Federico Raimondo`_ and `Denis Engemann`_

- Deprecated and renamed ``mne.viz.plot_image_epochs`` for ``mne.plot.plot_epochs_image`` by `Teon Brooks`_

- ``picks`` argument has been added to :func:`mne.time_frequency.tfr_morlet`, :func:`mne.time_frequency.tfr_multitaper` by `Teon Brooks`_

- ``mne.io.Raw.preload_data`` has been deprecated for :func:`mne.io.Raw.load_data` by `Teon Brooks`_

- ``RawBrainVision`` objects now always have event channel ``'STI 014'``, and recordings with no events will have this channel set to zero by `Eric Larson`_

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

* 273  Eric Larson
* 270  Jaakko Leppakangas
* 194  Alexandre Gramfort
* 128  Denis A. Engemann
* 114  Jona Sassenhagen
* 107  Mark Wronkiewicz
*  97  Teon Brooks
*  81  Lorenzo De Santis
*  55  Yousra Bekhti
*  54  Jean-Remi King
*  48  Romain Trachel
*  45  Mainak Jas
*  40  Alexandre Barachant
*  32  Marijn van Vliet
*  26  Jair Montoya
*  22  Chris Holdgraf
*  16  Christopher J. Bailey
*   7  Christian Brodbeck
*   5  Natalie Klein
*   5  Fede Raimondo
*   5  Alan Leggitt
*   5  Roan LaPlante
*   5  Ross Maddox
*   4  Dan G. Wakeman
*   3  Daniel McCloy
*   3  Daniel Strohmeier
*   1  Jussi Nurminen

.. _changes_0_9:

Version 0.9
-----------

Changelog
~~~~~~~~~

- Add support for mayavi figures in ``add_section`` method in Report by `Mainak Jas`_

- Add extract volumes of interest from freesurfer segmentation and setup as volume source space by `Alan Leggitt`_

- Add support to combine source spaces of different types by `Alan Leggitt`_

- Add support for source estimate for mixed source spaces by `Alan Leggitt`_

- Add ``SourceSpaces.save_as_volume`` method by `Alan Leggitt`_

- Automatically compute proper box sizes when generating layouts on the fly by `Marijn van Vliet`_

- Average evoked topographies across time points by `Denis Engemann`_

- Add option to Report class to save images as vector graphics (SVG) by `Denis Engemann`_

- Add events count to ``mne.viz.plot_events`` by `Denis Engemann`_

- Add support for stereotactic EEG (sEEG) channel type by `Marmaduke Woodman`_

- Add support for montage files by `Denis Engemann`_, `Marijn van Vliet`_, `Jona Sassenhagen`_, `Alex Gramfort`_ and `Teon Brooks`_

- Add support for spatiotemporal permutation clustering on sensors by `Denis Engemann`_

- Add support for multitaper time-frequency analysis by `Hari Bharadwaj`_

- Add Stockwell (S) transform for time-frequency representations by `Denis Engemann`_ and `Alex Gramfort`_

- Add reading and writing support for time frequency data (AverageTFR objects) by  `Denis Engemann`_

- Add reading and writing support for digitizer data, and function for adding dig points to info by `Teon Brooks`_

- Add  ``plot_projs_topomap`` method to ``Raw``, ``Epochs`` and ``Evoked`` objects by `Teon Brooks`_

- Add EEG (based on spherical splines) and MEG (based on field interpolation) bad channel interpolation method to ``Raw``, ``Epochs`` and ``Evoked`` objects by `Denis Engemann`_ and `Mainak Jas`_

- Add parameter to ``whiten_evoked``, ``compute_whitener`` and ``prepare_noise_cov`` to set the exact rank by `Martin Luessi`_ and `Denis Engemann`_

- Add fiff I/O for processing history and MaxFilter info by `Denis Engemann`_ and `Eric Larson`_

- Add automated regularization with support for multiple sensor types to ``compute_covariance`` by `Denis Engemann`_ and `Alex Gramfort`_

- Add ``Evoked.plot_white`` method to diagnose the quality of the estimated noise covariance and its impact on spatial whitening by `Denis Engemann`_ and `Alex Gramfort`_

- Add ``mne.evoked.grand_average`` function to compute grand average of Evoked data while interpolating bad EEG channels if necessary by `Mads Jensen`_ and `Alex Gramfort`_

- Improve EEG referencing support and add support for bipolar referencing by `Marijn van Vliet`_ and `Alex Gramfort`_

- Enable TFR calculation on Evoked objects by `Eric Larson`_

- Add support for combining Evoked datasets with arbitrary weights (e.g., for oddball paradigms) by `Eric Larson`_ and `Alex Gramfort`_

- Add support for concatenating a list of Epochs objects by `Denis Engemann`_

- Labels support subtraction (``label_1 - label_2``) by `Christian Brodbeck`_

- Add GeneralizationAcrossTime object with support for cross-condition generalization by `Jean-Remi King`_ and `Denis Engemann`_

- Add support for single dipole fitting by `Eric Larson`_

- Add support for spherical models in forward calculations by `Eric Larson`_

- Add support for SNR estimation by `Eric Larson`_

- Add support for Savitsky-Golay filtering of Evoked and Epochs by `Eric Larson`_

- Add support for adding an empty reference channel to data by `Teon Brooks`_

- Add reader function ``mne.io.read_raw_fif`` for Raw FIF files by `Teon Brooks`_

- Add example of creating MNE objects from arbitrary data and NEO files by `Jaakko Leppakangas`_

- Add ``plot_psd`` and ``plot_psd_topomap`` methods to epochs by `Yousra Bekhti`_, `Eric Larson`_ and `Denis Engemann`_

- ``evoked.pick_types``, ``epochs.pick_types``, and ``tfr.pick_types`` added by `Eric Larson`_

- ``rename_channels`` and ``set_channel_types`` added as methods to ``Raw``, ``Epochs`` and ``Evoked`` objects by `Teon Brooks`_

- Add RAP-MUSIC inverse method by `Yousra Bekhti`_ and `Alex Gramfort`_

- Add ``evoked.as_type`` to  allow remapping data in MEG channels to virtual magnetometer or gradiometer channels by `Mainak Jas`_

- Add :meth:`mne.Report.add_bem_to_section`, :meth:`mne.Report.add_htmls_to_section` methods to :class:`mne.Report` by `Teon Brooks`_

- Add support for KIT epochs files with ``read_epochs_kit`` by `Teon Brooks`_

- Add whitening plots for evokeds to ``mne.Report`` by `Mainak Jas`_

- Add ``DigMontage`` class and reader to interface with digitization info by `Teon Brooks`_ and `Christian Brodbeck`_

- Add ``set_montage`` method to the ``Raw``, ``Epochs``, and ``Evoked`` objects by `Teon Brooks`_ and `Denis Engemann`_

- Add support for capturing sensor positions when clicking on an image by `Chris Holdgraf`_

- Add support for custom sensor positions when creating Layout objects by `Chris Holdgraf`_

BUG
~~~

- Fix energy conservation for STFT with tight frames by `Daniel Strohmeier`_

- Fix incorrect data matrix when tfr was plotted with parameters ``tmin``, ``tmax``, ``fmin`` and ``fmax`` by `Mainak Jas`_

- Fix channel names in topomaps by `Alex Gramfort`_

- Fix mapping of ``l_trans_bandwidth`` (to low frequency) and ``h_trans_bandwidth`` (to high frequency) in ``_BaseRaw.filter`` by `Denis Engemann`_

- Fix scaling source spaces when distances have to be recomputed by `Christian Brodbeck`_

- Fix repeated samples in client to FieldTrip buffer by `Mainak Jas`_ and `Federico Raimondo`_

- Fix highpass and lowpass units read from Brainvision vhdr files by `Alex Gramfort`_

- Add missing attributes for BrainVision and KIT systems needed for resample by `Teon Brooks`_

- Fix file extensions of SSP projection files written by mne commands (from _proj.fif to -prof.fif) by `Alex Gramfort`_

- Generating EEG layouts no longer requires digitization points by `Marijn van Vliet`_

- Add missing attributes to BTI, KIT, and BrainVision by `Eric Larson`_

- The API change to the edf, brainvision, and egi break backwards compatibility for when importing eeg data by `Teon Brooks`_

- Fix bug in ``mne.viz.plot_topo`` if ylim was passed for single sensor layouts by `Denis Engemann`_

- Average reference projections will no longer by automatically added after applying a custom EEG reference by `Marijn van Vliet`_

- Fix picks argument to filter in n dimensions (affects FilterEstimator), and highpass filter in FilterEstimator by `Mainak Jas`_

- Fix beamformer code LCMV/DICS for CTF data with reference channels by `Denis Engemann`_ and `Alex Gramfort`_

- Fix scalings for bad EEG channels in ``mne.viz.plot_topo`` by `Marijn van Vliet`_

- Fix EGI reading when no events are present by `Federico Raimondo`_

- Add functionality to determine plot limits automatically or by data percentiles by `Mark Wronkiewicz`_

- Fix bug in mne.io.edf where the channel offsets were omitted in the voltage calculations by `Teon Brooks`_

- Decouple section ordering in command-line from python interface for mne-report by `Mainak Jas`_

- Fix bug with ICA resetting by `Denis Engemann`_

API
~~~

- apply_inverse functions have a new boolean parameter ``prepared`` which saves computation time by calling ``prepare_inverse_operator`` only if it is False

- find_events and read_events functions have a new parameter ``mask`` to set some bits to a don't care state by `Teon Brooks`_

- New channels module including layouts, electrode montages, and neighbor definitions of sensors which deprecates ``mne.layouts`` by `Denis Engemann`_

- ``read_raw_brainvision``, ``read_raw_edf``, ``read_raw_egi`` all use a standard montage import by `Teon Brooks`_

- Fix missing calibration factors for ``mne.io.egi.read_raw_egi`` by `Denis Engemann`_ and `Federico Raimondo`_

- Allow multiple filename patterns as a list (e.g., \*raw.fif and \*-eve.fif) to be parsed by mne report in ``Report.parse_folder()`` by `Mainak Jas`_

- ``read_hsp``, ``read_elp``, and ``write_hsp``, ``write_mrk`` were removed and made private by `Teon Brooks`_

- When computing the noise covariance or MNE inverse solutions, the rank is estimated empirically using more sensitive thresholds, which stabilizes results by `Denis Engemann`_ and `Eric Larson`_ and `Alex Gramfort`_

- Raw FIFF files can be preloaded after class instantiation using ``raw.preload_data()``

- Add ``label`` parameter to ``apply_inverse`` by `Teon Brooks`_

- Deprecated ``label_time_courses`` for ``in_label`` method in `SourceEstimate` by `Teon Brooks`_

- Deprecated ``as_data_frame`` for ``to_data_frame`` by `Chris Holdgraf`_

- Add ``transform``, ``unit`` parameters to ``read_montage`` by `Teon Brooks`_

- Deprecated ``fmin, fmid, fmax`` in stc.plot and added ``clim`` by `Mark Wronkiewicz`_

- Use ``scipy.signal.welch`` instead of matplotlib.psd inside ``compute_raw_psd`` and ``compute_epochs_psd`` by `Yousra Bekhti`_ `Eric Larson`_ and `Denis Engemann`_. As a consquence, ``Raw.plot_raw_psds`` has been deprecated.

- ``Raw`` instances returned by ``mne.forward.apply_forward_raw`` now always have times starting from
  zero to be consistent with all other ``Raw`` instances. To get the former ``start`` and ``stop`` times,
  use ``raw.first_samp / raw.info['sfreq']`` and ``raw.last_samp / raw.info['sfreq']``.

- ``pick_types_evoked`` has been deprecated in favor of ``evoked.pick_types``.

- Deprecated changing the sensor type of channels in ``rename_channels`` by `Teon Brooks`_

- CUDA is no longer initialized at module import, but only when first used.

- ``add_figs_to_section`` and ``add_images_to_section`` now have a ``textbox`` parameter to add comments to the image by `Teon Brooks`_

- Deprecated ``iir_filter_raw`` for ``fit_iir_model_raw``.

- Add ``montage`` parameter to the ``create_info`` function to create the info using montages by `Teon Brooks`_

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

* 515  Eric Larson
* 343  Denis A. Engemann
* 304  Alexandre Gramfort
* 300  Teon Brooks
* 142  Mainak Jas
* 119  Jean-Remi King
*  77  Alan Leggitt
*  75  Marijn van Vliet
*  63  Chris Holdgraf
*  57  Yousra Bekhti
*  49  Mark Wronkiewicz
*  44  Christian Brodbeck
*  30  Jona Sassenhagen
*  29  Hari Bharadwaj
*  27  Clément Moutard
*  24  Ingoo Lee
*  18  Marmaduke Woodman
*  16  Martin Luessi
*  10  Jaakko Leppakangas
*   9  Andrew Dykstra
*   9  Daniel Strohmeier
*   7  kjs
*   6  Dan G. Wakeman
*   5  Federico Raimondo
*   3  Basile Pinsard
*   3  Christoph Dinh
*   3  Hafeza Anevar
*   2  Martin Billinger
*   2  Roan LaPlante
*   1  Manoj Kumar
*   1  Matt Tucker
*   1  Romain Trachel
*   1  mads jensen
*   1  sviter

.. _changes_0_8:

Version 0.8
-----------

Changelog
~~~~~~~~~

- Add Python3 support by `Nick Ward`_, `Alex Gramfort`_, `Denis Engemann`_, and `Eric Larson`_

- Add ``get_peak`` method for evoked and stc objects by  `Denis Engemann`_

- Add ``iter_topography`` function for radically simplified custom sensor topography plotting by `Denis Engemann`_

- Add field line interpolation by `Eric Larson`_

- Add full provenance tacking for epochs and improve ``drop_log`` by `Tal Linzen`_, `Alex Gramfort`_ and `Denis Engemann`_

- Add systematic contains method to ``Raw``, ``Epochs`` and ``Evoked`` for channel type membership testing by `Denis Engemann`_

- Add fiff unicode writing and reading support by `Denis Engemann`_

- Add 3D MEG/EEG field plotting function and evoked method by `Denis Engemann`_ and  `Alex Gramfort`_

- Add consistent channel-dropping methods to ``Raw``, ``Epochs`` and ``Evoked`` by `Denis Engemann`_ and  `Alex Gramfort`_

- Add ``equalize_channnels`` function to set common channels for a list of ``Raw``, ``Epochs``, or ``Evoked`` objects by `Denis Engemann`_

- Add ``plot_events`` function to visually display paradigm by `Alex Gramfort`_

- Improved connectivity circle plot by `Martin Luessi`_

- Add ability to anonymize measurement info by `Eric Larson`_

- Add callback to connectivity circle plot to isolate connections to clicked nodes `Roan LaPlante`_

- Add ability to add patch information to source spaces by `Eric Larson`_

- Add ``split_label`` function to divide labels into multiple parts by `Christian Brodbeck`_

- Add ``color`` attribute to ``Label`` objects by `Christian Brodbeck`_

- Add ``max`` mode for ``extract_label_time_course`` by `Mads Jensen`_

- Add ``rename_channels`` function to change channel names and types in info object by `Dan Wakeman`_ and `Denis Engemann`_

- Add  ``compute_ems`` function to extract the time course of experimental effects by `Denis Engemann`_, `Sébastien Marti`_ and `Alex Gramfort`_

- Add option to expand Labels defined in a source space to the original surface (``Label.fill()``) by `Christian Brodbeck`_

- GUIs can be invoked form the command line using `$ mne coreg` and `$ mne kit2fiff` by `Christian Brodbeck`_

- Add ``add_channels_epochs`` function to combine different recordings at the Epochs level by `Christian Brodbeck`_ and `Denis Engemann`_

- Add support for EGI Netstation simple binary files by `Denis Engemann`_

- Add support for treating arbitrary data (numpy ndarray) as a Raw instance by `Eric Larson`_

- Support for parsing the EDF+ annotation channel by `Martin Billinger`_

- Add EpochsArray constructor for creating epochs from numpy arrays by `Denis Engemann`_ and `Federico Raimondo`_

- Add connector to FieldTrip realtime client by `Mainak Jas`_

- Add color and event_id with legend options in plot_events in viz.py by `Cathy Nangini`_

- Add ``events_list`` parameter to ``mne.concatenate_raws`` to concatenate events corresponding to runs by `Denis Engemann`_

- Add ``read_ch_connectivity`` function to read FieldTrip neighbor template .mat files and obtain sensor adjacency matrices by `Denis Engemann`_

- Add display of head in helmet from -trans.fif file to check coregistration quality by `Mainak Jas`_

- Add ``raw.add_events`` to allow adding events to a raw file by `Eric Larson`_

- Add ``plot_image`` method to Evoked object to display data as images by `Jean-Remi King`_ and `Alex Gramfort`_ and `Denis Engemann`_

- Add BCI demo with CSP on motor imagery by `Martin Billinger`_

- New ICA API with unified methods for processing ``Raw``, ``Epochs`` and ``Evoked`` objects by `Denis Engemann`_

- Apply ICA at the evoked stage by `Denis Engemann`_

- New ICA methods for visualizing unmixing quality, artifact detection and rejection by `Denis Engemann`_

- Add ``pick_channels`` and ``drop_channels`` mixin class to pick and drop channels from ``Raw``, ``Epochs``, and ``Evoked`` objects by `Andrew Dykstra`_ and `Denis Engemann`_

- Add ``EvokedArray`` class to create an Evoked object from an array by `Andrew Dykstra`_

- Add ``plot_bem`` method to visualize BEM contours on MRI anatomical images by `Mainak Jas`_ and `Alex Gramfort`_

- Add automated ECG detection using cross-trial phase statistics by `Denis Engemann`_ and `Juergen Dammers`_

- Add Forward class to succintly display gain matrix info by `Andrew Dykstra`_

- Add reading and writing of split raw files by `Martin Luessi`_

- Add OLS regression function by `Tal Linzen`_, `Teon Brooks`_ and `Denis Engemann`_

- Add computation of point spread and cross-talk functions for MNE type solutions by `Alex Gramfort`_ and `Olaf Hauk`_

- Add mask parameter to `plot_evoked_topomap` and ``evoked.plot_topomap`` by `Denis Engemann`_ and `Alex Gramfort`_

- Add infomax and extended infomax ICA by `Denis Engemann`_, `Juergen Dammers`_ and `Lukas Breuer`_ and `Federico Raimondo`_

- Aesthetically redesign interpolated topography plots by `Denis Engemann`_ and `Alex Gramfort`_

- Simplify sensor space time-frequency analysis API with ``tfr_morlet`` function by `Alex Gramfort`_ and `Denis Engemann`_

- Add new somatosensory MEG dataset with nice time-frequency content by `Alex Gramfort`_

- Add HDF5 write/read support for SourceEstimates by `Eric Larson`_

- Add InverseOperator class to display inverse operator info by `Mainak Jas`_

- Add `$ mne report` command to generate html reports of MEG/EEG data analysis pipelines by `Mainak Jas`_, `Alex Gramfort`_ and `Denis Engemann`_

- Improve ICA verbosity with regard to rank reduction by `Denis Engemann`_

BUG
~~~

- Fix incorrect ``times`` attribute when stc was computed using ``apply_inverse`` after decimation at epochs stage for certain, arbitrary sample frequencies by `Denis Engemann`_

- Fix corner case error for step-down-in-jumps permutation test (when step-down threshold was high enough to include all clusters) by `Eric Larson`_

- Fix selection of total number of components via float when picking ICA sources by `Denis Engemann`_ and `Qunxi Dong`_

- Fix writing and reading transforms after modification in measurement info by `Denis Engemann`_ and `Martin Luessi`_ and `Eric Larson`_

- Fix pre-whitening / rescaling when estimating ICA on multiple channels without covariance by `Denis Engemann`_

- Fix ICA pre-whitening, avoid recomputation when applying ICA to new data by `Denis Engemann`_

API
~~~

- The minimum numpy version has been increased to 1.6 from 1.4.

- Epochs object now has a selection attribute to track provenance of selected Epochs. The length of the drop_log attribute is now the same as the length of the original events passed to Epochs. In earlier versions it had the length of the events filtered by event_id. Epochs has also now a plot_drop_log method.

- Deprecate Epochs.drop_picks in favor of a new method called drop_channels

- Deprecate ``labels_from_parc`` and ``parc_from_labels`` in favor of ``read_labels_from_annot`` and ``write_labels_to_annot``

- The default of the new add_dist option of ``setup_source_space`` to add patch information will change from False to True in MNE-Python 0.9

- Deprecate ``read_evoked`` and ``write_evoked`` in favor of ``read_evokeds`` and ``write_evokeds``. read_evokeds will return all `Evoked` instances in a file by default.

- Deprecate ``setno`` in favor of ``condition`` in the initialization of an Evoked instance. This affects ``mne.fiff.Evoked`` and ``read_evokeds``, but not ``read_evoked``.

- Deprecate ``mne.fiff`` module, use ``mne.io`` instead e.g. ``mne.io.Raw`` instead of ``mne.fiff.Raw``.

- Pick functions (e.g., ``pick_types``) are now in the mne namespace (e.g. use ``mne.pick_types``).

- Deprecated ICA methods specific to one container type. Use ICA.fit, ICA.get_sources ICA.apply and ``ICA.plot_*`` for processing Raw, Epochs and Evoked objects.

- The default smoothing method for ``mne.stc_to_label`` will change in v0.9, and the old method is deprecated.

- As default, for ICA the maximum number of PCA components equals the number of channels passed. The number of PCA components used to reconstruct the sensor space signals now defaults to the maximum number of PCA components estimated.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

* 418  Denis A. Engemann
* 284  Alexandre Gramfort
* 242  Eric Larson
* 155  Christian Brodbeck
* 144  Mainak Jas
* 49  Martin Billinger
* 49  Andrew Dykstra
* 44  Tal Linzen
* 37  Dan G. Wakeman
* 36  Martin Luessi
* 26  Teon Brooks
* 20  Cathy Nangini
* 15  Hari Bharadwaj
* 15  Roman Goj
* 10  Ross Maddox
* 9  Marmaduke Woodman
* 8  Praveen Sripad
* 8  Tanay
* 8  Roan LaPlante
* 5  Saket Choudhary
* 4  Nick Ward
* 4  Mads Jensen
* 3  Olaf Hauk
* 3  Brad Buran
* 2  Daniel Strohmeier
* 2  Federico Raimondo
* 2  Alan Leggitt
* 1  Jean-Remi King
* 1  Matti Hamalainen


.. _changes_0_7:

Version 0.7
-----------

Changelog
~~~~~~~~~

- Add capability for real-time feedback via trigger codes using StimServer and StimClient classes by `Mainak Jas`_

- New decoding module for MEG analysis containing sklearn compatible transformers by `Mainak Jas`_ and `Alex Gramfort`_

- New realtime module containing RtEpochs, RtClient and MockRtClient class by `Martin Luessi`_, `Christopher Dinh`_, `Alex Gramfort`_, `Denis Engemann`_ and `Mainak Jas`_

- Allow picking normal orientation in LCMV beamformers by `Roman Goj`_, `Alex Gramfort`_, `Denis Engemann`_ and `Martin Luessi`_

- Add printing summary to terminal for measurement info by `Denis Engemann`_

- Add read and write info attribute ICA objects by `Denis Engemann`_

- Decoding with Common Spatial Patterns (CSP) by `Romain Trachel`_ and `Alex Gramfort`_

- Add ICA ``plot_topomap`` function and method for displaying the spatial sensitivity of ICA sources by `Denis Engemann`_

- Plotting multiple brain views at once by `Eric Larson`_

- Reading head positions from raw FIFF files by `Eric Larson`_

- Add decimation parameter to ICA.decompose*  methods by `Denis Engemann`_ and `Alex Gramfort`_

- Add rejection buffer to ICA.decompose* methods by `Denis Engemann`_ and `Alex Gramfort`_

- Improve ICA computation speed and memory usage by `Denis Engemann`_ and `Alex Gramfort`_

- Add polygonal surface decimation function to preprocess head surfaces for coregistration by `Denis Engemann`_ and `Alex Gramfort`_

- DICS time-frequency beamforming for epochs, evoked and for estimating source power by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Add method for computing cross-spectral density (CSD) from epochs and class for storing CSD data by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Add trellis plot function and method for visualizing single epochs by `Denis Engemann`_

- Add fiducials read/write support by `Christian Brodbeck`_ and `Alex Gramfort`_

- Add select / drop bad channels in `plot_raw` on click by `Denis Engemann`_

- Add `ico` and `oct` source space creation in native Python by `Eric Larson`_

- Add interactive rejection of bad trials in ``plot_epochs`` by `Denis Engemann`_

- Add morph map calculation by `Eric Larson`_ and `Martin Luessi`_

- Add volume and discrete source space creation and I/O support by `Eric Larson`_

- Time-frequency beamforming to obtain spectrograms in source space using LCMV and DICS by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Compute epochs power spectral density function by `Denis Engemann`_

- Plot raw power spectral density by `Eric Larson`_

- Computing of distances along the cortical surface by `Eric Larson`_

- Add reading BEM solutions by `Eric Larson`_

- Add forward solution calculation in native Python by `Eric Larson`_

- Add (Neuro)debian license compatibility by `Eric Larson`_

- Automatic QRS threshold selection for ECG events by `Eric Larson`_

- Add Travis continuous integration service by `Denis Engemann`_

- Add SPM face data set by `Denis Engemann`_ `Martin Luessi`_ and `Alex Gramfort`_

- Support reading of EDF+,BDF data by `Teon Brooks`_

- Tools for scaling MRIs (mne.scale_mri) by `Christian Brodbeck`_

- GUI for head-MRI coregistration (mne.gui.coregistration) by `Christian Brodbeck`_

- GUI for ki2fiff conversion (mne.gui.kit2fiff) by `Christian Brodbeck`_

- Support reading of EEG BrainVision data by `Teon Brooks`_

- Improve CTF compensation handling by `Martin Luessi`_ and `Eric Larson`_

- Improve and extend automated layout guessing by `Denis Engemann`_

- Add Continuum Analytics Anaconda support by `Denis Engemann`_

- Add `subtract evoked` option to beamformers by `Andrew Dykstra`_

- Add new `transform` method to SourceEstimate(s) by `Andrew Dykstra`_

API
~~~

- The pick_normal parameter for minimum norm solvers has been renamed as ``pick_ori`` and normal orientation picking is now achieved by passing the value "normal" for the `pick_ori` parameter.

- ICA objects now expose the measurement info of the object fitted.

- Average EEG reference is now added by default to Raw instances.

- Removed deprecated read/write_stc/w, use SourceEstimate methods instead

- The ``chs`` argument in ``mne.layouts.find_layout`` is deprecated and will be removed in MNE-Python 0.9. Use ``info`` instead.

- ``plot_evoked`` and ``Epochs.plot`` now open a new figure by default. To plot on an existing figure please specify the `axes` parameter.


Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 336  Denis A. Engemann
* 202  Eric Larson
* 193  Roman Goj
* 138  Alexandre Gramfort
*  99  Mainak Jas
*  75  Christian Brodbeck
*  60  Martin Luessi
*  40  Teon Brooks
*  29  Romain Trachel
*  28  Andrew Dykstra
*  12  Mark Wronkiewicz
*  10  Christoph Dinh
*   8  Alan Leggitt
*   3  Yaroslav Halchenko
*   3  Daniel Strohmeier
*   2  Mads Jensen
*   2  Praveen Sripad
*   1  Luke Bloy
*   1  Emanuele Olivetti
*   1  Yousra BEKHTI


.. _changes_0_6:

Version 0.6
-----------

Changelog
~~~~~~~~~

- Linear (and zeroth-order) detrending for Epochs and Evoked by `Eric Larson`_

- Label morphing between subjects by `Eric Larson`_

- Define events based on time lag between reference and target event by `Denis Engemann`_

- ICA convenience function implementing an automated artifact removal workflow by `Denis Engemann`_

- Bad channels no longer included in epochs by default by `Eric Larson`_

- Support for diagonal noise covariances in inverse methods and rank computation by `Eric Larson`_

- Support for using CUDA in FFT-based FIR filtering (method='fft') and resampling by `Eric Larson`_

- Optimized FFT length selection for faster overlap-add filtering by `Martin Luessi`_

- Ability to exclude bad channels from evoked plots or shown them in red by `Martin Luessi`_

- Option to show both hemispheres when plotting SourceEstimate with PySurfer by `Martin Luessi`_

- Optimized Raw reading and epoching routines to limit memory copies by `Eric Larson`_

- Advanced options to save raw files in short or double precision by `Eric Larson`_

- Option to detect decreasing events using find_events by `Simon Kornblith`_

- Option to change default stim_channel used for finding events by `Eric Larson`_

- Use average patch normal from surface-oriented forward solution in inverse calculation when possible by `Eric Larson`_

- Function to plot drop_log from Epochs instance by `Eric Larson`_

- Estimate rank of Raw data by `Eric Larson`_

- Support reading of BTi/4D data by `Denis Engemann`_

- Wrapper for generating forward solutions by `Eric Larson`_

- Averaging forward solutions by `Eric Larson`_

- Events now contain the pre-event stim channel value in the middle column, by `Christian Brodbeck`_

- New function ``mne.find_stim_steps`` for finding all steps in a stim channel by `Christian Brodbeck`_

- Get information about FIFF files using mne.fiff.show_fiff() by `Eric Larson`_

- Compute forward fields sensitivity maps by `Alex Gramfort`_ and `Eric Larson`_

- Support reading of KIT data by `Teon Brooks`_ and `Christian Brodbeck`_

- Raw data visualization by `Eric Larson`_

- Smarter SourceEstimate object that contains linear inverse kernel and sensor space data for fast time-frequency transforms in source space by `Martin Luessi`_

- Add example of decoding/MVPA on MEG sensor data by `Alex Gramfort`_

- Add support for non-paired tests in spatiotemporal cluster stats by `Alex Gramfort`_

- Add unified SSP-projector API for Raw, Epochs and Evoked objects by `Denis Engemann`_, `Alex Gramfort`_ `Eric Larson`_ and `Martin Luessi`_

- Add support for delayed SSP application at evoked stage `Denis Engemann`_, `Alex Gramfort`_, `Eric Larson`_ and `Martin Luessi`_

- Support selective parameter updating in functions taking dicts as arguments by `Denis Engemann`_

- New ICA method ``sources_as_epochs`` to create Epochs in ICA space by `Denis Engemann`_

- New method in Evoked and Epoch classes to shift time scale by `Mainak Jas`_

- Added option to specify EOG channel(s) when computing PCA/SSP projections for EOG artifacts by `Mainak Jas`_

- Improved connectivity interface to allow combinations of signals, e.g., seed time series and source estimates, by `Martin Luessi`_

- Effective connectivity estimation using Phase Slope Index (PSI) by `Martin Luessi`_

- Support for threshold-free cluster enhancement (TFCE) by `Eric Larson`_

- Support for "hat" variance regularization by `Eric Larson`_

- Access source estimates as Pandas DataFrame by `Denis Engemann`_.

- Add example of decoding/MVPA on MEG source space data by `Denis Engemann`_

- Add support for --tstart option in mne_compute_proj_eog.py by `Alex Gramfort`_

- Add two-way repeated measures ANOVA for mass-univariate statistics by `Denis Engemann`_, `Eric Larson`_ and `Alex Gramfort`_

- Add function for summarizing clusters from spatio-temporal-cluster permutation tests by `Denis Engemann`_ and `Eric Larson`_

- Add generator support for ``lcmv_epochs`` by `Denis Engemann`_

- Gamma-MAP sparse source localization method by `Martin Luessi`_ and `Alex Gramfort`_

- Add regular expression and substring support for selecting parcellation labels by `Denis Engemann`_

- New plot_evoked option for interactive and reversible selection of SSP projection vectors by `Denis Engemann`_

- Plot 2D flat topographies with interpolation for evoked and SSPs by `Christian Brodbeck`_ and `Alex Gramfort`_

- Support delayed SSP applicationon for 2D flat topographies by `Denis Engemann`_ and `Christian Brodbeck`_ and `Alex Gramfort`_

- Allow picking maximum power source, a.k.a. "optimal", orientation in LCMV beamformers by `Roman Goj`_, `Alex Gramfort`_, `Denis Engemann`_ and `Martin Luessi`_

- Add sensor type scaling parameter to plot_topo by `Andrew Dykstra`_, `Denis Engemann`_  and `Eric Larson`_

- Support delayed SSP application in plot_topo by `Denis Engemann`_

API
~~~

- Deprecated use of fiff.pick_types without specifying exclude -- use either [] (none), ``bads`` (bad channels), or a list of string (channel names).

- Depth bias correction in dSPM/MNE/sLORETA make_inverse_operator is now done like in the C code using only gradiometers if present, else magnetometers, and EEG if no MEG channels are present.

- Fixed-orientation inverse solutions need to be made using `fixed=True` option (using non-surface-oriented forward solutions if no depth weighting is used) to maintain compatibility with MNE C code.

- Raw.save() will only overwrite the destination file, if it exists, if option overwrite=True is set.

- mne.utils.set_config(), get_config(), get_config_path() moved to mne namespace.

- Raw constructor argument proj_active deprecated -- use proj argument instead.

- Functions from the mne.mixed_norm module have been moved to the mne.inverse_sparse module.

- Deprecate CTF compensation (keep_comp and dest_comp) in Epochs and move it to Raw with a single compensation parameter.

- Remove artifacts module. Artifacts- and preprocessing related functions can now be found in mne.preprocessing.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 340  Eric Larson
* 330  Denis A. Engemann
* 204  Alexandre Gramfort
*  72  Christian Brodbeck
*  66  Roman Goj
*  65  Martin Luessi
*  37  Teon Brooks
*  18  Mainak Jas
*   9  Simon Kornblith
*   7  Daniel Strohmeier
*   6  Romain Trachel
*   5  Yousra BEKHTI
*   5  Brad Buran
*   1  Andrew Dykstra
*   1  Christoph Dinh

.. _changes_0_5:

Version 0.5
-----------

Changelog
~~~~~~~~~

- Multi-taper PSD estimation for single epochs in source space using minimum norm by `Martin Luessi`_

- Read and visualize .dip files obtained with xfit or mne_dipole_fit by `Alex Gramfort`_

- Make EEG layout by `Eric Larson`_

- Ability to specify SSP projectors when computing covariance from raw by `Eric Larson`_

- Read and write txt based event files (.eve or .txt) by `Eric Larson`_

- Pass qrs threshold to preprocessing functions by `Eric Larson`_

- Compute SSP projections from continuous raw data by `Eric Larson`_

- Support for applied SSP projections when loading Raw by `Eric Larson`_ and `Alex Gramfort`_

- Support for loading Raw stored in different fif files by `Eric Larson`_

- IO of many Evoked in a single fif file + compute Epochs.standard_error by `Eric Larson`_ and `Alex Gramfort`_

- ICA computation on Raw and Epochs with automatic component selection by `Denis Engemann`_ and `Alex Gramfort`_

- Saving ICA sources to fif files and creating ICA topography layouts by `Denis Engemann`_

- Save and restore ICA session to and from fif by `Denis Engemann`_

- Export raw, epochs and evoked data as data frame to the pandas library by `Denis Engemann`_

- Export raw, epochs and evoked data to the nitime library by `Denis Engemann`_

- Copy methods for raw and epochs objects by `Denis Engemann`_, `Martin Luessi`_ and `Alex Gramfort`_

- New raw objects method to get the time at certain indices by `Denis Engemann`_ and `Alex Gramfort`_

- Plot method for evoked objects by `Denis Engemann`_

- Enhancement of cluster-level stats (speed and memory efficiency) by `Eric Larson`_ and `Martin Luessi`_

- Reading of source space distances by `Eric Larson`_

- Support for filling / smoothing labels and speedup of morphing by `Eric Larson`_

- Adding options for morphing by `Eric Larson`_

- Plotting functions for time frequency and epochs image topographies by `Denis Engemann`_ and `Alex Gramfort`_

- Plotting ERP/ERF images by `Alex Gramfort`_

- See detailed subplot when cliking on a channel inside a topography plot by `Martin Luessi`_, `Eric Larson`_ and `Denis Engemann`_

- Misc channel type support plotting functions by `Denis Engemann`_

- Improved logging support by `Eric Larson`_

- Whitening of evoked data for plotting and quality checking by `Alex Gramfort`_

- Transparent I/O of gzipped fif files (as .fif.gz) by `Eric Larson`_

- Spectral connectivity estimation in sensor and source space by `Martin Luessi`_

- Read and write Epochs in FIF files by `Alex Gramfort`_

- Resampling of Raw, Epochs, and Evoked by `Eric Larson`_

- Creating epochs objects for different conditions and accessing conditions via user-defined name by `Denis Engemann`_ , `Eric Larson`_, `Alex Gramfort`_ and `Christian Brodbeck`_

- Visualizing evoked responses from different conditions in one topography plot by `Denis Engemann`_ and `Alex Gramfort`_

- Support for L21 MxNE solver using coordinate descent using scikit-learn by `Alex Gramfort`_ and `Daniel Strohmeier`_

- Support IIR filters (butterworth, chebyshev, bessel, etc.) by `Eric Larson`_

- Read labels from FreeSurfer parcellation by  `Martin Luessi`_

- Combining labels in source space by `Christian Brodbeck`_

- Read and write source spaces, surfaces and coordinate transforms to and from files by `Christian Brodbeck`_

- Downsample epochs by `Christian Brodbeck`_ and `Eric Larson`_

- New labels class for handling source estimates by `Christian Brodbeck`_, `Martin Luessi`_  and `Alex Gramfort`_

- New plotting routines to easily display SourceEstimates using PySurfer by `Alex Gramfort`_

- Function to extract label time courses from SourceEstimate(s) by `Martin Luessi`_

- Function to visualize connectivity as circular graph by `Martin Luessi`_ and `Alex Gramfort`_

- Time-frequency Mixed Norm Estimates (TF-MxNE) by `Alex Gramfort`_ and `Daniel Strohmeier`_


API
~~~
- Added nave parameter to source_induced_power() and source_band_induced_power(), use nave=1 by default (wrong nave was used before).

- Use mne.layout.read_layout instead of mne.layout.Layout to read a layout file (.lout)

- Use raw.time_as_index instead of time_to_index (still works but is deprecated).

- The artifacts module (mne.artifacts) is now merged into mne.preprocessing

- Epochs objects now also take dicts as values for the event_id argument. They now can represent multiple conditions.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 313  Eric Larson
* 226  Alexandre Gramfort
* 219  Denis A. Engemann
* 104  Christian Brodbeck
*  85  Martin Luessi
*   6  Daniel Strohmeier
*   4  Teon Brooks
*   1  Dan G. Wakeman


.. _changes_0_4:

Version 0.4
-----------

Changelog
~~~~~~~~~

- Add function to compute source PSD using minimum norm by `Alex Gramfort`_

- L21 Mixed Norm Estimates (MxNE) by `Alex Gramfort`_ and `Daniel Strohmeier`_

- Generation of simulated evoked responses by `Alex Gramfort`_, `Daniel Strohmeier`_, and `Martin Luessi`_

- Fit AR models to raw data for temporal whitening by `Alex Gramfort`_.

- speedup + reduce memory of mne.morph_data by `Alex Gramfort`_.

- Backporting scipy.signal.firwin2 so filtering works with old scipy by `Alex Gramfort`_.

- LCMV Beamformer for evoked data, single trials, and raw data by `Alex Gramfort`_ and `Martin Luessi`_.

- Add support for reading named channel selections by `Martin Luessi`_.

- Add Raw.filter method to more easily band pass data by `Alex Gramfort`_.

- Add tmin + tmax parameters in mne.compute_covariance to estimate noise covariance in epochs baseline without creating new epochs by `Alex Gramfort`_.

- Add support for sLORETA in apply_inverse, apply_inverse_raw, apply_inverse_epochs (API Change) by `Alex Gramfort`_.

- Add method to regularize a noise covariance by `Alex Gramfort`_.

- Read and write measurement info in forward and inverse operators for interactive visualization in mne_analyze by `Alex Gramfort`_.

- New mne_compute_proj_ecg.py and mne_compute_proj_eog.py scripts to estimate ECG/EOG PCA/SSP vectors by `Alex Gramfort`_ and `Martin Luessi`_.

- Wrapper function and script (mne_maxfilter.py) for Elekta Neuromag MaxFilter(TM) by `Martin Luessi`_

- Add method to eliminate stimulation artifacts from raw data by linear interpolation or windowing by `Daniel Strohmeier`_.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 118 Alexandre Gramfort
* 81  Martin Luessi
* 15  Daniel Strohmeier
*  4  Christian Brodbeck
*  4  Louis Thibault
*  2  Brad Buran

.. _changes_0_3:

Version 0.3
-----------

Changelog
~~~~~~~~~

- Sign flip computation for robust label average of signed values by `Alex Gramfort`_.

- Reading and writing of .w files by `Martin Luessi`_.

- Support for modifying Raw object and allow raw data preloading with memory mapping by `Martin Luessi`_ and `Alex Gramfort`_.

- Support of arithmetic of Evoked data (useful to concatenate between runs and compute contrasts) by `Alex Gramfort`_.

- Support for computing sensor space data from a source estimate using an MNE forward solution by `Martin Luessi`_.

- Support of arithmetic of Covariance by `Alex Gramfort`_.

- Write BEM surfaces in Python  by `Alex Gramfort`_.

- Filtering operations and apply_function interface for Raw object by `Martin Luessi`_.

- Support for complex valued raw fiff files and computation of analytic signal for Raw object by `Martin Luessi`_.

- Write inverse operators (surface and volume) by `Alex Gramfort`_.

- Covariance matrix computation with multiple event types by `Martin Luessi`_.

- New tutorial in the documentation and new classes and functions reference page by `Alex Gramfort`_.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 80  Alexandre Gramfort
* 51  Martin Luessi

Version 0.2
-----------

Changelog
~~~~~~~~~

- New stats functions for FDR correction and Bonferroni by `Alex Gramfort`_.

- Faster time-frequency using downsampling trick by `Alex Gramfort`_.

- Support for volume source spaces by `Alex Gramfort`_ (requires next MNE release or nightly).

- Improved Epochs handling by `Martin Luessi`_ (slicing, drop_bad_epochs).

- Bug fix in Epochs + ECG detection by Manfred Kitzbichler.

- New pick_types_evoked function by `Alex Gramfort`_.

- SourceEstimate now supports algebra by `Alex Gramfort`_.

API changes summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are the code migration instructions when upgrading from mne-python
version 0.1:

- New return values for the function find_ecg_events

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 33  Alexandre Gramfort
* 12  Martin Luessi
*  2  Yaroslav Halchenko
*  1  Manfred Kitzbichler

.. _Alex Gramfort: http://alexandre.gramfort.net

.. _Martin Luessi: https://www.martinos.org/user/8245

.. _Yaroslav Halchenko: http://www.onerussian.com/

.. _Daniel Strohmeier: http://www.tu-ilmenau.de/bmti/fachgebiete/biomedizinische-technik/dipl-ing-daniel-strohmeier/

.. _Eric Larson: http://larsoner.com

.. _Denis Engemann: http://denis-engemann.de

.. _Christian Brodbeck: https://github.com/christianbrodbeck

.. _Simon Kornblith: http://simonster.com

.. _Teon Brooks: https://teonbrooks.github.io

.. _Mainak Jas: http://ltl.tkk.fi/wiki/Mainak_Jas

.. _Roman Goj: http://romanmne.blogspot.co.uk

.. _Andrew Dykstra: https://github.com/adykstra

.. _Romain Trachel: http://www.lscp.net/braware/trachelBr.html

.. _Christopher Dinh: https://github.com/chdinh

.. _Nick Ward: http://www.ucl.ac.uk/ion/departments/sobell/Research/NWard

.. _Tal Linzen: http://tallinzen.net/

.. _Roan LaPlante: https://github.com/aestrivex

.. _Mads Jensen: https://github.com/MadsJensen

.. _Dan Wakeman: https://github.com/dgwakeman

.. _Qunxi Dong: https://github.com/dongqunxi

.. _Martin Billinger: https://github.com/kazemakase

.. _Federico Raimondo: https://github.com/fraimondo

.. _Cathy Nangini: https://github.com/KatiRG

.. _Jean-Remi King: https://github.com/kingjr

.. _Juergen Dammers: https://github.com/jdammers

.. _Olaf Hauk: http://www.neuroscience.cam.ac.uk/directory/profile.php?olafhauk

.. _Lukas Breuer: http://www.researchgate.net/profile/Lukas_Breuer

.. _Federico Raimondo: https://github.com/fraimondo

.. _Alan Leggitt: https://github.com/leggitta

.. _Marijn van Vliet: https://github.com/wmvanvliet

.. _Marmaduke Woodman: https://github.com/maedoc

.. _Jona Sassenhagen: https://github.com/jona-sassenhagen

.. _Hari Bharadwaj: http://www.haribharadwaj.com

.. _Chris Holdgraf: http://chrisholdgraf.com

.. _Jaakko Leppakangas: https://github.com/jaeilepp

.. _Yousra Bekhti: https://www.linkedin.com/pub/yousra-bekhti/56/886/421

.. _Mark Wronkiewicz: http://ilabs.washington.edu/graduate-students/bio/i-labs-mark-wronkiewicz

.. _Sébastien Marti: http://www.researchgate.net/profile/Sebastien_Marti

.. _Chris Bailey: https://github.com/cjayb

.. _Ross Maddox: https://www.urmc.rochester.edu/labs/maddox-lab.aspx

.. _Alexandre Barachant: http://alexandre.barachant.org

.. _Daniel McCloy: http://dan.mccloy.info

.. _Jair Montoya Martinez: https://github.com/jmontoyam

.. _Samu Taulu: http://ilabs.washington.edu/institute-faculty/bio/i-labs-samu-taulu-dsc

.. _Lorenzo Desantis: https://github.com/lorenzo-desantis/

.. _Jukka Nenonen: https://www.linkedin.com/pub/jukka-nenonen/28/b5a/684

.. _Jussi Nurminen: https://scholar.google.fi/citations?user=R6CQz5wAAAAJ&hl=en

.. _Clemens Brunner: https://github.com/cle1109

.. _Asish Panda: https://github.com/kaichogami

.. _Natalie Klein: http://www.stat.cmu.edu/people/students/neklein

.. _Jon Houck: https://scholar.google.com/citations?user=DNoS05IAAAAJ&hl=en

.. _Pablo-Arias: https://github.com/Pablo-Arias

.. _Alexander Rudiuk: https://github.com/ARudiuk

.. _Mikołaj Magnuski: https://github.com/mmagnuski

.. _Felix Raimundo: https://github.com/gamazeps

.. _Nick Foti: http://nfoti.github.io

.. _Guillaume Dumas: http://www.extrospection.eu

.. _Chris Mullins: http://crmullins.com

.. _Phillip Alday: https://palday.bitbucket.io

.. _Andreas Hojlund: https://github.com/ahoejlund

.. _Johannes Niediek: https://github.com/jniediek

.. _Sheraz Khan: https://github.com/SherazKhan

.. _Antti Rantala: https://github.com/Odingod

.. _Keith Doelling: http://science.keithdoelling.com

.. _Paul Pasler: https://github.com/ppasler

.. _Niklas Wilming: https://github.com/nwilming

.. _Annalisa Pascarella: http://www.iac.rm.cnr.it/~pasca/

.. _Luke Bloy: https://scholar.google.com/citations?hl=en&user=Ad_slYcAAAAJ&view_op=list_works&sortby=pubdate

.. _Leonardo Barbosa: https://github.com/noreun

.. _Erkka Heinila: https://github.com/Teekuningas

.. _Andrea Brovelli: http://www.int.univ-amu.fr/_BROVELLI-Andrea_?lang=en

.. _Richard Höchenberger: http://hoechenberger.name

.. _Matt Boggess: https://github.com/mattboggess

.. _Jean-Baptiste Schiratti: https://github.com/jbschiratti

.. _Laura Gwilliams: http://lauragwilliams.github.io

.. _Jesper Duemose Nielsen: https://github.com/jdue

.. _Mathurin Massias: https://mathurinm.github.io/

.. _ramonapariciog: https://github.com/ramonapariciog

.. _Britta Westner: https://github.com/britta-wstnr

.. _Lukáš Hejtmánek: https://github.com/hejtmy

.. _Stefan Repplinger: https://github.com/stfnrpplngr

.. _Okba Bekhelifi: https://github.com/okbalefthanded

.. _Nicolas Barascud: https://github.com/nbara

.. _Alejandro Weinstein: http://ocam.cl

.. _Emily Stephen: http://github.com/emilyps14

.. _Nathalie Gayraud: https://github.com/ngayraud

.. _Anne-Sophie Dubarry: https://github.com/annesodub

.. _Stefan Appelhoff: http://stefanappelhoff.com

.. _Tommy Clausner: https://github.com/TommyClausner

.. _Pierre Ablin: https://pierreablin.com

.. _Oleh Kozynets: https://github.com/OlehKSS

.. _Susanna Aro: https://www.linkedin.com/in/susanna-aro

.. _Joan Massich: https://github.com/massich

.. _Henrich Kolkhorst: https://github.com/hekolk

.. _Steven Bethard: https://github.com/bethard

.. _Thomas Hartmann: https://github.com/thht

.. _Steven Gutstein: http://robust.cs.utep.edu/~gutstein

.. _Peter Molfese: https://github.com/pmolfese

.. _Dirk Gütlin: https://github.com/DiGyt

.. _Jasper van den Bosch: https://github.com/ilogue

.. _Ezequiel Mikulan: https://github.com/ezemikulan

.. _Rasmus Zetter: https://people.aalto.fi/rasmus.zetter

.. _Marcin Koculak: https://github.com/mkoculak

.. _David Sabbagh: https://github.com/DavidSabbagh

.. _Hubert Banville: https://github.com/hubertjb

.. _buildqa: https://github.com/buildqa

.. _jeythekey: https://github.com/jeythekey

.. _Sara Sommariva: http://www.dima.unige.it/~sommariva/

.. _Cristóbal Moënne-Loccoz: https://github.com/cmmoenne

.. _David Haslacher: https://github.com/davidhaslacher

.. _Larry Eisenman:  https://github.com/lneisenman

.. _Stanislas Chambon: https://github.com/Slasnista

.. _Jeff Hanna: https://github.com/jshanna100

.. _kalenkovich: https://github.com/kalenkovich

.. _Antoine Gauthier: https://github.com/Okamille

.. _Samuel Deslauriers-Gauthier: https://github.com/sdeslauriers

.. _Sebastian Castano: https://github.com/jscastanoc

.. _Guillaume Favelier: https://github.com/GuillaumeFavelier

.. _Katarina Slama: https://katarinaslama.github.io

.. _Bruno Nicenboim: http://nicenboim.org

.. _Ivana Kojcic: https://github.com/ikojcic

.. _Nikolas Chalas: https://github.com/Nichalas

.. _Quentin Bertrand: https://github.com/QB3

.. _Alexander Kovrig: https://github.com/OpenSatori

.. _Kostiantyn Maksymenko: https://github.com/makkostya
