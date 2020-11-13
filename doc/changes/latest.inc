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

- Update ``surfaces`` argument in :func:`mne.viz.plot_alignment` to allow dict for transparency values, and set default for sEEG data to have transparency (:gh:`8445` by `Keith Doelling`_)

- Add ``n_pca_components`` argument to :func:`mne.viz.plot_ica_overlay` (:gh:`8351` by `Eric Larson`_)

- Add :func:`mne.stc_near_sensors` to facilitate plotting ECoG data (:gh:`8190` by `Eric Larson`_)

- Add ``proj`` argument to :func:`mne.make_fixed_length_epochs` (:gh:`8351` by `Eric Larson`_)

- Reduce memory usage of volume source spaces (:gh:`8379` by `Eric Larson`_)

- Speed up heavy use of :meth:`mne.SourceMorph.apply` for volumetric source spaces by use of the method :meth:`mne.SourceMorph.compute_vol_morph_mat` (:gh:`8366` by `Eric Larson`_)

- Add support for non-uniform ``zooms`` (e.g., when using a surrogate MRI via :func:`mne.scale_mri`) in volumetric morphing  (:gh:`8642` by `Eric Larson`_)

- In :func:`mne.compute_source_morph` ``zooms`` are no longer required to match the spacing of ``src_to``, which is useful to ensure the morphing is accurate when the ``src_to`` spacing is large (e.g., 1 cm) (:gh:`8642` by `Eric Larson`_)

- Add volumetric source space support to :func:`mne.labels_to_stc` (:gh:`8447` by `Eric Larson`_)

- Speed up :class:`mne.decoding.TimeDelayingRidge` with edge correction using Numba (:gh:`8323` by `Eric Larson`_)

- Add :meth:`mne.Epochs.reset_drop_log_selection` to facilitate writing epochs with many ignored entries in their drop log (:gh:`8449` by `Eric Larson`_)

- Add sEEG source visualization using :func:`mne.stc_near_sensors` and sEEG working tutorial (:gh:`8402` by `Eric Larson`_ and `Adam Li`_)

- Add :meth:`mne.channels.DigMontage.get_positions`, which will return a dictionary of channel positions, coordinate frame and fiducial locations (:gh:`8460` by `Adam Li`_)

- Add ``picks`` parameter to :func:`mne.preprocessing.fix_stim_artifact` to specify which channel needs to be fixed (:gh:`8482` by `Alex Gramfort`_)

- Further improved documentation building instructions and execution on Windows (:gh:`8502` by `kalenkovich`_ and `Eric Larson`_)

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

- Fix bug in :func:`mne.io.read_raw_eeglab` where empty event durations led to an error (:gh:`8384` by `Mikołaj Magnuski`_)

API changes
~~~~~~~~~~~

- Added `mne.SourceEstimate.apply_baseline` method for baseline-correction of source estimates, (:gh:`8452` by `Olaf Hauk`_)

- Minimum required versions were increased for core dependencies NumPy (1.15.4), SciPy (1.1.0), and Matplotlib (3.0) and for the optional dependencies scikit-learn (0.20.2) and pandas (0.23.4) (:gh:`8374` by `Eric Larson`_)

- The parameter ``on_split_missing`` has been added to :func:`mne.io.read_raw_fif` and its default will change from ``'warn'`` to ``'raise'`` in 0.23, by (:gh:`8357` `Eric Larson`_)

- The ``max_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_components`` during initialization and ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` instead (:gh:`8351` by `Eric Larson`_)

- The ``n_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` (:gh:`8356` by `Eric Larson`_)

- The ``trans`` argument of :func:`mne.extract_label_time_course` is deprecated and will be removed in 0.23 as it is no longer necessary (:gh:`8389` by `Eric Larson`_)

- New `mne.viz.Brain.set_time` method to set the displayed time in seconds (:gh:`8415` by `Daniel McCloy`_)

- Update the ``backend`` parameter of :func:`mne.viz.plot_source_estimates` to integrate ``pyvista`` (:gh:`8395` by `Guillaume Favelier`_)

- Add ``group_by`` parameter to `mne.viz.plot_epochs` and `mne.Epochs.plot` to allow displaying channel data by sensor position (:gh:`8381` by `Daniel McCloy`_)

- Parameter ``event_colors`` in `mne.viz.plot_epochs` and `mne.Epochs.plot` is deprecated, replaced by ``event_color`` which is consistent with `mne.viz.plot_raw` and provides greater flexibility (:gh:`8381` by `Daniel McCloy`_)
