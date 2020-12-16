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

Enhancements
~~~~~~~~~~~~
- Update ``surfaces`` argument in :func:`mne.viz.plot_alignment` to allow dict for transparency values, and set default for sEEG data to have transparency (by `Keith Doelling`_ (:gh:`8445`))

- Add ``n_pca_components`` argument to :func:`mne.viz.plot_ica_overlay` by `Eric Larson`_ (:gh:`8351`)

- Add :func:`mne.stc_near_sensors` to facilitate plotting ECoG data by `Eric Larson`_ (:gh:`8190`)

- Add ``proj`` argument to :func:`mne.make_fixed_length_epochs` by `Eric Larson`_ (:gh:`8351`)

- Reduce memory usage of volume source spaces by `Eric Larson`_ (:gh:`8379`)

- Speed up heavy use of :meth:`mne.SourceMorph.apply` for volumetric source spaces by use of the method :meth:`mne.SourceMorph.compute_vol_morph_mat` by `Eric Larson`_ (:gh:`8366`)

- Add support for non-uniform ``zooms`` (e.g., when using a surrogate MRI via :func:`mne.scale_mri`) in volumetric morphing by `Eric Larson`_  (:gh:`8642`)

- In :func:`mne.compute_source_morph` ``zooms`` are no longer required to match the spacing of ``src_to``, which is useful to ensure the morphing is accurate when the ``src_to`` spacing is large (e.g., 1 cm) by `Eric Larson`_ (:gh:`8642`)

- Add volumetric source space support to :func:`mne.labels_to_stc` by `Eric Larson`_ (:gh:`8447`)

- Speed up :class:`mne.decoding.TimeDelayingRidge` with edge correction using Numba by `Eric Larson`_ (:gh:`8323`)

- Add :meth:`mne.Epochs.reset_drop_log_selection` to facilitate writing epochs with many ignored entries in their drop log by `Eric Larson`_ (:gh:`8449`)

- Add sEEG source visualization using :func:`mne.stc_near_sensors` and sEEG working tutorial by `Eric Larson`_ and `Adam Li`_ (:gh:`8402`)

- Add :meth:`mne.channels.DigMontage.get_positions`, which will return a dictionary of channel positions, coordinate frame and fiducial locations by `Adam Li`_ (:gh:`8460`)

Bugs
~~~~
- Fix bug with mne.io.egi.tests/test_egi.py where it mandatorily downloaded testing data when it was not necessary **by new contributor** |Aniket Pradhan|_ (:gh:`8474`)

- Fix bug with reading split files that have dashes in the filename **by new contributor** |Eduard Ort|_ (:gh:`8339`)

- Fix bug with `~mne.viz.plot_epochs_image` when ``order`` is supplied and multiple conditions are plotted by `Daniel McCloy`_ (:gh:`8377`)

- Fix bug with :func:`mne.viz.plot_source_estimates` when using the PyVista backend where singleton time points were not handled properly by `Eric Larson`_ (:gh:`8285`)

- Fix bug when passing ``axes`` to plotting functions, :func:`matplotlib.pyplot.tight_layout` will not be called when the figure was created using a constrained layout by `Eric Larson`_ (:gh:`8344`)

- Fix bug with compensated CTF data when picking channels without preload by `Eric Larson`_ (:gh:`8318`)

- Fix bug when merging fNIRS channels in :func:`mne.viz.plot_evoked_topomap` and related functions by `Robert Luke`_ (:gh:`8306`)

- Fix bug where extrapolation points created artifacts in :func:`mne.viz.plot_evoked_topomap` and related functions (:gh:`8425` by `_Mikołaj Magnuski`_)

- Fix bug where events could overflow when writing to FIF by `Eric Larson`_ (:gh:`8448`)

- :func:`mne.io.read_raw_edf` now supports EDF files with invalid recording dates by `Clemens Brunner`_ (:gh:`8283`)

- Fix bug with :func:`mne.io.Raw.save` when using ``split_naming='bids'`` where non-split files would still be named ``name_split-01_meg.fif`` instead of the requested ``name_meg.fif`` by `Alex Gramfort`_ and `Eric Larson`_ (:gh:`8464`)

- Fix bug with :class:`mne.preprocessing.ICA` where ``n_pca_components`` as a :class:`python:float` would give the number of components that explained less than or equal to the given variance. It now gives greater than the given number for better usability and consistency with :class:`sklearn.decomposition.PCA`. Generally this will mean that one more component will be included, by `Eric Larson`_ (:gh:`8326`)

- Fix bug with :class:`mne.preprocessing.ICA` where projections were not tracked properly by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.read_ica_eeglab` where full-rank data were not handled properly by `Eric Larson`_ (:gh:`8326`)

- Fix bug with :ref:`somato-dataset` where the BEM was not included by `Eric Larson`_ (:gh:`8317`)

- Fix bug with coordinate frames when performing volumetric morphs via :func:`mne.compute_source_morph` and :meth:`mne.SourceMorph.apply` that could lead to ~5 mm bias by `Eric Larson`_ (:gh:`8642`)

- Fix missing documentation of :func:`mne.io.read_raw_nihon` in :ref:`tut-imorting-eeg-data` by `Adam Li`_ (:gh`8320`)

- Fix bug with :func:`mne.add_reference_channels` when :func:`mne.io.Raw.reorder_channels` or related methods are used afterward by `Eric Larson`_ (:gh:`8303`)

- Fix bug where the ``verbose`` arguments to :meth:`mne.Evoked.apply_baseline` and :meth:`mne.Epochs.apply_baseline` were not keyword-only by `Eric Larson`_ (:gh:`8349`)

- ``ICA.max_pca_components`` will not be altered by calling `~mne.preprocessing.ICA.fit` anymore. Instead, the new attribute ``ICA.max_pca_components_`` will be set, by `Richard Höchenberger`_ (:gh:`8321`)

- Fix bug that `~mne.viz.plot_ica_overlay` would sometimes not create red traces, by `Richard Höchenberger`_ (:gh:`8341`)

- Fix bug with :class:`~mne.preprocessing.ICA` where ``n_components=None, n_pca_components=None`` could lead to unstable unmixing matrix inversion by making ``n_components=None`` also use the lesser of ``n_components=0.999999`` and ``n_components=n_pca_components`` by `Eric Larson`_ (:gh:`8351`)

- The ``ica.n_pca_components`` property is no longer be updated during :meth:`mne.preprocessing.ICA.fit`, instead ``ica.n_components_`` will be added to the instance by `Eric Larson`_ (:gh:`8351`)

- Pass ``rank`` everyhwere in forward preparation for source imaging. This bug affected sparse solvers when using maxfilter data, by `Alex Gramfort`_ (:gh:`8368`)

- Fix bug in :func:`mne.viz.plot_alignment` where ECoG and sEEG channels were not plotted and fNIRS channels were always plotted in the head coordinate frame by `Eric Larson`_ (:gh:`8393`)

- Fix bug in :func:`mne.set_bipolar_reference` where ``ch_info`` could contain invalid channel information keys by `Eric Larson`_

- When reading BrainVision raw data, the channel units and types were sometimes not inferred correctly by `Richard Höchenberger`_ (:gh:`8434`)

- Attempting to remove baseline correction from preloaded `~mne.Epochs` will now raise an exception by `Richard Höchenberger`_ (:gh:`8435`)

API changes
~~~~~~~~~~~

- Minimum required versions were increased for core dependencies NumPy (1.15.4), SciPy (1.1.0), and Matplotlib (3.0) and for the optional dependencies scikit-learn (0.20.2) and pandas (0.23.4), by `Eric Larson`_

- The parameter ``on_split_missing`` has been added to :func:`mne.io.read_raw_fif` and its default will change from ``'warn'`` to ``'raise'`` in 0.23, by `Eric Larson`_ (:gh:`8357`)

- The ``max_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_components`` during initialization and ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` instead by `Eric Larson`_ (:gh:`8351`)

- The ``n_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` by `Eric Larson`_ (:gh:`8356`)

- The ``trans`` argument of :func:`mne.extract_label_time_course` is deprecated and will be removed in 0.23 as it is no longer necessary by `Eric Larson`_

- New `mne.viz.Brain.set_time` method to set the displayed time in seconds, by `Daniel McCloy`_ (:gh:`8415`)

- Update the ``backend`` parameter of :func:`mne.viz.plot_source_estimates` to integrate ``pyvista`` by `Guillaume Favelier`_
