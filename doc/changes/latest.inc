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

Enhancements
~~~~~~~~~~~~

- Add ``n_pca_components`` argument to :func:`mne.viz.plot_ica_overlay` by `Eric Larson`_ (:gh:`8351`)

- Add :func:`mne.stc_near_sensors` to facilitate plotting ECoG data by `Eric Larson`_ (:gh:`8190`)

- Add ``proj`` argument to :func:`mne.make_fixed_length_epochs` by `Eric Larson`_ (:gh:`8351`)

- Speed up :class:`mne.decoding.TimeDelayingRidge` with edge correction using Numba by `Eric Larson`_ (:gh:`8323`)

Bugs
~~~~
- Fix bug with reading split files that have dashes in the filename **by new contributor** |Eduard Ort|_ (:gh:`8339`)

- Fix bug with `~mne.viz.plot_epochs_image` when ``order`` is supplied and multiple conditions are plotted by `Daniel McCloy`_ (:gh:`8377`)

- Fix bug with :func:`mne.viz.plot_source_estimates` when using the PyVista backend where singleton time points were not handled properly by `Eric Larson`_ (:gh:`8285`)

- Fix bug when passing ``axes`` to plotting functions, :func:`matplotlib.pyplot.tight_layout` will not be called when the figure was created using a constrained layout by `Eric Larson`_ (:gh:`8344`)

- Fix bug with compensated CTF data when picking channels without preload by `Eric Larson`_ (:gh:`8318`)

- Fix bug when merging fNIRS channels in :func:`mne.viz.plot_evoked_topomap` and related functions by `Robert Luke`_ (:gh:`8306`)

- :func:`mne.io.read_raw_edf` now supports EDF files with invalid recording dates by `Clemens Brunner`_ (:gh:`8283`)

- Fix bug with :class:`mne.preprocessing.ICA` where ``n_pca_components`` as a :class:`python:float` would give the number of components that explained less than or equal to the given variance. It now gives greater than the given number for better usability and consistency with :class:`sklearn.decomposition.PCA`. Generally this will mean that one more component will be included, by `Eric Larson`_ (:gh:`8326`)

- Fix bug with :class:`mne.preprocessing.ICA` where projections were not tracked properly by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.read_ica_eeglab` where full-rank data were not handled properly by `Eric Larson`_ (:gh:`8326`)

- Fix bug with :ref:`somato-dataset` where the BEM was not included by `Eric Larson`_ (:gh:`8317`)

- Fix missing documentation of :func:`mne.io.read_raw_nihon` in :ref:`tut-imorting-eeg-data` by `Adam Li`_ (:gh`8320`)

- Fix bug with :func:`mne.add_reference_channels` when :func:`mne.io.Raw.reorder_channels` or related methods are used afterward by `Eric Larson`_ (:gh:`8303`)

- Fix bug where the ``verbose`` arguments to :meth:`mne.Evoked.apply_baseline` and :meth:`mne.Epochs.apply_baseline` were not keyword-only by `Eric Larson`_ (:gh:`8349`)

- ``ICA.max_pca_components`` will not be altered by calling `~mne.preprocessing.ICA.fit` anymore. Instead, the new attribute ``ICA.max_pca_components_`` will be set, by `Richard Höchenberger`_ (:gh:`8321`)

- Fix bug that `~mne.viz.plot_ica_overlay` would sometimes not create red traces, by `Richard Höchenberger`_ (:gh:`8341`)

- Fix bug with :class:`~mne.preprocessing.ICA` where ``n_components=None, n_pca_components=None`` could lead to unstable unmixing matrix inversion by making ``n_components=None`` also use the lesser of ``n_components=0.999999`` and ``n_components=n_pca_components`` by `Eric Larson`_ (:gh:`8351`)

- The ``ica.n_pca_components`` property is no longer be updated during :meth:`mne.preprocessing.ICA.fit`, instead ``ica.n_components_`` will be added to the instance by `Eric Larson`_ (:gh:`8351`)

- Pass ``rank`` everyhwere in forward preparation for source imaging. This bug affected sparse solvers when using maxfilter data, by `Alex Gramfort`_ (:gh:`8368`)

- Fix :func:`mne.io.read_raw_curry` to deal with Curry datasets that have channels that are listed in the labels file, but which are absent from the saved data file (e.g. 'Ref' channel).  Also now populates info['meas_date'] if possible by `Tod Flak` _ (:gh:`8400`)

API changes
~~~~~~~~~~~

- Minimum required versions were increased for core dependencies NumPy (1.15.4), SciPy (1.1.0), and Matplotlib (3.0) and for the optional dependencies scikit-learn (0.20.2) and pandas (0.23.4), by `Eric Larson`_

- The parameter ``on_split_missing`` has been added to :func:`mne.io.read_raw_fif` and its default will change from ``'warn'`` to ``'raise'`` in 0.23, by `Eric Larson`_ (:gh:`8357`)

- The ``max_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_components`` during initialization and ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` instead by `Eric Larson`_ (:gh:`8351`)

- The ``n_pca_components`` argument of :class:`~mne.preprocessing.ICA` has been deprecated, use ``n_pca_components`` in :meth:`~mne.preprocessing.ICA.apply` by `Eric Larson`_ (:gh:`8356`)

- The ``trans`` argument of :func:`mne.extract_label_time_course` is deprecated and will be removed in 0.23 as it is no longer necessary by `Eric Larson`_
