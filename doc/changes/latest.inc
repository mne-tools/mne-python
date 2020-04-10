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

- Add support for signals in mV for :func:`mne.io.read_raw_brainvision` by `Clemens Brunner`_

- Allow resampling raw data with :func:`mne.io.Raw.resample` without preloading data, by `Eric Larson`_

- :func:`mne.viz.plot_dipole_locations` and :meth:`mne.Dipole.plot_locations` gained a ``title`` argument to specify a custom figure title in ``orthoview`` mode by `Richard Höchenberger`_

- Added temporal derivative distribution repair :func:`mne.preprocessing.nirs.temporal_derivative_distribution_repair` by `Robert Luke`_

- Add functionality to interpolate bad NIRS channels by `Robert Luke`_

- Add function :func:`mne.preprocessing.annotate_muscle_zscore` to annotate periods with muscle artifacts. by `Adonay Nunes`_

Bug
~~~

- Fix bug with :func:`mne.minimum_norm.compute_source_psd_epochs` and :func:`mne.minimum_norm.source_band_induced_power` raised errors when ``method='eLORETA'`` by `Eric Larson`_

- Fix bug with :func:`mne.io.Raw.plot` where toggling all projectors did not actually take effect by `Eric Larson`_

- Fix bug with :func:`mne.read_epochs` when loading data in complex format with ``preload=False`` by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.nirs.scalp_coupling_index` where filter transition was incorrectly assigned by `Robert Luke`_

- Fix bug with :func:`mne.make_forward_dipole` where :func:`mne.write_forward_solution` could not be used by `Eric Larson`_

- Fix bug that prevents ``n_jobs`` from being a NumPy integer type, by `Daniel McCloy`_.

- Fix bug with :func:`mne.io.read_raw_gdf` where birthdays were not parsed properly, leading to an error by `Svea Marie Meyer`_



API
~~~

- Nothing yet
