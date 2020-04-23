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

- :meth:`mne.Epochs.plot_psd_topomap` and :func:`mne.viz.plot_epochs_psd_topomap` now allow joint colorbar limits across subplots, by `Daniel McCloy`_.

- Add :class:`mne.MixedVectorSourceEstimate` for vector source estimates for mixed source spaces, by `Eric Larson`_

- Add :meth:`mne.MixedSourceEstimate.surface` and :meth:`mne.MixedSourceEstimate.volume` methods to allow surface and volume extraction by `Eric Larson`_

- Add support to :func:`mne.extract_label_time_course` for vector-valued :class:`VectorSourceEstimate` and :class:`mne.MixedVectorSourceEstimate` by `Eric Larson`_

- Allow resampling raw data with :func:`mne.io.Raw.resample` without preloading data, by `Eric Larson`_

- Allow using ``pick_ori='vector'`` with a fixed-orientation inverse to facilitate visualization with :func:`mne.viz.plot_vector_source_estimates` by `Eric Larson`_

- :func:`mne.viz.plot_dipole_locations` and :meth:`mne.Dipole.plot_locations` gained a ``title`` argument to specify a custom figure title in ``orthoview`` mode by `Richard Höchenberger`_

- Added temporal derivative distribution repair :func:`mne.preprocessing.nirs.temporal_derivative_distribution_repair` by `Robert Luke`_

- Add functionality to interpolate bad NIRS channels by `Robert Luke`_

- Add support to in :meth:`mne.io.Raw.plot` for passing ``clipping`` as a float to clip to a proportion of the dedicated channel range by `Eric Larson`_

- Add function :func:`mne.preprocessing.annotate_muscle_zscore` to annotate periods with muscle artifacts. by `Adonay Nunes`_

- Support for saving movies of source time courses (STCs) with ``brain.save_movie`` method and from graphical user interface by `Guillaume Favelier`_

- Add "on_missing='raise'" to :meth:`mne.io.Raw.set_montage` and related functions to allow ignoring of missing electrode coordinates by `Adam Li`_

Bug
~~~

- Fix bug with :func:`mne.setup_volume_source_space` when ``volume_label`` was supplied where voxels slightly (in a worst case, about 37% times ``pos`` in distance) outside the voxel-grid-based bounds of regions were errantly included, by `Eric Larson`_

- Fix bug with :func:`mne.minimum_norm.compute_source_psd_epochs` and :func:`mne.minimum_norm.source_band_induced_power` raised errors when ``method='eLORETA'`` by `Eric Larson`_

- The default plotting mode for :func:`mne.io.Raw.plot` and :ref:`gen_mne_browse_raw` has been changed to ``clipping=3.`` to facilitate data analysis with large deflections, by `Eric Larson`_

- Fix handling of NaN when using TFCE in clustering functions such as :func:`mne.stats.spatio_temporal_cluster_1samp_test` by `Eric Larson`_

- The :class:`mne.MixedSourceEstimate` class has been clarified to contain two cortical surface source spaces, plus at least one other source space. Creating source estimates in other orderings is not supported, by `Eric Larson`_

- Fix bug where :class:`VolSourceEstimate.vertices <mne.VolSourceEstimate>` was an instance of :class:`~numpy.ndarray` instead of :class:`python:list` of one :class:`~numpy.ndarray`, by `Eric Larson`_

- Fix bug with :func:`mne.io.Raw.plot` where toggling all projectors did not actually take effect by `Eric Larson`_

- Fix bug with :func:`mne.read_epochs` when loading data in complex format with ``preload=False`` by `Eric Larson`_

- Fix bug with :class:`mne.Report` where the BEM section could not be toggled by `Eric Larson`_

- Fix bug when using :meth:`mne.Epochs.crop` to exclude the baseline period would break :func:`mne.Epochs.save` / :func:`mne.read_epochs` round-trip by `Eric Larson`_

- Fix :ref:`gen_mne_setup_forward_model` to have it actually compute the BEM solution in addition to creating the BEM model by `Eric Larson`_

- Fix bug with :func:`mne.io.read_raw_edf` where null bytes were not properly handled, causing an error when opening a file by `Eric Larson`_

- Fix bug with :func:`mne.preprocessing.nirs.scalp_coupling_index` where filter transition was incorrectly assigned by `Robert Luke`_

- Fix bug with :func:`mne.make_forward_dipole` where :func:`mne.write_forward_solution` could not be used by `Eric Larson`_

- Fix bug with :meth:`mne.io.Raw.plot` when ``scalings='auto'`` where bad data would prevent channel plotting by `Eric Larson`_

- Default ``border`` and ``extrapolate`` arguments for :func:`mne.Evoked.plot_topomap` and related functions were changed from ``0.`` to ``'mean'`` and ``'box'`` to ``'auto'``, respectively, to help more accurately reflect sensor geometries and boundary conditions. ``extrapolate='auto'`` uses ``extrapolate='local'`` for MEG data and ``extrapolate='head'`` otherwise, by `Eric Larson`_

- Fix bug that prevents ``n_jobs`` from being a NumPy integer type, by `Daniel McCloy`_.

- Fix bug with :func:`mne.io.read_raw_gdf` where birthdays were not parsed properly, leading to an error by `Svea Marie Meyer`_

- Fix bug with :func:`mne.io.read_raw_edf` where recording ID was not read properly for non-ASCII characters by `Lx37`_

- Make :func:`mne.set_config` accept path-like input values by `Richard Höchenberger`_

- Fix bug with :func:`mne.beamformer.make_dics` where complex conjugates were not applied properly by `Britta Westner`_ and `Eric Larson`_

- Fix bug with :func:`mne.bem.make_watershed_bem` where the RAS coordinates of watershed bem surfaces were not updated correctly from the volume file by `Yu-Han Luo`_

API
~~~

- Python 3.5 is no longer supported, Python 3.6+ is required, by `Eric Larson`_

- ``vmin`` and ``vmax`` parameters are deprecated in :meth:`mne.Epochs.plot_psd_topomap` and :func:`mne.viz.plot_epochs_psd_topomap`; use new ``vlim`` parameter instead, by `Daniel McCloy`_.

- The method ``stc_mixed.plot_surface`` for a :class:`mne.MixedSourceEstimate` has been deprecated in favor of :meth:`stc.surface().plot(...) <mne.MixedSourceEstimate.surface>` by `Eric Larson`_

- The function ``mne.channels.read_dig_captrack`` will be deprecated in version 0.22 in favor of :func:`mne.channels.read_dig_captrak` to correct the spelling error: "captraCK" -> "captraK", by `Stefan Appelhoff`_
