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
- Add command :ref:`gen_mne_setup_source_space` to quickly set up bilateral hemisphere surface-based source space with subsampling by `Victor Ferat`_.

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

- Add function to convert NIRS data to optical density :func:`mne.preprocessing.optical_density` by `Robert Luke`_

- Add function to convert NIRS data to haemoglobin concentration :func:`mne.preprocessing.beer_lambert_law` by `Robert Luke`_

- Add reader for ``*.dat`` electrode position files :func:`mne.channels.read_dig_dat` by `Christian Brodbeck`_

- Improved :ref:`limo-dataset` usage and :ref:`example <ex-limo-data>` for usage of :func:`mne.stats.linear_regression` by `Jose Alanis`_

- For KIT systems without built-in layout, :func:`mne.channels.find_layout` now falls back on an automatically generated layout, by `Christian Brodbeck`_

- :meth:`mne.Epochs.plot` now takes a ``epochs_colors`` parameter to color specific epoch segments by `Mainak Jas`_

- Add command :ref:`gen_mne_setup_forward_model` to quickly create a BEM model for a subject by `Victor Ferat`_.

- Add command :ref:`gen_mne_prepare_bem_model` to quickly create a BEM solution using the linear collocation approach by `Victor Ferat`_.

Bug
~~~

- Fix :meth:`mne.Epochs.shift_time` and :meth:`mne.Evoked.shift_time` to return the modified :class:`~mne.Epochs` or :class:`~mne.Evoked` instance (instead of ``None``) by `Daniel McCloy`_.

- Fix bug in :class:`~mne.preprocessing.ICA` where requesting extended infomax via ``fit_params={'extended': True}`` was overridden, by `Daniel McCloy`_.

- Fix bug in :meth:`mne.Epochs.interpolate_bads` where the ``origin`` was not used during MEG or EEG data interpolation by `Eric Larson`_. Old behavior can be achieved using ``origin=(0., 0., 0.)`` for EEG and ``origin=(0., 0., 0.04)`` for MEG, and the new default is ``origin='auto'``, which uses a head-digitization-based fit.

- Fix bug in :func:`mne.write_evokeds` where ``evoked.nave`` was not saved properly when multiple :class:`~mne.Evoked` instances were written to a single file, by `Eric Larson`_

- Fix bug in :func:`mne.preprocessing.mark_flat` where acquisition skips were not handled proeprly, by `Eric Larson`_

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

API
~~~

- The APIs of :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` are now more similar to :func:`mne.viz.plot_projs_topomap` by `Daniel McCloy`_.

- :func:`mne.viz.plot_projs_topomap` now accepts both :class:`~mne.channels.Layout` and :class:`~mne.Info` (previously one or the other), and will ignore :class:`~mne.Info` if :class:`~mne.channels.Layout` is provided `Daniel McCloy`_.

- :func:`mne.viz.plot_projs_topomap` and the related methods :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` now accept parameter ``vlim`` to control the colormap, with keyword ``'joint'`` computing the colormap jointly across all projectors of a given channel type, by `Daniel McCloy`_.

- New methods :meth:`mne.io.Raw.get_channel_types`, :meth:`mne.Epochs.get_channel_types`, :meth:`mne.Evoked.get_channel_types` by `Daniel McCloy`_.

- Deprecate ``mne.minimum_norm.point_spread_function`` and ``mne.minimum_norm.cross_talk_function`` by `Alex Gramfort`_
