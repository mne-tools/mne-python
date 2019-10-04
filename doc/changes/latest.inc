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

- Add support for computing resolution matrix to get point spread functions (PSF) and cross-talk functions (CTF) in :func:`mne.minimum_norm.make_resolution_matrix`, :func:`mne.minimum_norm.get_cross_talk`, :func:`mne.minimum_norm.get_point_spread` by `Olaf Hauk`_

- Add keyboard functionality to interactive colorbar plotting TFRs by `Stefan Repplinger`_

- Add the ability to ``return_event_id`` in :func:`mne.read_events` for use with MNE-C produced ``-annot.fif`` files, by `Eric Larson`_

- Add option to :func:`mne.connectivity.spectral_connectivity` to compute corrected imaginary PLV by `Adonay Nunes`_

- Add reader for NIRx data in :func:`mne.io.read_raw_nirx` by `Robert Luke`_

- For KIT systems without built-in layout, :func:`mne.channels.find_layout` now falls back on an automatically generated layout, by `Christian Brodbeck`_

Bug
~~~

- Fix TAL channel parsing (annotations) for EDF-D files by `Clemens Brunner`_

- Fix handling of repeated events in :class:`mne.Epochs` by `Fahimeh Mamashli`_ and `Alex Gramfort`_

- Fix reading of cardinals in .htps files (identifier are int not strings) by `Alex Gramfort`_

- Fix IO of TFRs when event_id contain a / in one of the keys by `Alex Gramfort`_

- Fix ``info['sfreq']`` when decimating in :func:`mne.time_frequency.tfr_multitaper` and :func:`mne.time_frequency.tfr_morlet` and make sure an error is raised when exceed Nyquist frequency by `Adonay Nunes`_

API
~~~

- The APIs of :meth:`mne.io.Raw.plot_projs_topomap`, :meth:`mne.Epochs.plot_projs_topomap` and :meth:`mne.Evoked.plot_projs_topomap` are now more similar to :func:`mne.viz.plot_projs_topomap` by `Daniel McCloy`_.

- :func:`mne.viz.plot_projs_topomap` now accepts both :class:`~mne.channels.Layout` and :class:`~mne.Info` (previously one or the other), and will ignore :class:`~mne.Info` if :class:`~mne.channels.Layout` is provided `Daniel McCloy`_.

- New methods :meth:`mne.io.Raw.get_channel_types`, :meth:`mne.Epochs.get_channel_types`, :meth:`mne.Evoked.get_channel_types` by `Daniel McCloy`_.

- Deprecate ``mne.minimum_norm.point_spread_function`` and ``mne.minimum_norm.cross_talk_function`` by `Alex Gramfort`_
