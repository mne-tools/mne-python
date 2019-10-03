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
- Add command ``mne setup_source_space`` to quickly set up bilateral hemisphere surface-based source space with subsampling by `Victor Ferat`_.

- Add function :func:`mne.make_fixed_length_epochs` to segment raw into fixed length epochs by `Mohammad Daneshzand`_

- Add support for passing a destination source space ``src_to`` in :func:`mne.compute_source_morph` to ensure morphing for multiple subjects results in consistent STCs, by `Eric Larson`_

- Add support for plotting fNIRS channels in :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add keyboard functionality to interactive colorbar plotting TFRs by `Stefan Repplinger`_

- Add the ability to ``return_event_id`` in :func:`mne.read_events` for use with MNE-C produced ``-annot.fif`` files, by `Eric Larson`_

- Add option to :func:`mne.connectivity.spectral_connectivity` to compute corrected imaginary PLV by `Adonay Nunes`_

- Add reader for NIRx data in :func:`mne.io.read_raw_nirx` by `Robert Luke`_

Bug
~~~

- Fix TAL channel parsing (annotations) for EDF-D files by `Clemens Brunner`_

- Fix handling of repeated events in :class:`mne.Epochs` by `Fahimeh Mamashli`_ and `Alex Gramfort`_

- Fix IO of TFRs when event_id contain a / in one of the keys by `Alex Gramfort`_

API
~~~
