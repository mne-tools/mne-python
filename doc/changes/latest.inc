.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: changes from first-time contributors should be added to the TOP of
   the relevant section (Enhancements / Bugs / API changes), and should look
   like this (where xxxx is the pull request number):

       - description of enhancement/bugfix/API change (:gh:`xxxx` by
         :newcontrib:`Firstname Lastname`)

   Also add a corresponding entry for yourself in doc/changes/names.inc

.. _current:

Current (1.2.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- EEGLAB files (saved as MAT versions less than v7.3) can now be imported with :func:`mne.io.read_raw_eeglab` without the optional dependency ``pymatreader`` (:gh:`11006` by `Clemens Brunner`_)
- Add :func:`mne.time_frequency.csd_tfr` to compute cross-spectral density from :class:`mne.time_frequency.EpochsTFR` (:gh:`10986` by `Alex Rockhill`_)
- :meth:`mne.Epochs.plot_psd_topomap` now suppresses redundant colorbars when ``vlim='joint'`` (:gh:`11051` by `Daniel McCloy`_)
- Add ``starting_affine`` keyword argument to :func:`mne.transforms.compute_volume_registration` to initialize an alignment with an affine (:gh:`11020` by `Alex Rockhill`_)

Bugs
~~~~
- Fix bug in :func:`mne.decoding.TimeFrequency` that prevented cloning if constructor arguments were modified (:gh:`11004` by :newcontrib:`Daniel Carlström Schad`)
- Document ``height`` and ``weight`` keys of  ``subject_info`` entry in :class:`mne.Info` (:gh:`11019` by :newcontrib:`Sena Er`)
- Fixed bug in :func:`mne.viz.plot_filter` when plotting filters created using ``output='ba'`` mode with ``compensation`` turned on. (by `Marian Dovgialo`_)

API changes
~~~~~~~~~~~
- None yet
