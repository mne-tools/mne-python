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

Current (1.4.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Add support for UCL/FIL OPM data using :func:`mne.io.read_raw_fil` (:gh:`11366` by :newcontrib:`George O'Neill` and `Robert Seymour`_)
- Added ability to read stimulus durations from SNIRF files when using :func:`mne.io.read_raw_snirf` (:gh:`11397` by `Robert Luke`_)
- Add :meth:`mne.Info.save` to save an :class:`mne.Info` object to a fif file (:gh:`11401` by `Alex Rockhill`_)
- Improved error message when downloads are corrupted for :func:`mne.datasets.sample.data_path` and related functions (:gh:`11407` by `Eric Larson`_)
- Add support for ``skip_by_annotation`` in :func:`mne.io.Raw.notch_filter` (:gh:`11388` by `Mainak Jas`_)
- Slightly adjusted the window title for :func:`mne.Epochs.plot` (:gh:`11419` by `Richard Höchenberger`_ and `Daniel McCloy`_)
- Add :func:`mne.count_events` to count unique event types in a given event array (:gh:`11430` by `Clemens Brunner`_)
- Add a video to :ref:`tut-freesurfer-mne` of a brain inflating from the pial surface to aid in understanding the inflated brain (:gh:`11440` by `Alex Rockhill`_)
- Add automatic projection of sEEG contact onto the inflated surface for :meth:`mne.viz.Brain.add_sensors` (:gh:`11436` by `Alex Rockhill`_)

Bugs
~~~~
- Fix bug where :func:`mne.preprocessing.regress_artifact` and :class:`mne.preprocessing.EOGRegression` incorrectly tracked ``picks`` (:gh:`11366` by `Eric Larson`_)
- Fix bug where channel names were not properly sanitized in :func:`mne.write_evokeds` and related functions (:gh:`11399` by `Eric Larson`_)
- Fix bug where splash screen would not always disappear (:gh:`11398` by `Eric Larson`_)
- Fix bug where having a different combination of volumes loaded into ``freeview`` caused different affines to be returned by :func:`mne.read_lta` for the same Linear Transform Array (LTA) (:gh:`11402` by `Alex Rockhill`_)
- Fix how :class:`mne.channels.DigMontage` is set when using :func:`mne.gui.locate_ieeg` so that :func:`mne.Info.get_montage` works and does not return ``None`` (:gh:`11421` by `Alex Rockhill`_)
- Fix :func:`mne.io.read_raw_edf` when reading EDF data with different sampling rates and a mix of data channels when using ``infer_types=True`` (:gh:`11427` by `Alex Gramfort`_)
- Fix how :class:`mne.channels.DigMontage` is set when using :func:`mne.preprocessing.ieeg.project_sensors_onto_brain` so that :func:`mne.Info.get_montage` works and does not return ``None`` (:gh:`11436` by `Alex Rockhill`_)
- Fix configuration folder discovery on Windows, which would fail in certain edge cases; and produce a helpful error message if discovery still fails (:gh:`11441` by `Richard Höchenberger`_)

API changes
~~~~~~~~~~~
- None yet
