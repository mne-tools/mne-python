.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: changes from first-time contributors should be added to the TOP of
   the relevant section (Enhancements / Bugs / API changes), and should look
   like this (where xxxx is the pull request number):
   - description of enhancement/bugfix/API change (:gh:`xxxx` **by new contributor** |Firstname Lastname|_)

.. _current:

Current (1.0.dev0)
-------------------

.. |New Contributor| replace:: **New Contributor**

.. |Jan Zerfowski| replace:: **Jan Zerfowski**
.. |Senwen Deng| replace:: **Senwen Deng**

Enhancements
~~~~~~~~~~~~
.. - Add something cool (:gh:`9192` **by new contributor** |New Contributor|_)

- Speed up :func:`mne.preprocessing.annotate_muscle_zscore`, :func:`mne.preprocessing.annotate_movement`, and :func:`mne.preprocessing.annotate_nan` through better annotation creation (:gh:`10089` **by new contributor** |Senwen Deng|_)

- Fix some unused variables in time_frequency_erds.py example (:gh:`10076` **by new contributor** |Jan Zerfowski|_)

- Add show local maxima toggling button to :func:`mne.gui.locate_ieeg` (:gh:`9952` by `Alex Rockhill`_)

- Improve docstring of :class:`mne.Info` and add attributes that were not covered (:gh:`9922` by `Mathieu Scheltienne`_)

- Add an alternate, manual procedure for aligning a CT to an MR procedure to :ref:`tut-ieeg-localize` (:gh:`9978` by `Alex Rockhill`_)

- Improve docstring of export functions :func:`mne.export.export_raw`, :func:`mne.export.export_epochs`, :func:`mne.export.export_evokeds`, :func:`mne.export.export_evokeds_mff` and issue a warning when there are unapplied projectors (:gh:`9994` by `Mathieu Scheltienne`_)

- Add support for reading haemoglobin fNIRS data to :func:`mne.io.read_raw_snirf` (:gh:`9929` by `Robert Luke`_)

- Add ``fill_hole_size`` keyword argument to :func:`mne.viz.Brain.add_volume_labels` to close holes in the mesh (:gh:`10024` by `Alex Rockhill`_)

- Changed :class:`mne.Epochs` and :class:`mne.Evoked` to have a more concise ``__repr__`` to improve interactive MNE usage in Python Interactive Console, IDEs, and debuggers when many events are handled. (:gh:`10042` by `Jan Sosulski`_)

- Improve docstring of ``events`` arguments and cross-referencing to :term:`events` (:gh:`10056` by `Mathieu Scheltienne`_)

- Speed up repeated surface-smoothing operation (e.g., in repeated calls to :meth:`stc.plot() <mne.SourceEstimate.plot>`) (:gh:`10077` by `Eric Larson`_)

- Add ``verbose`` parameter to :func:`mne.io.Raw.load_bad_channels` and log information on how bad channels are updated (:gh:`10102` by `Stefan Appelhoff`_)

- Add ``infer_type`` argument to :func:`mne.io.read_raw_edf` and :func:`mne.io.read_raw_bdf` to automatically infer channel types from channel labels (:gh:`10058` by `Clemens Brunner`_)

Bugs
~~~~

.. |Adina Wagner| replace:: **Adina Wagner**

- Teach :func:`mne.io.read_raw_bti` to use its ``eog_ch`` parameter (:gh:`10093` **by new contributor** |Adina Wagner|_)

- Fix default of :func:`mne.io.Raw.plot` to be ``use_opengl=None``, which will act like False unless ``MNE_BROWSER_USE_OPENGL=true`` is set in the user configuration (:gh:`9957` by `Eric Larson`_)

- Fix bug with :class:`mne.Report` where figures were saved with ``bbox_inches='tight'``, which led to inconsistent sizes in sliders (:gh:`9966` by `Eric Larson`_)

- When opening a saved report and saving it to a different filename again, don't change ``Report.fname`` to avoid a regression when using :func:`~mne.open_report` as a context manager (:gh:`9998` by `Marijn van Vliet`_)

- Fix bug in :func:`mne.make_forward_solution` where sensor-sphere geometry check was incorrect (:gh:`9968` by `Eric Larson`_)

- Use single char alphanumeric suffix when renaming long channel names (over 15-characters) when writing to FIF format. (:gh:`10002` by `Luke Bloy`_)

- Add argument ``overwrite`` to :func:`mne.export.export_raw`, :func:`mne.export.export_epochs`, :func:`mne.export.export_evokeds` and :func:`mne.export.export_evokeds_mff` (:gh:`9975` by `Mathieu Scheltienne`_)

- :func:`mne.gui.coregistration` and the ``mne coreg`` command didn't respect the ``inspect`` parameter (:gh:`9972` by `Richard Höchenberger`_)

- Fix incorrect projection of source space onto white matter surface instead of pial in :ref:`tut-working-with-ecog` (:gh:`9980` by `Alex Rockhill`_)

- Fix channel type support when reading from EEGLAB ``.set`` format with :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` (:gh:`9990` by `Mathieu Scheltienne`_)

- Fix suboptimal alignment using :func:`mne.transforms.compute_volume_registration` (:gh:`9991` by `Alex Rockhill`_)

- Only warn if header is missing in BrainVision files instead of raising an error (:gh:`10001` by `Clemens Brunner`_)

- Add argument ``overwrite`` to `mne.preprocessing.ICA.save` to check for existing file (:gh:`10004` by `Mathieu Scheltienne`_)

- :class:`mne.Report` now raises an exception if invalid tags were passed (:gh:`9970` by `Richard Höchenberger`_)

- Fix bug in :func:`mne.get_montage_volume_labels` that set the maximum number of voxels to be included too low causing unwanted capping of the included voxel labels (:gh:`10021` by `Alex Rockhill`_)

- :func:`~mne.sys_info` output now contains the installed version of ``pooch``, too; this output had been accidentally removed previously (:gh:`10047` by `Richard Höchenberger`_)

- Fix automatic channel type detection from channel labels in :func:`mne.io.read_raw_edf` and :func:`mne.io.read_raw_bdf` (and disable this functionality from :func:`mne.io.read_raw_gdf`) (:gh:`10058` by `Clemens Brunner`_)

- Fix :func:`~mne.stats.permutation_cluster_1samp_test` to properly handle 2-dimensional data in combination with TFCE (:gh:`10073` by `Richard Höchenberger`_)

- Fix channel grouping error when using "butterfly mode" with :meth:`mne.io.Raw.plot` (:gh:`10087` by `Daniel McCloy`_)

- Fix bug that appears during automatic calculation of the colormap of `mne.viz.Brain` when data values of ``fmin`` and ``fmax`` are too close (:gh:`10074` by `Guillaume Favelier`_)

- We now display a scrollbar in the tags dropdown of a `~mne.Report` if many tags have been added, granting access to all tags instead of "hiding" them below the bottom of the page (:gh:`10082` by `Richard Höchenberger`_)

- Remove repeated logging output when overwriting an existing `~mne.io.Raw` file (:gh:`10095` by `Richard Höchenberger`_ and `Stefan Appelhoff`_)

API changes
~~~~~~~~~~~
- ``mne.Info.pick_channels`` has been deprecated. Use ``inst.pick_channels`` to pick channels from Raw|Epochs|Evoked. Use :func:`mne.pick_info` to pick channels from :class:`mne.Info` (:gh:`10039` by `Mathieu Scheltienne`_)

- Argument ``event_list`` has been deprecated in favor of ``events`` in :func:`mne.write_events` (:gh:`10056` by `Mathieu Scheltienne`_)
