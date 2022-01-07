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

Current (1.0.dev0)
-------------------

Enhancements
~~~~~~~~~~~~

- Speed up :func:`mne.preprocessing.annotate_muscle_zscore`, :func:`mne.preprocessing.annotate_movement`, and :func:`mne.preprocessing.annotate_nan` through better annotation creation (:gh:`10089` by :newcontrib:`Senwen Deng`)

- Fix some unused variables in time_frequency_erds.py example (:gh:`10076` by :newcontrib:`Jan Zerfowski`)

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

- Reduce the time it takes to generate a :class:`mne.io.Raw`, :class:`~mne.Epochs`, or :class:`~mne.preprocessing.ICA` figure if a ``scalings`` parameter is provided (:gh:`10109` by `Richard Höchenberger`_ and `Eric Larson`_)

- :meth:`mne.Report.add_raw` gained a new ``scalings`` parameter to provide custom data scalings for the butterfly plots (:gh:`10109` by `Richard Höchenberger`_)

- The ``butterfly`` parameter of :meth:`mne.Report.add_raw` now also accepts numbers to specify how many segments to plot (:gh:`10115` by `Richard Höchenberger`_)

- The ``psd`` parameter of :meth:`mne.Report.add_epochs` now also accepts numbers to specify the signal duration used for PSD calculation (:gh:`10119` by `Richard Höchenberger`_)

- Drastically speed up butterfly plot generation in :meth:`mne.Report.add_raw`. We now don't plot annotations anymore; however, we feel that the speed improvements justify this change, also considering the annotations were of limited use in the displayed one-second time slices anyway (:gh:`10114`, :gh:`10116` by `Richard Höchenberger`_)

- In :class:`mne.Report`, limit the width of automatically generated figures to a maximum of 850 pixels (450 pixels for :class:`mne.SourceEstimate` plots), and the resolution to 100 DPI to reduce file size, memory consumption, and – in some cases like :meth:`mne.Report.add_stc` – processing time (:gh:`10126`, :gh:`10129`, :gh:`10135`, :gh:`10142` by `Richard Höchenberger`_)

- Annotations from a :class:`~mne.io.Raw` object are now preserved by the :class:`~mne.Epochs` constructor (:gh:`9969` by `Adam Li`_)

- :class:`~mne.Epochs` metadata tables are now included in :class:`mne.Report` (:gh:`10166` by `Richard Höchenberger`_)

- :func:`mne.get_head_surf` and :func:`mne.dig_mri_distances` gained a new parameter, ``on_defects``, controlling how to handle surfaces with topological defects (:gh:`10175` by `Richard Höchenberger`_)

- :meth:`mne.Report.add_epochs` gained a new parameter, ``drop_log_ignore``, to control which drop reasons to omit when creating the drop log plot (:gh:`10182` by `Richard Höchenberger`_)

Bugs
~~~~

- Teach :func:`mne.io.read_raw_bti` to use its ``eog_ch`` parameter (:gh:`10093` by :newcontrib:`Adina Wagner`)

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

- Fix annotation cropping and I/O roundtrip when there is no measurement date available (:gh:`10040` by `Mathieu Scheltienne`_ and `Alex Gramfort`_).

- :func:`~mne.sys_info` output now contains the installed version of ``pooch``, too; this output had been accidentally removed previously (:gh:`10047` by `Richard Höchenberger`_)

- Fix automatic channel type detection from channel labels in :func:`mne.io.read_raw_edf` and :func:`mne.io.read_raw_bdf` (and disable this functionality from :func:`mne.io.read_raw_gdf`) (:gh:`10058` by `Clemens Brunner`_)

- Fix :func:`~mne.stats.permutation_cluster_1samp_test` to properly handle 2-dimensional data in combination with TFCE (:gh:`10073` by `Richard Höchenberger`_)

- Fix channel grouping error when using "butterfly mode" with :meth:`mne.io.Raw.plot` (:gh:`10087` by `Daniel McCloy`_)

- Fix inconsistent behavior of ``mne.preprocessing.annotate_*`` functions by making them all return :class:`mne.Annotations` objects with the ``orig_time`` attribute set to ``raw.info["meas_time"]`` (:gh:`10067` and :gh:`10118` by `Stefan Appelhoff`_, `Eric Larson`_, and `Alex Gramfort`_)

- Fix bug that appears during automatic calculation of the colormap of `mne.viz.Brain` when data values of ``fmin`` and ``fmax`` are too close (:gh:`10074` by `Guillaume Favelier`_)

- We now display a scrollbar in the tags dropdown of a `~mne.Report` if many tags have been added, granting access to all tags instead of "hiding" them below the bottom of the page (:gh:`10082` by `Richard Höchenberger`_)

- Creating :class:`mne.Epochs` now provides clearer logging (less ambiguous, no duplicates) when the ``preload`` and/or ``metadata`` parameters are set (:gh:`10112` by `Stefan Appelhoff`_)

- Fix bug with :class:`mne.Epochs` where save-load round-trip with FIF would cause :meth:`mne.Epochs.apply_baseline` to no longer work (:gh:`10177` by `Eric Larson`_)

- Fix functions by adding missing ``overwrite`` parameters: :func:`mne.write_events`, :func:`mne.write_cov`, :func:`mne.write_evokeds`, :meth:`mne.SourceEstimate.save`, :func:`mne.minimum_norm.write_inverse_operator`, :func:`mne.write_proj`, and related methods (:gh:`10127` by `Eric Larson`_)

- Fix bug with :func:`mne.transforms.compute_volume_registration` and :func:`mne.compute_source_morph` (volumetric) where the smoothing factors were not scaled based on ``zooms`` (:gh:`10132` by `Eric Larson`_)

- Remove repeated logging output when overwriting an existing `~mne.io.Raw` file (:gh:`10095` by `Richard Höchenberger`_ and `Stefan Appelhoff`_)

- In the plots generated by :meth:`mne.Report.add_stc`, we now only add 5 labels to the color bar to reduce the chance of overlap, which could previously cause the labels to become unreadable (:gh:`10135` by `Richard Höchenberger`_)

- :meth:`mne.Report.add_trans` now allows you to add sensor alignment plots for head surfaces that have topological defects (:gh:`10175` by `Richard Höchenberger`_)

- :meth:`mne.Report.add_trans` now also works if no digitization points are present in the data (:gh:`10176` by `Jeff Stout`_)

API changes
~~~~~~~~~~~
- ``mne.Info.pick_channels`` has been deprecated. Use ``inst.pick_channels`` to pick channels from :class:`~mne.io.Raw`, :class:`~mne.Epochs`, and :class:`~mne.Evoked`. Use :func:`mne.pick_info` to pick channels from :class:`mne.Info` (:gh:`10039` by `Mathieu Scheltienne`_)

- Argument ``event_list`` has been deprecated in favor of ``events`` in :func:`mne.write_events` (:gh:`10056` by `Mathieu Scheltienne`_)

- ``mne.preprocessing.annotate_flat`` has been deprecated in favor of :func`mne.preprocessing.annotate_amplitude`, that covers both minimum and maximum peak-to-peak variation. (:gh:`10143` by `Mathieu Scheltienne`_)
