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


Enhancements
~~~~~~~~~~~~
.. - Add something cool (:gh:`9192` **by new contributor** |New Contributor|_)

- Add show local maxima toggling button to :func:`mne.gui.locate_ieeg` (:gh:`9952` by `Alex Rockhill`_)

- Improve docstring of :class:`mne.Info` and add attributes that were not covered (:gh:`9922` by `Mathieu Scheltienne`_)

- Add an alternate, manual procedure for aligning a CT to an MR procedure to :ref:`tut-ieeg-localize` (:gh:`9978` by `Alex Rockhill`_)

- Improve docstring of export functions :func:`mne.export.export_raw`, :func:`mne.export.export_epochs`, :func:`mne.export.export_evokeds`, :func:`mne.export.export_evokeds_mff` and issue a warning when there are unapplied projectors (:gh:`9994` by `Mathieu Scheltienne`_)

Bugs
~~~~
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

API changes
~~~~~~~~~~~
- Nothing yet
