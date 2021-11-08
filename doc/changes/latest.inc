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

Bugs
~~~~
- Fix default of :func:`mne.io.Raw.plot` to be ``use_opengl=None``, which will act like False unless ``MNE_BROWSE_USE_OPENGL=true`` is set in the user configuration (:gh:`9957` by `Eric Larson`_)

- Fix bug with :class:`mne.Report` where figures were saved with ``bbox_inches='tight'``, which led to inconsistent sizes in sliders (:gh:`9966` by `Eric Larson`_)

- Fix bug in :func:`mne.make_forward_solution` where sensor-sphere geometry check was incorrect (:gh:`9968` by `Eric Larson`_)

- Add argument ``overwrite`` to :func:`mne.export.export_raw` and :func:`mne.export.export_epochs` (:gh:`9975` by `Mathieu Scheltienne`_)


API changes
~~~~~~~~~~~
- Nothing yet
