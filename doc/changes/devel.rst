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

Version 1.6.dev0 (development)
------------------------------

Enhancements
~~~~~~~~~~~~
- Improve tests for saving splits with `Epochs` (:gh:`11884` by `Dmitrii Altukhov`_) 
- Added functionality for linking interactive figures together, such that changing one figure will affect another, see :ref:`tut-ui-events` and :mod:`mne.viz.ui_events` (:gh:`11685` by `Marijn van Vliet`_)
- HTML anchors for :class:`mne.Report` now reflect the ``section-title`` of the report items rather than using a global incrementor ``global-N`` (:gh:`11890` by `Eric Larson`_)

Bugs
~~~~
- Fix bugs with saving splits for :class:`~mne.Epochs` (:gh:`11876` by `Dmitrii Altukhov`_) 

API changes
~~~~~~~~~~~
- None yet
