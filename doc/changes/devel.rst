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

Version 1.7.dev0 (development)
------------------------------

Enhancements
~~~~~~~~~~~~
- :meth:`mne.Evoked.apply_function` can now also work on full data array, instead of just channel wise (analogous to :meth:`mne.io.Raw.apply_function` and :meth:`mne.Epochs.apply_function`). (:gh:`12206` by `Dominik Welke`_)
- Custom functions applied via :meth:`mne.io.Raw.apply_function`, :meth:`mne.Epochs.apply_function` or :meth:`mne.Evoked.apply_function` can now use ``ch_idx`` or ``ch_name`` to get access to the currently processed channel (during channel wise processing). (:gh:`12206` by `Dominik Welke`_)

Bugs
~~~~
- Fix bug in :meth:`mne.Epochs.apply_function` where data was handed down incorrectly in parallel processing (:gh:`12206` by `Dominik Welke`_)

API changes
~~~~~~~~~~~
- Optional input argument ``channel_wise`` added to :meth:`mne.Evoked.apply_function` (analogous to :meth:`mne.io.Raw.apply_function` and :meth:`mne.Epochs.apply_function`). (:gh:`12206` by `Dominik Welke`_)
