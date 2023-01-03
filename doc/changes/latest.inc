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
- Added ability to read stimulus durations from SNIRF files when using :func:`mne.io.read_raw_snirf` (:gh:`11397` by `Robert Luke`_)
- Add :meth:`mne.Info.save` to save an :class:`mne.Info` object to a fif file (:gh:`11401` by `Alex Rockhill`_)

Bugs
~~~~
- Fix bug where channel names were not properly sanitized in :func:`mne.write_evokeds` and related functions (:gh:`11399` by `Eric Larson`_)
- Fix bug where loading different combinations of volumes were in ``freeview`` when rotating and translating a volume to make a linear transformation array (LTA), different ``dst`` (destination) volumes were used in the LTA file, resulting in different affines for :func:`mne.read_lta` when the rotation and translations were the same (:gh:`11402` by `Alex Rockhill`_)

API changes
~~~~~~~~~~~
- None yet
