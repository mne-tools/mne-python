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

Current (1.3.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Mixed, cortical + discrete source spaces with fixed orientations are now allowed. (:gh:`11241` by `Jevri Hanna`_)

Bugs
~~~~
- Fix bug in documentation of :func:`mne.channels.make_dig_montage` (:gh:`11235` by :newcontrib:`Daniel Hasegan`)
- Add support for bad channel handling in :func:`mne.set_bipolar_reference` (:gh:`11245` by `Moritz Gerster`_, :newcontrib:`Dinara Issagaliyeva`, :newcontrib:`Jennifer Behnke`, :newcontrib:`Hakimeh Aslsardroud`, and :newcontrib:`Pavel Navratil`)
- Fix bug where ``ica.reject_`` was not saved to disk, and the ``ica.reject_`` property was not inherited from ``Epochs`` when doing ``ICA.fit(epochs)`` (:gh:`11244` by `Eric Larson`_)

API changes
~~~~~~~~~~~
- None yet
