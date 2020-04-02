.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Changelog" for new features
   - "Bug" for bug fixes
   - "API" for backward-incompatible changes

.. _current:

Current (0.21.dev0)
-------------------

Changelog
~~~~~~~~~

- Add support for signals in mV for :func:`mne.io.read_raw_brainvision` by `Clemens Brunner`_

Bug
~~~

- Fix bug with :func:`mne.preprocessing.nirs.scalp_coupling_index` where filter transition was incorrectly assigned by `Robert Luke`_

- Fix bug with :func:`mne.make_forward_dipole` where :func:`mne.write_forward_solution` could not be used by `Eric Larson`_

API
~~~

- Nothing yet
