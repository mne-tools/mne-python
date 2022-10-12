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
- None yet

Bugs
~~~~
- None yet

API changes
~~~~~~~~~~~
- Add the argument ``max_iter`` to :func:`~mne.time_frequency.psd_array_multitaper` and to :func:`~mne.time_frequency.csd_array_multitaper` to increase the maximum number of iterations to reach convergence when using ``adaptive=True`` (:gh:`11237` by `Mathieu Scheltienne`_)

Authors
~~~~~~~
* Mathieu Scheltienne
