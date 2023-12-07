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

In this version, we started adding type hints (also known as "type annotations") to select parts of the codebase.
This meta information will be used by development environments (IDEs) like VS Code and PyCharm automatically to provide
better assistance such as tab completion or error detection even before running your code.

So far, we've only added return type hints to :func:`mne.read_evokeds` and :func:`mne.io.read_raw`. Now your editors will know:
these functions return evoked and raw data, respectively. We are planning add type hints to more functions after careful
evaluation in the future.

You don't need to do anything to benefit from these changes – your editor will pick them up automatically and provide the
enhanced experience if it supports it!

Enhancements
~~~~~~~~~~~~
- Speed up export to .edf in :func:`mne.export.export_raw` by using ``edfio`` instead of ``EDFlib-Python`` (:gh:`12218` by :newcontrib:`Florian Hofer`)
- We added type hints for the return values of :func:`mne.read_evokeds` and :func:`mne.io.read_raw`. Development environments like VS Code or PyCharm will now provide more help when using these functions in your code. (:gh:`12250` by `Richard Höchenberger`_ and `Eric Larson`_)
- Add ``method="polyphase"`` to :meth:`mne.io.Raw.resample` and related functions to allow resampling using :func:`scipy.signal.upfirdn` (:gh:`12268` by `Eric Larson`_)
- The package build backend was switched from ``setuptools`` to ``hatchling``. This will only affect users who build and install MNE-Python from source. (:gh:`12269` by `Richard Höchenberger`_)

Bugs
~~~~
- Allow :func:`mne.viz.plot_compare_evokeds` to plot eyetracking channels, and improve error handling (:gh:`12190` by `Scott Huberty`_)
- Fix bug with accessing the last data sample using ``raw[:, -1]`` where an empty array was returned (:gh:`12248` by `Eric Larson`_)
- Remove incorrect type hints in :func:`mne.io.read_raw_neuralynx` (:gh:`12236` by `Richard Höchenberger`_)
- ``defusedxml`` is now an optional (rather than required) dependency and needed when reading EGI-MFF data, NEDF data, and BrainVision montages (:gh:`12264` by `Eric Larson`_)

API changes
~~~~~~~~~~~
- None yet
