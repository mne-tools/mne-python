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

Current (1.1.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Add support for ``overview_mode`` in :meth:`raw.plot() <mne.io.Raw.plot>` and related functions/methods (:gh:`10501` by `Eric Larson`_)

- The ``pick_channels`` method gained a ``verbose`` parameter, allowing e.g. to suppress messages about removed projectors (:gh:`10544` by `Richard Höchenberger`_)

- Add :meth:`mne.preprocessing.ICA.find_bads_muscle` to find muscle-related ICA components with an example, :ref:`ex-muscle-ica` (:gh:`10534` by `Alex Rockhill`)

Bugs
~~~~
- Fix bug in :func:`mne.io.read_raw_brainvision` when BrainVision data are acquired with the Brain Products "V-Amp" amplifier and disabled lowpass filter is marked with value ``0`` (:gh:`10517` by :newcontrib:`Alessandro Tonin`)

- Fix bug in :func:`mne.pick_types` and related methods where ``csd=True`` was not passed handled properly (:gh:`10470` by :newcontrib:`Matthias Dold`)

- Fix bug where plots produced using the ``'qt'`` / ``mne_qt_browser`` backend could not be added using :meth:`mne.Report.add_figure` (:gh:`10485` by `Eric Larson`_)

- Fix bug where ``theme`` was not handled properly in :meth:`mne.io.Raw.plot` (:gh:`10487`, :gh:`10500` by `Mathieu Scheltienne`_ and `Eric Larson`_)

- Rendering issues with recent MESA releases can be avoided by setting the new environment variable``MNE_3D_OPTION_MULTI_SAMPLES=1`` or using :func:`mne.viz.set_3d_options` (:gh:`10513` by `Eric Larson`_)

- Fix behavior for the ``pyvista`` 3D renderer's ``quiver3D`` function so that default arguments plot a glyph in ``arrow`` mode (:gh:`10493` by `Alex Rockhill`_)

- Retain epochs metadata when using :func:`mne.channels.combine_channels` (:gh:`10504` by `Clemens Brunner`_)

- Fix reading of fiducial locations in :func:`mne.io.read_raw_eeglab` (:gh:`10521` by `Alex Gramfort`_)

API and behavior changes
~~~~~~~~~~~~~~~~~~~~~~~~
- When creating BEM surfaces via :func:`mne.bem.make_watershed_bem` and :func:`mne.bem.make_flash_bem`, the ``copy`` parameter now defaults to ``True``. This means that instead of creating symbolic links inside the FreeSurfer subject's ``bem`` folder, we now create "actual" files. This should avoid troubles when sharing files across different operating systems and file systems (:gh:`10531` by `Richard Höchenberger`_)
