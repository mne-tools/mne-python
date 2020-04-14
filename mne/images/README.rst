.. -*- mode: rst -*-


Documentation
=============

The images are used in `mne/viz/_brain` as icons in the toolbar of `_TimeViewer`.
It is necessary to build those images into a resource file for proper usage by
the application.

The resource configuration file `mne.qrc` describes the location of the resources
in the filesystem and also define aliases for their use in the code.

To generate automatically the resource file in `mne/viz/_brain`:

.. code-block:: bash

    pyrcc5 -o mne/viz/_brain/resources.py mne.qrc


Patching
========

The output file imports `PyQt5` globally which is not consistent with MNE core
structure so unit testing will fail. It is then strongly recommended to modify
it as follows:

- Refactor the Qt version checking into a `_check_version` function
- Use local import of `QtCore` in each function definition instead of globally
- Do not call `qInitResources()` from the resource file itself, this function is
  called externally

Note
====

The output file does not follow PEP8 guidelines so minor code formatting is
expected. 
