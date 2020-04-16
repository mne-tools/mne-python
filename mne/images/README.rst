.. -*- mode: rst -*-


Documentation
=============

The images are used in ``mne/viz/_brain`` as icons in the toolbar of ``_TimeViewer``.
It is necessary to compile those images into a resource file for proper use by
the application.

The resource configuration file ``mne.qrc`` describes the location of the resources
in the filesystem and also defines aliases for their use in the code.

To automatically generate the resource file in ``mne/viz/_brain``:

.. code-block:: bash

    pyrcc5 -o mne/viz/_brain/resources.py mne.qrc
