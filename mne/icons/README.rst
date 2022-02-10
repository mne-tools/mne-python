.. -*- mode: rst -*-


Documentation
=============

The icons are used in ``mne/viz/_brain/_brain.py`` for the toolbar.
It is necessary to compile those icons into a resource file for proper use by
the application.

The resource configuration file ``mne/icons/mne.qrc`` describes the location of
the resources in the filesystem and also defines aliases for their use in the code.

To automatically generate the resource file in ``mne/icons``:

.. code-block:: bash

    pyrcc5 -o mne/icons/resources.py mne/icons/mne.qrc

These Material design icons are provided by Google under the `Apache 2.0`_ license.


.. _Apache 2.0: https://github.com/google/material-design-icons/blob/master/LICENSE
