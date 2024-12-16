.. _manual-install:
.. _standard-instructions:

Install via :code:`pip` or :code:`conda`
========================================

.. hint::
   If you're unfamiliar with Python, we recommend using our :ref:`installers`
   instead.

MNE-Python requires Python version |min_python_version| or higher. If you
need help installing Python, please refer to our :ref:`install-python` guide.

Installing MNE-Python with all dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you use Anaconda, we suggest installing MNE-Python into its own ``conda`` environment.

First, please ensure you're using a recent version of ``conda``. Run in your terminal:

.. code-block:: console

   $ conda update --name=base conda  # update conda
   $ conda --version

The installed ``conda`` version should be ``23.10.0`` or newer.

Now, you can install MNE-Python:

.. code-block:: console

   $ conda create --channel=conda-forge --strict-channel-priority --name=mne mne

This will create a new ``conda`` environment called ``mne`` (you can adjust
this by passing a different name via ``--name``) and install all
dependencies into it.

If you need to convert structural MRI scans into models
of the scalp, inner/outer skull, and cortical surfaces, you will also need
to install :doc:`FreeSurfer <freesurfer>`.

Installing MNE-Python with core dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you only need MNE-Python's core functionality, which includes 2D plotting
(but does not support 3D visualization), install via :code:`pip`:

.. code-block:: console

   $ pip install mne

or via :code:`conda`:

.. code-block:: console

   $ conda create --channel=conda-forge --strict-channel-priority --name=mne mne-base

This will create a new ``conda`` environment called ``mne`` (you can adjust
this by passing a different name via ``--name``).

This minimal installation requires only a few dependencies. If you need additional
functionality later on, you can install individual packages as needed.

Installing MNE-Python with HDF5 support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you plan to use MNE-Python's functions that require
`HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ I/O (this
includes :func:`mne.io.read_raw_eeglab`, :meth:`mne.SourceMorph.save`, and
others), you should run via :code:`pip`:

.. code-block:: console

   $ pip install "mne[hdf5]"

or via :code:`conda`:

.. code-block:: console

   $ conda create --override-channels --channel=conda-forge --name=mne mne-base h5io h5py pymatreader

This will create a new ``conda`` environment called ``mne`` (you can adjust
this by passing a different name via ``--name``).

If you have already installed MNE-Python with core dependencies (e.g. via ``pip install mne``),
you can install these two packages to unlock HDF5 support:

.. code-block:: console

   $ pip install h5io pymatreader

Installing MNE-Python for other scenarios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :ref:`advanced_setup` page has additional
tips and tricks for special situations (servers, notebooks, CUDA, installing
the development version, etc). The :ref:`contributing` has additional
installation instructions for (future) contributors to MNE-Python (e.g, extra
dependencies for running our tests and building our documentation).
