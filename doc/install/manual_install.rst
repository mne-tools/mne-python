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

.. _ide_setup:

Python IDE integration
======================

Most users find it convenient to write and run their code in an `Integrated
Development Environment`_ (IDE). Some popular choices for scientific
Python development are:

- `Visual Studio Code`_ (often shortened to "VS Code" or "vscode") is a
  development-focused text editor that supports many programming languages in
  addition to Python, includes an integrated terminal console, and has a rich
  extension ecosystem. Installing
  `Microsoft's Python Extension
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__ is
  enough to get most Python users up and running. VS Code is free and
  open-source.

- `Spyder`_ is a free and open-source IDE developed by and for scientists who
  use Python. It can be installed via a
  `standalone Spyder installer <https://docs.spyder-ide.org/current/installation.html#downloading-and-installing>`__.
  To avoid dependency conflicts with Spyder, you should install ``mne`` in a
  separate environment, as explained in previous sections or using our dedicated
  installer. Then, instruct
  Spyder to use the MNE-Python interpreter by opening
  Spyder and `navigating to <https://docs.spyder-ide.org/current/faq.html#using-existing-environment>`__
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.

- `PyCharm`_ is an IDE specifically for Python development that provides an
  all-in-one solution (no extension packages needed). PyCharm comes in a
  free and open-source Community edition as well as a paid Professional edition.

For these IDEs, you'll need to provide the path to the Python interpreter you want it
to use. If you're using the MNE-Python installers, on Linux and macOS opening the
**Prompt** will display several lines of information, including a line that will read
something like:

.. code-block:: output

   Using Python: /some/directory/mne-python_1.7.1_0/bin/python

Altertatively (or on Windows), you can find that path by opening the Python interpreter
you want to use (e.g., the one from the MNE-Python installer, or a ``conda`` environment
that you have activated) and running::

   >>> import sys
   >>> print(sys.executable) # doctest:+SKIP

This should print something like
``C:\Program Files\MNE-Python\1.7.0_0\bin\python.exe`` (Windows) or
``/Users/user/Applications/MNE-Python/1.7.0_0/.mne-python/bin/python`` (macOS).

For Spyder, if the console cannot start because ``spyder-kernels`` is missing,
install the required version in the conda environment. For example, with the
environment you want to use activated, run ``conda install spyder-kernels``.
