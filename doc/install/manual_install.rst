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

The dependency stack is large and may take a long time (several tens of
minutes) to resolve on some systems via the default ``conda`` solver. We
therefore highly recommend using the new `libmamba <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`__
solver instead, which is **much** faster. To permanently change to this solver,
you can set ``CONDA_SOLVER=libmamba`` in your environment or run
``conda config --set solver libmamba``. Below we just use ``--solver`` in each command.

Run in your terminal:

.. code-block:: console

    $ conda install --channel=conda-forge --name=base conda-libmamba-solver
    $ conda create --solver=libmamba --override-channels --channel=conda-forge --name=mne mne

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

   $ conda create --override-channels --channel=conda-forge --name=mne mne-base

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

   $ pip install mne[hdf5]

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

Python IDEs
===========

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
  use Python. It is included by default in the ``base`` environment when you
  install Anaconda, and can be started from a terminal with the command
  ``spyder`` (or on Windows or macOS, launched from the Anaconda Navigator GUI).
  It can also be installed with `dedicated installers <https://www.spyder-ide.org/#section-download>`_.
  To avoid dependency conflicts with Spyder, you should install ``mne`` in a
  separate environment, as explained in previous sections. Then, instruct
  Spyder to use the ``mne`` environment as its default interpreter by opening
  Spyder and navigating to
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.
  There, paste the output of the following terminal commands:

  .. code-block:: console

      $ conda activate mne
      $ python -c "import sys; print(sys.executable)"

  It should be something like ``C:\Users\user\anaconda3\envs\mne\python.exe``
  (Windows) or ``/Users/user/opt/anaconda3/envs/mne/bin/python`` (macOS).

  If the Spyder console can not start because ``spyder-kernels`` is missing,
  install the required version in the ``mne`` environment with the following
  commands in the terminal, where you replace ``...`` with the exact version of
  ``spyder-kernels`` that Spyder tells you it requires.

  .. code-block:: console

      $ conda activate mne
      $ conda install --override-channels --channel=conda-forge spyder-kernels=...

  Refer to the `Spyder documentation <https://docs.spyder-ide.org/current/troubleshooting/common-illnesses.html#spyder-kernels-not-installed-incompatible>`_
  for more information about ``spyder-kernels`` and the version matching.

  If the Spyder graphic backend is not set to ``inline`` but to e.g. ``Qt5``,
  ``PyQt5`` (``pip``) or ``pyqt`` (``conda``) must be installed in the ``mne``
  environment.

- `PyCharm`_ is an IDE specifically for Python development that provides an
  all-in-one solution (no extension packages needed). PyCharm comes in a
  free and open-source Community edition as well as a paid Professional edition.
