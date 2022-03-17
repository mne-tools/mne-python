.. include:: ../links.inc

.. _manual-install:

Manual installation
===================

MNE-Python requires Python version |min_python_version| or higher. If you've
never worked with Python before, skip ahead to the last paragraph of this page.
For users already familiar with Python:

- If you only need MNE-Python's computational functions, only hard dependencies
  will be included when running:

  .. code-block:: console

      $ pip install mne

- If you plan to use MNE-Python's functions that use HDF5-based I/O (e.g.,
  :func:`mne.io.read_raw_eeglab`, :meth:`mne.SourceMorph.save`, etc.),
  you should run:

  .. code-block:: console

      $ pip install mne[hdf5]

  This will pull in additional dependencies pymatreader_, h5io_, and h5py_.

- If you need MNE-Python's 3D rendering capabilities (e.g., plotting estimated
  source activity on a cortical surface) it is a good idea to install
  MNE-Python into its own virtual environment. To do this with
  `conda <anaconda_>`_:

  .. code-block:: console

      $ conda create --name=mne --channel=conda-forge mne
      $ #                   ↑↑↑                       ↑↑↑
      $ #             environment name            package name

  This will create a new ``conda`` environment called ``mne`` and install all
  dependencies into it. If you need to convert structural MRI scans into models
  of the scalp, inner/outer skull, and cortical surfaces you also need
  :doc:`FreeSurfer <freesurfer>`.

Python IDEs
===========

Most users find it convenient to write and run their code in an `Integrated
Development Environment`_ (IDE). Some popular choices for scientific
Python development are:

- `Spyder`_ is a free and open-source IDE developed by and for scientists who
  use Python. It is included by default in the ``base`` environment when you
  install Anaconda, and can be started from a terminal with the command
  ``spyder`` (or on Windows or macOS, launched from the Anaconda Navigator GUI).
  It can also be installed with `dedicated installers <https://www.spyder-ide.org/#section-download>`_.
  To avoid dependency conflicts with Spyder, you should install ``mne`` in a
  separate environment, like explained in the earlier sections. Then, set
  Spyder to use the ``mne`` environment as its default interpreter by opening
  Spyder and navigating to
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.
  There, paste the output of the following terminal commands::

      $ conda activate mne
      $ python -c "import sys; print(sys.executable)"

  It should be something like ``C:\Users\user\anaconda3\envs\mne\python.exe``
  (Windows) or ``/Users/user/opt/anaconda3/envs/mne/bin/python`` (macOS).

  If the Spyder console can not start because ``spyder-kernels`` is missing,
  install the required version in the ``mne`` environment with the following
  commands in the terminal::

      $ conda activate mne
      $ conda install spyder-kernels=HERE_EXACT_VERSION -c conda-forge

  Refer to the `spyder documentation <https://docs.spyder-ide.org/current/troubleshooting/common-illnesses.html#spyder-kernels-not-installed-incompatible>`_
  for more information about ``spyder-kernels`` and the version matching.

  If the Spyder graphic backend is not set to ``inline`` but to e.g. ``Qt5``,
  ``pyqt`` must be installed in the ``mne`` environment.

- `Visual Studio Code`_ (often shortened to "VS Code" or "vscode") is a
  development-focused text editor that supports many programming languages in
  addition to Python, includes an integrated terminal console, and has a rich
  ecosystem of packages to extend its capabilities. Installing
  `Microsoft's Python Extension
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__ is
  enough to get most Python users up and running. VS Code is free and
  open-source.
- `Atom`_ is a text editor similar to vscode, with a package ecosystem that
  includes a `Python IDE package <https://atom.io/packages/ide-python>`__ as
  well as `several <https://atom.io/packages/atom-terminal>`__
  `packages <https://atom.io/packages/atom-terminal-panel>`__
  `for <https://atom.io/packages/terminal-plus>`__
  `integrated <https://atom.io/packages/platformio-ide-terminal>`__
  `terminals <https://atom.io/packages/term3>`__. Atom is free and open-source.
- `SublimeText`_ is a general-purpose text editor that is fast and lightweight,
  and also has a rich package ecosystem. There is a package called `Terminus`_
  that provides an integrated terminal console, and a (confusingly named)
  package called "anaconda"
  (`found here <https://packagecontrol.io/packages/Anaconda>`__) that provides
  many Python-specific features. SublimeText is free (closed-source shareware).
- `PyCharm`_ is an IDE specifically for Python development that provides an
  all-in-one installation (no extension packages needed). PyCharm comes in a
  free "community" edition and a paid "professional" edition, and is
  closed-source.


For users unfamiliar with Python, the :ref:`standard-instructions` page has
detailed instructions for different
operating systems, and there are instructions for :ref:`install-python`
if you don't already have it. The :ref:`advanced_setup` page has additional
tips and tricks for special situations (servers, notebooks, CUDA, installing
the development version, etc). The :ref:`contributing` has additional
installation instructions for (future) contributors to MNE-Python (e.g, extra
dependencies for running our tests and building our docs).

.. include:: manual_install_python.inc

.. include:: manual_install_mne.inc

.. include:: manual_install_advanced.inc
