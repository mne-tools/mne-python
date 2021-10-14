.. include:: ../links.inc

Quick start
===========

MNE-Python requires Python version |min_python_version| or higher. If you've
never worked with Python before, skip ahead to the last paragraph of this page.
For users already familiar with Python:

- If you only need MNE-Python's computational functions (not plotting), the
  only hard dependencies are `NumPy`_ and `SciPy`_ which will be included
  automatically when running:

  .. code-block:: console

      $ pip install mne

- If you plan to use MNE-Python's functions that download example datasets,
  you should run:

  .. code-block:: console

      $ pip install mne[data]

  This will pull in additional dependencies `Pooch`_ and `tqdm`_.

- If you need 2D plotting capabilities (i.e., most EEG/ERP or other
  sensor-level analyses), you should also install `Matplotlib`_:

  .. code-block:: console

      $ pip install mne matplotlib

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

For users unfamiliar with Python, the :ref:`standard_instructions` page has
detailed instructions for different
operating systems, and there are instructions for :ref:`install-python`
if you don't already have it. The :ref:`advanced_setup` page has additional
tips and tricks for special situations (servers, notebooks, CUDA, installing
the development version, etc). The :ref:`contributing` has additional
installation instructions for (future) contributors to MNE-Python (e.g, extra
dependencies for running our tests and building our docs).

.. toctree::
    :hidden:

    pre_install
    install_python
    mne_python
    updating
    freesurfer
    advanced
