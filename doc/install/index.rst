.. include:: ../links.inc

Quick start
===========

MNE-Python requires Python version |min_python_version| or higher. If you've
never worked with Python before, skip ahead to the last paragraph of this page.
For users already familiar with Python:

- If you only need 2D plotting capabilities with MNE-Python (i.e., most EEG/ERP
  or other sensor-level analyses), you can install MNE-Python using ``pip``:

  .. code-block:: console

      $ pip install mne

  The only hard dependencies are `NumPy`_ and `SciPy`_, though most users will
  want to install `Matplotlib`_ too (for plotting).

- If you need MNE-Python's 3D rendering capabilities (e.g., plotting estimated
  source activity on a cortical surface) it is a good idea to install
  MNE-Python into its own virtual environment. To do this with
  `conda <anaconda_>`_:

  .. code-block:: console

      $ conda create --name=mne --channel=conda-forge mne
      $ #                   ↑↑↑                       ↑↑↑
      $ #             environment name            package name

  This will create a new ``conda`` environment called ``mne``.
  If you need to convert structural MRI scans into models of the scalp,
  inner/outer skull, and cortical surfaces you also need
  :doc:`FreeSurfer <freesurfer>`.

For users unfamiliar with Python, the :ref:`standard_instructions` page has detailed instructions for different
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
