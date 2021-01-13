Quick start
===========

MNE-Python requires Python version 3.6 or higher.
For users already familiar with Python:

- If you only need 2D plotting capabilities with MNE-Python (i.e., most EEG/ERP
  or other sensor-level analyses), you can install MNE-Python using ``pip``:

  .. code-block:: console

      $ pip install mne  # dependencies are numpy, scipy, matplotlib

- If you need MNE-Python's 3D plotting capabilities (e.g., plotting estimated
  source activity on a cortical surface) it is a good idea to install
  MNE-Python into its own virtual environment. To do this with ``conda`` (this
  will create a conda environment called ``mne``):

  .. code-block:: console

      $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
      $ conda env update --file environment.yml

The :ref:`install_python_and_mne_python` page has more detailed instructions
for different operating systems (including instructions for installing Python
if you don't already have it). The :ref:`advanced_setup` page has additional
tips and tricks for special situations (servers, notebooks, CUDA, installing
the development version, etc). The :ref:`contributing` has additional
installation instructions for (future) contributors to MNE-Python (extra
dependencies, etc).

.. toctree::
    :hidden:

    pre_install
    mne_python
    freesurfer
    advanced
    contributing
