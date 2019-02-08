.. include:: links.inc

.. _install_python_and_mne_python:

Installing MNE-python
=====================

.. contents::
   :local:
   :depth: 1

Installing Python
^^^^^^^^^^^^^^^^^

MNE-python runs within python, and depends on several other python packages.
We strongly recommend the `Anaconda`_ or `Miniconda`_ distributions of python.
The main difference is that Anaconda comes with more than 250 scientific
packages pre-bundled, whereas Miniconda starts off with a minimal set of around
30 packages. Both distributions include the ``conda`` command line tool for
installing new packages and managing different package sets ("environments")
for different projects. Follow the installation instructions for `Anaconda
<anaconda-install>`__ or `Miniconda <miniconda-install>`__; when you are done,
you should see a similar output if you type the following command in a
terminal:

.. code-block:: console

        $ conda --version && python --version
        conda 4.6.2
        Python 3.6.7 :: Anaconda, Inc.

If you get an error message, consult the Anaconda documentation and search for
Anaconda install tips (`Stack Overflow`_ results are often helpful).

Installing MNE-python and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have Anaconda or Miniconda installed, the easiest way to install
MNE-python is to use the provided `environment file`_ to install MNE-python
and its dependencies into a new conda environment:

.. code-block:: console

    $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
    $ conda env create -f environment.yml
    $ conda activate mne

(You can also use a web browser to download the required `environment file`_ if
you do not have ``curl``.) These commands will create a new environment called
`mne` and then activate it.

.. note::

    The name of the environment is built into the environment file, but can be
    changed on the command line with the ``-n`` flag; see ``conda env create
    --help`` for more info.

Make sure you activate the environment (`conda activate mne`) each
time you open a terminal, or put the activation command in your ``.bashrc``
or ``.profile`` so that it happens automatically.

.. admonition:: |apple| macOS
  :class: note

  If you are on macOS, you need to manually update PyQt5. This step is not
  needed on Linux, and even breaks things on Windows.

  .. code-block:: console

    $ pip install --upgrade "pyqt5>=5.10"

Testing your installation
^^^^^^^^^^^^^^^^^^^^^^^^^

To make sure MNE-python installed correctly, type the following command in a
terminal:

.. code-block:: console

    $ python -c 'import mne; mne.sys_info()'

This should display some system information along with the versions of
MNE-python and its dependencies. Typical output looks like this::

    Platform:      Linux-4.15.0-44-generic-x86_64-with-debian-buster-sid
    Python:        3.6.7 |Anaconda, Inc.| (default, Oct 23 2018, 19:16:44)  [GCC 7.3.0]
    Executable:    /opt/miniconda3/envs/mne/bin/python
    CPU:           x86_64: 8 cores
    Memory:        7.6 GB

    mne:           0.18.dev0
    numpy:         1.15.4 {blas=mkl_rt, lapack=mkl_rt}
    scipy:         1.1.0
    matplotlib:    3.0.2 {backend=Qt5Agg}

    sklearn:       0.20.2
    nibabel:       2.3.1
    mayavi:        4.7.0.dev0 {qt_api=pyqt5, PyQt5=5.9.2}
    cupy:          Not found
    pandas:        0.23.4
    dipy:          0.15.0

Troubleshooting MNE-python installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If something went wrong during installation and you can't figure it out
yourself, the `MNE mailing list`_ and `MNE gitter channel`_ are good resources
for troubleshooting installation problems.

**Next:** :doc:`advanced_setup`
