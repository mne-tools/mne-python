.. include:: links.inc

.. _install_python_and_mne_python:

Installing MNE-Python
=====================

.. contents::
   :local:
   :depth: 1

Installing Python
^^^^^^^^^^^^^^^^^

MNE-Python runs within Python, and depends on several other Python packages.
MNE-Python 0.18 only supports Python version 3.5 or higher. We strongly
recommend the `Anaconda`_ distribution of Python, which comes with more than
250 scientific packages pre-bundled, and includes the ``conda`` command line
tool for installing new packages and managing different package sets
("environments") for different projects. Follow the installation instructions
for `Anaconda <anaconda-install>`__; when you are done, you should see a
similar output if you type the following command in a terminal:

.. code-block:: console

        $ conda --version && python --version
        conda 4.6.2
        Python 3.6.7 :: Anaconda, Inc.

If you get an error message, consult the Anaconda documentation and search for
Anaconda install tips (`Stack Overflow`_ results are often helpful).

Installing MNE-Python and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have Anaconda installed, the easiest way to install
MNE-Python is to use the provided `environment file`_ to install MNE-Python
and its dependencies into a new conda environment:

.. code-block:: console

    $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
    $ conda env create -f environment.yml
    $ conda activate mne

(You can also use a web browser to download the required `environment file`_ if
you do not have ``curl``.) These commands will create a new environment called
``mne`` and then activate it.

Make sure you activate the environment (``conda activate mne``) each time you
open a terminal, or put the activation command in your ``.bashrc`` or
``.profile`` so that it happens automatically.

.. admonition:: |apple| macOS
  :class: note

  If you are on macOS, you need to manually update PyQt5. This step is not
  needed on Linux, and even breaks things on Windows.

  .. code-block:: console

    $ pip install --upgrade "pyqt5>=5.10"

Testing your installation
^^^^^^^^^^^^^^^^^^^^^^^^^

To make sure MNE-Python installed correctly, type the following command in a
terminal:

.. code-block:: console

    $ python -c 'import mne; mne.sys_info()'

This should display some system information along with the versions of
MNE-Python and its dependencies. Typical output looks like this:

.. code-block:: console

    Platform:      Linux-4.18.0-13-generic-x86_64-with-debian-buster-sid
    Python:        3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34)  [GCC 7.3.0]
    Executable:    /home/travis/miniconda/envs/test/bin/python
    CPU:           x86_64: 48 cores
    Memory:        62.7 GB

    mne:           0.17.0
    numpy:         1.15.4 {blas=mkl_rt, lapack=mkl_rt}
    scipy:         1.2.0
    matplotlib:    3.0.2 {backend=Qt5Agg}

    sklearn:       0.20.2
    nibabel:       2.3.3
    mayavi:        4.7.0.dev0 {qt_api=pyqt5, PyQt5=5.10.1}
    cupy:          Not found
    pandas:        0.24.0
    dipy:          0.15.0

Troubleshooting MNE-Python installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If something went wrong during installation and you can't figure it out
yourself, check out the :doc:`advanced_setup` page to see if your problem is
discussed there. If not, the `MNE mailing list`_ and `MNE gitter channel`_ are
good resources for troubleshooting installation problems.

**Next:** :doc:`install_freesurfer`
