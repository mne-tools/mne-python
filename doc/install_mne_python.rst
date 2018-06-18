.. include:: links.inc

.. _install_python_and_mne_python:

Installing MNE-Python
---------------------

There are many ways to install a Python interpreter and MNE. Here we show a simple well tested solution.

1. Get Python
#############

We recommend the `Anaconda distribution <https://www.anaconda.com/distribution/>`_.
Follow the `installation instructions <http://docs.continuum.io/anaconda/install>`_.
When you are done, you should see a similar output if you type the following command in a terminal:

.. code-block:: console

    $ conda --version && python --version
    conda 4.5.4
    Python 3.6.5 :: Anaconda, Inc.

If you get an error message, consult the Anaconda documentation and search for Anaconda install
tips (`Stack Overflow <https://stackoverflow.com/>`_ results are often helpful).

.. note:: MNE-Python 0.17 will be the last release to support Python 2. From MNE-Python 0.18, only Python 3 will be supported.

2. Get MNE and its dependencies
###############################

From the command line, install the MNE dependencies to a dedicated ``mne`` Anaconda environment.

.. code-block:: console

    $ curl -O https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
    $ conda env create -f environment.yml
    $ conda activate mne

You can also use a web browser to `download the required environment file <https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml>`_ if you do not have ``curl``.

.. admonition:: |apple| macOS
  :class: note

  If you are on macOS, you need to manually update PyQt5. This step is not needed on Linux, and even breaks things on Windows.

  .. code-block:: console

    $ pip install --upgrade pyqt5>=5.10


3. Check that everything works
##############################

To make sure everything installed correctly, type the following command in a terminal:

.. code-block:: console

    $ python

This should open an interactive Python prompt, where you can type:

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!

.. admonition:: |windows| Windows
  :class: note

  If you are on Windows, you might have to use the IPython magic command ``%gui qt``
  after importing MNE, Mayavi or PySurfer (see `here <https://github.com/ipython/ipython/issues/10384>`_):

  .. code-block:: ipython

     In [1]: from mayavi import mlab
     In [2]: %gui qt

For advanced topics like how to get :ref:`CUDA` support or if you are experiencing other issues, check out :ref:`advanced_setup`.

