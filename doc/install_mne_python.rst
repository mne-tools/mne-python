.. include:: links.inc

.. _install_python_and_mne_python:

Installing MNE-Python
---------------------

There are many possible ways to install a Python interpreter and MNE.
Here we provide guidance for the simplest, most well tested solution.

1. Get a Python interpreter
###########################

* We recommend the Anaconda Python 3+ distribution.
  Follow `their installation instructions <http://docs.continuum.io/anaconda/install>`_.
  When you are done, you should see some variant of this in a terminal:

  .. code-block:: console

      $ conda --version && python --version
      conda 4.4.10
      Python 3.6.4 :: Continuum Analytics, Inc.

  If it doesn't, something went wrong.
  Look through the Anaconda documentation and Google Anaconda install
  tips (StackExchange results are often helpful).

  Note that MNE-Python 0.17 will be the last release to support Python 3. From MNE-Python 0.18, only Python 3 will be supported.

2. Get MNE and its dependencies
###############################

* From the command line, install the MNE dependencies to a dedicated ``mne`` Anaconda environment.

  .. code-block:: console

      $ curl -O https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
      $ conda env create -f environment.yml
      $ source activate mne

  Use any web browser to download ``environment.yml`` if you do not have ``curl``

.. raw:: html

  <ul><li><p class="first"><b><i class="fa fa-apple"></i> macOS users only</b></p>

Manually update PyQt5. This step is not needed on Linux, and breaks things on Windows.

.. code-block:: console

    $ pip install --upgrade pyqt5>=5.10

.. raw:: html

  </li></ul>


3. Check that everything works
##############################

* To check that everything worked, do:

  .. code-block:: console

      $ python

  This should open an Anaconda Python prompt, where you can now do::

      >>> import mne

  If you get a new prompt with no error messages, you should be good to go!


.. raw:: html

  <ul><li><p class="first"><b><i class="fa fa-windows"></i> Windows users only</b></p>

In IPython, using the magic ``%gui qt`` after importing MNE, Mayavi, or PySurfer might be
`necessary <https://github.com/ipython/ipython/issues/10384>`_, e.g.:

.. code-block:: ipython

   In [1]: from mayavi import mlab
   In [2]: %gui qt

.. raw:: html

  </li></ul>


* For advanced topics like how to get :ref:`CUDA` support or if you're
  having trouble, visit :ref:`advanced_setup`.

