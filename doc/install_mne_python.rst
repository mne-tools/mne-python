.. include:: links.inc

.. _install_python_and_mne_python:

Installing MNE-Python
---------------------

There are many possible ways to install a Python interpreter and MNE.
Here we provide guidance for the simplest, most well tested solution.

1. Get a Python interpreter
###########################

* **We recommend the
  Anaconda Python 3+ distribution**. Follow the installation instructions
  for your operating system
  `here <http://docs.continuum.io/anaconda/install>`_.

* Check the installation, which should look like a variant of this:

  .. code-block:: console

      $ conda --version && python --version
      conda 4.4.10
      Python 3.6.4 :: Continuum Analytics, Inc.

  If it doesn't, **something went wrong**.
  Look through the Anaconda documentation and Google Anaconda install
  tips (StackExchange results are often helpful).

2. Get MNE and its dependencies
###############################

* From the command line, install the MNE dependencies to a dedicated ``mne`` Anaconda environment:

  .. code-block:: console

      $ curl -O https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
      $ conda env create -f environment.yml
      $ source activate mne

  .. note:: If you are on OSX, for now you also will need to do:

            .. code-block:: console

               $ pip install --upgrade --no-deps pyqt5>=5.10

.. raw:: html

    <div class="row container">
      <div class="col-sm-7 container">

* To check that everything worked, do:

  .. code-block:: console

      $ python

  This should open an Anaconda Python prompt, where you can now do::

      >>> import mne

  If you get a new prompt with no error messages, you should be good to go!

.. raw:: html

      </div>
      <div class="col-sm-4 container">

.. note::

   .. raw:: html

     <i class="fa fa-windows"></i><b>Windows users:</b>

   If 3D plotting in Jupyter Notebooks doesn't work
   well, using the IPython magic ``%gui qt`` after importing
   MNE, Mayavi, or PySurfer should
   `help <https://github.com/ipython/ipython/issues/10384>`_, e.g.:

   .. code:: ipython

      from mayavi import mlab
      %gui qt

.. raw:: html

      </div>
    </div>

* For advanced topics like how to get :ref:`CUDA` support or if you're
  having trouble, visit :ref:`advanced_setup`.

