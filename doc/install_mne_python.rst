.. include:: links.inc

.. _install_python_and_mne_python:

Install Python and MNE-Python
-----------------------------

To use MNE-Python, you need two things:

1. A working Python interpreter and dependencies

2. The MNE-Python package installed to the Python distribution

Step-by-step instructions to accomplish this are given below.

.. contents:: Steps
   :local:
   :depth: 1

.. note:: Users who work at a facility with a site-wide install of
          MNE-Python (e.g. Martinos center) are encouraged to contact
          their technical staff about how to access and use MNE-Python,
          as the instructions might differ.

.. _install_interpreter:

1. Install a Python interpreter and dependencies
################################################

There are multiple options available for getting a suitable Python interpreter
running on your system. However, for a fast and up to date scientific Python
environment that resolves all dependencies, we highly recommend following
installation instructions for the Anaconda Python distribution. Go here
to download, and follow the installation instructions:

* https://www.continuum.io/downloads
* http://docs.continuum.io/anaconda/install

.. note:: Intel's `Math Kernel Library <https://software.intel.com/en-us/intel-mkl>`_
          speeds up many linear algebra comptutations, some by an order of
          magnitude. It is included by default in Anaconda, which makes it
          preferable to some other potentially viable Python interpreter
          options (e.g., using ``brew`` in OSX or ``sudo apt-get install``
          on Ubuntu to get a usable Python interpreter). 

If everything is set up correctly, you should be able to check the version
of ``conda`` that is installed (your version number will probably be newer)
and which ``python`` will be called when you run ``python``:

.. code-block:: bash

    $ conda --version
    conda 3.19.1
    $ which python
    /home/agramfort/anaconda/bin/python

If your installation doesn't look something like this,
*something went wrong* and you should try to fix it. We recommend looking
through the Anaconda documentation a bit, and Googling for Anaconda
install tips (StackExchange results are often helpful).

Once Anaconda works properly, you can do this to resolve
the MNE-Python dependencies:

.. code-block:: bash

    $ conda install scipy matplotlib scikit-learn

To test that everything works properly, open up IPython:

.. code-block:: bash

    $ ipython --matplotlib=qt

Now that you have a working Python environment you can install MNE-Python.

If you want to have an environment with a clean MATLAB-like interface,
consider using Spyder_, which can easily be installed with Anaconda
as:

.. code-block:: bash

    $ conda install spyder

.. _install_mne_python:

2. Install the MNE-Python package
#################################

There are a many options for installing MNE-Python, but two of the most
useful and common are:

1. **Use the stable release version of MNE-Python.** It can be installed as:

   .. code-block:: bash

       $ pip install mne --upgrade

   MNE-Python tends to release about once every six months, and this
   command can be used to update the install after each release.

.. _installing_master:

2. **Use the development master version of MNE-Python.** If you want to
   be able to update your MNE-Python version between releases (e.g., for
   bugfixes or new features), this will set you up for frequent updates:

   .. code-block:: bash

       $ git clone git://github.com/mne-tools/mne-python.git
       $ cd mne-python
       $ python setup.py develop

   A cool feature of ``python setup.py develop`` is that any changes made to
   the files (e.g., by updating to latest ``master``) will be reflected in
   ``mne`` as soon as you restart your Python interpreter. So to update to
   the latest version of the ``master`` development branch, you can do:

   .. code-block:: bash

       $ git pull origin master

   and your MNE-Python will be updated to have the latest changes.
   If you plan to contribute to MNE-Python, you should follow a variant
   of this approach outlined in the
   :ref:`contribution instructions <contributing>`. 

3. Check your installation
##########################

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!

A good place to start is on our :ref:`tutorials` page or with our
:ref:`general_examples`.

Along the way, make frequent use of :ref:`api_reference` and
:ref:`manual` to understand the capabilities of MNE.

4. Optional advanced setup
##########################

.. _CUDA:

CUDA
^^^^

We have developed specialized routines to make use of
`NVIDIA CUDA GPU processing <http://www.nvidia.com/object/cuda_home_new.html>`_
to speed up some operations (e.g. FIR filtering) by up to 10x. 
If you want to use NVIDIA CUDA, you should install:

1. `the NVIDIA toolkit on your system <https://developer.nvidia.com/cuda-downloads>`_
2. `PyCUDA <http://wiki.tiker.net/PyCuda/Installation/>`_
3. `skcuda <https://github.com/lebedov/scikits.cuda>`_

For example, on Ubuntu 15.10, a combination of system packages and ``git``
packages can be used to install the CUDA stack:

.. code-block:: bash

    # install system packages for CUDA
    $ sudo apt-get install nvidia-cuda-dev nvidia-modprobe
    # install PyCUDA
    $ git clone http://git.tiker.net/trees/pycuda.git
    $ cd pycuda
    $ ./configure.py --cuda-enable-gl
    $ git submodule update --init
    $ make -j 4
    $ python setup.py install
    # install skcuda
    $ cd ..
    $ git clone https://github.com/lebedov/scikit-cuda.git
    $ cd scikit-cuda
    $ python setup.py install

To initialize mne-python cuda support, after installing these dependencies
and running their associated unit tests (to ensure your installation is correct)
you can run:

.. code-block:: bash

    $ MNE_USE_CUDA=true MNE_LOGGING_LEVEL=info python -c "import mne; mne.cuda.init_cuda()"
    Enabling CUDA with 1.55 GB available memory

If you have everything installed correctly, you should see an INFO-level log
message telling you your CUDA hardware's available memory. To have CUDA
initialized on startup, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can test if MNE CUDA support is working by running the associated test:

.. code-block:: bash

    $ nosetests mne/tests/test_filter.py

If ``MNE_USE_CUDA=true`` and all tests pass with none skipped, then
MNE-Python CUDA support works.

IPython (and notebooks)
^^^^^^^^^^^^^^^^^^^^^^^

In IPython, we strongly recommend using the Qt matplotlib backend for
fast and correct rendering:

.. code-block:: bash

    $ ipython --matplotlib=qt

On Linux, for example, QT is the only matplotlib backend for which 3D rendering
will work correctly. On Mac OS X for other backends certain matplotlib
functions might not work as expected.

To take full advantage of MNE-Python's visualization capacities in combination
with IPython notebooks and inline displaying, please explicitly add the
following magic method invocation to your notebook or configure your notebook
runtime accordingly::

    In [1]: %matplotlib inline

If you use another Python setup and you encounter some difficulties please
report them on the MNE mailing list or on github to get assistance.

Mayavi and PySurfer
^^^^^^^^^^^^^^^^^^^

Mayavi is currently only available for Python2.7.
The easiest way to install `mayavi` is to do the following with Anaconda:

.. code-block:: bash

    $ conda install mayavi

For other methods of installation, please consult the
`Mayavi documentation <http://docs.enthought.com/mayavi/mayavi/installation.html>`_.

The PySurfer package, which is used for visualizing cortical source estimates,
uses Mayavi and can be installed using:

.. code-block:: bash

    $ pip install PySurfer

Some users may need to configure PySurfer before they can make full use of
our visualization capabilities. Please refer to the
`PySurfer installation page <https://pysurfer.github.io/install.html>`_
for up to date information.

Troubleshooting
###############

If you run into trouble when visualizing source estimates (or anything else
using mayavi), you can try setting ETS_TOOLKIT environment variable::

    >>> import os
    >>> os.environ['ETS_TOOLKIT'] = 'qt4'
    >>> os.environ['QT_API'] = 'pyqt'

This will tell Traits that we will use Qt with PyQt bindings.

If you get an error saying::

    ValueError: API 'QDate' has already been set to version 1

you have run into a conflict with Traits. You can work around this by telling
the interpreter to use QtGui and QtCore from pyface::

    >>> from pyface.qt import QtGui, QtCore

This line should be added before any imports from mne-python.

For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html.
