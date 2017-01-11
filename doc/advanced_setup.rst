:orphan:

.. include:: links.inc

.. _advanced_setup:

Advanced setup and troubleshooting
----------------------------------

.. contents:: Steps
   :local:
   :depth: 1

.. _CUDA:

CUDA
####

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
#######################

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

Troubleshooting
###############

If you run into trouble when visualizing source estimates (or anything else
using mayavi), you can try setting the ``ETS_TOOLKIT`` environment variable::

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
