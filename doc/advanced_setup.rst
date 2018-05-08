:orphan:

.. include:: links.inc

.. _advanced_setup:

Advanced setup and troubleshooting
----------------------------------

.. contents:: Steps
   :local:
   :depth: 1

.. _installing_master:

Using the development version of MNE (latest master)
####################################################

It is possible to update your version of MNE between releases for
bugfixes or new features.

.. warning:: In between releases, function and class APIs can change without
             warning.

You can use ``pip`` for a one-time update:

.. code-block:: console

   $ pip install --upgrade --no-deps git+https://github.com/mne-tools/mne-python.git

Or, if you prefer to be set up for frequent updates, you can use ``git`` directly:

.. code-block:: console

   $ git clone git://github.com/mne-tools/mne-python.git
   $ cd mne-python
   $ python setup.py develop

A feature of ``python setup.py develop`` is that any changes made to
the files (e.g., by updating to latest ``master``) will be reflected in
``mne`` as soon as you restart your Python interpreter. So to update to
the latest version of the ``master`` development branch, you can do:

.. code-block:: console

   $ git pull origin master

and MNE will be updated to have the latest changes.


If you plan to contribute to MNE, please continue reading how to
:ref:`contribute_to_mne`.

.. _CUDA:

CUDA (NVIDIA GPU acceleration)
##############################

We have developed specialized routines to make use of
`NVIDIA CUDA GPU processing <http://www.nvidia.com/object/cuda_home_new.html>`_
to speed up some operations (e.g. FIR filtering) by up to 10x.
If you want to use NVIDIA CUDA, you should install:

1. `the NVIDIA toolkit on your system <https://developer.nvidia.com/cuda-downloads>`_
2. `PyCUDA <http://wiki.tiker.net/PyCuda/Installation/>`_
3. `skcuda <https://github.com/lebedov/scikits.cuda>`_

For example, on Ubuntu 15.10, a combination of system packages and ``git``
packages can be used to install the CUDA stack:

.. code-block:: console

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

.. code-block:: console

    $ MNE_USE_CUDA=true MNE_LOGGING_LEVEL=info python -c "import mne; mne.cuda.init_cuda()"
    Enabling CUDA with 1.55 GB available memory

If you have everything installed correctly, you should see an INFO-level log
message telling you your CUDA hardware's available memory. To have CUDA
initialized on startup, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can test if MNE CUDA support is working by running the associated test:

.. code-block:: console

    $ pytest mne/tests/test_filter.py

If ``MNE_USE_CUDA=true`` and all tests pass with none skipped, then
MNE-Python CUDA support works.

IPython / Jupyter notebooks
###########################

In Jupyter, we strongly recommend using the Qt matplotlib backend for
fast and correct rendering:

.. code-block:: console

    $ ipython --matplotlib=qt

On Linux, for example, QT is the only matplotlib backend for which 3D rendering
will work correctly. On Mac OS X for other backends certain matplotlib
functions might not work as expected.

To take full advantage of MNE-Python's visualization capacities in combination
with IPython notebooks and inline displaying, please explicitly add the
following magic method invocation to your notebook or configure your notebook
runtime accordingly:

.. code-block:: ipython

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

For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html.
