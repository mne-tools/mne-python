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
If you want to use NVIDIA CUDA, you should ensure that you are running the
NVIDIA proprietary drivers, and then do the following (assuming you are using
``conda`` as your Python environment):

.. code-block: console

    $ conda install cupy

To initialize and test mne-python cuda support, after installing cupy_
you can run the following, which should give you an INFO-level log
message telling you your CUDA hardware's available memory:

.. code-block:: console

    $ MNE_USE_CUDA=true python -c "import mne; mne.cuda.init_cuda(verbose=True)"
    Enabling CUDA with 1.55 GB available memory

To have CUDA initialized automatically when necessary by MNE, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can test if MNE CUDA support is working by running the associated test:

.. code-block:: console

    $ pytest mne/tests/test_filter.py

If ``MNE_USE_CUDA=true`` and all tests pass with none skipped due to missing
CUDA, then MNE-Python CUDA support works.

To use CUDA, look for functions and methods that state that tehy allow
passing ``n_jobs='cuda'``, such as :meth:`mne.io.Raw.filter` and
:meth:`mne.io.Raw.resample`.

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
    >>> os.environ['QT_API'] = 'pyqt5'

This will tell Traits that we will use Qt with PyQt bindings.

For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html.

Off-screen rendering on Linux with MESA
#######################################

On remote systems, it might be possible to use MESA software rendering
(such as ``llvmpipe`` or ``swr``) for 3D visualization with some tweaks.
For example, on CentOS 7.5 you might be able to use the environment variable
to force MESA to use modern OpenGL by using this before executing
``spyder`` or ``python``:

.. code-block:: console

    $ export MESA_GL_VERSION_OVERRIDE=3.3

Also, it's possible that different software rending backends might perform
better than others, such as using the ``llvmpipe`` backend rather than ``swr``.