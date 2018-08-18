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

Some routines can utilize
`NVIDIA CUDA GPU processing <https://developer.nvidia.com/cuda-zone>`_
to speed up some operations (e.g. FIR filtering) by roughly an order of magnitude.
To use CUDA, first  ensure that you are running the NVIDIA proprietary drivers
on your operating system, and then do:

.. code-block:: console

    $ conda install cupy
    $ MNE_USE_CUDA=true python -c "import mne; mne.cuda.init_cuda(verbose=True)"
    Enabling CUDA with 1.55 GB available memory

If you receieve a message reporting the GPU's available memory, ``cupy``
is woriking properly. To permanently enable CUDA in MNE, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can then test MNE CUDA support by running the associated test:

.. code-block:: console

    $ pytest mne/tests/test_filter.py -k cuda

If the tests pass, then CUDA should work in MNE. You can use CUDA in methods
that state that they allow passing ``n_jobs='cuda'``, such as
:meth:`mne.io.Raw.filter` and :meth:`mne.io.Raw.resample`,
and they should run faster than the CPU-based multithreading such as
``n_jobs=8``.

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