.. include:: links.inc

.. _advanced_setup:

Advanced setup of MNE-python
============================

.. contents::
   :local:
   :depth: 1

IPython / Jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Jupyter, we strongly recommend using the Qt matplotlib backend for
fast and correct rendering:

.. code-block:: console

    $ ipython --matplotlib=qt

On Linux, for example, QT is the only matplotlib backend for which 3D rendering
will work correctly. On macOS, certain matplotlib functions might not work as
expected on backends other than QT.

To take full advantage of MNE-python's visualization capacities in combination
with IPython notebooks and inline displaying, please explicitly add the
following magic method invocation to your notebook or configure your notebook
runtime accordingly:

.. code-block:: ipython

    In [1]: %matplotlib inline


.. admonition:: |windows| Windows
  :class: note

  If you are using MNE-python on Windows through IPython or Jupyter,
  you might have to use the IPython magic command ``%gui qt`` after importing
  MNE-python, Mayavi or PySurfer (see `here
  <https://github.com/ipython/ipython/issues/10384>`_). For example:

  .. code-block:: ipython

     In [1]: from mayavi import mlab
     In [2]: %gui qt

If you use another Python setup and you encounter some difficulties please
report them on the `MNE mailing list`_ or on the `GitHub issues page`_ to get
assistance.

.. _installing_master:

Using the development version of MNE-python (latest master)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want access to the latest features and bugfixes, you can easily switch
from the stable version of MNE-python to the current development version.

.. warning:: In between releases, function and class APIs can change without
             warning.

For a one-time update to latest master, make sure you're in the ``mne`` conda
environment (``conda activate mne``), and use ``pip``:

.. code-block:: console

   $ pip install --upgrade --no-deps git+https://github.com/mne-tools/mne-python.git

If you plan to contribute to MNE-python, or if you prefer to update frequently,
you can use ``git`` directly (again, within the ``mne`` conda environment):

.. code-block:: console

   $ cd <path_to_where_you_want_mne-python_source_code_installed>
   $ git clone git://github.com/mne-tools/mne-python.git
   $ cd mne-python
   $ python setup.py develop

A feature of ``python setup.py develop`` is that any changes made to
the files (e.g., by updating to latest ``master``) will be reflected in
``mne`` as soon as you restart your Python interpreter. So to update to
the latest version of the ``master`` development branch, you can do:

.. code-block:: console

   $ git pull origin master

from within the mne-python source folder, and MNE will be automatically updated
to have the latest changes.

If you plan to contribute to MNE-python, please continue reading how to
:doc:`contribute to MNE-python <contributing>`.

.. _CUDA:

Using MNE-python with CUDA (NVIDIA GPU acceleration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations in MNE-python can utilize `NVIDIA CUDA GPU processing`_ to
speed up some operations (e.g. FIR filtering) by roughly an order of magnitude.
To use CUDA, first  ensure that you are running the `NVIDIA proprietary
drivers`_ on your operating system, and then do:

.. code-block:: console

    $ conda install cupy
    $ MNE_USE_CUDA=true python -c "import mne; mne.cuda.init_cuda(verbose=True)"
    Enabling CUDA with 1.55 GB available memory

If you receive a message reporting the GPU's available memory, CuPy_
is working properly. To permanently enable CUDA in MNE, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true')  # doctest: +SKIP

You can then test MNE CUDA support by running the associated test:

.. code-block:: console

    $ pytest mne/tests/test_filter.py -k cuda

If the tests pass, then CUDA should work in MNE. You can use CUDA in methods
that state that they allow passing ``n_jobs='cuda'``, such as
:meth:`mne.io.Raw.filter` and :meth:`mne.io.Raw.resample`,
and they should run faster than the CPU-based multithreading such as
``n_jobs=8``.

Off-screen rendering on Linux with MESA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On remote systems, it might be possible to use MESA software rendering
(such as ``llvmpipe`` or ``swr``) for 3D visualization (with some tweaks).
For example, on CentOS 7.5 you might be able to use an environment variable
to force MESA to use modern OpenGL by using this before executing
``spyder`` or ``python``:

.. code-block:: console

    $ export MESA_GL_VERSION_OVERRIDE=3.3

Also, it's possible that different software rending backends might perform
better than others, such as using the ``llvmpipe`` backend rather than ``swr``.

Troubleshooting 3D plots
^^^^^^^^^^^^^^^^^^^^^^^^

If you run into trouble when visualizing source estimates (or anything else
using mayavi), you can try setting a couple of environment variables at the
beginning of your script, session, or notebook::

    >>> import os
    >>> os.environ['ETS_TOOLKIT'] = 'qt4'
    >>> os.environ['QT_API'] = 'pyqt5'

This will tell mayavi to use Qt backend with PyQt bindings, instead of the
default PySide. For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html#integrating-in-a-qt-application.
