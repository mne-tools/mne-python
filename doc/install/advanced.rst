.. include:: ../links.inc

.. _advanced_setup:

Advanced setup of MNE-Python
============================

.. contents::
   :local:
   :depth: 1

Using MNE-Python with IPython / Jupyter notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using MNE-Python within IPython or a Jupyter notebook, we strongly
recommend using the Qt matplotlib backend for fast and correct rendering. On
Linux, for example, Qt is the only matplotlib backend for which 3D rendering
will work correctly. On macOS, certain matplotlib functions might not work as
expected on backends other than Qt. Enabling Qt can be accomplished when
starting IPython from a terminal:

.. code-block:: console

    $ ipython --matplotlib=qt

or in a Jupyter Notebook, you can use the "magic" command:

.. code-block:: ipython

    In [1]: %matplotlib qt

This will create separate pop-up windows for each figure, and has the advantage
that the 3D plots will retain rich interactivity (so, for example, you can
click-and-drag to rotate cortical surface activation maps).

If you are creating a static notebook or simply prefer Jupyter's inline plot
display, MNE-Python will work with the standard "inline" magic:

.. code-block:: ipython

    In [1]: %matplotlib inline

but some functionality will be lost. For example, mayavi scenes will still
pop-up a separate window, but only one window at a time is possible, and
interactivity within the scene is limited in non-blocking plot calls.

.. admonition:: |windows| Windows
  :class: note

  If you are using MNE-Python on Windows through IPython or Jupyter, you might
  also have to use the IPython magic command ``%gui qt`` after importing
  MNE-Python, Mayavi or PySurfer (see `here
  <https://github.com/ipython/ipython/issues/10384>`_). For example:

  .. code-block:: ipython

     In [1]: from mayavi import mlab
     In [2]: %gui qt

If you use another Python setup and you encounter some difficulties please
report them on the `MNE mailing list`_ or on the `GitHub issues page`_ to get
assistance.

.. _installing_master:

Using the development version of MNE-Python (latest master)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want access to the latest features and bugfixes, you can easily switch
from the stable version of MNE-Python to the current development version.

.. warning:: In between releases, function and class APIs can change without
             warning.

For a one-time update to latest master, make sure you're in the conda
environment where MNE-Python is installed (if you followed the default install
instructions, this will be ``base``), and use ``pip`` to upgrade:

.. code-block:: console

   $ conda activate name_of_my_mne_environment
   $ pip install --upgrade --no-deps https://api.github.com/repos/mne-tools/mne-python/zipball/master

If you plan to contribute to MNE-Python, or just prefer to use git rather than
pip to make frequent updates, check out the :ref:`contributing guide
<contributing>`.

.. _other-py-distros:

Using MNE-Python with other Python distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the `Anaconda`_ Python distribution provides many conveniences, other
distributions of Python should also work with MNE-Python.  In particular,
`Miniconda`_ is a lightweight alternative to Anaconda that is fully compatible;
like Anaconda, Miniconda includes the ``conda`` command line tool for
installing new packages and managing environments; unlike Anaconda, Miniconda
starts off with a minimal set of around 30 packages instead of Anaconda's
hundreds. See the `installation instructions for Miniconda`_ for more info.

It is also possible to use a system-level installation of Python (version 3.5
or higher) and use ``pip`` to install MNE-Python and its dependencies, using
the provided `requirements file`_:

.. code-block:: console

    curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/requirements.txt
    pip install --user requirements.txt

Other configurations will probably also work, but we may be unable to offer
support if you encounter difficulties related to your particular Python
installation choices.

.. _CUDA:

Using MNE-Python with CUDA (NVIDIA GPU acceleration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some operations in MNE-Python can utilize `NVIDIA CUDA GPU processing`_ to
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

Off-screen rendering in MNE-Python on Linux with MESA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On remote systems, it might be possible to use MESA software rendering
(such as ``llvmpipe`` or ``swr``) for 3D visualization (with some tweaks).
For example, on CentOS 7.5 you might be able to use an environment variable
to force MESA to use modern OpenGL by using this before executing
``spyder`` or ``python``:

.. code-block:: console

    $ export MESA_GL_VERSION_OVERRIDE=3.3

Also, it's possible that different software rending backends might perform
better than others, such as using the ``llvmpipe`` backend rather than ``swr``.

Troubleshooting 3D plots in MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you run into trouble when visualizing source estimates (or anything else
using mayavi), you can try setting a couple of environment variables at the
beginning of your script, session, or notebook::

    >>> import os
    >>> os.environ['ETS_TOOLKIT'] = 'qt4'
    >>> os.environ['QT_API'] = 'pyqt5'

This will tell mayavi to use Qt backend with PyQt bindings, instead of the
default PySide. For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html#integrating-in-a-qt-application.
