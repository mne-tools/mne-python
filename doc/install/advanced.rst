.. include:: ../links.inc

.. _advanced_setup:

Advanced setup of MNE-Python
============================

.. contents::
   :local:
   :depth: 2

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

If you installed the ``nb_conda_kernels`` package into your ``base``
environment (as recommended), you should be able to launch ``mne``-capable
notebooks from within the Anaconda Navigator GUI without having to explicitly
switch to the ``mne`` environment first; look for ``Python [conda env:mne]``
when choosing which notebook kernel to use. Otherwise, be sure to activate the
``mne`` environment before launching the notebook.

If you use another Python setup and you encounter some difficulties please
report them on the `MNE Forum`_ or on the `GitHub issues page`_ to get
assistance.

It is also possible to interact with the 3D plots without installing Qt by using
the notebook 3d backend:

.. code-block:: ipython

   In [1]: import mne
   In [2]: mne.viz.set_3d_backend("notebook")


The notebook 3d backend requires PyVista to be installed along with other packages,
please follow :doc:`mne_python`


.. _installing_main:

Using the development version of MNE-Python (latest main)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want access to the latest features and bugfixes, you can easily switch
from the stable version of MNE-Python to the current development version.

.. warning:: In between releases, function and class APIs can change without
             warning.

For a one-time update to latest main, make sure you're in the conda
environment where MNE-Python is installed (if you followed the default install
instructions, this will be ``base``), and use ``pip`` to upgrade:

.. code-block:: console

   $ conda activate name_of_my_mne_environment
   $ pip install --upgrade --no-deps https://github.com/mne-tools/mne-python/archive/main.zip

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

    curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/main/requirements.txt
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

MESA also can have trouble with full-screen antialiasing, which you can
disable with:

.. code-block:: console

    $ export MNE_3D_OPTION_ANTIALIAS=false

or by doing
:func:`mne.viz.set_3d_options(antialias=False) <mne.viz.set_3d_options>` within
a given Python session.

Another issue that may come up is that the MESA software itself may be out of date
in certain operating systems, for example CentOS. This may lead to incomplete
rendering of some 3D plots. A solution is described in this `Github comment <https://github.com/mne-tools/mne-python/issues/7977#issuecomment-729921035>`_.
It boils down to building a newer version (e.g., 18.3.6)
locally following a variant of `these instructions <https://xorg-team.pages.debian.net/xorg/howto/build-mesa.html#_preparing_mesa_sources>`_.
If you have CentOS 7 or newer, you can also try some `prebuilt binaries <https://osf.io/sp9qg/download>`_ we made.
After downloading the files, untar them and add them to the appropriate library paths
using the following commands:

.. code-block:: console

    $ tar xzvf mesa_18.3.6_centos_lib.tgz
    $ export LIBGL_DRIVERS_PATH="${PWD}/lib"
    $ export LD_LIBRARY_PATH="${PWD}/lib"

To check that everything went well, type the following:

.. code-block:: console

    $ glxinfo | grep "OpenGL core profile version"

which should give::

    OpenGL core profile version string: 3.3 (Core Profile) Mesa 18.3.6

Another way to check is to type:

.. code-block:: console

    $ mne sys_info

and it should show the right version of MESA::

    ...
    pyvista:       0.27.4 {pyvistaqt=0.2.0, OpenGL 3.3 (Core Profile) Mesa 18.3.6 via llvmpipe (LLVM 3.4, 256 bits)}
    ...

.. _troubleshoot_3d:

Troubleshooting 3D plots in MNE-Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3D plotting trouble after version 0.20 upgrade on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When upgrading MNE-Python to version 0.20, some macOS users may end up with
conflicting versions of some of the 3D plotting dependencies. If you plot using
the pyvista 3D backend and find that you can click-drag to rotate the brain,
but cannot adjust any of the settings sliders, it is likely that your versions
of VTK and/or QT are incompatible. This series of commands should fix it:

.. code-block:: console

    $ conda uninstall vtk
    $ pip uninstall -y pyvista
    $ conda install vtk
    $ pip install --no-cache pyvista

If you installed VTK using ``pip`` rather than ``conda``, substitute the first
line for ``pip uninstall -y vtk``.


3D plotting trouble using mayavi 3D backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run into trouble when visualizing source estimates (or anything else)
using mayavi, you can try setting a couple of environment variables at the
beginning of your script, session, or notebook::

    >>> import os
    >>> os.environ['ETS_TOOLKIT'] = 'qt4'
    >>> os.environ['QT_API'] = 'pyqt5'

This will tell mayavi to use Qt backend with PyQt bindings, instead of the
default PySide. For more information, see
http://docs.enthought.com/mayavi/mayavi/building_applications.html#integrating-in-a-qt-application.
