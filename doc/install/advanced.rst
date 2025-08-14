.. _advanced_setup:

Advanced setup
==============

Working with Jupyter Notebooks and JupyterLab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you like using Jupyter notebooks, you should also update the "base"
conda environment to include the ``nb_conda_kernels`` package; this will
make it easier to use MNE-Python in Jupyter Notebooks launched from the
Anaconda GUI:

.. code-block:: console

    $ conda install --name=base nb_conda_kernels

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

but some functionality will be lost. For example, PyVista scenes will still
pop-up a separate window, but only one window at a time is possible, and
interactivity within the scene is limited in non-blocking plot calls.

.. admonition:: |windows| Windows
  :class: note

  If you are using MNE-Python on Windows through IPython or Jupyter, you might
  also have to use the IPython magic command ``%gui qt`` (see `here
  <https://github.com/ipython/ipython/issues/10384>`_). For example:

  .. code-block:: ipython

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
please follow :ref:`standard-instructions`.

Installing to a headless Linux server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, follow the standard installation instructions. Next, you can choose
to install the ``osmesa`` (off-screen MESA) VTK variant, which avoids the need
to use Xvfb to start a virtual display (and have a sufficiently updated
MESA to render properly):

.. code-block:: console

    $ conda install -c conda-forge "vtk>=9.2=*osmesa*" "mesalib=21.2.5"

Using the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`installing_main` for how to do a one-time update to the latest
development version of MNE-Python. If you plan to contribute to MNE-Python, or
just prefer to use git rather than pip to make frequent updates, there are
instructions for installing from a ``git clone`` in the :ref:`contributing`.

Choosing the Qt framework
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``conda-forge`` version of MNE-Python ships with PyQt5. If you would like to
use a different binding, you can instead install MNE-Python via ``pip``:

.. code-block:: console

    $ pip install "mne[full]"          # uses PyQt6
    $ pip install "mne[full-pyqt6]"    # same as above
    $ pip install "mne[full-pyside6]"  # use PySide6
    $ pip install "mne[full-no-qt]"    # don't install any Qt binding

.. _CUDA:

Fixing dock icons on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^

On newer versions of Ubuntu (e.g., 24.04), applications must supply a ``.desktop``
file associated with them, otherwise a generic icon will be used like:

.. image:: ../_static/default_linux_dock_icon.png
   :alt: Default Linux dock icon

To fix this, you can create a ``.desktop`` file for MNE-Python. Here is an example
file that you can save as ``~/.local/share/applications/mne-python.desktop`` after
fixing the path to the MNE-Python icon, which you can download
`here <https://github.com/mne-tools/mne-python/blob/922a7801a0ca6af225c7b861fe6bd97b1518af3a/mne/icons/mne_default_icon.png>`__
if needed:

.. code-block:: ini

    [Desktop Entry]
    Type=Application
    Version=1.5
    Name=MNE-Python
    StartupWMClass=MNE-Python
    Icon=/path/to/mne-python/mne/icons/mne_default_icon.png
    SingleMainWindow=true
    NoDisplay=true

It should make the icon appear correctly in the dock:

.. image:: ../_static/mne_python_dock_icon.png
   :alt: MNE-Python dock icon

GPU acceleration with CUDA
^^^^^^^^^^^^^^^^^^^^^^^^^^

MNE-Python can utilize `NVIDIA CUDA GPU processing`_ to speed up some
operations (e.g. FIR filtering) by roughly an order of magnitude. To use CUDA,
first  ensure that you are running the `NVIDIA proprietary drivers`_ on your
operating system, and then do:

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

Off-screen rendering with MESA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On remote Linux systems, it might be possible to use MESA software rendering
(such as ``llvmpipe`` or ``swr``) for 3D visualization (with some tweaks).
For example, on CentOS 7.5 you might be able to use an environment variable
to force MESA to use modern OpenGL by using this before executing
``spyder`` or ``python``:

.. code-block:: console

    $ export MESA_GL_VERSION_OVERRIDE=3.3

Also, it's possible that different software rending backends might perform
better than others, such as using the ``llvmpipe`` backend rather than ``swr``.
In newer MESA (21+), rendering can be incorrect when using MSAA, so consider
setting:

.. code-block:: console

    $ export MNE_3D_OPTION_MULTI_SAMPLES=1

MESA also can have trouble with full-screen antialiasing, which you can
disable with:

.. code-block:: console

    $ export MNE_3D_OPTION_ANTIALIAS=false

or by doing
:func:`mne.viz.set_3d_options(antialias=False) <mne.viz.set_3d_options>` within
a given Python session.

Some hardware-accelerated graphics on linux (e.g., some Intel graphics cards)
provide an insufficient implementation of OpenGL, and in those cases it can help to
force software rendering instead with something like:

.. code-block:: console

    $ export LIBGL_ALWAYS_SOFTWARE=true

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

Troubleshooting 3D plots
^^^^^^^^^^^^^^^^^^^^^^^^

3D plotting trouble after upgrade on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When upgrading MNE-Python from version 0.19 or lower, some macOS users may end
up with
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

3D plotting trouble on Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are having trouble with 3D plotting on Linux, one possibility is that you
are using Wayland for graphics. To check, you can do:

.. code-block:: console

    $ echo $XDG_SESSION_TYPE
    wayland

If so, you will need to tell Qt to use X11 instead of Wayland. You can do this
by setting ``export QT_QPA_PLATFORM=xcb`` in your terminal session. To make it
permanent for your logins, you can set it for example in ``~/.profile``.

.. LINKS

.. _`pyvista`: https://docs.pyvista.org/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
.. _`xvfb`: https://en.wikipedia.org/wiki/Xvfb
