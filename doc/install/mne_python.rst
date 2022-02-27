.. include:: ../links.inc

.. _standard_instructions:

Installing MNE-Python
=====================

.. highlight:: console

Once you have Python/Anaconda installed, you have a few choices for how to
install MNE-Python.

2D plotting and sensor-level analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you only need 2D plotting capabilities with MNE-Python (i.e., most EEG/ERP
or other sensor-level analyses), you can install all you need by running
``pip install mne`` in a terminal window (on Windows, use the "Anaconda Prompt"
from the Start menu, or the "CMD.exe prompt" from within the Anaconda Navigator
GUI). This will install MNE-Python into the "base" conda environment, which
should be active by default and should already have the necessary dependencies
(``numpy``, ``scipy``, and ``matplotlib``). If you want to make use of
MNE-Python's dataset downloading functions, run ``pip install mne[data]``
instead.

A second option is to install MNE-Python into its own virtual environment
(instead of installing into conda's "base" environment). This can be done via::

    $ conda create --name=new_environment_name python=3
    $ conda activate new_environment_name
    $ pip install mne

This approach is a good choice if you want to keep a separate virtual
environment for each project. This helps with reproducibility, since each
project-specific environment will have a record of which versions of the
various software packages are installed in it (accessible with ``conda list``).

3D plotting and source analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you need MNE-Python's 3D rendering capabilities (e.g., plotting estimated
source activity on a cortical surface) it is best to install
MNE-Python into its own virtual environment, so that the extra dependencies
needed for 3D plotting stay in sync (i.e., they only get updated to versions
that are compatible with MNE-Python). See the detailed instructions below for
your operating system.

.. collapse:: |linux| Linux

   Install MNE-Python from conda-forge::

       $ conda create --name=mne --channel=conda-forge mne

.. collapse:: |apple| macOS

    Install MNE-Python into a new environment (here called ``mne``, but you can
    name the environment whatever you want)::

        $ conda create --name=mne --channel=conda-forge mne

    If you like using Jupyter notebooks, you should also update the "base"
    conda environment to include the ``nb_conda_kernels`` package; this will
    make it easier to use MNE-Python in Jupyter Notebooks launched from the
    Anaconda GUI::

        $ conda install --name=base nb_conda_kernels


.. collapse:: |windows| Windows

    Open an Anaconda command prompt, and run:

    .. code-block:: doscon

        > conda create --name=mne --channel=conda-forge mne

    If you like using Jupyter notebooks, you should also update the "base"
    conda environment to include the ``nb_conda_kernels`` package; this will
    make it easier to use MNE-Python in Jupyter Notebooks launched from the
    Anaconda GUI:

    .. code-block:: doscon

        > conda install --name base nb_conda_kernels

.. raw:: html

   <div width="100%" height="0 px" style="margin: 0 0 15px;"></div>

.. javascript below adapted from nilearn

.. raw:: html

    <script type="text/javascript">
    var OSName="linux-linux";
    if (navigator.userAgent.indexOf("Win")!=-1) OSName="windows-windows";
    if (navigator.userAgent.indexOf("Mac")!=-1) OSName="apple-macos";
    $(document).ready(function(){
        var element = document.getElementById("collapse_" + OSName);
        element.className += " show";
        element.setAttribute("aria-expanded", "true");
    });
    </script>

Installing to a headless server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. collapse:: |server| If you are installing on a headless server...
    :class: danger

    With `pyvista`_:
    Download the `server environment file`_ and use it to create the conda
    environment::

        $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/main/server_environment.yml
        $ conda create --name=mne --file=server_environment.yml

Testing your installation
^^^^^^^^^^^^^^^^^^^^^^^^^

To make sure MNE-Python installed itself and its dependencies correctly,
type the following command in a terminal::

    $ python -c "import mne; mne.sys_info()"

This should display some system information along with the versions of
MNE-Python and its dependencies. Typical output looks like this::

    Platform:      Linux-5.0.0-1031-gcp-x86_64-with-glibc2.2.5
    Python:        3.8.1 (default, Dec 20 2019, 10:06:11)  [GCC 7.4.0]
    Executable:    /home/travis/virtualenv/python3.8.1/bin/python
    CPU:           x86_64: 2 cores
    Memory:        7.8 GB

    mne:           0.21.dev0
    numpy:         1.19.0.dev0+8dfaa4a {blas=openblas, lapack=openblas}
    scipy:         1.5.0.dev0+f614064
    matplotlib:    3.2.1 {backend=Qt5Agg}

    sklearn:       0.22.2.post1
    numba:         0.49.0
    nibabel:       3.1.0
    cupy:          Not found
    pandas:        1.0.3
    dipy:          1.1.1
    pyvista:       0.25.2 {pyvistaqt=0.1.0}
    vtk:           9.0.0
    PyQt5:         5.14.1


.. collapse:: |hand-paper| If you get an error...
    :class: danger

    .. rubric:: If you see an error like:

    ::

        Traceback (most recent call last):
          File "<string>", line 1, in <module>
        ModuleNotFoundError: No module named 'mne'

    This suggests that your environment containing MNE-Python is not active.
    If you followed the setup for 3D plotting/source analysis (i.e., you
    installed to a new ``mne`` environment instead of the ``base`` environment)
    try running ``conda activate mne`` first, and try again. If this works,
    you might want to set your terminal to automatically activate the
    ``mne`` environment each time you open a terminal::

        $ echo conda activate mne >> ~/.bashrc    # for bash shells
        $ echo conda activate mne >> ~/.zprofile  # for zsh shells

If something else went wrong during installation and you can't figure it out,
check out the :doc:`advanced` page to see if your problem is discussed there.
If not, the `MNE Forum`_ is a good resources for troubleshooting installation
problems.


Python IDEs
^^^^^^^^^^^

Most users find it convenient to write and run their code in an `Integrated
Development Environment`_ (IDE). Some popular choices for scientific
Python development are:

- `Spyder`_ is a free and open-source IDE developed by and for scientists who
  use Python. It is included by default in the ``base`` environment when you
  install Anaconda, and can be started from a terminal with the command
  ``spyder`` (or on Windows or macOS, launched from the Anaconda Navigator GUI).
  It can also be installed with `dedicated installers <https://www.spyder-ide.org/#section-download>`_.
  To avoid dependency conflicts with Spyder, you should install ``mne`` in a
  separate environment, like explained in the earlier sections. Then, set
  Spyder to use the ``mne`` environment as its default interpreter by opening
  Spyder and navigating to
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.
  There, paste the output of the following terminal commands::

      $ conda activate mne
      $ python -c "import sys; print(sys.executable)"

  It should be something like ``C:\Users\user\anaconda3\envs\mne\python.exe``
  (Windows) or ``/Users/user/opt/anaconda3/envs/mne/bin/python`` (macOS).

  If the Spyder console can not start because ``spyder-kernels`` is missing,
  install the required version in the ``mne`` environment with the following
  commands in the terminal::

      $ conda activate mne
      $ conda install spyder-kernels=HERE_EXACT_VERSION -c conda-forge

  Refer to the `spyder documentation <https://docs.spyder-ide.org/current/troubleshooting/common-illnesses.html#spyder-kernels-not-installed-incompatible>`_
  for more information about ``spyder-kernels`` and the version matching.

  If the Spyder graphic backend is not set to ``inline`` but to e.g. ``Qt5``,
  ``pyqt`` must be installed in the ``mne`` environment.

- `Visual Studio Code`_ (often shortened to "VS Code" or "vscode") is a
  development-focused text editor that supports many programming languages in
  addition to Python, includes an integrated terminal console, and has a rich
  ecosystem of packages to extend its capabilities. Installing
  `Microsoft's Python Extension
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__ is
  enough to get most Python users up and running. VS Code is free and
  open-source.
- `Atom`_ is a text editor similar to vscode, with a package ecosystem that
  includes a `Python IDE package <https://atom.io/packages/ide-python>`__ as
  well as `several <https://atom.io/packages/atom-terminal>`__
  `packages <https://atom.io/packages/atom-terminal-panel>`__
  `for <https://atom.io/packages/terminal-plus>`__
  `integrated <https://atom.io/packages/platformio-ide-terminal>`__
  `terminals <https://atom.io/packages/term3>`__. Atom is free and open-source.
- `SublimeText`_ is a general-purpose text editor that is fast and lightweight,
  and also has a rich package ecosystem. There is a package called `Terminus`_
  that provides an integrated terminal console, and a (confusingly named)
  package called "anaconda"
  (`found here <https://packagecontrol.io/packages/Anaconda>`__) that provides
  many Python-specific features. SublimeText is free (closed-source shareware).
- `PyCharm`_ is an IDE specifically for Python development that provides an
  all-in-one installation (no extension packages needed). PyCharm comes in a
  free "community" edition and a paid "professional" edition, and is
  closed-source.

.. highlight:: python

.. LINKS

.. _environment file: https://raw.githubusercontent.com/mne-tools/mne-python/main/environment.yml
.. _server environment file: https://raw.githubusercontent.com/mne-tools/mne-python/main/server_environment.yml
.. _`pyvista`: https://docs.pyvista.org/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
.. _`xvfb`: https://en.wikipedia.org/wiki/Xvfb
.. _`integrated development environment`: https://en.wikipedia.org/wiki/Integrated_development_environment
.. _`spyder`: https://www.spyder-ide.org/
.. _`visual studio code`: https://code.visualstudio.com/
.. _`sublimetext`: https://www.sublimetext.com/
.. _`terminus`: https://packagecontrol.io/packages/Terminus
.. _`pycharm`: https://www.jetbrains.com/pycharm/
.. _`atom`: https://atom.io/
