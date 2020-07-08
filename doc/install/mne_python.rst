.. include:: ../links.inc

.. _install_python_and_mne_python:

Installing MNE-Python
=====================

.. contents:: Page contents
   :local:
   :depth: 1

.. highlight:: console

.. _install-python:

Installing Python
^^^^^^^^^^^^^^^^^

MNE-Python runs within Python, and depends on several other Python packages.
Starting with version 0.21, MNE-Python only supports Python version 3.6 or
higher. We strongly
recommend the `Anaconda`_ distribution of Python, which comes with more than
250 scientific packages pre-bundled, and includes the ``conda`` command line
tool for installing new packages and managing different package sets
("environments") for different projects.

To get started, follow the `installation instructions for Anaconda`_.

.. warning::
   If you have the ``PYTHONPATH`` or ``PYTHONHOME`` environment variables set,
   you may run into difficulty using Anaconda. See the
   `Anaconda troubleshooting guide`_ for more information. Note that it is
   easy to switch between ``conda``-managed Python installations and the system
   Python installation using the ``conda activate`` and ``conda deactivate``
   commands, so you may find that after adopting Anaconda it is possible
   (indeed, preferable) to leave ``PYTHONPATH`` and ``PYTHONHOME`` permanently
   unset.

When you are done, if you type the following commands in a command shell,
you should see outputs similar to the following (assuming you installed
conda to ``/home/user/anaconda3``)::

    $ conda --version && python --version
    conda 4.6.2
    Python 3.6.7 :: Anaconda, Inc.
    $ which python
    /home/user/anaconda3/bin/python
    $ which pip
    /home/user/anaconda3/bin/pip

.. collapse:: |hand-stop-o| If you get an error or these look incorrect...
    :class: danger

    .. rubric:: If you are on a |windows| Windows command prompt:

    Most of our instructions start with ``$``, which indicates
    that the commands are designed to be run from a ``bash`` command shell.

    Windows command prompts do not expose the same command-line tools as
    ``bash`` shells, so commands like ``which`` will not work. You can test
    your installation in Windows ``cmd.exe`` shells with ``where`` instead::

        > where python
        C:\Users\user\anaconda3\python.exe
        > where pip
        C:\Users\user\anaconda3\Scripts\pip.exe

    .. rubric:: If you see something like:

    ::

        conda: command not found

    It means that your ``PATH`` variable (what the system uses to find
    programs) is not set properly. In a correct installation, doing::

        $ echo $PATH
        ...:/home/user/anaconda3/bin:...

    Will show the Anaconda binary path (above) somewhere in the output
    (probably at or near the beginning), but the ``command not found`` error
    suggests that it is missing.

    On Linux or macOS, the installer should have put something
    like the following in your ``~/.bashrc`` or ``~/.bash_profile`` (or your
    ``.zprofile`` if you're using macOS Catalina or later, where the default
    shell is ``zsh``):

    .. code-block:: bash

        # >>> conda initialize >>>
        # !! Contents within this block are managed by 'conda init' !!
        __conda_setup= ...
        ...
        # <<< conda initialize <<<

    If this is missing, it is possible that you are not on the same shell that
    was used during the installation. You can verify which shell you are on by
    using the command::

        $ echo $SHELL

    If you do not find this line in the configuration file for the shell you
    are using (bash, zsh, tcsh, etc.), try running::

        conda init

    in your command shell. If your shell is not ``cmd.exe`` (Windows) or
    ``bash`` (Linux, macOS) you will need to pass the name of the shell to the
    ``conda init`` command. See ``conda init --help`` for more info and
    supported shells.

    You can also consult the Anaconda documentation and search for
    Anaconda install tips (`Stack Overflow`_ results are often helpful)
    to fix these or other problems when ``conda`` does not work.

.. _standard_instructions:

Installing MNE-Python and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have Python/Anaconda installed, you have a few choices for how to
install MNE-Python.

For sensor-level analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

If you only need 2D plotting capabilities with MNE-Python (i.e., most EEG/ERP
or other sensor-level analyses), you can install all you need by running
``pip install mne`` in a terminal window (on Windows, use the "Anaconda Prompt"
from the Start menu, or the "CMD.exe prompt" from within the Anaconda Navigator
GUI). This will install MNE-Python into the "base" conda environment, which
should be active by default and should already have the necessary dependencies
(``numpy``, ``scipy``, and ``matplotlib``).

For 3D plotting and source analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need MNE-Python's 3D plotting capabilities (e.g., plotting estimated
source activity on a cortical surface) it is a good idea to install
MNE-Python into its own virtual environment, so that the extra dependencies
needed for 3D plotting stay in sync (i.e., they only get updated to versions
that are compatible with MNE-Python). See the detailed instructions below for
your operating system.

.. collapse:: |linux| Linux

   Download the MNE-Python `environment file`_ (done here with ``curl``) and
   use it to create a new environment (named ``mne`` by default)::

       $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
       $ conda env update --file environment.yml

   .. collapse:: |hand-stop-o| If you get errors building mayavi...
       :class: danger

       Installing `mayavi`_ needs OpenGL support. On debian-like systems this
       means installing ``libopengl0``, i.e., ``sudo apt install libopengl0``.

.. collapse:: |apple| macOS

   Update the ``base`` conda environment to include the ``nb_conda_kernels``
   package, so you can use MNE-Python in Jupyter Notebooks launched from the
   Anaconda GUI. Then download the MNE-Python `environment file`_ (done here
   with ``curl``) and use it to create a new environment (named ``mne`` by
   default)::

       $ conda install --name base nb_conda_kernels
       $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
       $ conda env update --file environment.yml

.. collapse:: |windows| Windows

   - Download the `environment file`_
   - Open an Anaconda command prompt
   - Run :samp:`conda install --name base nb_conda_kernels`
   - :samp:`cd` to the directory where you downloaded the file
   - Run :samp:`conda env update --file environment.yml`

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
        element.className += " in";
        element.setAttribute("aria-expanded", "true");
    });
    </script>

Installing to a headless server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. collapse:: |server| If you are installing on a headless server...
    :class: danger

    With `pyvista`_:
    Follow the steps described in :ref:`standard_instructions`
    but use the `server environment file`_ instead of the `environment file`_.

    With `mayavi`_:
    Installing `mayavi`_ requires a running `X server`_. If you are
    installing MNE-Python into a computer with no display connected to it, you
    can try removing `mayavi`_ from the :file:`environment.yml` file before
    running :samp:`conda env create --file environment.yml`, activating the new
    environment, and then installing `mayavi`_ using `xvfb`_ (e.g.,
    :samp:`xvfb-run pip install mayavi`). Be sure to read Mayavi's instructions
    on `off-screen rendering`_ and `rendering with a virtual framebuffer`_.

    Note: if :samp:`xvfb` is not already installed on your server, you will
    need administrator privileges to install it.


Testing MNE-Python installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    mayavi:        4.7.2.dev0
    pyvista:       0.25.2 {pyvistaqt=0.1.0}
    vtk:           9.0.0
    PyQt5:         5.14.1


.. collapse:: |hand-stop-o| If you get an error...
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
If not, the `MNE mailing list`_ and `MNE gitter channel`_ are
good resources for troubleshooting installation problems.


Installing a Python IDE
^^^^^^^^^^^^^^^^^^^^^^^

Most users find it convenient to write and run their code in an `Integrated
Development Environment`_ (IDE). Some popular choices for scientific
Python development are:

- `Spyder`_ is a free and open-source IDE developed by and for scientists who
  use Python. It is included by default in the ``base`` environment when you
  install Anaconda, and can be started from a terminal with the command
  ``spyder`` (or on Windows or macOS, launched from the Anaconda Navigator GUI).
  If you installed MNE-Python into a separate ``mne`` environment (not the
  ``base`` Anaconda environment), you can set up Spyder to use the ``mne``
  environment automatically, by opening Spyder and navigating to
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.
  There, paste the output of the following terminal command::

      $ conda activate mne && python -c "import sys; print(sys.executable)"

  It should be something like ``C:\Users\user\anaconda3\envs\mne\python.exe``
  (Windows) or ``/Users/user/anaconda3/envs/mne/bin/python`` (macOS).
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

**Next:** :doc:`freesurfer`


.. LINKS

.. _environment file: https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
.. _server environment file: https://raw.githubusercontent.com/mne-tools/mne-python/master/server_environment.yml
.. _`mayavi`: https://docs.enthought.com/mayavi/mayavi/
.. _`pyvista`: https://docs.pyvista.org/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
.. _`xvfb`: https://en.wikipedia.org/wiki/Xvfb
.. _`off-screen rendering`: https://docs.enthought.com/mayavi/mayavi/tips.html#off-screen-rendering
.. _`rendering with a virtual framebuffer`: https://docs.enthought.com/mayavi/mayavi/tips.html#rendering-using-the-virtual-framebuffer
.. _`integrated development environment`: https://en.wikipedia.org/wiki/Integrated_development_environment
.. _`spyder`: https://www.spyder-ide.org/
.. _`visual studio code`: https://code.visualstudio.com/
.. _`sublimetext`: https://www.sublimetext.com/
.. _`terminus`: https://packagecontrol.io/packages/Terminus
.. _`pycharm`: https://www.jetbrains.com/pycharm/
.. _`atom`: https://atom.io/
