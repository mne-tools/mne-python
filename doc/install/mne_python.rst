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

When you are done, if you type the following commands in a ``bash`` terminal,
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
    that the commands are designed to be run from a Bash command prompt.

    Windows command prompts do not expose the same command-line tools as Bash
    shells, so things like ``which`` will not work, and you need to use
    alternatives, such as::

        > where mne
        C:\Users\mneuser\Anaconda3\Scripts\mne

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

    On Linux or OSX, the installer should have put something
    like the following in your ``~/.bashrc`` or ``~/.bash_profile``
    (or somewhere else if you are using a non-bash terminal):

    .. code-block:: bash

        . ~/anaconda3/etc/profile.d/conda.sh

    If this is missing, it is possible that you are not on the same shell that
    was used during the installation. You can verify which shell you are on by
    using the command::

        $ echo $SHELL

    If you do not find this line in the configuration file for the shell you
    are using (bash, tcsh, etc.), add the line to that shell's ``rc`` or
    ``profile`` file to fix the problem.

    .. rubric:: If you see an error like:

    ::

        CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.

    It means that you have used an old method to set up Anaconda. This
    means that you have something like::

        PATH=~/anaconda3/bin:$PATH

    in your ``~/.bash_profile``. You should update this line to use
    the modern way using ``anaconda3/etc/profile.d/conda.sh`` above.

    You can also consult the Anaconda documentation and search for
    Anaconda install tips (`Stack Overflow`_ results are often helpful)
    to fix these or other problems when ``conda`` does not work.

.. _standard_instructions:

Installing MNE-Python and its dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have Anaconda installed, the easiest way to install
MNE-Python with all dependencies is update your base Anaconda environment:

.. _environment file: https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
.. _server environment file: https://raw.githubusercontent.com/mne-tools/mne-python/master/server_environment.yml

.. collapse:: |linux| Linux

   Use the base `environment file`_, e.g.::

       $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
       $ conda env update --file environment.yml

   .. collapse:: |hand-stop-o| If you get errors building mayavi...
       :class: danger

       Installing `mayavi`_ needs OpenGL support. On debian-like systems this
       means installing ``libopengl0``, i.e., ``sudo apt install libopengl0``.

.. collapse:: |apple| macOS

   Use the base `environment file`_, e.g.::

       $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
       $ conda env update --file environment.yml

.. collapse:: |windows| Windows

   - Download the base `environment file`_
   - Open an Anaconda command prompt
   - :samp:`cd` to the directory where you downloaded the file
   - Run :samp:`conda env update --file environment.yml`

.. raw:: html

   <div width="100%" height="0 px" style="margin: 0 0 15px;"></div>

If you prefer an isolated Anaconda environment, instead of using\
:samp:`conda env update` to modify your "base" environment,
you can create a new dedicated environment (here called "mne") with
:samp:`conda env create --name mne --file environment.yml`.

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

.. collapse:: |hand-stop-o| If you are installing on a headless server...
    :class: danger

    With `pyvista`_:
    Follow the steps described in :ref:`standard_instructions`
    but use `server environment file`_ instead of `environment file`_.

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

    $ mne sys_info

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

    This suggests that your environment containing ``mne`` is not active.
    If you installed to the ``mne`` instead of ``base`` environment, try doing
    ``conda activate mne`` and try again. If this works, you might want to
    add ``conda activate mne`` to the end of your ``~/.bashrc`` or
    ``~/.bash_profile`` files so that it gets executed automatically.

If something else went wrong during installation and you can't figure it out,
check out the :doc:`advanced` page to see if your problem is discussed there.
If not, the `MNE mailing list`_ and `MNE gitter channel`_ are
good resources for troubleshooting installation problems.


Installing a Python IDE
^^^^^^^^^^^^^^^^^^^^^^^

Most users find it convenient to write and run their code in an `Integrated
Development Environment <ide>`_ (IDE). Some popular choices for scientific
Python development are:

- `Spyder`_ is an IDE developed by and for scientists who use Python. It is
  included by default when you install anaconda and can be started from a
  terminal with the command ``spyder``. If you installed MNE-Python into a
  separate environment from the ``base`` anaconda environment, you can add
  Spyder to your MNE-Python environment by running ``conda install spyder``
  while your MNE-Python environment is active. Spyder is free and open-source.
- `Visual Studio Code`_ (often shortened to "vscode") is a development-focused
  text editor that supports many programming languages in addition to Python,
  and has a rich ecosystem of packages to extend its capabilities. Installing
  `Microsoft's Python Extension
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__ is
  enough to get most Python users up and running. Visual Studio Code is free
  and open-source.
- `Atom`_ is a text editor similar to vscode, with a package ecosystem that
  includes a `Python IDE package <https://atom.io/packages/ide-python>`__ as
  well as `several <https://atom.io/packages/atom-terminal>`__
  `packages <https://atom.io/packages/atom-terminal-panel>`__
  `for <https://atom.io/packages/terminal-plus>`__
  `integrated <https://atom.io/packages/platformio-ide-terminal>`__
  `terminals <https://atom.io/packages/term3>`__.
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

.. _`mayavi`: https://docs.enthought.com/mayavi/mayavi/
.. _`pyvista`: https://docs.pyvista.org/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
.. _`xvfb`: https://en.wikipedia.org/wiki/Xvfb
.. _`off-screen rendering`: https://docs.enthought.com/mayavi/mayavi/tips.html#off-screen-rendering
.. _`rendering with a virtual framebuffer`: https://docs.enthought.com/mayavi/mayavi/tips.html#rendering-using-the-virtual-framebuffer
.. _`ide`: https://en.wikipedia.org/wiki/Integrated_development_environment
.. _`spyder`: https://www.spyder-ide.org/
.. _`visual studio code`: https://code.visualstudio.com/
.. _`sublimetext`: https://www.sublimetext.com/
.. _`terminus`: https://packagecontrol.io/packages/Terminus
.. _`pycharm`: https://www.jetbrains.com/pycharm/
.. _`atom`: https://atom.io/
