.. include:: ../links.inc

.. _install_mne_c:

Installing MNE-C
================

.. contents::
   :local:
   :depth: 1

System requirements
^^^^^^^^^^^^^^^^^^^

MNE-C runs on macOS (version 10.5 "Leopard" or later) and Linux (kernel 2.6.9
or later). Both 32- and 64-bit operating systems are supported; a PowerPC
version for macOS can be provided upon request. At least 2 GB of memory is
required, 4 GB or more is recommended. The software requires at least 80 MB of
disk space. MATLAB is an optional dependency; the free `MATLAB runtime`_ is
sufficient. If MATLAB is not present, the utilities ``mne_convert_mne_data``,
``mne_epochs2mat``, ``mne_raw2mat``, and ``mne_simu`` will not work.

For boundary-element model (BEM) mesh generation, and for accessing the ``tkmedit``
program from ``mne_analyze``, MNE-C needs access to a
working installation of :doc:`FreeSurfer <freesurfer>`, including the
environment variables ``FREESURFER_HOME``, ``SUBJECTS_DIR``, and ``SUBJECT``.

.. admonition:: |apple| macOS
  :class: note

  For installation on macOS, you also need:

  - the `XCode developer tools`_.
  - an X Window System such as XQuartz_. Version 2.7.9 of XQuartz should work
    out of the box; the most current version (2.7.11, as of May 2019) may
    require these additional steps to work:

    .. code-block:: console

        $ cd /opt/X11/lib
        $ sudo cp libXt.6.dylib libXt.6.dylib.bak
        $ cd flat_namespace/
        $ sudo cp libXt.6.dylib ../.

  - the netpbm_ library. The recommended way to get netpbm is to install
    Homebrew_, and run ``brew install netpbm`` in the Terminal app.
    Alternatively, if you prefer to use MacPorts_, you can run
    ``sudo port install netpbm`` in the Terminal app.


Downloading and Installing MNE-C
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MNE-C is distributed as either a compressed tar archive (.tar.gz) or a macOS
disk image (.dmg). The `MNE-C download page`_ requires registration with a
valid email address.  The current stable version is 2.7.3; "nightly" builds of
the development version are also available on the download page.

To install from the compressed tar archive, change directory to the desired
install location, and unpack the software using ``tar``:

.. code-block:: console

    $ cd <path_to_desired_install_location>
    $ tar zxvf <path_to_archive_file>

To install from the macOS disk image, double-click the downloaded .dmg file. In
the window that opens, double-click the installer package file (.pkg) to launch
the installer, and follow its instructions. In newer versions of macOS, if you
see an error that the app is from an untrusted developer, you can override this
warning by opening it anyway from the Security & Privacy pane within the
computer's System Preferences.

.. _user_environment:

Configuring MNE-C
^^^^^^^^^^^^^^^^^

MNE-C requires two environment variables to be defined manually:

- ``MNE_ROOT`` should give the path to the folder where MNE-C is installed
- ``MATLAB_ROOT`` should give the path to your MATLAB binary (e.g.,
  ``/opt/MATLAB/R2018b`` or similar).  If you do not have MATLAB or the MATLAB
  runtime, leave ``MATLAB_ROOT`` undefined.

Other environment variables are defined by setup scripts provided with MNE-C.
You may either run the setup script each time you use MNE-C, or (recommended)
configure your shell to run it automatically each time you open a terminal. For
bash compatible shells, e.g., sh/bash/zsh, the script to source is
``$MNE_ROOT/bin/mne_setup_sh``.  For C shells, e.g., csh/tcsh, the script to
source is ``$MNE_ROOT/bin/mne_setup``.  If you don't know what shell you are
using, you can run the following command to find out:

.. code-block:: console

    $ echo $SHELL

To configure MNE-C automatically for ``bash`` or ``sh`` shells, add this to
your ``.bashrc``:

.. code-block:: sh

    export MNE_ROOT=<path_to_MNE>
    export MATLAB_ROOT=<path_to_MATLAB>
    source $MNE_ROOT/bin/mne_setup_sh

where ``<path_to_MNE>`` and ``<path_to_MATLAB>`` are replaced by the absolute
paths to MNE-C and MATLAB, respectively. If you don't have MATLAB, you should
still include the ``export MATLAB_ROOT=`` statement, but leave
``<path_to_MATLAB>`` blank.

To configure MNE-C automatically for ``zsh``, use the built-in ``emulate``
command in your ``.zshrc`` file:

.. code-block:: sh

    export MNE_ROOT=<path_to_MNE>
    export MATLAB_ROOT=<path_to_MATLAB>
    emulate sh -c 'source $MNE_ROOT/bin/mne_setup_sh'

To configure MNE-C automatically for ``csh`` or ``tcsh`` shells, the
corresponding commands in the ``.cshrc`` / ``.tcshrc`` file are:

.. code-block:: tcsh

    setenv MNE_ROOT <path_to_MNE>
    setenv MATLAB_ROOT <path_to_MATLAB>
    source $MNE_ROOT/bin/mne_setup

If you have done this correctly, the command ``ls $MNE_ROOT/bin/mne_setup_sh``
should succeed when run in a new terminal.

Testing MNE-C installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

An easy way to verify whether your installation of MNE-C is working is to test
the OpenGL graphics performance:

.. code-block:: console

    $ $MNE_ROOT/bin/mne_opengl_test

This will render an inflated brain surface repeatedly, rotating it by 5 degrees
around the z-axis between redraws. The time spent for each full revolution is
printed to the terminal window where ``mne_opengl_test`` was invoked.  Switch
focus to that terminal window and use the interrupt key (usually control-c) to
halt the test.

The best graphics performance occurs when MNE-C renders to a local display on a
computer with hardware acceleration enabled. The ``mne_analyze`` GUI has a menu
item "On GLX..." in the Help menu; if the GLX dialog says "Direct rendering
context" then hardware acceleration is in use. If you are rendering to a local
display and see "Nondirect rendering context", it is recommended that you
enable hardware acceleration (consult a search engine or your local IT support
staff for assistance). If you are rendering to a remote display or using a VNC
connection, "Nondirect rendering context" is normal.

On the fastest graphics cards, the time per revolution in the
``mne_opengl_test`` is well below 1 second. If your time per revolution is
longer than 10 seconds, either the graphics hardware acceleration is not in
effect or you need a faster graphics adapter.

Troubleshooting MNE-C installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If MNE-C can't find ``libxp.so.6``, download libxp6 from ubuntu_ or debian_ and
install with ``dpkg`` or similar:

.. code-block:: console

    $ sudo dpkg -i libxp6_1.0.2-1ubuntu1_amd64.deb

If MNE-C can't find ``libgfortran.so.1``, you can probably safely link that
filename to the current version of libfortran that came with your system. On
a typical 64-bit Ubuntu-like system this would be accomplished by:

.. code-block:: console

    $ cd /usr/lib/x86_64-linux-gnu
    $ sudo ln -s libgfortran.so.1 $(find . -maxdepth 1 -type f -name libgfortran.so*)

If you encounter other errors installing MNE-C, please send a message to the
`MNE mailing list`_.

.. links

.. _MNE-C download page: http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php
.. _MATLAB runtime: https://www.mathworks.com/products/compiler/matlab-runtime.html
.. _netpbm: http://netpbm.sourceforge.net/
.. _MacPorts: https://www.macports.org/
.. _Homebrew: https://brew.sh/
.. _XCode developer tools: http://appstore.com/mac/apple/xcode
.. _xquartz: https://www.xquartz.org/
.. _ubuntu: https://packages.ubuntu.com/search?keywords=libxp6
.. _debian: https://packages.debian.org/search?keywords=libxp6
