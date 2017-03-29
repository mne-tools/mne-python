.. include:: links.inc

.. _install_mne_c:

Install MNE-C
-------------

The MNE-C commands can be downloaded
`here <http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php>`_.
The :ref:`c_reference` gives an overview of the MNE-C tools.

.. contents:: Contents
   :local:
   :depth: 1

System requirements and installation
####################################

The MNE-C runs on Mac OSX and LINUX, and requires:

- Mac OSX version 10.5 (Leopard) or later.
- LINUX kernel 2.6.9 or later
- On both LINUX and Mac OSX 32-bit and 64-bit Intel platforms
  are supported. PowerPC version on Mac OSX can be provided upon request.
- At least 2 GB of memory, 4 GB or more recommended.
- Disk space required for the MNE software: 80 MB
- Additional open source software on Mac OSX, see :ref:`BABDBCJE`.

The MNE software is distributed as a compressed tar archive
(Mac OSX and LINUX) or a Mac OSX disk image (dmg).

The file names follow the convention:

::

  MNE-* <*version*>*- <*rev*> -* <*Operating system*>*-* <*Processor*>*.* <*ext*>*

The present version number is 2.7.4. The <*rev*> field
is the SVN revision number at the time this package was created.
The <*Operating system*> field
is either Linux or MacOSX. The <*processor*> field
is either i386 or x86_64. The <*ext*> field
is 'gz' for compressed tar archive files and 'dmg' for
Mac OSX disk images.

Installing from a compressed tar archive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Go to the directory where you want the software to be installed:

.. code-block:: bash

    $ cd <dir>

Unpack the tar archive:

.. code-block:: bash

    $ tar zxvf <software package>

The name of the software directory under <*dir*> will
be the same as the package file without the .gz extension.

Installing from a Mac OSX disk image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Double click on the disk image file.
  A window opens with the installer package ( <*name*> .pkg)
  inside.

- Double click the the package file. The installer starts.

- Follow the instructions in the installer.

.. note::

    The software will be installed to /Applications/ <*name*> by default.
    If you want another location, select Choose Folder... on the Select a
    Destination screen in the installer.

.. _user_environment:

Setting up MNE-C environment
############################

The system-dependent location of the MNE Software will be
here referred to by the environment variable MNE_ROOT. There are
two scripts for setting up user environment so that the software
can be used conveniently:

.. code-block:: bash

    $ $MNE_ROOT/bin/mne_setup_sh

and

.. code-block:: bash

    $ $MNE_ROOT/bin/mne_setup

compatible with the POSIX and csh/tcsh shells, respectively. Since
the scripts set environment variables they should be 'sourced' to
the present shell. You can find which type of a shell you are using
by saying

.. code-block:: bash

    $ echo $SHELL

If the output indicates a POSIX shell (bash or sh) you should issue
the three commands:

.. code-block:: bash

    $ export MNE_ROOT=<MNE>
    $ export MATLAB_ROOT=<Matlab>
    $ . $MNE_ROOT/bin/mne_setup_sh

with ``<MNE>`` replaced
by the directory where you have installed the MNE software and ``<Matlab>`` is
the directory where Matlab is installed. If you do not have Matlab,
leave MATLAB_ROOT undefined. If Matlab is not available, the utilities
mne_convert_mne_data, mne_epochs2mat, mne_raw2mat, nd mne_simu will not work.

For csh/tcsh the corresponding commands are:

.. code-block:: csh

    $ setenv MNE_ROOT <MNE>
    $ setenv MATLAB_ROOT <Matlab>
    $ source $MNE_ROOT/bin/mne_setup

For BEM mesh generation using the watershed algorithm or
on the basis of multi-echo FLASH MRI data (see :ref:`create_bem_model`) and
for accessing the tkmedit program
from mne_analyze, see :ref:`CACCHCBF`,
the MNE software needs access to a FreeSurfer license
and software. Therefore, to use these features it is mandatory that
you set up the FreeSurfer environment
as described in the FreeSurfer documentation.

The environment variables relevant to the MNE software are
listed in :ref:`CIHDGFAA`.

.. tabularcolumns:: |p{0.3\linewidth}|p{0.55\linewidth}|
.. _CIHDGFAA:
.. table:: Environment variables

    +-------------------------+--------------------------------------------+
    | Name of the variable    |   Description                              |
    +=========================+============================================+
    | ``MNE_ROOT``            | Location of the MNE software, see above.   |
    +-------------------------+--------------------------------------------+
    | ``FREESURFER_HOME``     | Location of the FreeSurfer software.       |
    |                         | Needed during FreeSurfer reconstruction    |
    |                         | and if the FreeSurfer MRI viewer is used   |
    |                         | with mne_analyze, see :ref:`CACCHCBF`.     |
    +-------------------------+--------------------------------------------+
    | ``SUBJECTS_DIR``        | Location of the MRI data.                  |
    +-------------------------+--------------------------------------------+
    | ``SUBJECT``             | Name of the current subject.               |
    +-------------------------+--------------------------------------------+
    | ``MNE_TRIGGER_CH_NAME`` | Name of the trigger channel in raw data,   |
    |                         | see :ref:`mne_process_raw`.                |
    +-------------------------+--------------------------------------------+
    | ``MNE_TRIGGER_CH_MASK`` | Mask to be applied to the trigger channel  |
    |                         | values, see :ref:`mne_process_raw`.        |
    +-------------------------+--------------------------------------------+

.. _BABDBCJE:

Additional options
##################

Additional software
^^^^^^^^^^^^^^^^^^^

MNE uses the `Netpbm package <http://netpbm.sourceforge.net/>`_
to create image files in formats other than tif and rgb from
``mne_analyze`` and ``mne_browse_raw``.
This package is usually present on LINUX systems. On Mac OSX, you
need to install the netpbm package. The recommended way to do this
is to use the MacPorts Project tools, see http://www.macports.org/:

- If you have not installed the MacPorts
  software, goto http://www.macports.org/install.php and follow the
  instructions to install MacPorts.

- Install the netpbm package by saying: ``sudo port install netpbm``

MacPorts requires that you have the XCode developer tools
and X11 windowing environment installed. X11 is also needed by MNE.
For Mac OSX Leopard, we recommend using XQuartz (http://xquartz.macosforge.org/).
As of this writing, XQuartz does not yet exist for SnowLeopard;
the X11 included with the operating system is sufficient.

.. _CIHIIBDA:

Testing the performance of your OpenGL graphics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The graphics performance of mne_analyze depends
on your graphics software and hardware configuration. You get the
best performance if you are using mne_analyze locally
on a computer and the hardware acceleration capabilities are in
use. You can check the On GLX... item
in the help menu of mne_analyze to
see whether the hardware acceleration is in effect. If the dialog
popping up says Direct rendering context ,
you are using hardware acceleration. If this dialog indicates Nondirect rendering context , you are either using software
emulation locally, rendering to a remote display, or employing VNC
connection. If you are rendering to a local display and get an indication
of Nondirect rendering context ,
software emulation is in effect and you should contact your local
computer support to enable hardware acceleration for GLX. In some
cases, this may require acquiring a new graphics display card. Fortunately,
relatively high-performance OpenGL-capable graphics cards are not expensive.

There is also an utility mne_opengl_test to
assess the graphics performance more quantitatively. This utility
renders an inflated brain surface repeatedly, rotating it by 5 degrees
around the *z* axis between redraws. At each
revolution, the time spent for the full revolution is reported on
the terminal window where mne_opengl_test was
started from. The program renders the surface until the interrupt
key (usually control-c) is pressed on the terminal window.

mne_opengl_test is located
in the ``bin`` directory and is thus started as:

.. code-block:: bash

    $ $MNE_ROOT/bin/mne_opengl_test

On the fastest graphics cards, the time per revolution is
well below 1 second. If this time longer than 10 seconds either
the graphics hardware acceleration is not in effect or you need
a faster graphics adapter.
