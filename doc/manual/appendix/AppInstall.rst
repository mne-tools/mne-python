

.. _install_config:

==============================
Installation and configuration
==============================

System requirements
###################

The MNE software runs on Mac OSX and LINUX operating systems.
The hardware and software requirements are:

- Mac OSX version 10.5 (Leopard) or later.

- LINUX kernel 2.6.9 or later

- On both LINUX and Mac OSX 32-bit and 64-bit Intel platforms
  are supported. PowerPC version on Mac OSX can be provided upon request.

- At least 2 GB of memory, 4 GB or more recommended.

- Disk space required for the MNE software: 80 MB

- Additional open source software on Mac OSX, see :ref:`BABDBCJE`.

Installation
############

The MNE software is distributed as a compressed tar archive
(Mac OSX and LINUX) or a Mac OSX disk image (dmg).

Download the software
=====================

Download the software package of interest. The file names
follow the convention:

MNE-* <*version*>*- <*rev*> -* <*Operating
system*>*-* <*Processor*>*.* <*ext*>*

The present version number is 2.7.0. The <*rev*> field
is the SVN revision number at the time this package was created.
The <*Operating system*> field
is either Linux or MacOSX. The <*processor*> field
is either i386 or x86_64. The <*ext*> field
is 'gz' for compressed tar archive files and 'dmg' for
Mac OSX disk images.

Installing from a compressed tar archive
========================================

Go to the directory where you want the software to be installed:

``cd`` <*dir*>

Unpack the tar archive:

``tar zxvf`` <*software package*>

The name of the software directory under <*dir*> will
be the same as the package file without the .gz extension.

Installing from a Mac OSX disk  image
=====================================

- Double click on the disk image file.
  A window opens with the installer package ( <*name*> .pkg)
  inside.

- Double click the the package file. The installer starts.

- Follow the instructions in the installer.

.. note:: The software will be installed to /Applications/ <*name*> by    default. If you want another location, select Choose Folder... on the Select a Destination screen    in the installer.

.. note:: To provide centralized support in an environment    with

.. _BABDBCJE:

Additional software
===================

MNE uses the 'Netpbm' package (http://netpbm.sourceforge.net/)
to create image files in formats other than tif and rgb from mne_analyze and mne_browse_raw .
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
===============================================

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
relatively high-performance OpenGL-capable graphics cards very inexpensive.

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

``$MNE_ROOT/bin/mne_opengl_test``

On the fastest graphics cards, the time per revolution is
well below 1 second. If this time longer than 10 seconds either
the graphics hardware acceleration is not in effect or you need
a faster graphics adapter.

Obtain FreeSurfer
#################

The MNE software relies on the FreeSurfer software for cortical
surface reconstruction and other MRI-related tasks. Please consult
the FreeSurfer home page site at ``http://surfer.nmr.mgh.harvard.edu/`` .

How to get started
##################

After you have installed the software, a good place to start
is to look at the manual:

- Source the correct setup script, see :ref:`user_environment`,
  and

- Say: ``mne_view_manual`` .

Chapters of interest for a novice user include:

- :ref:`getting_started` contains introduction
  to the software and setup instructions.

- :ref:`ch_cookbook` is an overview of the necessary steps to
  compute the cortically constrained minimum-norm solutions.

- :ref:`ch_sample_data` is a hands-on exercise demonstrating analysis
  of the sample data set.

- :ref:`ch_reading` contains a list of useful references for
  understanding the methods implemented in the MNE software.
