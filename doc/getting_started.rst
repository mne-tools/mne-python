.. include:: links.inc

.. _getting_started:

Getting started
===============

.. contents:: Contents
   :local:
   :depth: 2

MNE is an academic software package that aims to provide data analysis
pipelines encompassing all phases of M/EEG data processing.
It consists of two subpackages which are fully integrated
and compatible: the original MNE-C (distributed as compiled C code)
and MNE-Python. A basic :ref:`ch_matlab` is also available mostly
to allow reading and write MNE files. For source localization
the software depends on anatomical MRI processing tools provided
by the FreeSurfer_ software.

Install Python and MNE-Python
-----------------------------

To use MNE-Python, you need two things:

1. :ref:`A working Python interpreter and dependencies<install_interpreter>`

2. :ref:`MNE-Python installed to the Python distribution <install_mne_python>`

.. note:: Users who work at a facility with a site-wide install of
          MNE-Python (e.g. Martinos center) are encouraged to contact
          their technical staff about how to access and use MNE-Python,
          as the instructions might differ from those below.


.. _install_interpreter:

1. Install a Python interpreter and dependencies
################################################

There are multiple options available for getting a suitable Python interpreter
running on your system. However, for a fast and up to date scientific Python
environment that resolves all dependencies, we highly recommend following
installation instructions for the Anaconda Python distribution. Go here
to download, and follow the installation instructions:

* https://www.continuum.io/downloads
* http://docs.continuum.io/anaconda/install

If everything is set up correctly, you should be able to check the version
of ``conda`` that is installed (your version number will probably be newer)
and which ``python`` will be called when you run ``python``:

.. code-block:: bash

    $ conda --version
    conda 3.19.1
    $ which python
    /home/agramfort/anaconda/bin/python

If your installation doesn't look something like this,
*something went wrong* and you should try to fix it. We recommend looking
through the Anaconda documentation a bit, and Googling for Anaconda
install tips (StackExchange results are often helpful).

Once Anaconda works properly, you can do this to resolve
the MNE-Python dependencies:

.. code-block:: bash

    $ conda install scipy matplotlib scikit-learn

To test that everything works properly, open up IPython:

.. code-block:: bash

    $ ipython --matplotlib=qt

Now that you have a working Python environment you can install MNE-Python.

If you want to have an environment with a clean MATLAB-like interface,
consider using Spyder_, which can easily be installed with Anaconda
as:

.. code-block:: bash

    $ conda install spyder

.. _install_mne_python:

2. Install the MNE-Python package
#################################

There are a many options for installing MNE-Python, but two of the most
useful and common are:

1. **Use the stable release version of MNE-Python.** It can be installed as:

   .. code-block:: bash

       $ pip install mne --upgrade

   MNE-Python tends to release about once every six months, and this
   command can be used to update the install after each release.

.. _installing_master:

2. **Use the development master version of MNE-Python.** If you want to
   be able to update your MNE-Python version between releases (e.g., for
   bugfixes or new features), this will set you up for frequent updates:

   .. code-block:: bash

       $ git clone git://github.com/mne-tools/mne-python.git
       $ cd mne-python
       $ python setup.py develop

   A cool feature of ``python setup.py develop`` is that any changes made to
   the files (e.g., by updating to latest ``master``) will be reflected in
   ``mne`` as soon as you restart your Python interpreter. So to update to
   the latest version of the ``master`` development branch, you can do:

   .. code-block:: bash

       $ git pull origin master

   and your MNE-Python will be updated to have the latest changes.
   If you plan to contribute to MNE-Python, you should follow a variant
   of this approach outlined in the
   :ref:`contribution instructions <contributing>`. 

3. Check your installation
##########################

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!

4. Familiarize yourself with Python
###################################

Here are a few great resources:

* `SciPy lectures <http://scipy-lectures.github.io>`_
* `Learn X in Y minutes: Python <https://learnxinyminutes.com/docs/python/>`_
* `NumPy for MATLAB users <https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html>`_

We highly recommend watching the Scipy videos and reading through these
sites to get a sense of how scientific computing is done in Python.

5. Run some MNE-Python examples
###############################

Once MNE-python is set up, you should try running:

1. :ref:`Beginning tutorials <tutorials>`
2. `More advanced examples <auto_examples/index.html>`_

Along the way, make frequent use :ref:`api_reference` and
:ref:`manual` to understand the capabilities of MNE.

6. Optional advanced setup
##########################

Intel MKL
^^^^^^^^^

Intel's `Math Kernel Library <https://software.intel.com/en-us/intel-mkl>`_
speeds up many linear algebra comptutations, some up to 10x.

With Anaconda Python, you can `sign up for a subscription free for academic use <https://www.continuum.io/anaconda-academic-subscriptions-available>`_
that allows installation of the `accelerate <http://docs.continuum.io/accelerate/index>`_
package, which enables use of MKL. After getting a subscription set up and installing
the license file, you can use:

.. code-block:: bash

    $ conda install accelerate

and your NumPy_, SciPy_, `scikit-learn`_, and therefore ``mne``
should all work faster.

CUDA
^^^^

We have developed specialized routines to make use of
`NVIDIA CUDA GPU processing <http://www.nvidia.com/object/cuda_home_new.html>`_
to speed up some operations (e.g. FIR filtering) by up to 10x. 
If you want to use NVIDIA CUDA, you should install:

1. `the NVIDIA toolkit on your system <https://developer.nvidia.com/cuda-downloads>`_
2. `PyCUDA <http://wiki.tiker.net/PyCuda/Installation/>`_
3. `skcuda <https://github.com/lebedov/scikits.cuda>`_

For example, on Ubutnu 15.10, a combination of system packages and ``git``
packages can be used to install the CUDA stack:

.. code-block:: bash

    # install system packages for CUDA
    $ sudo apt-get install nvidia-cuda-dev nvidia-modprobe
    # install PyCUDA
    $ git clone http://git.tiker.net/trees/pycuda.git
    $ cd pycuda
    $ ./configure.py --cuda-enable-gl
    $ git submodule update --init
    $ make -j 4
    $ python setup.py install
    # install skcuda
    $ cd ..
    $ git clone https://github.com/lebedov/scikit-cuda.git
    $ cd scikit-cuda
    $ python setup.py install

To initialize mne-python cuda support, after installing these dependencies
and running their associated unit tests (to ensure your installation is correct)
you can run:

.. code-block:: bash

    $ MNE_USE_CUDA=true MNE_LOGGING_LEVEL=info python -c "import mne; mne.cuda.init_cuda()"
    Enabling CUDA with 1.55 GB available memory

If you have everything installed correctly, you should see an INFO-level log
message telling you your CUDA hardware's available memory. To have CUDA
initialized on startup, you can do::

    >>> mne.utils.set_config('MNE_USE_CUDA', 'true') # doctest: +SKIP

You can test if MNE CUDA support is working by running the associated test:

.. code-block:: bash

    $ nosetests mne/tests/test_filter.py

If ``MNE_USE_CUDA=true`` and all tests pass with none skipped, then
MNE-Python CUDA support works.

IPython (and notebooks)
^^^^^^^^^^^^^^^^^^^^^^^

In IPython, we strongly recommend using the Qt matplotlib backend for
fast and correct rendering:

.. code-block:: bash

    $ ipython --matplotlib=qt

On Linux, for example, QT is the only matplotlib backend for which 3D rendering
will work correctly. On Mac OS X for other backends certain matplotlib
functions might not work as expected.

To take full advantage of MNE-Python's visualization capacities in combination
with IPython notebooks and inline displaying, please explicitly add the
following magic method invocation to your notebook or configure your notebook
runtime accordingly::

    >>> %matplotlib inline

If you use another Python setup and you encounter some difficulties please
report them on the MNE mailing list or on github to get assistance.

Mayavi and PySurfer
^^^^^^^^^^^^^^^^^^^

Mayavi is currently only available for Python2.7.
The easiest way to install `mayavi` is to do the following with Anaconda:

.. code-block:: bash

    $ conda install mayavi

For other methods of installation, please consult the
`Mayavi documentation <http://docs.enthought.com/mayavi/mayavi/installation.html>`_.

The PySurfer package, which is used for visualizing cortical source estimates,
uses Mayavi and can be installed using:

.. code-block:: bash

    $ pip install PySurfer

Some users may need to configure PySurfer before they can make full use of
our visualization capabilities. Please refer to the
`PySurfer installation page <https://pysurfer.github.io/install.html>`_
for up to date information.


Install MNE-C
-------------

Some advanced functionality is provided by the MNE-C command-line tools.
It is not strictly necessary to have the MNE-C tools installed to use
MNE-Python, but it can be helpful.

The MNE Unix commands can be downloaded at:

* `Download MNE <http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php>`_

:ref:`c_reference` gives an overview of the command line
tools provided with MNE.

System requirements
###################

The MNE Unix commands runs on Mac OSX and LINUX operating systems.
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

The file names follow the convention:

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

Setting up MNE Unix commands environment
########################################

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
    $ $MNE_ROOT/bin/mne_setup_sh

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

Additional software
###################

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
###############################################

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

.. code-block:: bash

    $ $MNE_ROOT/bin/mne_opengl_test

On the fastest graphics cards, the time per revolution is
well below 1 second. If this time longer than 10 seconds either
the graphics hardware acceleration is not in effect or you need
a faster graphics adapter.

Obtain FreeSurfer
#################

The MNE software relies on the FreeSurfer software for cortical
surface reconstruction and other MRI-related tasks. Please consult the
FreeSurfer_ home page.
