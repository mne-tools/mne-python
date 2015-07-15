.. _getting_started:

Getting Started
===============

.. contents:: Contents
   :local:

.. XXX do a Getting for both C and Python

MNE is an academic software package that aims to provide data analysis
pipelines encompassing all phases of M/EEG data processing.
It consists of two subpackages which are fully integrated
and compatible: the original MNE-C (distributed as compiled C code)
and MNE-Python. A basic :ref:`ch_matlab` is also available mostly
to allow reading and write MNE files. For source localization
the software depends on anatomical MRI processing tools provided
by the `FreeSurfer`_ software.

.. _FreeSurfer: http://surfer.nmr.mgh.harvard.edu

Downloading and installing the Unix commands
--------------------------------------------

.. note::

    If you are working at the Martinos Center see :ref:`setup_martinos`
    for instructions to work with MNE and to access the Neuromag software.

We want to thank all MNE Software users at the Martinos Center and
in other institutions for their collaboration during the creation
of this software as well as for useful comments on the software
and its documentation.

The MNE Unix commands can be downloaded at:

* `Download <http://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php>`_ MNE

:ref:`commands_list` gives an overview of the command line
tools provided with MNE.

.. _user_environment:

Setting up MNE Unix commands environment
########################################

The system-dependent location of the MNE Software will be
here referred to by the environment variable MNE_ROOT. There are
two scripts for setting up user environment so that the software
can be used conveniently:

``$MNE_ROOT/bin/mne_setup_sh``

and

``$MNE_ROOT/bin/mne_setup``

compatible with the POSIX and csh/tcsh shells, respectively. Since
the scripts set environment variables they should be 'sourced' to
the present shell. You can find which type of a shell you are using
by saying

``echo $SHELL``

If the output indicates a POSIX shell (bash or sh) you should issue
the three commands:

``export MNE_ROOT=`` <*MNE*> ``export MATLAB_ROOT=`` <*Matlab*> ``. $MNE_ROOT/bin/mne_setup_sh``

with <*MNE*> replaced
by the directory where you have installed the MNE software and <*Matlab*> is
the directory where Matlab is installed. If you do not have Matlab,
leave MATLAB_ROOT undefined. If Matlab is not available, the utilities
mne_convert_mne_data , mne_epochs2mat , mne_raw2mat ,
and mne_simu will not work.

For csh/tcsh the corresponding commands are:

``setenv MNE_ROOT`` <*MNE*> ``setenv MATLAB_ROOT`` <*Matlab*> ``source $MNE_ROOT/bin/mne_setup``

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
    | MNE_ROOT                | Location of the MNE software, see above.   |
    +-------------------------+--------------------------------------------+
    | FREESURFER_HOME         | Location of the FreeSurfer software.       |
    |                         | Needed during FreeSurfer reconstruction    |
    |                         | and if the FreeSurfer MRI viewer is used   |
    |                         | with mne_analyze, see :ref:`CACCHCBF`.     |
    +-------------------------+--------------------------------------------+
    | SUBJECTS_DIR            | Location of the MRI data.                  |
    +-------------------------+--------------------------------------------+
    | SUBJECT                 | Name of the current subject.               |
    +-------------------------+--------------------------------------------+
    | MNE_TRIGGER_CH_NAME     | Name of the trigger channel in raw data,   |
    |                         | see :ref:`BABBGJEA`.                       |
    +-------------------------+--------------------------------------------+
    | MNE_TRIGGER_CH_MASK     | Mask to be applied to the trigger channel  |
    |                         | values, see :ref:`BABBGJEA`.               |
    +-------------------------+--------------------------------------------+


Downloading and installing MNE-Python
-------------------------------------

.. note::

    If you are at the Martinos Center, please see this section :ref:`inside_martinos`.

New to the Python programming language?
#######################################

This is a very good place to get started: http://scipy-lectures.github.io.

Installing the Python interpreter
#################################

For a fast and up to date scientific Python environment that resolves all
dependencies, we recommend the Anaconda Python distribution:

https://store.continuum.io/cshop/anaconda/

Anaconda is free for academic purposes.

To test that everything works properly, open up IPython::

    ipython --pylab qt

Now that you have a working Python environment you can install MNE-Python.

mne-python installation
#######################

Most users should start with the "stable" version of mne-python, which can
be installed this way:

    pip install mne --upgrade

For the newest features (and potentially more bugs), you can instead install
the development version by:

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev

If you plan to contribute to the project, please follow the git instructions: 
:ref:`contributing`.

If you would like to use a custom installation of python (or have specific
questions about integrating special tools like IPython notebooks), please
see this section :ref:`detailed_notes`.

Checking your installation
##########################

To check that everything went fine, in ipython, type::

    >>> import mne

If you get a new prompt with no error messages, you should be good to go!
Consider reading the :ref:`detailed_notes` for more advanced options and
speed-related enhancements.

Going beyond
------------

Now you're ready to read our:

  * :ref:`tutorials`
  * :ref:`examples`
  * :ref:`manual`
