Installing FreeSurfer
=====================

`FreeSurfer <fs-wiki_>`_ is software for analysis and visualization of MRI data.
In the MNE ecosystem, freesurfer is used to convert structural MRI scans into
models of the scalp, inner/outer skull, and cortical surfaces, which are used
to

1. model how changes in the electrical and magnetic field caused by neural
   activity propagate to the sensor locations (part of computing the "forward
   solution"), and

2. constrain the estimates of where brain activity may have occurred (in the
   "inverse imaging" step of source localization).

System requirements, setup instructions, and test scripts are provided on the
`FreeSurfer download page`_. Note that if you don't already have it, you will
need to install ``tcsh`` for FreeSurfer to work; ``tcsh`` is usually
pre-installed with macOS, and is available in the package repositories for
Linux-based systems (e.g., ``sudo apt install tcsh`` on Ubuntu-like systems).

**Next:** :doc:`advanced`

.. LINKS

.. _fs-wiki: https://surfer.nmr.mgh.harvard.edu/fswiki/
.. _`FreeSurfer download page`: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
