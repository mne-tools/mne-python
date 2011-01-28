.. -*- mode: rst -*-

About
=====

MNE is a python module for processing MEG and EEG data.

It is a project initiated:

Athinoula A. Martinos Center for Biomedical Imaging
Massachusetts General Hospital
Charlestown, MA, USA

Available under the BSD (3-clause) license.

It is mainly a reimplementation of the Matlab code written by Matti Hämäläinen.

Download
========

TODO

Dependencies
============

The required dependencies to build the software are python >= 2.5,
NumPy >= 1.1, SciPy >= 0.6.

To run the tests you will also need nose >= 0.10.

Install
=======

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install


Mailing list
============

None

Development
===========

Code
----

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com:mne-tools/mne-python.git

or if you have write privileges::

    git clone git@github.com:mne-tools/mne-python.git

Bugs
----

Please report bugs you might encounter to:
gramfort@nmr.mgh.harvard.edu

Testing
-------

You can launch the test suite using nosetests from the source folder.

You'll need before to have run the MNE tutorial to have the required files
on your drive.
