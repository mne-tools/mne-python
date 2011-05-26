.. -*- mode: rst -*-

About
=====

MNE is a Python module for processing MEG and EEG data.

It is a project initiated:

Athinoula A. Martinos Center for Biomedical Imaging
Massachusetts General Hospital
Charlestown, MA, USA

Available under the very permissive BSD (3-clause) license.

Even if this code is primarily developed at the Martinos Center,
the purpose of opening it is to welcome contributions and feedback
from users. This code is not the property of MGH / Martinos Center,
so feel free to use it, modify it and add your name to this project.

Download
========

Just click on the *Downloads* button at https://github.com/mne-tools/mne-python

Dependencies
============

The required dependencies to build the software are python >= 2.5,
NumPy >= 1.4, SciPy >= 0.7.

To run the tests you will also need nose >= 0.10.
and the MNE sample dataset (will be downloaded automatically
when you run an example ... but be patient)

Install
=======

This packages uses distutils, which is the default way of installing
python modules. The install command is::

  python setup.py install


Mailing list
============

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis

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

