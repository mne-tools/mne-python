.. -*- mode: rst -*-

`mne-python <http://martinos.org/mne/mne-python.html>`_
=======================================================

This package is designed for sensor- and source-space analysis of M-EEG
data, including frequency-domain and time-frequency analyses and
non-parametric statistics. This package is presently evolving quickly and
thanks to the adopted open development environment user contributions can
be easily incorporated.

Get more information
^^^^^^^^^^^^^^^^^^^^

This page only contains bare-bones instructions for installing mne-python.

If you're familiar with MNE and you're looking for information on using
mne-python specifically, jump right to the `mne-python homepage
<http://martinos.org/mne/mne-python.html>`_. This website includes a
`tutorial <http://martinos.org/mne/python_tutorial.html>`_,
helpful `examples <http://martinos.org/mne/auto_examples/index.html>`_, and
a handy `function reference <http://martinos.org/mne/python_reference.html>`_,
among other things.

If you're unfamiliar with MNE, you can visit the
`MNE homepage <http://martinos.org/mne>`_ for full user documentation.

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git://github.com/mne-tools/mne-python.git

If you don't have git installed, you can download a zip or tarball
of the latest code: http://github.com/mne-tools/mne-python/archives/master

Install mne-python
^^^^^^^^^^^^^^^^^^

As any Python packages, to install MNE-Python, go in the mne-python source
code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

You can also install the latest release version with easy_install::

    easy_install -U mne

or with pip::

    pip install mne --upgrade

or for the latest development version (the most up to date)::

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are python >= 2.6,
NumPy >= 1.4, SciPy >= 0.7.2 and matplotlib >= 0.98.4.

Some isolated functions require pandas >= 7.3 and nitime (multitaper analysis).

To run the tests you will also need nose >= 0.10.
and the MNE sample dataset (will be downloaded automatically
when you run an example ... but be patient)

Contribute to mne-python
^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the mne-python homepage:

http://martinos.org/mne/contributing.html

Mailing list
^^^^^^^^^^^^

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis

Running the test suite
^^^^^^^^^^^^^^^^^^^^^^

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.

Making a release and uploading it to PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This command is only run by project manager, to make a release, and
upload in to PyPI::

    python setup.py sdist bdist_egg register upload


Licensing
^^^^^^^^^

MNE-Python is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011, authors of MNE-Python
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the names of MNE-Python authors nor the names of any
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    **This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness for
    a particular purpose are disclaimed. In no event shall the copyright
    owner or contributors be liable for any direct, indirect, incidental,
    special, exemplary, or consequential damages (including, but not
    limited to, procurement of substitute goods or services; loss of use,
    data, or profits; or business interruption) however caused and on any
    theory of liability, whether in contract, strict liability, or tort
    (including negligence or otherwise) arising in any way out of the use
    of this software, even if advised of the possibility of such
    damage.**
