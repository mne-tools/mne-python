.. -*- mode: rst -*-

The homepage of MNE with user documentation is located on:

http://martinos.org/mne

Getting the latest code
=========================

To get the latest code using git, simply type::

    git clone git://github.com/mne-tools/mne-python.git

If you don't have git installed, you can download a zip or tarball
of the latest code: http://github.com/mne-tools/mne-python/archives/master

Installing
==========

As any Python packages, to install mne-python, simply do::

    python setup.py install

in the source code directory.

You can also install the latest release with easy_install::

    easy_install -U mne

or with pip::

    pip install mne --upgrade

Workflow to contribute
=========================

To contribute to mne-python, first create an account on `github
<http://github.com/>`_. Once this is done, fork the `mne-python repository
<http://github.com/mne-tools/mne-python>`_ to have you own repository,
clone it using 'git clone' on the computers where you want to work. Make
your changes in your clone, push them to your github account, test them
on several computer, and when you are happy with them, send a pull
request to the main repository.

Dependencies
============

The required dependencies to build the software are python >= 2.5,
NumPy >= 1.4, SciPy >= 0.7.

To run the tests you will also need nose >= 0.10.
and the MNE sample dataset (will be downloaded automatically
when you run an example ... but be patient)

Mailing list
============

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis

Running the test suite
=========================

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.

Making a release and uploading it to PyPI
==================================================

This command is only run by project manager, to make a release, and
upload in to PyPI::

    python setup.py sdist bdist_egg register upload


Licensing
----------

mne-python is **BSD-licenced** (3 clause):

    This software is OSI Certified Open Source Software.
    OSI Certified is a certification mark of the Open Source Initiative.

    Copyright (c) 2011, Alexandre Gramfort
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of Alexandre Gramfort. nor the names of other mne
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
