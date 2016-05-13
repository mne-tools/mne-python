.. -*- mode: rst -*-


|Travis|_ |Appveyor|_ |Coveralls|_ |Zenodo|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-python.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-python

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/reccwk3filrasumg/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/Eric89GXL/mne-python/branch/master

.. |Coveralls| image:: https://coveralls.io/repos/mne-tools/mne-python/badge.png?branch=master
.. _Coveralls: https://coveralls.io/r/mne-tools/mne-python?branch=master

.. |Zenodo| image:: https://zenodo.org/badge/5822/mne-tools/mne-python.svg
.. _Zenodo: https://zenodo.org/badge/latestdoi/5822/mne-tools/mne-python

`mne-python <http://mne-tools.github.io/>`_
=======================================================

This package is designed for sensor- and source-space analysis of [M/E]EG
data, including frequency-domain and time-frequency analyses, MVPA/decoding
and non-parametric statistics. This package is presently evolving quickly and
thanks to the adopted open development environment user contributions can
be easily incorporated.

Get more information
^^^^^^^^^^^^^^^^^^^^

This page only contains bare-bones instructions for installing mne-python.

If you're familiar with MNE and you're looking for information on using
mne-python specifically, jump right to the `mne-python homepage
<http://mne-tools.github.io/stable/python_reference.html>`_. This website includes
`tutorials <http://mne-tools.github.io/stable/tutorials.html>`_,
helpful `examples <http://mne-tools.github.io/stable/auto_examples/index.html>`_, and
a handy `function reference <http://mne-tools.github.io/stable/python_reference.html>`_,
among other things.

If you're unfamiliar with MNE, you can visit the
`MNE homepage <http://martinos.org/mne>`_ for full user documentation.

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using git, simply type::

    git clone git://github.com/mne-tools/mne-python.git

If you don't have git installed, you can download a zip
of the latest code: https://github.com/mne-tools/mne-python/archive/master.zip

Install mne-python
^^^^^^^^^^^^^^^^^^

As any Python packages, to install MNE-Python, after obtaining the source code
(e.g. from git), go in the mne-python source code directory and do::

    python setup.py install

or if you don't have admin access to your python setup (permission denied
when install) use::

    python setup.py install --user

You can also install the latest release version with easy_install::

    easy_install -U mne

or with pip::

    pip install mne
    
for an update of an already installed version use::

    pip install mne --upgrade

or for the latest development version (the most up to date)::

    pip install -e git+https://github.com/mne-tools/mne-python#egg=mne-dev --user

Dependencies
^^^^^^^^^^^^

The required dependencies to build the software are python >= 2.6,
NumPy >= 1.6, SciPy >= 0.7.2 and matplotlib >= 0.98.4.

Some isolated functions require pandas >= 0.7.3.
Decoding relies on scikit-learn >= 0.15.

To run the tests you will also need nose >= 0.10.
and the MNE sample dataset (will be downloaded automatically
when you run an example ... but be patient).

To use NVIDIA CUDA for resampling and FFT FIR filtering, you will also need
to install the NVIDIA CUDA SDK, pycuda, and scikits.cuda. The difficulty of this
varies by platform; consider reading the following site for help getting pycuda
to work (typically the most difficult to configure):

http://wiki.tiker.net/PyCuda/Installation/

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
