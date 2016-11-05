.. -*- mode: rst -*-


|Travis|_ |Appveyor|_ |Circle|_ |Codecov|_ |Zenodo|_

|MNE|_

.. |Travis| image:: https://api.travis-ci.org/mne-tools/mne-python.png?branch=master
.. _Travis: https://travis-ci.org/mne-tools/mne-python

.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/reccwk3filrasumg/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/Eric89GXL/mne-python/branch/master

.. |Circle| image:: https://circleci.com/gh/mne-tools/mne-python.svg?style=svg
.. _Circle: https://circleci.com/gh/mne-tools/mne-python

.. |Codecov| image:: https://codecov.io/gh/mne-tools/mne-python/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/mne-tools/mne-python

.. |Zenodo| image:: https://zenodo.org/badge/5822/mne-tools/mne-python.svg
.. _Zenodo: https://zenodo.org/badge/latestdoi/5822/mne-tools/mne-python

.. |MNE| image:: http://mne-tools.github.io/dev/_static/mne_logo.png
.. _MNE: https://mne-tools.github.io

`MNE-Python <http://mne-tools.github.io/>`_
=======================================================

This package is designed for sensor- and source-space analysis of [M/E]EG
data, including frequency-domain and time-frequency analyses, MVPA/decoding
and non-parametric statistics. This package generally evolves quickly and
user contributions can easily be incorporated thanks to the open
development environment .

Get more information
^^^^^^^^^^^^^^^^^^^^

If you're unfamiliar with MNE or MNE-Python, you can visit the
`MNE homepage <http://mne-tools.github.io/>`_ for full user documentation.

Get the latest code
^^^^^^^^^^^^^^^^^^^

To get the latest code using `git <https://git-scm.com/>`_, simply type:

.. code-block:: bash

    $ git clone git://github.com/mne-tools/mne-python.git

If you don't have git installed, you can download a
`zip of the latest code <https://github.com/mne-tools/mne-python/archive/master.zip>`_.

Install mne-python
^^^^^^^^^^^^^^^^^^

As with most Python packages, to install the latest stable version of
MNE-Python, you can do:

.. code-block:: bash

    $ pip install mne

For more complete instructions and more advanced install methods (e.g. for
the latest development version), see the
`getting started page <http://mne-tools.github.io/stable/getting_started.html>`_
page.

Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run the software are:

  - Python >= 2.7
  - NumPy >= 1.8
  - SciPy >= 0.12
  - matplotlib >= 1.3

For full functionality, some functions require:

  - scikit-learn >= 0.18
  - nibabel >= 2.1.0
  - pandas >= 0.12

To use NVIDIA CUDA for resampling and FFT FIR filtering, you will also need
to install the NVIDIA CUDA SDK, pycuda, and scikits.cuda. See the
`getting started page <http://mne-tools.github.io/stable/getting_started.html>`_
for more information.

Contribute to mne-python
^^^^^^^^^^^^^^^^^^^^^^^^

Please see the documentation on the mne-python homepage:

http://martinos.org/mne/contributing.html

Mailing list
^^^^^^^^^^^^

http://mail.nmr.mgh.harvard.edu/mailman/listinfo/mne_analysis

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
