.. -*- mode: rst -*-

|MNE|

MNE-Python
==========

MNE-Python is an open-source Python package for exploring,
visualizing, and analyzing human neurophysiological data such as MEG, EEG, sEEG,
ECoG, and more. It includes modules for data input/output, preprocessing,
visualization, source estimation, time-frequency analysis, connectivity analysis,
machine learning, statistics, and more.


Documentation
^^^^^^^^^^^^^

`Documentation`_ for MNE-Python encompasses installation instructions, tutorials,
and examples for a wide variety of topics, contributing guidelines, and an API
reference.


Forum
^^^^^^

The `user forum`_ is the best place to ask questions about MNE-Python usage or
the contribution process. The forum also features job opportunities and other
announcements.

If you find a bug or have an idea for a new feature that should be added to
MNE-Python, please use the
`issue tracker <https://github.com/mne-tools/mne-python/issues/new/choose>`__ of
our GitHub repository.


Installation
^^^^^^^^^^^^

To install the latest stable version of MNE-Python with minimal dependencies
only, use pip_ in a terminal:

.. code-block:: console

    $ pip install --upgrade mne

For more complete instructions, including our standalone installers and more
advanced installation methods, please refer to the `installation guide`_.


Get the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest development version of MNE-Python using pip_, open a
terminal and type:

.. code-block:: console

    $ pip install --upgrade https://github.com/mne-tools/mne-python/archive/refs/heads/main.zip

To clone the repository with `git <https://git-scm.com/>`__, open a terminal
and type:

.. code-block:: console

    $ git clone https://github.com/mne-tools/mne-python.git


Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

.. ↓↓↓ BEGIN CORE DEPS LIST. DO NOT EDIT! HANDLED BY PRE-COMMIT HOOK ↓↓↓

- `Python <https://www.python.org>`__ ≥ 3.10
- `NumPy <https://numpy.org>`__ ≥ 1.23
- `SciPy <https://scipy.org>`__ ≥ 1.9
- `Matplotlib <https://matplotlib.org>`__ ≥ 3.6
- `Pooch <https://www.fatiando.org/pooch/latest/>`__ ≥ 1.5
- `tqdm <https://tqdm.github.io>`__
- `Jinja2 <https://palletsprojects.com/p/jinja/>`__
- `decorator <https://github.com/micheles/decorator>`__
- `lazy-loader <https://pypi.org/project/lazy_loader>`__ ≥ 0.3
- `packaging <https://packaging.pypa.io/en/stable/>`__

.. ↑↑↑ END CORE DEPS LIST. DO NOT EDIT! HANDLED BY PRE-COMMIT HOOK ↑↑↑

Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines <https://mne.tools/dev/development/contributing.html>`__ on our documentation website.


About
^^^^^

+---------+------------+----------------+
| CI      | |Codecov|  | |Bandit|       |
+---------+------------+----------------+
| Package | |PyPI|     | |conda-forge|  |
+---------+------------+----------------+
| Docs    | |Docs|     | |Discourse|    |
+---------+------------+----------------+
| Meta    | |Zenodo|   | |OpenSSF|      |
+---------+------------+----------------+


License
^^^^^^^

MNE-Python is licensed under the BSD-3-Clause license.


.. _Documentation: https://mne.tools/dev/
.. _user forum: https://mne.discourse.group
.. _installation guide: https://mne.tools/dev/install/index.html
.. _pip: https://pip.pypa.io/en/stable/

.. |PyPI| image:: https://img.shields.io/pypi/dm/mne.svg?label=PyPI
   :target: https://pypi.org/project/mne/

.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/mne.svg?label=Conda
   :target: https://anaconda.org/conda-forge/mne

.. |Docs| image:: https://img.shields.io/badge/Docs-online-green?label=Documentation
   :target: https://mne.tools/dev/

.. |Zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.592483.svg
   :target: https://doi.org/10.5281/zenodo.592483

.. |Discourse| image:: https://img.shields.io/discourse/status?label=Forum&server=https%3A%2F%2Fmne.discourse.group%2F
   :target: https://mne.discourse.group/

.. |Codecov| image:: https://img.shields.io/codecov/c/github/mne-tools/mne-python?label=Coverage
   :target: https://codecov.io/gh/mne-tools/mne-python

.. |Bandit| image:: https://img.shields.io/badge/Security-Bandit-yellow.svg
   :target: https://github.com/PyCQA/bandit

.. |OpenSSF| image:: https://www.bestpractices.dev/projects/7783/badge
   :target: https://www.bestpractices.dev/projects/7783

.. |MNE| image:: https://mne.tools/dev/_static/mne_logo_gray.svg
   :target: https://mne.tools/dev/
