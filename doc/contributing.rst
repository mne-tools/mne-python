:orphan:

.. include:: links.inc

.. _contribute_to_mne:

How to contribute to MNE
========================

.. contents:: Contents
   :local:
   :depth: 1

We are open to many types of contributions, from bugfixes to functionality
enhancements. mne-python_ is meant to be maintained by a community of labs,
and as such, we seek enhancements that will likely benefit a large proportion
of the users who use the package.

*Before starting new code*, we highly recommend opening an issue on
`mne-python GitHub`_ to discuss potential changes. Getting on the same
page as the maintainers about changes or enhancements before too much
coding is done saves everyone time and effort!

Code guidelines
---------------

* Standard python style guidelines set by pep8_ and pyflakes_ are followed
  with very few exceptions. We recommend using an editor that calls out
  style violations automatcally, such as Spyder_. From the MNE code root, you
  can check for violations using flake8_ with:

  .. code-block:: bash

     $ make flake

* Use `numpy style`_ for docstrings. Follow existing examples for simplest
  guidance.

* New functionality must be covered by tests. For example, a
  :class:`mne.Evoked` method in ``mne/evoked.py`` should have a corresponding
  test in ``mne/tests/test_evoked.py``.

* Changes must be accompanied by updated documentation, including
  :doc:`doc/whats_new.rst <whats_new>` and
  :doc:`doc/python_reference.rst <python_reference>`.

* After making changes, **ensure all tests pass**. This can be done
  by running:

  .. code-block:: bash

     $ make test

  To run individual tests, you can also run e.g.:

  .. code-block:: bash

     $ nosetests mne/tests/test_evoked:test_io_evoked -x --verbose

  Make sure you have the testing dataset, which you can get by doing::

     >>> mne.datasets.testing.data_path(verbose=True)  # doctest: +SKIP

MNE-specific coding guidelines
------------------------------

These are guidelines that are generally followed:

Pull requests
^^^^^^^^^^^^^
* Address one issue per pull request (PR).
* Avoid unnecessary cosmetic changes in PRs.
* Minimize test timing while maximizing coverage. Use ``nosetests --with-timer`` on modified tests.
* Update the ``doc/whats_new.rst`` file last, just before merge to avoid merge conflicts.

Naming
^^^^^^
* Classes should be named using CamelCase.
* Functions and instances/variables should be snake_case (n_samples rather than nsamples).
* Avoid single-character variable names.

Importing
^^^^^^^^^
* Import modules in this order:
  1. builtin
  2. standard scientific (``numpy as np``, ``scipy`` submodules)
  3. others
  4. mne imports (relative within the MNE module, absolute in the examples)
* Imports for ``matplotlib`` and optional modules (``sklearn``, and ``pandas``, etc.) within the MNE module should be nested (i.e., within a function or method, not at the top of a file).

Vizualization
^^^^^^^^^^^^^
* Add public functions to the :mod:`mne.viz` package and use these in the corresponding methods.
* All visualization functions must accept a ``show`` parameter and return a ``fig`` handle.
* Use ``RdBu_r`` colormap for signed data with a meaningful middle (zero-point) and ``Reds`` otherwise in visualization functions and examples.

Return types
^^^^^^^^^^^^
* Methods should modify inplace and return ``self``, functions should return copies (where applicable).

Style
^^^^^
* Use single quotes whenever possible.
* Prefer generator or list comprehensions over ``filter``, ``map`` and other functional idioms.
* Use explicit functional constructors for builtin containers to improve readability (e.g., ``list()``, ``dict``).
* Avoid nested functions or class methods if possible -- use private functions instead.
* Avoid ``**kwargs`` and ``*args`` in function signatures.
* Add brief docstrings to simple private functions and complete docstrings for complex ones.

Checking and building documentation
-----------------------------------

All changes to the codebase must be properly documented.
To ensure that documentation is rendered correctly, the best bet is to
follow the existing examples for class and function docstrings,
and examples and tutorials.

Our documentation (including docstring in code) uses ReStructuredText format,
see `Sphinx documentation`_ to learn more about editing them. Our code
follows the `NumPy docstring standard`_.

Documentation is automatically built remotely during pull requests. If
you want to also test documentation locally, you will need to install
``sphinx sphinx-gallery sphinx_bootstrap_theme numpydoc``, and then within
the ``mne/doc`` directory do:

.. code-block:: bash

   $ make html_dev-noplot

If you are working on examples or tutorials, you can build specific examples
with:

.. code-block:: bash

   $ PATTERN=plot_background_filtering.py make html_dev-pattern

Consult the `sphinx gallery documentation`_ for more details.

Deprecating
-----------
If you need to deprecate a function or a class, use the ``@deprecated`` decorator::

    from mne.utils import deprecated

    @deprecated('my_function will be deprecated in 0.XX, please use my_new_function instead.')
    def my_function():
       return 'foo'

If you need to deprecate a parameter, use the mne warning function.
For example to rename a parameter from `old_param` to `new_param` you can
use something like this::

    from mne.utils import warn

    def my_function(new_param, old_param=None):
        if old_param is not None:
             warn('old_param is deprecated and will be replaced by new_param in 0.XX.',
                  DeprecationWarning)
             new_param = old_param
        # Do what you have to do with new_param
        return 'foo'



Profiling
---------
To learn more about profiling python codes please see `the scikit learn profiling site <http://scikit-learn.org/stable/developers/performance.html#performance-howto>`_.

.. _troubleshooting:

Submitting changes
------------------
Changes to code can be submitted using a standard GitHub Pull Request, as
documented in :ref:`using_github`.
