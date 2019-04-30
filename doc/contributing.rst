.. include:: links.inc

.. contents::
   :local:
   :depth: 2

.. _contributing:

.. include:: ../CONTRIBUTING.rst

.. TO MERGE
.. http://martinos.org/mne/stable/configure_git.html#using-github

.. _`dev-setup`:

Overview of contribution process
================================

In general you'll be working with three different copies of the MNE-Python
codebase: the official copy at https://github.com/mne-tools/mne-python (usually
called "upstream"), your `fork <github-help-fork>`_ of the upstream repository
(similar URL, but with your username in place of ``mne-tools``, and usually
called "origin"), and the local copy of the codebase on your computer. The
typical contribution process is to:

1. synchronize your local copy with the upstream

2. make changes to your local copy

3. `push <github-help-push>`_ your changes to origin (your remote fork of the
   upstream)

4. submit a `pull request`_ from your fork into the upstream

The section :ref:`github-workflow` (below) describes this process in more
detail.


Setting up your local environment for MNE-Python development
============================================================

Configuring git
^^^^^^^^^^^^^^^

To get set up for contributing, make sure you have git installed on your local
computer:

- On Linux, the command ``sudo apt install git`` is usually sufficient; see the
  `official download instructions <git-install-nix>`_ for more options.

- On MacOS, download `the .dmg installer here <git-macos-download>`_; Atlassian
  also offers `more detailed instructions and alternatives <git-install-mac>`_
  such as using MacPorts or Homebrew.

- On Windows, we recommend `git Bash`_ rather than the `official Windows
  version of git <git-windows-download>`_, because git Bash provides its own
  shell that includes many Linux-equivalent command line programs that are
  useful for development. `GitHub desktop`_ is a GUI alternative to command
  line git that some users appreciate. Windows 10 also offers the `Windows
  subsystem for Linux`_ that offers similar functionality to git Bash, but has
  not been widely tested by MNE-Python developers yet.

The only absolutely necessary configuration step is identifying yourself and
your contact info:

.. code-block:: console

   $ git config --global user.name "Your Name"
   $ git config --global user.email you@yourdomain.example.com

Make sure that the same email address is associated with your GitHub account
and with your local git configuration. It is possible to associate multiple
emails with a GitHub account, so if you initially set them up with different
emails, just add the local email to the GitHub account.

Before creating your local copy of the codebase, go to the `MNE-Python GitHub`_
page and create a fork into your GitHub user account.


Setting up the Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    We strongly recommend the `Anaconda`_ or `Miniconda`_ environment managers
    for Python. Other setups are possible but are not officially supported by
    the MNE-Python development team; see discussion `here <other-py-distros>`_.

You can set up your local development environment by starting with the
`environment file`_ provided in the root of the MNE-Python repository. The
following commands will create the conda environment:

.. code-block:: console

    $ # change these variables for your situation
    $ INSTALL_LOCATION="/opt"
    $ GITHUB_USERNAME="new_mne_contributor"
    $ PREFERRED_ENVIRONMENT_NAME="mnedev"
    $ # make a local copy of your fork ("origin")
    $ cd $INSTALL_LOCATION
    $ git clone https://github.com/$GITHUB_USERNAME/mne-python.git
    $ # setup a link to the official repository ("upstream")
    $ cd mne-python
    $ git remote add upstream git://github.com/mne-tools/mne-python.git
    $ git fetch --all
    $ # create the conda environment and activate it
    $ conda env create -n $PREFERRED_ENVIRONMENT_NAME -f environment.yml
    $ conda activate $PREFERRED_ENVIRONMENT_NAME

.. note::

    When using the environment file to install with Anaconda or Miniconda, the
    name of the environment (``mne``) is built into the environment file
    itself, but can be changed on the command line with the ``-n`` flag (as
    shown above). This is helpful when maintaining separate environments for
    stable and development versions of MNE-Python, or when using the
    environment file as a starting point for new projects. See ``conda env
    create --help`` for more info.

The environment file installs the stable version of MNE-Python, so next we'll
remove that and replace it with the clone you just created:

.. code-block:: console

    $ pip uninstall -y mne  # or: conda remove --force mne
    $ pip install -e .      # or: python setup.py develop

Next, we'll do the same thing for Mayavi (one of our 3D plotting backends):

.. code-block:: console

    $ pip uninstall -y mayavi
    $ cd $INSTALL_LOCATION
    $ git clone git://github.com/enthought/mayavi.git
    $ cd mayavi
    $ pip install -e .  # or: python setup.py develop

Finally, add a few dependencies that are not needed for running MNE-Python but
are needed for locally running our test suite or building our documentation:

.. code-block:: console

    $ pip install sphinx sphinx-gallery sphinx_bootstrap_theme sphinx_fontawesome memory_profiler
    $ conda install sphinx-autobuild doc8  # linter for reStructuredText

.. TODO: no longer needed? pip install sphinx-gallery should be fine...
    $ cd $INSTALL_LOCATION
    $ git clone git://github.com/sphinx-gallery/sphinx-gallery.git
    $ cd sphinx-gallery
    $ pip install -e .

.. TODO: add this if sphinx-mermaid is officially adopted
    Our documentation includes diagrams that are built automatically from text
    using a tool called `mermaid`_. Originally a javascript tool, there is now
    a command-line interface ``mermaid.cli`` as well as a plugin for ``sphinx``
    (our documentation build tool) that allows mermaid blocks in the
    documentation to be rendered into diagrams automatically.
    .
    $ wget -O - "https://dl.yarnpkg.com/debian/pubkey.gpg" | sudo apt-key add -
    $ echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list > /dev/null
    $ sudo apt update
    $ sudo apt install yarn
    Once yarn is installed, you can use it to install mermaid
    $ cd $INSTALL_LOCATION
    $ yarn add mermaid
    $ yarn add mermaid.cli
    $ pip install sphinxcontrib-mermaid

.. note::

    The commands ``pip install -e .`` and ``python setup.py develop`` both
    install a python module into the current environment by creating a link to
    the source code directory (instead of copying the code to pip's
    ``site_packages`` directory, which is what normally happens. This means
    that any edits you make to the MNE-Python source code will be reflected the
    next time you open a Python interpreter and ``import mne`` (the ``-e`` flag
    of ``pip`` stands for an "editable" installation).


Basic git command reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Learning to work with git can take a long time, because it is a complex and
powerful tool for managing versions of files across multiple users, each of
whom have multiple copies of the codebase. We've already seen in the setup
commands above a few of the basic git commands useful to an MNE-Python
developer:

.. code-block:: shell

    # make a local copy of a repository:
    git clone <URL_OF_REMOTE_REPO>

    # connect a local copy to an additional remote:
    git remote add <NICKNAME_OF_REMOTE> <URL_OF_REMOTE_REPO>

    # get the current state of connected remote repos:
    git fetch --all

.. note::

    You may have noticed that earlier we used ``git://`` instead of
    ``https://`` in the address for the official "upstream" remote repository,
    and in the addresses for mayavi and sphinx-gallery.
    ``git://`` addresses are read-only, which means you can *pull* the official
    repository into your local copy (in order to stay up-to-date with changes
    made by other contributors) but you cannot *push* anything from your
    computer directly into the upstream remote. Instead, you must push your
    changes to your own remote fork first, and then create a pull request from
    your remote into the upstream remote (and even then, your changes are not
    automatically accepted into the upstream; the changes must be *approved* by
    maintainers and then *merged* into upstream).

Other commands that you will undoubtedly need relate to `branches
<github-help-branch>`_. Branches represent multiple copies of the codebase
*within a local clone or remote repo*. Branches are typically used to
experiment with new features while still keeping a clean, working copy of the
original codebase that you can switch back to at any time. The default branch
of any repo is always called ``master``, and it is recommended that you reserve
the ``master`` branch to be that clean copy of the working "upstream" codebase.
Therefore, if you want to add a new feature, you should first create a new
branch based off of ``master`` and then `check it out <git checkout>`_:

.. code-block:: shell

    # see what state the local codebase is in
    git status

    # view all (local and remote) branches that git knows about
    git branch --all    # or git branch -a

    # copy the current branch with a new name, and checkout the new branch
    git checkout -b <NAME_OF_NEW_BRANCH>

Git knows that sometimes you work on multiple changes in multiple files all at
once, but that ideally you would separate those changes into related
"changesets" that are grouped together based on common goals. For example, you
might want to group all the code changes together, separately from unit tests
or changes to the documentation. One method that git provides for organizing
your work in that way is the *stage* (or *staging area*).

Git keeps track of some files and ignores others, depending on whether you have
`added <git add>`_ the files to the repository. You use the same command (``git
add``) to tell git to start tracking a new file, or to

    # add a new file

`GitHub help`_ website
`GitHub Learning Lab`_ tutorial series
`pro git book`_


Connecting to GitHub with SSH (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One easy way to speed up development is to reduce the number of times you have
to type your password. SSH (secure shell) allows authentication with pre-shared
key pairs. The private half of your key pair is kept secret on your computer,
while the public half of your key pair is added to your GitHub account; when
you connect to GitHub from your computer, the local git client checks the
remote (public) key against your local (private) key, and grants access your
account only if the keys fit. GitHub has `several help pages
<github-help-ssh>`_ that guide you through the process.

Once you have set up GitHub to use SSH authentication, you should change the
addresses of your MNE-Python GitHub remotes, from ``https://`` addresses to
``git@`` addresses, so that git knows to connect via SSH instead of HTTPS. For
example:

.. code-block:: console

    $ git remote -v  # show existing remote addresses
    $ git remote set-url origin git@github.com:$GITHUB_USERNAME/mne-python.git
    $ git remote set-url upstream git@github.com:mne-tools/mne-python.git


MNE-Python coding conventions
=============================

General requirements
^^^^^^^^^^^^^^^^^^^^

All new functionality must have test coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, a new :class:`mne.Evoked` method in :doc:`mne/evoked.py` should
have a corresponding test in :doc:`mne/tests/test_evoked.py`.


All new functionality must be documented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This includes thorough docstring descriptions for all public API changes, as
well as how-to examples or longer tutorials for major contributions. Docstrings
for private functions may be more sparse, but should not be omitted.


New API elements should be added to the master reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes, functions, methods, and attributes cannot be cross-referenced unless
they are included in :doc:`doc/python_reference.rst <python_reference>`.


Describe your changes in the changelog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include in your changeset a brief description of the change in the
:doc:`changelog <doc/whats_new.rst>` (this can be skipped for very minor
changes like correcting typos in the documentation). Note that there are
sections of the changelog for each release, and separate subsections for
bugfixes, new features, and changes to the public API. It is usually best to do
this *after* your PR is finalized, to avoid merge conflicts (since this file is
updated with almost every PR).


Test locally before opening pull requests (PRs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python uses `continuous integration <ci>`_ (CI) to ensure code quality and
test across multiple installation targets. However, the CIs are often slower
than testing locally, especially when other contributors also have open PRs
(which is basically always the case). Therefore, do not rely on the CIs to
catch bugs and style errors for you; run the tests locally instead before
opening a new PR and before each time you push additional changes to an
already-open PR. See the :ref:`testing <run-tests>` section (below) for
examples of how to run tests locally, and make sure you have the testing
dataset installed.


Make tests fast and thorough
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever possible, use the testing dataset rather than one of the sample
datasets when writing tests; it includes small versions of most MNE-Python
objects (e.g., :class:`~mne.io.Raw` objects with short durations and few
channels). You can also check which lines are missed by the tests, then modify
existing tests (or write new ones) to target the missed lines. Here's an
example that reports which lines within ``mne.viz`` are missed when running
``test_evoked.py`` and ``test_topo.py``:

.. code-block:: console

    $ pytest --cov=mne.viz --cov-report=term-missing mne/viz/tests/test_evoked.py mne/viz/tests/test_topo.py

You can also use ``pytest --durations=20`` to ensure new or modified tests will
not slow down the test suite too much.


Code style
^^^^^^^^^^

Adhere to standard Python style guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pep8_ and pyflakes_ are followed with very few exceptions in MNE-Python. From
the ``mne-python`` root directory, you can check for style violations using
flake8_ by running:

.. code-block:: console

    $ make flake

in the shell. Several text editors also have integrated Python style checking
(either built-in or with a plugin), which can also catch style errors (and
train you to make them less frequently). Spyder_ and `Visual Studio Code
<vscode>`_ are two such editors.


Use consistent variable naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes should be named using ``CamelCase``. Functions and instances/variables
should be ``snake_case`` (``n_samples`` rather than ``nsamples``). Avoid
single-character variable names, unless inside a list- or dict-comprehension or
generator.


Follow `NumPy style`_ for docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases imitating existing docstrings will be sufficient, but consult the
`Numpy docstring style guidelines <numpy style>`_ for more complicated
formatting such as embedding example code, citing references, or including
rendered mathematics. Private function/method docstrings may be brief for
simple functions/methods, but complete docstrings are appropriate when private
functions/methods are relatively complex.


Cross-reference everywhere
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the docstrings and dedicated documentation pages (tutorials, how-to
examples, discussions, and glossary) should include cross-references to any
mentioned module, class, function, method, attribute, or documentation page.
There are sphinx directives for all of these (``:mod:``, ``:class:``,
``:func:``, ``:meth:``, ``:attr:``, ``:doc:``) as well as a generic
cross-reference directive (``:ref:``) for linking to specific sections of a
documentation page. MNE-Python also uses `intersphinx`_, so you can (and
should) cross-reference to Python built-in classes and functions as well as API
elements in NumPy, SciPy, PySurfer, etc. See :doc:`the sphinx configuration
file <doc/conf.py>` for the list of supported intersphinx projects.


Other style guidance
~~~~~~~~~~~~~~~~~~~~
- Use single quotes whenever possible.
- Prefer generators or comprehensions over ``filter``, ``map`` and other
  functional idioms.
- Use explicit functional constructors for builtin containers to improve
  readability (e.g., :func:`list`, :func:`dict`, :func:`set`).
- Avoid nested functions or class methods if possible — use private functions
  instead.
- Avoid ``*args`` and ``**kwargs`` in function signatures.


Code organization
^^^^^^^^^^^^^^^^^

Importing
~~~~~~~~~

Import modules in this order:

1. builtin
2. standard scientific (``numpy as np``, ``scipy`` submodules)
3. others
4. MNE-Python imports

When importing from other parts of MNE-Python, use relative imports in the main
codebase and absolute imports in the tutorials and how-to examples. Imports for
``matplotlib`` and optional modules (``sklearn``, ``pandas``, etc.) should be
nested (i.e., within a function or method, not at the top of a file).


Return types
~~~~~~~~~~~~

Methods should modify inplace and return ``self``, functions should return
copies (where applicable).


Vizualization
~~~~~~~~~~~~~

Visualization capabilities should be made available in both function and method
forms. Add public visualization functions to the :mod:`mne.viz` submodule, and
call those functions from the corresponding object methods. For example, the
method :meth:`mne.Epochs.plot` internally calls the function
:func:`mne.viz.plot_epochs`.

All visualization functions must accept a boolean ``show`` parameter and return
a :class:`~matplotlib.figure.Figure` handle.

Visualization functions should default to the colormap ``RdBu_r`` for signed
data with a meaningful middle (zero-point) and ``Reds`` otherwise. This applies
to both visualization functions and tutorials/examples.


.. _run_tests:

Running the test suite
^^^^^^^^^^^^^^^^^^^^^^

Running the full test suite is as simple as running

.. code-block:: console

    $ make test

from the ``mne-python`` root folder. Testing the entire module can be quite
slow, however, so to run individual tests while working on a new feature, you
can run, e.g.:

.. code-block:: console

    $ pytest mne/tests/test_evoked.py:test_io_evoked -x --verbose

Or alternatively:

.. code-block:: console

    $ pytest mne/tests/test_evoked.py -k test_io_evoked -x --verbose

Make sure you have the testing dataset, which you can get by running this in
a Python interpreter::

    >>> mne.datasets.testing.data_path(verbose=True)  # doctest: +SKIP


Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Our documentation (including docstrings in code files) is in `reStructuredText
<rst>`_ format and is built using `Sphinx`_ and `Sphinx-Gallery`_. The easiest
way to ensure that your contributions to the documentation are properly
formatted is to follow the style guidelines on this page, imitate existing
documentation examples, refer to the Sphinx and Sphinx-Gallery reference
materials when unsure how to format your contributions, and build the docs
locally to confirm that everything looks correct before submitting the changes
in a pull request.

You can build the documentation locally using `GNU Make`_ with :doc:`Makefile
<doc/Makefile>`. From within the ``mne/doc`` directory, a full documentation
build (of the current code state) can be triggered with

.. code-block:: console

   $ make html_dev

Additional recipes are available that build all docs but do not evaluate the
python code in the tutorials and how-to examples (``make html_dev-noplot``),
build all docs but only evaluate a regex-specified subset of the
examples/tutorials (``PATTERN="insert regex here" make html_dev-pattern``), or
build the docs for the most recent stable release (``make html_stable``). Run
``make help`` from the ``mne/doc`` directory for more options, or consult the
`Sphinx-Gallery`_ documentation for additional details.


.. _`github-workflow`:

GitHub workflow
^^^^^^^^^^^^^^^

- Search the `MNE-Python issues page <open-mne-issues>`_ (both open and closed
  issues) in case someone else has already started work on the same bugfix or
  feature. If you don't find anything, `open a new issue <new-mne-issue>`_ to
  discuss changes with maintainers before starting work on your proposed
  changes.

- Implement only one new feature or bugfix per pull request (PR). Occasionally
  it may make sense to fix a few related bugs at once, but this makes PRs
  harder to review and test, so check with MNE-Python maintainers first before
  doing this. Avoid purely cosmetic changes to the code; they make PRs harder
  to review.

- It is usually better to make PRs *from* branches other than your master
  branch, so that you can use your master branch to easily get back to the
  current state of the code if needed (e.g., if you're working on multiple
  changes at once, or need to pull in recent changes from someone else to get
  your new feature to work properly).

- In most cases you should make PRs *into* the upstream's master branch, unless
  you are specifically asked by a maintainer to PR into another branch (e.g.,
  for backports or maintenance bugfixes to the current stable version).

- Don't forget to include in your PR a brief description of the change in the
  :doc:`changelog <doc/whats_new.rst>`.

- Our community uses the following commit tags and conventions:

    - Work-in-progress PRs should be created as `draft PRs
      <github-help-draft-pr>`_ (preferred method), or failing that, the PR
      title should begin with ``WIP``

    - When you believe a PR is ready to be reviewed and merged, `convert it
      from a draft PR to a normal PR <github-help-convert-draft-pr>`_, and
      change its title to begin with ``MRG``

    - PRs that only affect documentation should additionally be labelled
      ``DOC``, bugfixes should be labelled ``FIX``, and new features should be
      labelled ``ENH`` (for "enhancement"). ``STY`` is used for style changes
      (i.e., improving docstring consistency or formatting without changing its
      content).

    - the following commit tags are supported: ``[ci skip]``, ``[skip
      travis]``, ``[skip appveyor]``, ``[skip azp]``, ``[skip circle]``, and
      ``[circle full]``. These should be used judiciously.

`This pull request <mne-model-pr>`_ exemplifies many of the conventions listed
above: it addresses only one problem; it started with an issue (#6112) to
discuss the problem and some possible solutions; it is a PR from the user's
non-master branch into the upstream master branch; it separates different kinds
of changes into separate commits and uses labels like ``DOC``, ``FIX``, and
``STY`` to make it easier for maintainers to review the changeset; etc. If you
are new to GitHub it can serve as a useful example of what to expect from the
PR review process.
