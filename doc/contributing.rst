.. contents:: Page contents
   :local:
   :depth: 2

.. _contributing:

.. include:: ../CONTRIBUTING.rst


Overview of contribution process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general you'll be working with three different copies of the MNE-Python
codebase: the official copy at https://github.com/mne-tools/mne-python (usually
called "upstream"), your `fork`_ of the upstream repository (similar URL, but
with your username in place of ``mne-tools``, and usually called "origin"), and
the local copy of the codebase on your computer. The typical contribution
process is to:

1. synchronize your local copy with ``upstream``

2. make changes to your local copy

3. `push`_ your changes to ``origin`` (your remote fork of the upstream)

4. submit a `pull request`_ from your fork into ``upstream``

The sections :ref:`basic-git` and :ref:`github-workflow` (below) describe this
process in more detail.


Setting up your local environment for MNE-Python development
============================================================

Configuring git
^^^^^^^^^^^^^^^

To get set up for contributing, make sure you have git installed on your local
computer:

- On Linux, the command ``sudo apt install git`` is usually sufficient; see the
  `official Linux instructions`_ for more options.

- On MacOS, download `the .dmg installer`_; Atlassian also provides `more
  detailed instructions and alternatives`_ such as using MacPorts or Homebrew.

- On Windows, we recommend `git Bash`_ rather than the `official Windows
  version of git`_, because git Bash provides its own shell that includes many
  Linux-equivalent command line programs that are useful for development.
  Windows 10 also offers the `Windows subsystem for Linux`_ that offers similar
  functionality to git Bash, but has not been widely tested by MNE-Python
  developers yet.

.. note::

    `GitHub desktop`_ is a GUI alternative to command line git that some users
    appreciate; it is available for Windows and MacOS.

Once git is installed, the only absolutely necessary configuration step is
identifying yourself and your contact info:

.. code-block:: console

   $ git config --global user.name "Your Name"
   $ git config --global user.email you@yourdomain.example.com

Make sure that the same email address is associated with your GitHub account
and with your local git configuration. It is possible to associate multiple
emails with a GitHub account, so if you initially set them up with different
emails, just add the local email to the GitHub account.

Sooner or later, git is going to ask you what text editor you want it to use
when writing commit messages, so you might as well configure that now too:

.. code-block:: console

   $ git config --global core.editor emacs    # or vim, or nano, or subl, or...

There are many other ways to customize git's behavior; see `configuring git`_
for more information. Once you have git installed and configured, and before
creating your local copy of the codebase, go to the `MNE-Python GitHub`_ page
and create a `fork`_ into your GitHub user account.


Setting up the Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    We strongly recommend the `Anaconda`_ or `Miniconda`_ environment managers
    for Python. Other setups are possible but are not officially supported by
    the MNE-Python development team; see discussion :ref:`here
    <other-py-distros>`.

The first step is to `clone`_ the MNE-Python repository from your fork, and
also connect the local copy to the ``upstream`` version of the codebase, so you
can stay up-to-date with changes from other contributors. First, edit these two
variables for your situation:

.. code-block:: console

    $ INSTALL_LOCATION="/opt"
    $ GITHUB_USERNAME="new_mne_contributor"

Then make a local clone of your remote fork ("origin"):

.. code-block:: console

    $ cd $INSTALL_LOCATION
    $ git clone https://github.com/$GITHUB_USERNAME/mne-python.git

Finally, set up a link between your local clone and the official repository
("upstream"):

.. code-block:: console

    $ cd mne-python
    $ git remote add upstream git://github.com/mne-tools/mne-python.git
    $ git fetch --all

.. note::

    We use ``git://`` instead of ``https://`` in the address for the official
    "upstream" remote repository. ``git://`` addresses are read-only, which
    means you can *pull* the official repository into your local copy (in order
    to stay up-to-date with changes made by other contributors) but you cannot
    *push* anything from your computer directly into the upstream remote.
    Instead, you must push your changes to your own remote fork first, and then
    create a pull request from your remote into the upstream remote (and even
    then, your changes are not automatically accepted into the upstream; the
    changes must be *approved* by maintainers and then *merged* into upstream).

Next, use the `environment file`_ provided in the root of the MNE-Python
repository to set up your local development environment. This will install all
of the dependencies needed for running MNE-Python. The following commands
create the conda environment:

.. code-block:: console

    $ PREFERRED_ENVIRONMENT_NAME="mnedev"
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

The environment file installs the *stable* version of MNE-Python, so next we'll
remove that and replace it with the *development* version (the clone we just
created):

.. code-block:: console

    $ cd $INSTALL_LOCATION/mne-python    # make sure we're in the right folder
    $ pip uninstall -y mne               # or: conda remove --force mne
    $ pip install -e .                   # or: python setup.py develop

.. note::

    The commands ``pip install -e .`` and ``python setup.py develop`` both
    install a python module into the current environment by creating a link to
    the source code directory (instead of copying the code to pip's
    ``site_packages`` directory, which is what normally happens). This means
    that any edits you make to the MNE-Python source code will be reflected the
    next time you open a Python interpreter and ``import mne`` (the ``-e`` flag
    of ``pip`` stands for an "editable" installation).

Finally, we'll add a few dependencies that are not needed for running
MNE-Python, but are needed for locally running our test suite or building our
documentation:

.. code-block:: console

    $ pip install sphinx sphinx-gallery sphinx_bootstrap_theme sphinx_fontawesome memory_profiler
    $ conda install sphinx-autobuild doc8  # linter packages for reStructuredText (optional)

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

    Occasionally, a bug emerges in one of the MNE-Python dependencies, and it
    temporarily becomes necessary to use the current master version of that
    dependency (until a new stable release is made that contains the bugfix).
    In such cases, you can do a one-time update of that dependency to its
    current master using pip + git (as shown for MNE-Python in
    :ref:`installing_master`). If you anticipate needing to update a dependency
    frequently, you can install the dependency in the same way you've just
    installed MNE-Python (i.e., cloning its repository and installing with
    ``pip install -e .`` from within the cloned repo), and then updating it
    periodically with ``git pull``.


.. _basic-git:

Basic git commands
^^^^^^^^^^^^^^^^^^

Learning to work with git can take a long time, because it is a complex and
powerful tool for managing versions of files across multiple users, each of
whom have multiple copies of the codebase. We've already seen in the setup
commands above a few of the basic git commands useful to an MNE-Python
developer:

- :samp:`git clone {<URL_OF_REMOTE_REPO>}` (make a local copy of a repository)

- :samp:`git remote add {<NICKNAME_OF_REMOTE>} {<URL_OF_REMOTE_REPO>}` (connect
  a local copy to an additional remote)

- ``git fetch --all`` (get the current state of connected remote repos)

Other commands that you will undoubtedly need relate to `branches`_. Branches
represent multiple copies of the codebase *within a local clone or remote
repo*. Branches are typically used to experiment with new features while still
keeping a clean, working copy of the original codebase that you can switch back
to at any time. The default branch of any repo is always called ``master``, and
it is recommended that you reserve the ``master`` branch to be that clean copy
of the working "upstream" codebase. Therefore, if you want to add a new
feature, you should first synchronize your local ``master`` branch with the
``upstream`` repository, then create a new branch based off of ``master`` and
`check it out`_ so that any changes you make will exist on that new branch
(instead of on ``master``):

.. code-block:: console

    $ git checkout master            # switch to master branch
    $ git fetch upstream             # get the current state of the upstream repo
    $ git merge upstream/master      # synchronize local master branch with upstream master branch
    $ git checkout -b new-feature-x  # create branch "new-feature-x" and check it out

.. note::

    You can save some typing by using ``git pull upstream/master`` to replace
    the ``fetch`` and ``merge`` lines above.

Now that you're on a new branch, you can fix a bug or add a new feature, add a
test, update the documentation, etc. When you're done, it's time to organize
your changes into a series of `commits`_. Commits are like snapshots of the
repository — actually, more like a description of what has to change to get
from the most recent snapshot to the current snapshot.

Git knows that people often work on multiple changes in multiple files all at
once, but that ultimately they should separate those changes into sets of
related changes that are grouped together based on common goals (so that it's
easier for their colleagues to understand and review the changes). For example,
you might want to group all the code changes together in one commit, put new
unit tests in another commit, and changes to the documentation in a third
commit.  Git makes this easy(ish) with something called the `stage`_ (or
*staging area*). After you've made some changes to the codebase, you'll have
what git calls "unstaged changes", which will show up with the `status`_
command:

.. code-block:: console

    $ # see what state the local copy of the codebase is in
    $ git status

Those unstaged changes can be `added`_ to the stage one by one, by either
adding a whole file's worth of changes, or by adding only certain lines
interactively:

.. code-block:: console

    $ # add a whole file's worth of changes
    $ # (same command works to add a completely new file):
    $ git add mne/some_file.py
    $ # enter interactive staging mode, to add only portions of the file:
    $ git add -p mne/viz/some_other_file.py

Once you've collected all the related changes together on the stage, the ``git
status`` command will now refer to them as "changes staged for commit". You can
commit them to the current branch with the `commit`_ command. If you just type
``git commit`` by itself, git will open the text editor you configured it to
use so that you can write a *commit message* — a short description of the
changes you've grouped together in this commit. You can bypass the text editor
by passing a commit message on the command line with the ``-m`` flag. For
example, if your first commit adds a new feature, your commit message might be:

.. code-block:: console

    $ git commit -m 'ENH: adds feature X to the Epochs class'

Once you've made the commit, the stage is now empty, and you can repeat the
cycle, adding the unit tests and documentation changes:

.. code-block:: console

    $ git add mne/tests/some_testing_file.py
    $ git commit -m 'add test of new feature X of the Epochs class'
    $ git add -p mne/some_file.py mne/viz/some_other_file.py
    $ git commit -m 'DOC: update Epochs and BaseEpochs docstrings'
    $ git add tutorials/new_tutorial_file.py
    $ git commit -m 'DOC: adds new tutorial about feature X'

When you're done, it's time to run the test suite to make sure your changes
haven't broken any existing functionality, and to make sure your new test
covers the lines of code you've added (see :ref:`run-tests` and
:ref:`build-docs`, below). Once everything looks good, it's time to push your
changes to your fork:

.. code-block:: console

    $ git push origin new-feature-x

Finally, go to the `MNE-Python GitHub`_ page, click on the pull requests tab,
click the "new pull request" button, and choose "compare across forks" to
select your new branch (``new-feature-x``) as the "head repository".  See the
GitHub help page on `creating a PR from a fork`_ for more information about
opening pull requests.

If any of the tests failed before you pushed your changes, try to fix them,
then add and commit the changes that fixed the tests, and push to your fork. If
you're stuck and can't figure out how to fix the tests, go ahead and push your
commits to your fork anyway and open a pull request (as described above), then
in the pull request you should describe how the tests are failing and ask for
advice about how to fix them.

To learn more about git, check out the `GitHub help`_ website, the `GitHub
Learning Lab`_ tutorial series, and the `pro git book`_.


Connecting to GitHub with SSH (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One easy way to speed up development is to reduce the number of times you have
to type your password. SSH (secure shell) allows authentication with pre-shared
key pairs. The private half of your key pair is kept secret on your computer,
while the public half of your key pair is added to your GitHub account; when
you connect to GitHub from your computer, the local git client checks the
remote (public) key against your local (private) key, and grants access your
account only if the keys fit. GitHub has `several help pages`_ that guide you
through the process.

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

For example, a new :class:`mne.Evoked` method in :file:`mne/evoked.py` should
have a corresponding test in :file:`mne/tests/test_evoked.py`.


All new functionality must be documented
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This includes thorough docstring descriptions for all public API changes, as
well as how-to examples or longer tutorials for major contributions. Docstrings
for private functions may be more sparse, but should not be omitted.


New API elements should be added to the master reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes, functions, methods, and attributes cannot be cross-referenced unless
they are included in the :doc:`python_reference`
(:file:`doc/python_reference.rst`).


Describe your changes in the changelog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include in your changeset a brief description of the change in the
:doc:`changelog <whats_new>` (:file:`doc/whats_new.rst`; this can be skipped
for very minor changes like correcting typos in the documentation). Note that
there are sections of the changelog for each release, and separate subsections
for bugfixes, new features, and changes to the public API. It is usually best
to wait to add a line to the changelog until your PR is finalized, to avoid
merge conflicts (since the changelog is updated with almost every PR).


Test locally before opening pull requests (PRs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python uses `continuous integration`_ (CI) to ensure code quality and
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

All contributions to MNE-Python are checked against style guidelines described
in `PEP 8`_. We also check for common coding errors (such as variables that are
defined but never used). We allow very few exceptions to these guidelines, and
use tools such as pep8_, pyflakes_, and flake8_ to check code style
automatically. From the :file:`mne-python` root directory, you can check for
style violations by running:

.. code-block:: console

    $ make flake

in the shell. Several text editors or IDEs also have Python style checking,
which can highlight style errors while you code (and train you to make those
errors less frequently). This functionality is built-in to the Spyder_ IDE, but
most editors have plug-ins that provide similar functionality. Search for
:samp:`python linter <name of your favorite editor>` to learn more.


Use consistent variable naming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes should be named using ``CamelCase``. Functions and instances/variables
should use ``snake_case`` (``n_samples`` rather than ``nsamples``). Avoid
single-character variable names, unless inside a :term:`comprehension <list
comprehension>` or :ref:`generator <tut-generators>`.


Follow NumPy style for docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases imitating existing docstrings will be sufficient, but consult the
`Numpy docstring style guidelines`_ for more complicated formatting such as
embedding example code, citing references, or including rendered mathematics.
Private function/method docstrings may be brief for simple functions/methods,
but complete docstrings are appropriate when private functions/methods are
relatively complex.


Cross-reference everywhere
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both the docstrings and dedicated documentation pages (tutorials, how-to
examples, discussions, and glossary) should include cross-references to any
mentioned module, class, function, method, attribute, or documentation page.
There are sphinx directives for all of these (``:mod:``, ``:class:``,
``:func:``, ``:meth:``, ``:attr:``, ``:doc:``) as well as a generic
cross-reference directive (``:ref:``) for linking to specific sections of a
documentation page. MNE-Python also uses Intersphinx_, so you can (and should)
cross-reference to Python built-in classes and functions as well as API
elements in :mod:`NumPy <numpy>`, :mod:`SciPy <scipy>`, etc. See the Sphinx
configuration file (:file:`doc/conf.py`) for the list of Intersphinx projects
we link to.


Other style guidance
~~~~~~~~~~~~~~~~~~~~

- Use single quotes whenever possible.

- Prefer :ref:`generators <tut-generators>` or
  :term:`comprehensions <list comprehension>` over :func:`filter`, :func:`map`
  and other functional idioms.

- Use explicit functional constructors for builtin containers to improve
  readability (e.g., :ref:`list() <func-list>`, :ref:`dict() <func-dict>`,
  :ref:`set() <func-set>`).

- Avoid nested functions or class methods if possible — use private functions
  instead.

- Avoid ``*args`` and ``**kwargs`` in function/method signatures.


Code organization
^^^^^^^^^^^^^^^^^

Importing
~~~~~~~~~

Import modules in this order:

1. Python built-in (``os``, ``copy``, ``functools``, etc)
2. standard scientific (``numpy as np``, ``scipy.signal``, etc)
3. others
4. MNE-Python imports (e.g., ``from .pick import pick_types``)

When importing from other parts of MNE-Python, use relative imports in the main
codebase and absolute imports in the tutorials and how-to examples. Imports for
``matplotlib`` and optional modules (``sklearn``, ``pandas``, etc.) should be
nested (i.e., within a function or method, not at the top of a file).


Return types
~~~~~~~~~~~~

Methods should modify inplace and return ``self``, functions should return
copies (where applicable). Docstrings should always give an informative name
for the return value, even if the function or method's return value is never
stored under that name in the code.


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


.. _build-docs:

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Our documentation (including docstrings in code files) is in
reStructuredText_ format and is built using Sphinx_ and `Sphinx-Gallery`_.
The easiest way to ensure that your contributions to the documentation are
properly formatted is to follow the style guidelines on this page, imitate
existing documentation examples, refer to the Sphinx and Sphinx-Gallery
reference materials when unsure how to format your contributions, and build the
docs locally to confirm that everything looks correct before submitting the
changes in a pull request.

You can build the documentation locally using `GNU Make`_ with
:file:`doc/Makefile`. From within the :file:`doc` directory, a full
documentation build (of the current code state) can be triggered with

.. code-block:: console

   $ make html_dev

Additional recipes are available that build all docs but do not evaluate the
python code in the tutorials and how-to examples (``make html_dev-noplot``),
build all docs but only evaluate a regex-specified subset of the
examples/tutorials (:samp:`PATTERN="{insert regex here}" make
html_dev-pattern`), or build the docs for the most recent stable release
(``make html_stable``). Run ``make help`` from the :file:`mne/doc` directory
for more options, or consult the `Sphinx-Gallery`_ documentation for additional
details.


.. _`github-workflow`:

GitHub workflow
^^^^^^^^^^^^^^^

Nearly everyone in the community of MNE-Python contributors and maintainers is
a working scientist, engineer, or student who contributes to MNE-Python in
their spare time. For that reason, a set of best practices have been adopted to
streamline the collaboration and review process. Most of these practices are
common to many open-source software projects, so learning to follow them while
working on MNE-Python will bear fruit when you contribute to other projects
down the road. Here are the guidelines:

- Search the `MNE-Python issues page`_ (both open and closed issues) in case
  someone else has already started work on the same bugfix or feature. If you
  don't find anything, `open a new issue`_ to discuss changes with maintainers
  before starting work on your proposed changes.

- Implement only one new feature or bugfix per pull request (PR). Occasionally
  it may make sense to fix a few related bugs at once, but this makes PRs
  harder to review and test, so check with MNE-Python maintainers first before
  doing this. Avoid purely cosmetic changes to the code; they make PRs harder
  to review.

- It is usually better to make PRs *from* branches other than your master
  branch, so that you can use your master branch to easily get back to a
  working state of the code if needed (e.g., if you're working on multiple
  changes at once, or need to pull in recent changes from someone else to get
  your new feature to work properly).

- In most cases you should make PRs *into* the upstream's master branch, unless
  you are specifically asked by a maintainer to PR into another branch (e.g.,
  for backports or maintenance bugfixes to the current stable version).

- Don't forget to include in your PR a brief description of the change in the
  :doc:`changelog <whats_new>` (:file:`doc/whats_new.rst`).

- Our community uses the following commit tags and conventions:

  - Work-in-progress PRs should be created as `draft PRs`_ and the PR title
    should begin with ``WIP``

  - When you believe a PR is ready to be reviewed and merged, `convert it
    from a draft PR to a normal PR`_, and change its title to begin with
    ``MRG``

  - PRs that only affect documentation should additionally be labelled
    ``DOC``, bugfixes should be labelled ``FIX``, and new features should be
    labelled ``ENH`` (for "enhancement"). ``STY`` is used for style changes
    (i.e., improving docstring consistency or formatting without changing its
    content).

  - the following commit tags are supported: ``[ci skip]``, ``[skip
    travis]``, ``[skip appveyor]``, ``[skip azp]``, ``[skip circle]``, and
    ``[circle full]``. These should be used judiciously.

`This sample pull request`_ exemplifies many of the conventions listed
above: it addresses only one problem; it started with an issue (#6112) to
discuss the problem and some possible solutions; it is a PR from the user's
non-master branch into the upstream master branch; it separates different kinds
of changes into separate commits and uses labels like ``DOC``, ``FIX``, and
``STY`` to make it easier for maintainers to review the changeset; etc. If you
are new to GitHub it can serve as a useful example of what to expect from the
PR review process.

.. MNE

.. _MNE-Python GitHub: https://github.com/mne-tools/mne-python
.. _MNE-Python issues page: https://github.com/mne-tools/mne-python/issues
.. _open a new issue: https://github.com/mne-tools/mne-python/issues/new/choose
.. _environment file:  https://raw.githubusercontent.com/mne-tools/mne-python/master/environment.yml
.. _This sample pull request: https://github.com/mne-tools/mne-python/pull/6230

.. git installation

.. _the .dmg installer: https://git-scm.com/download/mac
.. _official Windows version of git: https://git-scm.com/download/win
.. _official Linux instructions: https://git-scm.com/download/linux
.. _more detailed instructions and alternatives: https://www.atlassian.com/git/tutorials/install-git
.. _Windows subsystem for Linux: https://docs.microsoft.com/en-us/windows/wsl/about
.. _git bash: https://gitforwindows.org/
.. _GitHub desktop: https://desktop.github.com/

.. github help pages

.. _GitHub Help: https://help.github.com
.. _GitHub learning lab: https://lab.github.com/
.. _fork: https://help.github.com/en/articles/fork-a-repo
.. _clone: https://help.github.com/en/articles/cloning-a-repository
.. _push: https://help.github.com/en/articles/pushing-to-a-remote
.. _branches: https://help.github.com/en/articles/about-branches
.. _several help pages: https://help.github.com/en/articles/connecting-to-github-with-ssh
.. _draft PRs: https://help.github.com/en/articles/about-pull-requests#draft-pull-requests
.. _convert it from a draft PR to a normal PR: https://help.github.com/en/articles/changing-the-stage-of-a-pull-request
.. _pull request: https://help.github.com/en/articles/creating-a-pull-request-from-a-fork
.. _creating a PR from a fork: https://help.github.com/en/articles/creating-a-pull-request-from-a-fork

.. git docs

.. _check it out: https://git-scm.com/docs/git-checkout
.. _added: https://git-scm.com/docs/git-add
.. _commits: https://git-scm.com/docs/git-commit
.. _commit: https://git-scm.com/docs/git-commit
.. _status: https://git-scm.com/docs/git-status

.. git book

.. _pro git book: https://git-scm.com/book/
.. _stage: https://git-scm.com/book/en/v2/Git-Tools-Interactive-Staging
.. _configuring git: https://www.git-scm.com/book/en/v2/Customizing-Git-Git-Configuration

.. sphinx

.. _sphinx: http://www.sphinx-doc.org
.. _sphinx-gallery: https://sphinx-gallery.github.io
.. _reStructuredText: http://sphinx-doc.org/rest.html
.. _intersphinx: http://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

.. linting

.. _NumPy docstring style guidelines: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _pep8: https://pypi.org/project/pep8
.. _pyflakes: https://pypi.org/project/pyflakes
.. _Flake8: http://flake8.pycqa.org/

.. misc

.. _anaconda: https://www.anaconda.com/distribution/
.. _miniconda: https://conda.io/en/latest/miniconda.html
.. _Spyder: https://www.spyder-ide.org/
.. _GNU Make: https://www.gnu.org/software/make/
.. _continuous integration: https://en.wikipedia.org/wiki/Continuous_integration
.. _matplotlib: https://matplotlib.org/
