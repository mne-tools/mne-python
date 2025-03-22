.. _contributing:

Contributing guide
==================

.. highlight:: console

Thanks for taking the time to contribute! MNE-Python is an open-source project
sustained mostly by volunteer effort. We welcome contributions from anyone as
long as they abide by our `Code of Conduct`_.

There are lots of ways to contribute, such as:

.. rst-class:: icon-bullets

- |computer-mouse| Use the software, and when you find bugs, tell us about them! We can
  only fix the bugs we know about.
- |discourse| Answer questions on `our user forum`_.
- |comment| Tell us about parts of the documentation that you find confusing or
  unclear.
- |hand-sparkles| Tell us about things you wish MNE-Python could do, or things
  it can do but you wish they were easier.
- |universal-access| Improve the accessibility of our website.
- |bug-slash| Fix bugs.
- |text-slash| Fix mistakes in our function documentation strings.
- |wand-magic-sparkles| Implement new features.
- |pencil| Improve existing tutorials or write new ones.
- |python| Contribute to one of the many Python packages that MNE-Python
  depends on.

To *report* bugs, *request* new features, or *ask about* confusing
documentation, it's usually best to open a new issue on `our user forum`_
first; you'll probably get help fastest that way, and it helps keep our GitHub
issue tracker focused on things that we *know* will require changes to our
software (as opposed to problems that can be fixed in the user's code). We may
ultimately ask you to open an issue on GitHub too, but starting on the forum
helps us keep things organized. For fastest results, be sure to include
information about your operating system and MNE-Python version, and (if
applicable) include a reproducible code sample that is as short as possible and
ideally uses one of :ref:`our example datasets <datasets>`.

If you want to *fix* bugs, *add* new features, or *improve* our
docstrings/tutorials/website, those kinds of contributions are made through
`our GitHub repository <MNE-Python GitHub_>`_. The rest of this page explains
how to set up your workflow to make contributing via GitHub as easy as
possible.


.. dropdown:: Want an example to work through?
    :color: success
    :icon: rocket

    Feel free to just read through the rest of the page, but if you find it
    easier to "learn by doing", take a look at our
    `GitHub issues marked "easy"`_, pick one that looks interesting, and work
    through it while reading this guide!


Overview of contribution process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: Reminder: all contributors are expected to follow our
          `code of conduct`_.

Changes to MNE-Python are typically made by `forking`_ the MNE-Python
repository, making changes to your fork (usually by `cloning`_ it to your
personal computer, making the changes locally, and then `pushing`_ the local
changes up to your fork on GitHub), and finally creating a `pull request`_ to incorporate
your changes back into the shared "upstream" version of the codebase.

In general you'll be working with three different copies of the MNE-Python
codebase: the official remote copy at https://github.com/mne-tools/mne-python
(usually called ``upstream``), your remote `fork`_ of the upstream repository
(similar URL, but with your username in place of ``mne-tools``, and usually
called ``origin``), and the local copy of the codebase on your computer. The
typical contribution process is to:

1. synchronize your local copy with ``upstream``

2. make changes to your local copy

3. `push`_ your changes to ``origin`` (your remote fork of the upstream)

4. submit a `pull request`_ from your fork into ``upstream``

The sections :ref:`basic-git` and :ref:`github-workflow` (below) describe this
process in more detail.


Setting up your local development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuring git
~~~~~~~~~~~~~~~

.. admonition:: Git GUI alternative
    :class: sidebar note

    `GitHub desktop`_ is a GUI alternative to command line git that some users
    appreciate; it is available for |windows| Windows and |apple| MacOS.

To get set up for contributing, make sure you have git installed on your local
computer:

- On Linux, the command ``sudo apt install git`` is usually sufficient; see the
  `official Linux instructions`_ for more options.

- On MacOS, download `the .dmg installer`_; Atlassian also provides `more
  detailed instructions and alternatives`_ such as using MacPorts or Homebrew.

- On Windows, download and install `git for Windows`_. With Git BASH it provides its own shell that
  includes many Linux-equivalent command line programs that are useful for development.

  *Windows 10 also offers the* `Windows subsystem for Linux`_ *that offers similar
  functionality to git BASH, but has not been widely tested by MNE-Python
  developers yet and may still pose problems with graphical output (e.g. building the documentation)*


Once git is installed, the only absolutely necessary configuration step is
identifying yourself and your contact info::

   $ git config --global user.name "Your Name"
   $ git config --global user.email you@yourdomain.example.com

Make sure that the same email address is associated with your GitHub account
and with your local git configuration. It is possible to associate multiple
emails with a GitHub account, so if you initially set them up with different
emails, you can add the local email to the GitHub account.

Sooner or later, git is going to ask you what text editor you want it to use
when writing commit messages, so you might as well configure that now too::

   $ git config --global core.editor emacs    # or vim, or nano, or subl, or...

There are many other ways to customize git's behavior; see `configuring git`_
for more information.


GNU Make
~~~~~~~~

We use `GNU Make`_ to organize commands or short scripts that are often needed
in development. These are stored in files with the name :file:`Makefile`.
MNE-Python has two Makefiles, one in the package's root directory (containing
mainly testing commands) and one in :file:`doc/` (containing recipes for
building our documentation pages in different ways).

To check if make is already installed type ::

   $ make

into a terminal and you should see ::

   make: *** No targets specified and no makefile found.  Stop.

If you don't see this or something similar, you may not have ``make`` installed.

.. tab-set::

    .. tab-item:: Linux
        :class-content: text-center

        .. button-link:: https://www.gnu.org/software/make/
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Get make for Linux

    .. tab-item:: macOS
        :class-content: text-center

        .. button-link:: https://www.gnu.org/software/make/
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Get make for macOS

    .. tab-item:: Windows

        If you see: ::

            bash: make: command not found

        Install ``make`` for git BASH (which comes with `git for Windows`_):

        1. Download :file:`make-{newest.version}-without-guile-w32-bin.zip` from `ezwinports`_
        2. Extract zip-folder
        3. Copy the contents into :file:`{path_to_git}\\mingw64\\` (e.g. by merging the
           folders with the equivalent ones already inside)
        4. For the first time using git BASH, you need to run once (to be able to
           activate your ``mnedev`` environment): ::

            $ conda init bash

        If instead you see an error like: ::

                bash: conda: command not found

        at the top of your git BASH window, you need to add

        - :file:`{path_to_Anaconda}`
        - :file:`{path_to_Anaconda}\\Scripts`

        to Windows-PATH first.


Forking the MNE-Python repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have git installed and configured, and before creating your local copy
of the codebase, go to the `MNE-Python GitHub`_ page and create a `fork`_ into
your GitHub user account.

This will create a copy of the MNE-Python codebase inside your GitHub user
account (this is called "your fork"). Changes you make to MNE-Python will
eventually get "pushed" to your fork, and will be incorporated into the
official version of MNE-Python (often called the "upstream version") through a
"pull request". This process will be described in detail below; a summary
of how that structure is set up is given here:

.. graphviz:: ../_static/diagrams/git_setup.dot
   :alt: Diagram of recommended git setup
   :align: left


Creating the virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: Supported Python environments
    :class: sidebar note

    We strongly recommend the `Anaconda`_ or `Miniconda`_ environment managers
    for Python. Other setups are possible but are not officially supported by
    the MNE-Python development team; see discussion :ref:`here
    <other-py-distros>`. These instructions use  ``conda`` where possible;
    experts may replace those lines with some combination of ``git`` and
    ``pip``.

These instructions will set up a Python environment that is separated from your
system-level Python and any other managed Python environments on your computer.
This lets you switch between different versions of Python and also switch between
the stable and development
versions of MNE-Python (so you can, for example, use the same computer to
analyze your data with the stable release, and also work with the latest
development version to fix bugs or add new features). Even if you've already
followed the :ref:`installation instructions <install-python>` for the stable
version of MNE-Python, you should now repeat that process to create a new,
separate environment for MNE-Python development (here we'll give it the name
``mnedev``)::

    $ curl --remote-name https://raw.githubusercontent.com/mne-tools/mne-python/main/environment.yml
    $ conda env create --file environment.yml --name mnedev
    $ conda activate mnedev

Now you'll have *two* MNE-Python environments: ``mne`` (or whatever custom
name you used when installing the stable version of MNE-Python) and ``mnedev``
that we just created. At this point ``mnedev`` also has the stable version of
MNE-Python (that's what the :file:`environment.yml` file installs), but we're
about to remove the stable version from ``mnedev`` and replace it with the
development version. To do that, we'll `clone`_ the MNE-Python repository from
your remote fork, and also connect the local copy to the ``upstream`` version
of the codebase, so you can stay up-to-date with changes from other
contributors. First, edit these two variables for your situation::

    $ GITHUB_USERNAME="insert_your_actual_GitHub_username_here"
    $ # pick where to put your local copy of MNE-Python development version:
    $ INSTALL_LOCATION="/opt"

.. note::
   On Windows, add ``set`` before the variable names (``set GITHUB_USERNAME=...``, etc.).

Then make a local clone of your remote fork (``origin``)::

    $ cd $INSTALL_LOCATION
    $ git clone https://github.com/$GITHUB_USERNAME/mne-python.git

Finally, set up a link between your local clone and the official repository
(``upstream``) and set up ``git diff`` to work properly::

    $ cd mne-python
    $ git remote add upstream https://github.com/mne-tools/mne-python.git
    $ git fetch --all
    $ git config --local blame.ignoreRevsFile .git-blame-ignore-revs

Now we'll remove the *stable* version of MNE-Python and replace it with the
*development* version (the clone we just created with git). Make sure you're in
the correct environment first (``conda activate mnedev``), and then do::

    $ cd $INSTALL_LOCATION/mne-python    # make sure we're in the right folder
    $ conda remove --force mne-base  # the --force avoids dependency checking
    $ pip install -e .

The command ``pip install -e .`` installs a python module into the current
environment by creating a link to the source code directory (instead of copying
the code to pip's :file:`site_packages` directory, which is what normally
happens). This means that any edits you make to the MNE-Python source code will
be reflected the next time you open a Python interpreter and ``import mne``
(the ``-e`` flag of ``pip`` stands for an "editable" installation).

Finally, we'll add a few dependencies that are not needed for running
MNE-Python, but are needed for locally running our test suite::

    $ pip install -e ".[test]"

And for building our documentation::

    $ pip install -e ".[doc]"
    $ conda install graphviz

.. note::
   On Windows, if you installed graphviz using the conda command above but still get an error like this::

      WARNING: dot command 'dot' cannot be run (needed for graphviz output), check the graphviz_dot setting

   try adding the graphviz folder to path::

      $ PATH=$CONDA_PREFIX\\Library\\bin\\graphviz:$PATH

To build documentation, you will also require `optipng`_:

- On Linux, use the command ``sudo apt install optipng``.

- On MacOS, optipng can be installed using Homebrew.

- On Windows, unzip :file:`optipng.exe` from the `optipng for Windows`_ archive
  into the :file:`doc/` folder. This step is optional for Windows users.

There are additional optional dependencies needed to run various tests, such as
scikit-learn for decoding tests, or nibabel for MRI tests. If you want to run all the
tests, consider using our MNE installers (which provide these dependencies) or pay
attention to the skips that ``pytest`` reports and install the relevant libraries.
For example, this traceback::

    SKIPPED [2] mne/io/eyelink/tests/test_eyelink.py:14: could not import 'pandas': No module named 'pandas'

indicates that ``pandas`` needs to be installed in order to run the Eyelink tests.


.. _basic-git:

Basic git commands
~~~~~~~~~~~~~~~~~~

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
to at any time. The default branch of any repo is called ``main``, and
it is recommended that you reserve the ``main`` branch to be that clean copy
of the working ``upstream`` codebase. Therefore, if you want to add a new
feature, you should first synchronize your local ``main`` branch with the
``upstream`` repository, then create a new branch based off of ``main`` and
`check it out`_ so that any changes you make will exist on that new branch
(instead of on ``main``)::

    $ git checkout main            # switch to local main branch
    $ git fetch upstream             # get the current state of the remote upstream repo
    $ git merge upstream/main      # synchronize local main branch with remote upstream main branch
    $ git checkout -b new-feature-x  # create local branch "new-feature-x" and check it out

.. tip::
    :class: sidebar

    You can save some typing by using ``git pull upstream/main`` to replace
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
commit.  Git makes this possible with something called the `stage`_ (or
*staging area*). After you've made some changes to the codebase, you'll have
what git calls "unstaged changes", which will show up with the `status`_
command::

    $ git status    # see what state the local copy of the codebase is in

Those unstaged changes can be `added`_ to the stage one by one, by either
adding a whole file's worth of changes, or by adding only certain lines
interactively::

    $ git add mne/some_file.py      # add all the changes you made to this file
    $ git add mne/some_new_file.py  # add a completely new file in its entirety
    $ # enter interactive staging mode, to add only portions of a file:
    $ git add -p mne/viz/some_other_file.py

Once you've collected all the related changes together on the stage, the ``git
status`` command will now refer to them as "changes staged for commit". You can
commit them to the current branch with the `commit`_ command. If you just type
``git commit`` by itself, git will open the text editor you configured it to
use so that you can write a *commit message* — a short description of the
changes you've grouped together in this commit. You can bypass the text editor
by passing a commit message on the command line with the ``-m`` flag. For
example, if your first commit adds a new feature, your commit message might be::

    $ git commit -m 'ENH: adds feature X to the Epochs class'

Once you've made the commit, the stage is now empty, and you can repeat the
cycle, adding the unit tests and documentation changes::

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
changes to your fork::

    $ # push local changes to remote branch origin/new-feature-x
    $ # (this will create the remote branch if it doesn't already exist)
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
skills`_ tutorial series, and the `pro git book`_.


.. _github-ssh:

Connecting to GitHub with SSH (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
example::

    $ git remote -v  # show existing remote addresses
    $ git remote set-url origin git@github.com:$GITHUB_USERNAME/mne-python.git
    $ git remote set-url upstream git@github.com:mne-tools/mne-python.git


MNE-Python coding conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

General requirements
~~~~~~~~~~~~~~~~~~~~

All new functionality must have test coverage
---------------------------------------------

For example, a new `mne.Evoked` method in :file:`mne/evoked.py` should
have a corresponding test in :file:`mne/tests/test_evoked.py`.


All new functionality must be documented
----------------------------------------

This includes thorough docstring descriptions for all public API changes, as
well as how-to examples or longer tutorials for major contributions. Docstrings
for private functions may be more sparse, but should usually not be omitted.


Avoid API changes when possible
-------------------------------

Changes to the public API (e.g., class/function/method names and signatures)
should not be made lightly, as they can break existing user scripts. Changes to
the API require a deprecation cycle (with warnings) so that users have time to
adapt their code before API changes become default behavior. See :ref:`the
deprecation section <deprecating>` and `mne.utils.deprecated` for
instructions. Bug fixes (when something isn't doing what it says it will do) do
not require a deprecation cycle.

Note that any new API elements should be added to the main reference;
classes, functions, methods, and attributes cannot be cross-referenced unless
they are included in the :ref:`api_reference`
(:file:`doc/python_reference.rst`).


.. _deprecating:

Deprecate with a decorator or a warning
---------------------------------------

MNE-Python has a :func:`~mne.utils.deprecated` decorator for classes and
functions that will be removed in a future version:

.. code-block:: python

    from mne.utils import deprecated

    @deprecated('my_function is deprecated and will be removed in 0.XX; please '
                'use my_new_function instead.')
    def my_function():
       return 'foo'

If you need to deprecate a parameter, use :func:`mne.utils.warn`. For example,
to rename a parameter from ``old_param`` to ``new_param`` you can do something
like this:

.. code-block:: python

    from mne.utils import warn

    def my_other_function(new_param=None, old_param=None):
        if old_param is not None:
            depr_message = ('old_param is deprecated and will be replaced by '
                            'new_param in 0.XX.')
            if new_param is None:
                new_param = old_param
                warn(depr_message, FutureWarning)
            else:
                warn(depr_message + ' Since you passed values for both '
                     'old_param and new_param, old_param will be ignored.',
                     FutureWarning)
        # Do whatever you have to do with new_param
        return 'foo'

When deprecating, you should also add corresponding test(s) to the relevant
test file(s), to make sure that the warning(s) are being issued in the
conditions you expect:

.. code-block:: python

    # test deprecation warning for function
    with pytest.warns(FutureWarning, match='my_function is deprecated'):
        my_function()

    # test deprecation warning for parameter
    with pytest.warns(FutureWarning, match='values for both old_param'):
        my_other_function(new_param=1, old_param=2)
    with pytest.warns(FutureWarning, match='old_param is deprecated and'):
        my_other_function(old_param=2)

You should also search the codebase for any cases where the deprecated function
or parameter are being used internally, and update them immediately (don't wait
to the *end* of the deprecation cycle to do this). Later, at the end of the
deprecation period when the stated release is being prepared:

- delete the deprecated functions
- remove the deprecated parameters (along with the conditional branches of
  ``my_other_function`` that handle the presence of ``old_param``)
- remove the deprecation tests
- double-check for any other tests that relied on the deprecated test or
  parameter, and (if found) update them to use the new function / parameter.


Describe your changes in the changelog
--------------------------------------

Include in your changeset a brief description of the change in the
:ref:`changelog <whats_new>` using towncrier_ format, which aggregates small,
properly-named ``.rst`` files to create a changelog. This can be
skipped for very minor changes like correcting typos in the documentation.

There are six separate sections for changes, based on change type.
To add a changelog entry to a given section, name it as
:file:`doc/changes/devel/<PR-number>.<type>.rst`. The types are:

notable
    For overarching changes, e.g., adding type hints package-wide. These are rare.
dependency
    For changes to dependencies, e.g., adding a new dependency or changing
    the minimum version of an existing dependency.
bugfix
    For bug fixes. Can change code behavior with no deprecation period.
apichange
    Code behavior changes that require a deprecation period.
newfeature
    For new features.
other
    For changes that don't fit into any of the above categories, e.g.,
    internal refactorings.

For example, for an enhancement PR with number 12345, the changelog entry should be
added as a new file :file:`doc/changes/devel/12345.enhancement.rst`. The file should
contain:

1. A brief description of the change, typically in a single line of one or two
   sentences.
2. reST links to **public** API endpoints like functions (``:func:``),
   classes (``:class:``), and methods (``:meth:``). If changes are only internal
   to private functions/attributes, mention internal refactoring rather than name
   the private attributes changed.
3. Author credit. If you are a new contributor (we're very happy to have you here! 🤗),
   you should using the ``:newcontrib:`` reST role, whereas previous contributors should
   use a standard reST link to their name. For example, a new contributor could write:

   .. code-block:: rst

      Short description of the changes, by :newcontrib:`Firstname Lastname`.

   And an previous contributor could write:

   .. code-block:: rst

      Short description of the changes, by `Firstname Lastname`_.

Make sure that your name is included in the list of authors in
:file:`doc/changes/names.inc`, otherwise the documentation build will fail.
To add an author name, append a line with the following pattern (note
how the syntax is different from that used in the changelog):

.. code-block:: rst

  .. _Your Name: https://www.your-website.com/

Many contributors opt to link to their GitHub profile that way. Have a look
at the existing entries in the file to get some inspiration.

Sometimes, changes that shall appear as a single changelog entry are spread out
across multiple PRs. In this case, edit the existing towncrier file for the relevant
change, and append additional PR numbers in parentheticals with the ``:gh:`` role like:

.. code-block:: rst

    Short description of the changes, by `Firstname Lastname`_. (:gh:`12346`)

Test locally before opening pull requests (PRs)
-----------------------------------------------

MNE-Python uses `continuous integration`_ (CI) to ensure code quality and
test across multiple installation targets. However, the CIs are often slower
than testing locally, especially when other contributors also have open PRs
(which is basically always the case). Therefore, do not rely on the CIs to
catch bugs and style errors for you; :ref:`run the tests locally <run-tests>`
instead before opening a new PR and before each time you push additional
changes to an already-open PR.


Make tests fast and thorough
----------------------------

Whenever possible, use the testing dataset rather than one of the sample
datasets when writing tests; it includes small versions of most MNE-Python
objects (e.g., `~mne.io.Raw` objects with short durations and few
channels). You can also check which lines are missed by the tests, then modify
existing tests (or write new ones) to target the missed lines. Here's an
example that reports which lines within ``mne.viz`` are missed when running
:file:`test_evoked.py` and :file:`test_topo.py`::

    $ pytest --cov=mne.viz --cov-report=term-missing mne/viz/tests/test_evoked.py mne/viz/tests/test_topo.py

You can also use ``pytest --durations=5`` to ensure new or modified tests will
not slow down the test suite too much.


Code style
~~~~~~~~~~

Adhere to standard Python style guidelines
------------------------------------------

All contributions to MNE-Python are checked against style guidelines described
in `PEP 8`_. We also check for common coding errors (such as variables that are
defined but never used). We allow very few exceptions to these guidelines, and
use tools such as ruff_ to check code style
automatically. From the :file:`mne-python` root directory, you can check for
style violations by first installing our pre-commit hook::

    $ pip install pre-commit
    $ pre-commit install --install-hooks

Then running::

    $ make ruff  # alias for `pre-commit run -a`

in the shell. Several text editors or IDEs also have Python style checking,
which can highlight style errors while you code (and train you to make those
errors less frequently). This functionality is built-in to the Spyder_ IDE, but
most editors have plug-ins that provide similar functionality. Search for
:samp:`python linter <name of your favorite editor>` to learn more.


Use consistent variable naming
------------------------------

Classes should be named using ``CamelCase``. Functions and instances/variables
should use ``snake_case`` (``n_samples`` rather than ``nsamples``). Avoid
single-character variable names, unless inside a :term:`comprehension <list
comprehension>` or :ref:`generator <tut-generators>`.


We (mostly) follow NumPy style for docstrings
---------------------------------------------

In most cases you can look at existing MNE-Python docstrings to figure out how
yours should be formatted. If you can't find a relevant example, consult the
`Numpy docstring style guidelines`_ for examples of more complicated formatting
such as embedding example code, citing references, or including rendered
mathematics.  Note that we diverge from the NumPy docstring standard in a few
ways:

1. We use a module called ``sphinxcontrib-bibtex`` to render citations. Search
   our source code (``git grep footcite`` and ``git grep footbibliography``) to
   see examples of how to add in-text citations and formatted references to
   your docstrings, examples, or tutorials. The structured bibliographic data
   lives in :file:`doc/references.bib`; please follow the existing key scheme
   when adding new references (e.g., ``Singleauthor2019``,
   ``AuthoroneAuthortwo2020``, ``FirstauthorEtAl2021a``,
   ``FirstauthorEtAl2021b``).
2. We don't explicitly say "optional" for optional keyword parameters (because
   it's clear from the function or method signature which parameters have
   default values).
3. For parameters that may take multiple types, we use pipe characters instead
   of the word "or", like this: ``param_name : str | None``.
4. We don't include a ``Raises`` or ``Warns`` section describing
   errors/warnings that might occur.


Private function/method docstrings may be brief for simple functions/methods,
but complete docstrings are appropriate when private functions/methods are
relatively complex. To run some basic tests on documentation, you can use::

    $ pytest mne/tests/test_docstring_parameters.py
    $ make ruff


Cross-reference everywhere
--------------------------

Both the docstrings and dedicated documentation pages (tutorials, how-to
examples, discussions, and glossary) should include cross-references to any
mentioned module, class, function, method, attribute, or documentation page.
There are sphinx roles for all of these (``:mod:``, ``:class:``,
``:func:``, ``:meth:``, ``:attr:``, ``:doc:``) as well as a generic
cross-reference directive (``:ref:``) for linking to specific sections of a
documentation page.

.. warning::

    Some API elements have multiple exposure points (for example,
    ``mne.set_config`` and ``mne.utils.set_config``). For cross-references to
    work, they must match an entry in :file:`doc/python_reference.rst` (thus
    ``:func:`mne.set_config``` will work but ``:func:`mne.utils.set_config```
    will not).

MNE-Python also uses Intersphinx_, so you can (and should)
cross-reference to Python built-in classes and functions as well as API
elements in :mod:`NumPy <numpy>`, :mod:`SciPy <scipy>`, etc. See the Sphinx
configuration file (:file:`doc/conf.py`) for the list of Intersphinx projects
we link to. Their inventories can be examined using a tool like `sphobjinv`_ or
dumped to file with commands like::

    $ python -m sphinx.ext.intersphinx https://docs.python.org/3/objects.inv > python.txt

Note that anything surrounded by single backticks that is *not* preceded by one
of the API roles (``:class:``, ``:func:``, etc) will be assumed to be
in the MNE-Python namespace. This can save some typing especially in
tutorials; instead of ``see :func:`mne.io.Raw.plot_psd` for details`` you can
instead type ``see `mne.io.Raw.plot_psd` for details``.


Other style guidance
--------------------

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
~~~~~~~~~~~~~~~~~

Importing
---------

Import modules in this order, preferably alphabetized within each subsection:

1. Python built-in (``copy``, ``functools``, ``os``, etc.)
2. NumPy (``numpy as np``) and, in test files, pytest (``pytest``)
3. MNE-Python imports (e.g., ``from .pick import pick_types``)

When importing from other parts of MNE-Python, use relative imports in the main
codebase and absolute imports in tests, tutorials, and how-to examples. Imports
for ``matplotlib``, ``scipy``, and optional modules (``sklearn``, ``pandas``,
etc.) should be nested (i.e., within a function or method, not at the top of a
file). This helps reduce import time and limit hard requirements for using MNE.


Return types
------------

Methods should modify inplace and return ``self``, functions should return
copies (where applicable). Docstrings should always give an informative name
for the return value, even if the function or method's return value is never
stored under that name in the code.


Visualization
-------------

Visualization capabilities should be made available in both function and method
forms. Add public visualization functions to the :mod:`mne.viz` submodule, and
call those functions from the corresponding object methods. For example, the
method :meth:`mne.Epochs.plot` internally calls the function
:func:`mne.viz.plot_epochs`.

All visualization functions must accept a boolean ``show`` parameter and
typically return a :class:`matplotlib.figure.Figure` (or a list of
:class:`~matplotlib.figure.Figure` objects). 3D visualization functions return
a :class:`mne.viz.Figure3D`, :class:`mne.viz.Brain`, or other return type
as appropriate.

Visualization functions should default to the colormap ``RdBu_r`` for signed
data with a meaningful middle (zero-point) and ``Reds`` otherwise. This applies
to both visualization functions and tutorials/examples.


.. _run-tests:

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: pytest flags
    :class: sidebar tip

    The ``-x`` flag exits the pytest run when any test fails; this can speed
    up debugging when running all tests in a file or module.

    The ``--pdb`` flag will automatically start the python debugger upon test
    failure.

The full test suite can be run by calling ``pytest -m "not ultraslowtest" mne`` from the
``mne-python`` root folder. Testing the entire module can be quite
slow, however, so to run individual tests while working on a new feature, you
can run the following line::

    $ pytest mne/tests/test_evoked.py::test_io_evoked --verbose

Or alternatively::

    $ pytest mne/tests/test_evoked.py -k test_io_evoked --verbose

Make sure you have the testing dataset, which you can get by running this in
a Python interpreter:

.. code-block:: python

    >>> mne.datasets.testing.data_path(verbose=True)  # doctest: +SKIP


.. _build-docs:

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Our documentation (including docstrings in code files) is in
reStructuredText_ format and is built using Sphinx_ and `Sphinx-Gallery`_.
The easiest way to ensure that your contributions to the documentation are
properly formatted is to follow the style guidelines on this page, imitate
existing documentation examples, refer to the Sphinx and Sphinx-Gallery
reference materials when unsure how to format your contributions, and build the
docs locally to confirm that everything looks correct before submitting the
changes in a pull request.

You can build the documentation locally using `GNU Make`_ with
:file:`doc/Makefile`. From within the :file:`doc` directory, you can test
formatting and linking by running::

    $ make html-noplot

This will build the documentation *except* it will format (but not execute) the
tutorial and example files. If you have created or modified an example or
tutorial, you should instead run
:samp:`make html-pattern PATTERN={<REGEX_TO_SELECT_MY_TUTORIAL>}` to render
all the documentation and additionally execute just your example or tutorial
(so you can make sure it runs successfully and generates the output / figures
you expect).

After either of these commands completes, ``make show`` will open the
locally-rendered documentation site in your browser. If you see many warnings
that seem unrelated to your contributions, it might be that your output folder
for the documentation build contains old, now irrelevant, files. Running
``make clean`` will clean those up. Additional ``make`` recipes are available;
run ``make help`` from the :file:`doc` directory or consult the
`Sphinx-Gallery`_ documentation for additional details.


Modifying command-line tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MNE-Python provides support for a limited set of :ref:`python_commands`.
These are typically used with a call like::

    $ mne browse_raw ~/mne_data/MNE-sample-data/MEG/sample/sample_audvis_raw.fif

These are generally available for convenience, and can be useful for quick
debugging (in this case, for `mne.io.Raw.plot`).

If a given command-line function fails, they can also be executed as part of
the ``mne`` module with ``python -m``. For example::

    $ python -i -m mne browse_raw ...

Because this was launched with ``python -i``, once the script completes
it will drop to a Python terminal. This is useful when there are errors,
because then you can drop into a :func:`post-mortem debugger <python:pdb.pm>`:

.. code-block:: python

    >>> import pdb; pdb.pm()  # doctest:+SKIP


.. _`github-workflow`:

GitHub workflow
~~~~~~~~~~~~~~~

Nearly everyone in the community of MNE-Python contributors and maintainers is
a working scientist, engineer, or student who contributes to MNE-Python in
their spare time. For that reason, a set of best practices have been adopted to
streamline the collaboration and review process. Most of these practices are
common to many open-source software projects, so learning to follow them while
working on MNE-Python will bear fruit when you contribute to other projects
down the road. Here are the guidelines:

- Search the `GitHub issues page`_ (both open and closed issues) in case
  someone else has already started work on the same bugfix or feature. If you
  don't find anything, `open a new issue`_ to discuss changes with maintainers
  before starting work on your proposed changes.

- Implement only one new feature or bugfix per pull request (PR). Occasionally
  it may make sense to fix a few related bugs at once, but this makes PRs
  harder to review and test, so check with MNE-Python maintainers first before
  doing this. Avoid purely cosmetic changes to the code; they make PRs harder
  to review.

- It is usually better to make PRs *from* branches other than your main
  branch, so that you can use your main branch to easily get back to a
  working state of the code if needed (e.g., if you're working on multiple
  changes at once, or need to pull in recent changes from someone else to get
  your new feature to work properly).

- In most cases you should make PRs *into* the upstream's main branch, unless
  you are specifically asked by a maintainer to PR into another branch (e.g.,
  for backports or maintenance bugfixes to the current stable version).

- Don't forget to include in your PR a brief description of the change in the
  :ref:`changelog <whats_new>` (:file:`doc/whats_new.rst`).

- Our community uses the following commit tags and conventions:

  - Work-in-progress PRs should be created as `draft PRs`_ and the PR title
    should begin with ``WIP``.

  - When you believe a PR is ready to be reviewed and merged, `convert it
    from a draft PR to a normal PR`_, change its title to begin with ``MRG``,
    and add a comment to the PR asking for reviews (changing the title does not
    automatically notify maintainers).

  - PRs that only affect documentation should additionally be labelled
    ``DOC``, bugfixes should be labelled ``FIX``, and new features should be
    labelled ``ENH`` (for "enhancement"). ``STY`` is used for style changes
    (i.e., improving docstring consistency or formatting without changing its
    content).

  - the following commit tags are used to interact with our
    `continuous integration`_ (CI) providers. Use them judiciously; *do not
    skip tests simply because they are failing*:

    - ``[skip circle]`` Skip `CircleCI`_, which tests successful building of
      our documentation.

    - ``[skip actions]`` Skip our `GitHub Actions`_, which test installation
      and execution on Linux and macOS systems.

    - ``[skip azp]`` Skip `azure`_ which tests installation and execution on
      Windows systems.

    - ``[ci skip]`` is an alias for ``[skip actions][skip azp][skip circle]``.
      Notice that ``[skip ci]`` is not a valid tag.

    - ``[circle full]`` triggers a "full" documentation build, i.e., all code
      in tutorials and how-to examples will be *executed* (instead of just
      nicely formatted) and the resulting output and figures will be rendered
      as part of the tutorial/example.

- Examples and tutorials should execute as quickly and with as low memory usage as
  possible while still conveying necessary information. To see current execution
  times and memory usage, visit the `sg_execution_times page`_. To see unused API
  entries, see the `sg_api_usage page`_.

`This sample pull request`_ exemplifies many of the conventions listed above:
it addresses only one problem; it started with an issue to discuss the problem
and some possible solutions; it is a PR from the user's non-main branch into
the upstream main branch; it separates different kinds of changes into
separate commits and uses labels like ``DOC``, ``FIX``, and ``STY`` to make it
easier for maintainers to review the changeset; etc. If you are new to GitHub
it can serve as a useful example of what to expect from the PR review process.


.. MNE

.. _`GitHub issues marked "easy"`: https://github.com/mne-tools/mne-python/issues?q=is%3Aissue+is%3Aopen+label%3AEASY
.. _open a new issue: https://github.com/mne-tools/mne-python/issues/new/choose
.. _This sample pull request: https://github.com/mne-tools/mne-python/pull/6230
.. _our user forum: https://mne.discourse.group
.. _sg_execution_times page: https://mne.tools/dev/sg_execution_times.html
.. _sg_api_usage page: https://mne.tools/dev/sg_api_usage.html

.. git installation

.. _the .dmg installer: https://git-scm.com/download/mac
.. _official Linux instructions: https://git-scm.com/download/linux
.. _more detailed instructions and alternatives: https://www.atlassian.com/git/tutorials/install-git
.. _Windows subsystem for Linux: https://docs.microsoft.com/en-us/windows/wsl/about
.. _GitHub desktop: https://desktop.github.com/
.. _GNU Make: https://www.gnu.org/software/make/
.. _ezwinports: https://sourceforge.net/projects/ezwinports/files/

.. github help pages

.. _fork: https://help.github.com/en/articles/fork-a-repo
.. _clone: https://help.github.com/en/articles/cloning-a-repository
.. _push: https://help.github.com/en/articles/pushing-to-a-remote
.. _forking: https://help.github.com/en/articles/fork-a-repo
.. _cloning: https://help.github.com/en/articles/cloning-a-repository
.. _pushing: https://help.github.com/en/articles/pushing-to-a-remote
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

.. _stage: https://git-scm.com/book/en/v2/Git-Tools-Interactive-Staging
.. _configuring git: https://www.git-scm.com/book/en/v2/Customizing-Git-Git-Configuration

.. sphinx

.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _sphobjinv: https://sphobjinv.readthedocs.io/en/latest/

.. linting

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _ruff: https://beta.ruff.rs/docs

.. misc

.. _miniconda: https://conda.io/en/latest/miniconda.html
.. _Spyder: https://www.spyder-ide.org/
.. _continuous integration: https://en.wikipedia.org/wiki/Continuous_integration
.. _matplotlib: https://matplotlib.org/
.. _github actions: https://docs.github.com/en/free-pro-team@latest/actions/learn-github-actions
.. _azure: https://dev.azure.com/mne-tools/mne-python/_build/latest?definitionId=1&branchName=main
.. _CircleCI: https://circleci.com/gh/mne-tools/mne-python

.. optipng

.. _optipng: http://optipng.sourceforge.net/
.. _optipng for Windows: http://prdownloads.sourceforge.net/optipng/optipng-0.7.8-win64.zip?download

.. include:: ../links.inc
