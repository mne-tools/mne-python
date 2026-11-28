.. _ci-overview:

Overview of CI jobs
===================

Any jobs which run on pushes to pull requests are required to pass for the pull request to be merged.

For developers, the information in this table is pulled from information in the workflow files. That information should be kept up to date with the behaviour of the workflows, and new workflows must have this information added if they should be included in this table.

.. ci-overview::

Commit message tags
-------------------

You can include specific tags in your commit message to skip certain CI jobs. This is useful for documentation-only changes or minor updates that don't require full testing.

* ``[skip circle]``: Skips the CircleCI documentation build.
* ``[skip actions]``: Skips the GitHub Actions workflows.
* ``[ci skip]``: Skips most continuous integration checks.

PR labels for skipping checks
-----------------------------

In addition to commit messages, you can apply specific labels to your Pull Request to bypass certain checks:

* Add the **skip-changelog** label to skip the changelog check.
* Add the **skip-fragment** label to bypass the requirement for a news fragment (towncrier).
