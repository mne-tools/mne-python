.. ci-overview:

Overview of CI jobs
===================

.. container::

  .. raw:: html

    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script type="text/javascript" class="init">
    $(document).ready( function () {
        $('table.sg-datatable').DataTable({
            order: [[4, 'asc']], pageLength: 25
        });
    } );
    </script>

  .. list-table::
   :header-rows: 1
   :class: table table-striped sg-datatable

   * - Name
     - Description
     - When it runs
     - File
     - Category
   * - Test
     - Runs the unit test suite on Windows, the ultraslow tests on Linux, and tests for Qt bindings on Linux
     - On pushes to pull requests; on merges to ``main`` and ``maint/*`` branches
     - `azure-pipelines.yml <http://github.com/mne-tools/mne-python/blob/main/azure-pipelines.yml>`_
     - Azure Pipelines
   * - build_docs
     - Build the documentation
     - On pushes to pull requests; every day for ``main`` and ``maint/*`` branches
     - `.circleci/config.yml <http://github.com/mne-tools/mne-python/blob/main/.circleci/config.yml>`_
     - CircleCI
   * - linkcheck
     - Check for dead links in the documentation
     - First day of the month for ``main`` branch
     - `.circleci/config.yml <http://github.com/mne-tools/mne-python/blob/main/.circleci/config.yml>`_
     - CircleCI
   * - Tests
     - Runs the unit test suite across a range of Linux and macOS platforms, Python versions, and dependency versions
     - On pushes to pull requests; on merges to ``main`` and ``maint/*`` branches
     - `.github/workflows/tests.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/tests.yml>`_
     - GitHub Actions
   * - Changelog
     - Check that the towncrier changelog files are present and formatted correctly
     - On pushes to pull requests
     - `.github/workflows/check_changelog.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/check_changelog.yml>`_
     - GitHub Actions
   * - Scheduled updates
     - Opens pull requests to keep selected package dependencies and pre-commit hooks up to date
     - Every Monday
     - `.github/workflows/scheduled_updates.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/scheduled_updates.yml>`_
     - GitHub Actions
   * - CircleCI scheduled
     - Opens issues as alerts for failed scheduled CircleCI workflows
     - Every day
     - `.github/workflows/circleci_scheduled.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/circleci_scheduled.yml>`_
     - GitHub Actions
   * - Contributor credit
     - Opens pull requests to update information about who has contributed to the package, and how many people have cited the package in publications
     - First day of the month
     - `.github/workflows/credit.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/credit.yml>`_
     - GitHub Actions
   * - Sponsor review
     - Opens issues as reminders to check whether the current sponsors and supporting institutions listed on the website are up to date
     - 1st February, June, and October
     - `.github/workflows/sponsor_review.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/sponsor_review.yml>`_
     - GitHub Actions
   * - autofix.ci
     - Applies automatic fixes like adding PR names to towncrier files, ensuring new Python files have headers, and keeping stated dependency versions consistent across multiple sources
     - On pushes to pull requests
     - `.github/workflows/autofix.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/autofix.yml>`_
     - GitHub Actions
   * - CodeQL
     - Identifies security vulnerabilities
     - On merges to ``main`` branch
     - `.github/workflows/codeql-analysis.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/codeql-analysis.yml>`_
     - GitHub Actions
