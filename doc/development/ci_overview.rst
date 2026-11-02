.. ci-overview:

Overview of CI jobs
===================

| **Name**: Test
|   **Description**: Runs the unit test suite on Windows, the ultraslow tests on Linux, and tests for Qt bindings on Linux
|   **When it runs**: On pushes to pull requests; on merges to ``main`` and ``maint/*`` branches
|   **File**: `azure-pipelines.yml <http://github.com/mne-tools/mne-python/blob/main/azure-pipelines.yml>`_

| **Name**: build_docs
|   **Description**: Build the documentation
|   **When it runs**: On pushes to pull requests; every day for ``main`` and ``maint/*`` branches
|   **File**: `.circleci/config.yml <http://github.com/mne-tools/mne-python/blob/main/.circleci/config.yml>`_

| **Name**: linkcheck
|   **Description**: Check for dead links in the documentation
|   **When it runs**: First day of the month for ``main`` branch
|   **File**: `.circleci/config.yml <http://github.com/mne-tools/mne-python/blob/main/.circleci/config.yml>`_

| **Name**: autofix.ci
|   **Description**: Applies automatic fixes like adding PR names to towncrier files, ensuring new Python files have headers, and keeping stated dependency versions consistent across multiple sources
|   **When it runs**: On pushes to pull requests
|   **File**: `.github/workflows/autofix.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/autofix.yml>`_

| **Name**: Sponsor review
|   **Description**: Opens issues as reminders to check whether the current sponsors and supporting institutions listed on the website are up to date
|   **When it runs**: 1st February, June, and October
|   **File**: `.github/workflows/sponsor_review.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/sponsor_review.yml>`_

| **Name**: Tests
|   **Description**: Runs the unit test suite across a range of Linux and macOS platforms, Python versions, and dependency versions
|   **When it runs**: On pushes to pull requests; on merges to ``main`` and ``maint/*`` branches
|   **File**: `.github/workflows/tests.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/tests.yml>`_

| **Name**: CircleCI scheduled
|   **Description**: Opens issues as alerts for failed scheduled CircleCI workflows
|   **When it runs**: Every day
|   **File**: `.github/workflows/circleci_scheduled.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/circleci_scheduled.yml>`_

| **Name**: CodeQL
|   **Description**: Identifies security vulnerabilities
|   **When it runs**: On merges to ``main`` branch
|   **File**: `.github/workflows/codeql-analysis.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/codeql-analysis.yml>`_

| **Name**: Contributor credit
|   **Description**: Opens pull requests to update information about who has contributed to the package, and how many people have cited the package in publications
|   **When it runs**: First day of the month
|   **File**: `.github/workflows/credit.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/credit.yml>`_

| **Name**: Changelog
|   **Description**: Check that the towncrier changelog files are present and formatted correctly
|   **When it runs**: On pushes to pull requests
|   **File**: `.github/workflows/check_changelog.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/check_changelog.yml>`_

| **Name**: Scheduled updates
|   **Description**: Opens pull requests to keep selected package dependencies and pre-commit hooks up to date
|   **When it runs**: Every Monday
|   **File**: `.github/workflows/scheduled_updates.yml <http://github.com/mne-tools/mne-python/blob/main/.github/workflows/scheduled_updates.yml>`_
