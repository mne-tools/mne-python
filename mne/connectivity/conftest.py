# -*- coding: utf-8 -*-
# Author: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)


def pytest_configure(config):
    """Configure pytest options."""
    # Warnings
    # - Once mne-connectivity fully replaces mne.connectivity, then
    # this file and the "connectivity" submodule can be deleted.
    warning_lines = r"""
    ignore::DeprecationWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)
