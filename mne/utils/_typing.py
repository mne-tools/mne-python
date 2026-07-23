"""Shared objects used for type annotations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    # TODO VERSION: Remove this when Python 3.11+ is required (use typing.Self)
    from typing_extensions import Self

__all__ = ["Self"]
