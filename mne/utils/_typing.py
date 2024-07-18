"""Shared objects used for type annotations."""

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar

    Self = TypeVar("Self")
