# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Imported first by every decoding submodule that imports sklearn at module level
from ..utils import _soft_import

try:
    _sklearn = _soft_import("sklearn", "decoding")
except RuntimeError as exc:  # ImportError lets doctest collection skip us
    raise ImportError(str(exc)) from None
