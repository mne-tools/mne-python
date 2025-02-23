"""Non-Linear sparse inverse solvers."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "mxne_inverse": ["mixed_norm", "tf_mixed_norm", "make_stc_from_dipoles"],
        "_gamma_map": ["gamma_map"],
    },
)
