"""Non-Linear sparse inverse solvers."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "mxne_inverse": ["mixed_norm", "tf_mixed_norm", "make_stc_from_dipoles"],
        "_gamma_map": ["gamma_map"],
    },
)
