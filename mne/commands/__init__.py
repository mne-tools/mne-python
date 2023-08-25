"""Command-line utilities."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["utils"],
    submod_attrs={},
)
