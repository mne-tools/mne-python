"""Convenience functions for opening GUIs."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={"_gui": ["coregistration", "_GUIScraper"]},
)
