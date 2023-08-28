"""Functions for exporting data to non-FIF formats."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_export": ["export_raw", "export_epochs", "export_evokeds"],
        "_egimff": ["export_evokeds_mff"],
    },
)
