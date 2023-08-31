"""Report-generation functions and classes."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "report": ["Report", "open_report", "_ReportScraper"],
    },
)
