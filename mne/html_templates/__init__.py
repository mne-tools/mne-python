"""Jinja2 HTML templates."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={"_templates": ["_get_html_template"]},
)
