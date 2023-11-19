# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import functools

_COLLAPSED = False  # will override in doc build


@functools.lru_cache(maxsize=2)
def _get_html_templates_env(kind):
    # For _html_repr_() and mne.Report
    assert kind in ("repr", "report"), kind
    import jinja2

    templates_env = jinja2.Environment(
        loader=jinja2.PackageLoader(
            package_name="mne.html_templates", package_path=kind
        ),
        autoescape=jinja2.select_autoescape(default=True, default_for_string=True),
    )
    if kind == "report":
        templates_env.filters["zip"] = zip
    return templates_env


def _get_html_template(kind, name):
    return _RenderWrap(
        _get_html_templates_env(kind).get_template(name),
        collapsed=_COLLAPSED,
    )


class _RenderWrap:
    """Class that allows functools.partial-like wrapping of jinja2 Template.render()."""

    def __init__(self, template, **kwargs):
        self._template = template
        self._kwargs = kwargs

    def render(self, *args, **kwargs):
        return self._template.render(*args, **kwargs, **self._kwargs)
