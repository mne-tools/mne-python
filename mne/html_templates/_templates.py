# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import datetime
import functools
import uuid
from typing import Union

_COLLAPSED = False  # will override in doc build


def _format_big_number(value: Union[int, float]) -> str:
    """Insert thousand separators."""
    return f"{value:,}"


def _append_uuid(string: str, sep: str = "-") -> str:
    """Append a UUID to a string."""
    return f"{string}{sep}{uuid.uuid1()}"


def _data_type(obj) -> str:
    """Return the qualified name of a class."""
    return obj.__class__.__qualname__


def _dt_to_str(dt: datetime.datetime) -> str:
    return dt.strftime("%B %d, %Y    %H:%M:%S") + " UTC"


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

    templates_env.filters["format_number"] = _format_big_number
    templates_env.filters["append_uuid"] = _append_uuid
    templates_env.filters["data_type"] = _data_type
    templates_env.filters["dt_to_str"] = _dt_to_str
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
