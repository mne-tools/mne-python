# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations  # only needed for Python ≤ 3.9

import datetime
import functools
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .._fiff.pick import channel_type
from ..defaults import _handle_default

_COLLAPSED = False  # will override in doc build


def _format_number(value: int | float) -> str:
    """Insert thousand separators."""
    return f"{value:,}"


def _append_uuid(string: str, sep: str = "-") -> str:
    """Append a UUID to a string."""
    return f"{string}{sep}{uuid.uuid4()}"


def _data_type(obj) -> str:
    """Return the qualified name of a class."""
    return obj.__class__.__qualname__


def _dt_to_str(dt: datetime.datetime) -> str:
    """Convert a datetime object to a human-readable string representation."""
    return dt.strftime("%Y-%m-%d at %H:%M:%S %Z")


def _format_baseline(inst) -> str:
    """Format the baseline time period."""
    if inst.baseline is None:
        baseline = "off"
    else:
        baseline = (
            f"{round(inst.baseline[0], 3):.3f} – {round(inst.baseline[1], 3):.3f} s"
        )

    return baseline


def _format_metadata(inst) -> str:
    """Format metadata representation."""
    if inst.metadata is None:
        metadata = "No metadata set"
    else:
        metadata = f"{inst.metadata.shape[0]} rows × {inst.metadata.shape[1]} columns"

    return metadata


def _format_time_range(inst) -> str:
    """Format evoked and epochs time range."""
    tr = f"{round(inst.tmin, 3):.3f} – {round(inst.tmax, 3):.3f} s"
    return tr


def _format_projs(info) -> list[str]:
    """Format projectors."""
    projs = [f'{p["desc"]} ({"on" if p["active"] else "off"})' for p in info["projs"]]
    return projs


@dataclass
class _Channel:
    """A channel in a recording."""

    index: int
    name_html: str
    type: str
    type_pretty: str
    status: Literal["good", "bad"]


def _format_channels(info) -> dict[str, dict[Literal["good", "bad"], list[str]]]:
    """Format channel names."""
    ch_types_pretty: dict[str, str] = _handle_default("titles")
    channels = []

    if info.ch_names:
        for ch_index, ch_name in enumerate(info.ch_names):
            ch_type = channel_type(info, ch_index)
            ch_type_pretty = ch_types_pretty.get(ch_type, ch_type.upper())
            ch_status = "bad" if ch_name in info["bads"] else "good"
            channel = _Channel(
                index=ch_index,
                name_html=ch_name.replace(" ", "&nbsp;"),
                type=ch_type,
                type_pretty=ch_type_pretty,
                status=ch_status,
            )
            channels.append(channel)

    # Extract unique channel types and put them in the desired order.
    ch_types = list(set([c.type_pretty for c in channels]))
    ch_types = [c for c in ch_types_pretty.values() if c in ch_types]

    channels_formatted = {}
    for ch_type in ch_types:
        goods = [c for c in channels if c.type_pretty == ch_type and c.status == "good"]
        bads = [c for c in channels if c.type_pretty == ch_type and c.status == "bad"]
        if ch_type not in channels_formatted:
            channels_formatted[ch_type] = {"good": [], "bad": []}
        channels_formatted[ch_type]["good"] = goods
        channels_formatted[ch_type]["bad"] = bads

    return channels_formatted


def _has_attr(obj: Any, attr: str) -> bool:
    """Check if an object has an attribute `obj.attr`.

    This is needed because on dict-like objects, Jinja2's `obj.attr is defined` would
    check for `obj["attr"]`, which may not be what we want.
    """
    return hasattr(obj, attr)


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

    templates_env.filters["format_number"] = _format_number
    templates_env.filters["append_uuid"] = _append_uuid
    templates_env.filters["data_type"] = _data_type
    templates_env.filters["dt_to_str"] = _dt_to_str
    templates_env.filters["format_baseline"] = _format_baseline
    templates_env.filters["format_metadata"] = _format_metadata
    templates_env.filters["format_time_range"] = _format_time_range
    templates_env.filters["format_projs"] = _format_projs
    templates_env.filters["format_channels"] = _format_channels
    templates_env.filters["has_attr"] = _has_attr
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
