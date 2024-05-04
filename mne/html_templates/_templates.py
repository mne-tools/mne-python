# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
import datetime
import functools
import uuid
from collections import defaultdict
from typing import Any, Literal, Union

from .._fiff.pick import channel_type
from ..defaults import _handle_default

_COLLAPSED = True  # will override in doc build


def _format_number(value: Union[int, float]) -> str:
    """Insert thousand separators."""
    return f"{value:,}"


def _append_uuid(string: str, sep: str = "-") -> str:
    """Append a UUID to a string."""
    return f"{string}{sep}{uuid.uuid1()}"


def _data_type(obj) -> str:
    """Return the qualified name of a class."""
    return obj.__class__.__qualname__


def _dt_to_str(dt: datetime.datetime) -> str:
    """Convert a datetime object to a human-reaable string representation."""
    return dt.strftime("%B %d, %Y    %H:%M:%S") + " UTC"


def _format_baseline(inst) -> str:
    """Format the baseline time period."""
    if inst.baseline is None:
        baseline = "off"
    else:
        baseline = (
            f"{round(inst.baseline[0], 3):.3f} – {round(inst.baseline[1], 3):.3f} s"
        )

    return baseline


def _format_time_range(inst) -> str:
    """Format evoked and epochs time range."""
    tr = f"{round(inst.tmin, 3):.3f} – {round(inst.tmax, 3):.3f} s"
    return tr


def _format_projs(info) -> list[str]:
    """Format projectors."""
    projs = [f'{p["desc"]} ({"on" if p["active"] else "off"})' for p in info["projs"]]
    return projs


def _format_channels(info, ch_type: Literal["good", "bad", "ecg", "eog"]) -> str:
    """Format channel names."""
    titles = _handle_default("titles")

    if info.ch_names:
        # good channels
        good_names = defaultdict(lambda: list())
        for ci, ch_name in enumerate(info.ch_names):
            if ch_name in info["bads"]:
                continue
            channel_type_ = channel_type(info, ci)
            good_names[channel_type_].append(ch_name)
            del channel_type_
        good_channels = ", ".join(
            [f"{len(v)} {titles.get(k, k.upper())}" for k, v in good_names.items()]
        )
        for key in ("ecg", "eog"):  # ensure these are present
            if key not in good_names:
                good_names[key] = list()
        for key, val in good_names.items():
            good_names[key] = ", ".join(val) or "Not available"

        # bad channels
        bad_channels = ", ".join(info["bads"]) or "None"

        # ECG and EOG
        ecg = good_names["ecg"]
        eog = good_names["eog"]
    else:
        good_channels = bad_channels = ecg = eog = "None"

    channels = {"good": good_channels, "bad": bad_channels, "ecg": ecg, "eog": eog}
    return channels[ch_type]


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
