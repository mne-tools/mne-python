"""
Event API for inter-figure communication.

The event API allows figures to communicate with each other, such that a change
in one figure can trigger a change in another figure. For example, moving the
time cursor in one plot can update the current time in another plot. Another
scenario is two drawing routines drawing into the same window, using events to
stay in-sync.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations  # only needed for Python ≤ 3.9

import contextlib
import re
import weakref
from dataclasses import dataclass

from matplotlib.colors import Colormap

from ..utils import _validate_type, fill_doc, logger, verbose, warn

# Global dict {fig: channel} containing all currently active event channels.
_event_channels = weakref.WeakKeyDictionary()

# The event channels of figures can be linked together. This dict keeps track
# of these links. Links are bi-directional, so if {fig1: fig2} exists, then so
# must {fig2: fig1}.
_event_channel_links = weakref.WeakKeyDictionary()

# Event channels that are temporarily disabled by the disable_ui_events context
# manager.
_disabled_event_channels = weakref.WeakSet()

# Regex pattern used when converting CamelCase to snake_case.
# Detects all capital letters that are not at the beginning of a word.
_camel_to_snake = re.compile(r"(?<!^)(?=[A-Z])")


# List of events
@fill_doc
class UIEvent:
    """Abstract base class for all events.

    Attributes
    ----------
    %(ui_event_name_source)s
    """

    source = None

    @property
    def name(self):
        """The name of the event, which is the class name in snake case."""
        return _camel_to_snake.sub("_", self.__class__.__name__).lower()


@fill_doc
class FigureClosing(UIEvent):
    """Indicates that the user has requested to close a figure.

    Attributes
    ----------
    %(ui_event_name_source)s
    """

    pass


@dataclass
@fill_doc
class TimeChange(UIEvent):
    """Indicates that the user has selected a time.

    Parameters
    ----------
    time : float
        The new time in seconds.

    Attributes
    ----------
    %(ui_event_name_source)s
    time : float
        The new time in seconds.
    """

    time: float


@dataclass
@fill_doc
class PlaybackSpeed(UIEvent):
    """Indicates that the user has selected a different playback speed for videos.

    Parameters
    ----------
    speed : float
        The new speed in seconds per frame.

    Attributes
    ----------
    %(ui_event_name_source)s
    speed : float
        The new speed in seconds per frame.
    """

    speed: float


@dataclass
@fill_doc
class ColormapRange(UIEvent):
    """Indicates that the user has updated the bounds of the colormap.

    Parameters
    ----------
    kind : str
        Kind of colormap being updated. The Notes section of the drawing
        routine publishing this event should mention the possible kinds.
    ch_type : str
       Type of sensor the data originates from.
    %(fmin_fmid_fmax)s
    %(alpha)s
    cmap : str
        The colormap to use. Either string or matplotlib.colors.Colormap
        instance.

    Attributes
    ----------
    kind : str
        Kind of colormap being updated. The Notes section of the drawing
        routine publishing this event should mention the possible kinds.
    ch_type : str
        Type of sensor the data originates from.
    unit : str
        The unit of the values.
    %(ui_event_name_source)s
    %(fmin_fmid_fmax)s
    %(alpha)s
    cmap : str
        The colormap to use. Either string or matplotlib.colors.Colormap
        instance.
    """

    kind: str
    ch_type: str | None = None
    fmin: float | None = None
    fmid: float | None = None
    fmax: float | None = None
    alpha: bool | None = None
    cmap: Colormap | str | None = None


@dataclass
@fill_doc
class VertexSelect(UIEvent):
    """Indicates that the user has selected a vertex.

    Parameters
    ----------
    hemi : str
        The hemisphere the vertex was selected on.
        Can be ``"lh"``, ``"rh"``, or ``"vol"``.
    vertex_id : int
        The vertex number (in the high resolution mesh) that was selected.

    Attributes
    ----------
    %(ui_event_name_source)s
    hemi : str
        The hemisphere the vertex was selected on.
        Can be ``"lh"``, ``"rh"``, or ``"vol"``.
    vertex_id : int
        The vertex number (in the high resolution mesh) that was selected.
    """

    hemi: str
    vertex_id: int


@dataclass
@fill_doc
class Contours(UIEvent):
    """Indicates that the user has changed the contour lines.

    Parameters
    ----------
    kind : str
        The kind of contours lines being changed. The Notes section of the
        drawing routine publishing this event should mention the possible
        kinds.
    contours : list of float
        The new values at which contour lines need to be drawn.

    Attributes
    ----------
    %(ui_event_name_source)s
    kind : str
        The kind of contours lines being changed. The Notes section of the
        drawing routine publishing this event should mention the possible
        kinds.
    contours : list of float
        The new values at which contour lines need to be drawn.
    """

    kind: str
    contours: list[str]


@dataclass
@fill_doc
class ChannelsSelect(UIEvent):
    """Indicates that the user has selected one or more channels.

    Parameters
    ----------
    ch_names : list of str
        The names of the channels that were selected.

    Attributes
    ----------
    %(ui_event_name_source)s
    ch_names : list of str
        The names of the channels that were selected.
    """

    ch_names: list[str]


def _get_event_channel(fig):
    """Get the event channel associated with a figure.

    If the event channel doesn't exist yet, it gets created and added to the
    global ``_event_channels`` dict.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure to get the event channel for.

    Returns
    -------
    channel : dict[event -> list]
        The event channel. An event channel is a list mapping string event
        names to a list of callback representing all subscribers to the
        channel.
    """
    import matplotlib

    from ._brain import Brain
    from .evoked_field import EvokedField

    # Create the event channel if it doesn't exist yet
    if fig not in _event_channels:
        # The channel itself is a dict mapping string event names to a list of
        # subscribers. No subscribers yet for this new event channel.
        _event_channels[fig] = dict()

        weakfig = weakref.ref(fig)

        # When the figure is closed, its associated event channel should be
        # deleted. This is a good time to set this up.
        def delete_event_channel(event=None, *, weakfig=weakfig):
            """Delete the event channel (callback function)."""
            fig = weakfig()
            if fig is None:
                return
            publish(fig, event=FigureClosing())  # Notify subscribers of imminent close
            logger.debug(f"unlink(({fig})")
            unlink(fig)  # Remove channel from the _event_channel_links dict
            if fig in _event_channels:
                logger.debug(f"  del _event_channels[{fig}]")
                del _event_channels[fig]
            if fig in _disabled_event_channels:
                logger.debug(f"  _disabled_event_channels.remove({fig})")
                _disabled_event_channels.remove(fig)

        # Hook up the above callback function to the close event of the figure
        # window. How this is done exactly depends on the various figure types
        # MNE-Python has.
        _validate_type(fig, (matplotlib.figure.Figure, Brain, EvokedField), "fig")
        if isinstance(fig, matplotlib.figure.Figure):
            fig.canvas.mpl_connect("close_event", delete_event_channel)
        else:
            assert hasattr(fig, "_renderer")  # figures like Brain, EvokedField, etc.
            fig._renderer._window_close_connect(delete_event_channel, after=False)

    # Now the event channel exists for sure.
    return _event_channels[fig]


@verbose
def publish(fig, event, *, verbose=None):
    """Publish an event to all subscribers of the figure's channel.

    The figure's event channel and all linked event channels are searched for
    subscribers to the given event. Each subscriber had provided a callback
    function when subscribing, so we call that.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure that publishes the event.
    event : UIEvent
        Event to publish.
    %(verbose)s
    """
    if fig in _disabled_event_channels:
        return

    # Compile a list of all event channels that the event should be published
    # on.
    channels = [_get_event_channel(fig)]
    links = _event_channel_links.get(fig, None)
    if links is not None:
        for linked_fig, (include_events, exclude_events) in links.items():
            if (include_events is None or event.name in include_events) and (
                exclude_events is None or event.name not in exclude_events
            ):
                channels.append(_get_event_channel(linked_fig))

    # Publish the event by calling the registered callback functions.
    event.source = fig
    logger.debug(f"Publishing {event} on channel {fig}")
    for channel in channels:
        if event.name not in channel:
            channel[event.name] = set()
        for callback in channel[event.name]:
            callback(event=event)


@verbose
def subscribe(fig, event_name, callback, *, verbose=None):
    """Subscribe to an event on a figure's event channel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure of which event channel to subscribe.
    event_name : str
        The name of the event to listen for.
    callback : callable
        The function that should be called whenever the event is published.
    %(verbose)s
    """
    channel = _get_event_channel(fig)
    logger.debug(f"Subscribing to channel {channel}")
    if event_name not in channel:
        channel[event_name] = set()
    channel[event_name].add(callback)


@verbose
def unsubscribe(fig, event_names, callback=None, *, verbose=None):
    """Unsubscribe from an event on a figure's event channel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure of which event channel to unsubscribe from.
    event_names : str | list of str
        Select which events to stop subscribing to. Can be a single string
        event name, a list of event names or ``"all"`` which will unsubscribe
        from all events.
    callback : callable | None
        The callback function that should be unsubscribed, leaving all other
        callback functions that may be subscribed untouched. By default
        (``None``) all callback functions are unsubscribed from the event.
    %(verbose)s
    """
    channel = _get_event_channel(fig)

    # Determine which events to unsubscribe for.
    if event_names == "all":
        if callback is None:
            event_names = list(channel.keys())
        else:
            event_names = list(k for k, v in channel.items() if callback in v)
    elif isinstance(event_names, str):
        event_names = [event_names]

    for event_name in event_names:
        if event_name not in channel:
            warn(
                f'Cannot unsubscribe from event "{event_name}" as we have never '
                "subscribed to it."
            )
            continue

        if callback is None:
            del channel[event_name]
        else:
            # Unsubscribe specific callback function.
            subscribers = channel[event_name]
            if callback in subscribers:
                subscribers.remove(callback)
            else:
                warn(
                    f'Cannot unsubscribe {callback} from event "{event_name}" '
                    "as it was never subscribed to it."
                )
            if len(subscribers) == 0:
                del channel[event_name]  # keep things tidy


@verbose
def link(*figs, include_events=None, exclude_events=None, verbose=None):
    """Link the event channels of two figures together.

    When event channels are linked, any events that are published on one
    channel are simultaneously published on the other channel. Links are
    bi-directional.

    Parameters
    ----------
    *figs : tuple of matplotlib.figure.Figure | tuple of Figure3D
        The figures whose event channel will be linked.
    include_events : list of str | None
        Select which events to publish across figures. By default (``None``),
        both figures will receive all of each other's events. Passing a list of
        event names will restrict the events being shared across the figures to
        only the given ones.
    exclude_events : list of str | None
        Select which events not to publish across figures. By default (``None``),
        no events are excluded.
    %(verbose)s
    """
    if include_events is not None:
        include_events = set(include_events)
    if exclude_events is not None:
        exclude_events = set(exclude_events)

    # Make sure the event channels of the figures are setup properly.
    for fig in figs:
        _get_event_channel(fig)
        if fig not in _event_channel_links:
            _event_channel_links[fig] = weakref.WeakKeyDictionary()

    # Link the event channels
    for fig1 in figs:
        for fig2 in figs:
            if fig1 is not fig2:
                _event_channel_links[fig1][fig2] = (include_events, exclude_events)


@verbose
def unlink(fig, *, verbose=None):
    """Remove all links involving the event channel of the given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure whose event channel should be unlinked from all other event
        channels.
    %(verbose)s
    """
    linked_figs = _event_channel_links.get(fig)
    if linked_figs is not None:
        for linked_fig in linked_figs.keys():
            del _event_channel_links[linked_fig][fig]
            if len(_event_channel_links[linked_fig]) == 0:
                del _event_channel_links[linked_fig]
    if fig in _event_channel_links:  # need to check again because of weak refs
        del _event_channel_links[fig]


@contextlib.contextmanager
def disable_ui_events(fig):
    """Temporarily disable generation of UI events. Use as context manager.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure whose UI event generation should be temporarily disabled.
    """
    _disabled_event_channels.add(fig)
    try:
        yield
    finally:
        _disabled_event_channels.remove(fig)


def _cleanup_agg():
    """Call close_event for Agg canvases to help our doc build."""
    import matplotlib.backends.backend_agg
    import matplotlib.figure

    for key in list(_event_channels):  # we might remove keys as we go
        if isinstance(key, matplotlib.figure.Figure):
            canvas = key.canvas
            if isinstance(canvas, matplotlib.backends.backend_agg.FigureCanvasAgg):
                for cb in key.canvas.callbacks.callbacks["close_event"].values():
                    cb = cb()  # get the true ref
                    if cb is not None:
                        cb()
