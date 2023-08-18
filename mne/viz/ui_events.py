"""
Event API for inter-figure communication.

The event API allows figures to communicate with each other, such that a change
in one figure can trigger a change in another figure. For example, moving the
time cursor in one plot can update the current time in another plot. Another
scenario is two drawing routines drawing into the same window, using events to
stay in-sync.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import contextlib
from dataclasses import dataclass
from typing import Optional
from weakref import WeakKeyDictionary, WeakSet
import re

from ..utils import warn, fill_doc


# Global dict {fig: channel} containing all currently active event channels.
_event_channels = WeakKeyDictionary()

# The event channels of figures can be linked together. This dict keeps track
# of these links. Links are bi-directional, so if {fig1: fig2} exists, then so
# must {fig2: fig1}.
_event_channel_links = WeakKeyDictionary()

# Event channels that are temporarily disabled by the disable_ui_events context
# manager.
_disabled_event_channels = WeakSet()

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
    %(fmin_fmid_fmax)s
    %(alpha)s

    Attributes
    ----------
    %(ui_event_name_source)s
    %(fmin_fmid_fmax)s
    %(alpha)s
    """

    fmin: Optional[float] = None
    fmax: Optional[float] = None
    fmid: Optional[float] = None
    alpha: Optional[bool] = None


@dataclass
@fill_doc
class CameraMove(UIEvent):
    """Indicates that the user has moved the 3D camera.

    Parameters
    ----------
    %(roll)s
    %(distance)s
    %(azimuth)s
    %(elevation)s
    %(focalpoint)s

    Attributes
    ----------
    %(ui_event_name_source)s
    %(roll)s
    %(distance)s
    %(azimuth)s
    %(elevation)s
    %(focalpoint)s
    """

    roll: float = None
    distance: float = None
    azimuth: float = None
    elevation: float = None
    focalpoint: tuple = None


@dataclass
@fill_doc
class VertexSelect(UIEvent):
    """Indicates that the user has selected a vertex.

    Parameters
    ----------
    hemi : str
        The hemisphere the vertex was selected on.
        Can be ``"lh"``, ``"rh"``, or ``"vol"``
    vertex : int
        The vertex number (in the high resolution mesh) that was selected.

    Attributes
    ----------
    %(ui_event_name_source)s
    hemi : str
        The hemisphere the vertex was selected on.
        Can be ``"lh"``, ``"rh"``, or ``"vol"``
    vertex_id : int
        The vertex number (in the high resolution mesh) that was selected.
    """

    hemi: str
    vertex_id: int


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
    from . import Brain

    # Create the event channel if it doesn't exist yet
    if fig not in _event_channels:
        # The channel itself is a dict mapping string event names to a list of
        # subscribers. No subscribers yet for this new event channel.
        _event_channels[fig] = dict()

        # When the figure is closed, its associated event channel should be
        # deleted. This is a good time to set this up.
        def delete_event_channel(event=None):
            """Delete the event channel (callback function)."""
            publish(fig, event=FigureClosing())  # Notify subscribers of imminent close
            unlink(fig)  # Remove channel from the _event_channel_links dict
            if fig in _event_channels:
                del _event_channels[fig]

        # Hook up the above callback function to the close event of the figure
        # window. How this is done exactly depends on the various figure types
        # MNE-Python has.
        if isinstance(fig, matplotlib.figure.Figure):
            fig.canvas.mpl_connect("close_event", delete_event_channel)
        elif isinstance(fig, Brain):
            fig._renderer._window_close_connect(delete_event_channel, after=False)
        else:
            raise NotImplementedError("This figure type is not support yet.")

    # Now the event channel exists for sure.
    return _event_channels[fig]


def publish(fig, event):
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
    """
    if fig in _disabled_event_channels:
        return

    # Compile a list of all event channels that the event should be published
    # on.
    channels = [_get_event_channel(fig)]
    links = _event_channel_links.get(fig, None)
    if links is not None:
        for linked_fig, event_names in links.items():
            if event_names == "all" or event.name in event_names:
                channels.append(_get_event_channel(linked_fig))

    # Publish the event by calling the registered callback functions.
    event.source = fig
    for channel in channels:
        if event.name not in channel:
            channel[event.name] = set()
        for callback in channel[event.name]:
            callback(event=event)


def subscribe(fig, event_name, callback):
    """Subscribe to an event on a figure's event channel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure of which event channel to subscribe.
    event_name : str
        The name of the event to listen for.
    callback : callable
        The function that should be called whenever the event is published.
    """
    channel = _get_event_channel(fig)
    if event_name not in channel:
        channel[event_name] = set()
    channel[event_name].add(callback)


def unsubscribe(fig, event_names, callback=None):
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


def link(fig1, fig2, event_names="all"):
    """Link the event channels of two figures together.

    When event channels are linked, any events that are published on one
    channel are simultaneously published on the other channel. Links are
    bi-directional.

    Parameters
    ----------
    fig1 : matplotlib.figure.Figure | Figure3D
        The first figure whose event channel will be linked to the second.
    fig2 : matplotlib.figure.Figure | Figure3D
        The second figure whose event channel will be linked to the first.
    event_names : str | list of str
        Select which events to publish across figures. By default (``"all"``),
        both figures will receive all of each other's events. Passing a list of
        event names will restrict the events being shared across the figures to
        only the given ones.
    """
    if event_names != "all":
        event_names = set(event_names)

    if fig1 not in _event_channel_links:
        _event_channel_links[fig1] = WeakKeyDictionary()
    _event_channel_links[fig1][fig2] = event_names
    if fig2 not in _event_channel_links:
        _event_channel_links[fig2] = WeakKeyDictionary()
    _event_channel_links[fig2][fig1] = event_names


def unlink(fig):
    """Remove all links involving the event channel of the given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | Figure3D
        The figure whose event channel should be unlinked from all other event
        channels.
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
