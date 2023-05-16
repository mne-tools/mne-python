"""
Event API for inter-figure communication.

The event API allows figures to communicate with each other, such that a change
in one figure can trigger a change in another figure. For example, moving the
time cursor in a Brain plot can update the current time in an evoked plot.
Another scenario is two drawing routines drawing into the same window, using
events to stay in-sync.

Event channels and linking them
===============================
Each figure has an associated event channel. Drawing routines can publish
events on the channel and receive any events published by other subscribers.
When subscribing to an event, a callback function is given that will be called
whenever a drawing routine publishes that event on the event channel.

As a rule, drawing routines only subscribe and publish on the event channel
associated with the figure they are drawing. To broadcast events across
figures, we allow message channels to be "linked". When an event is published
on a channel, it is also published on all linked channels.

Event objects and their parameters
==================================
The events are modeled after matplotlib's event system. Each event has a string
name (the snake-case version of its class name) and a list of relevant values.
For example, the "time_change" event should have the new time as a value.
Values can be any python object. When publishing an event, the publisher
creates a new instance of the event's class. When subscribing to an event,
having to dig up the correct class is a bit of a hassle. Following matplotlib's
example, subscribers use the string name of the event to refer to it.

Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
from weakref import WeakKeyDictionary, WeakSet
import re

from ..utils import fill_doc


# Global dict {fig: channel} containing all currently active event channels.
_event_channels = WeakKeyDictionary()

# The event channels of figures can be linked together. This dict keeps track
# of these links. Links are bi-directional, so if {fig1: fig2} exists, then so
# must {fig2: fig1}.
_event_channel_links = WeakKeyDictionary()

# Regex pattern to convert CamelCase to snake_case
_camel_to_snake = re.compile(r"(?<!^)(?=[A-Z])")


# List of events
class UIEvent:
    """Abstract base class for all events."""

    def __init__(self):
        self.name = _camel_to_snake.sub("_", self.__class__.__name__).lower()


class FigureClosing(UIEvent):
    """Indicates that the user has requested to close a figure."""

    pass


class TimeChange(UIEvent):
    """Indicates that the user has selected a time."""

    def __init__(self, time):
        super().__init__()
        self.time = time


@fill_doc
def _get_event_channel(fig):
    """Get the event channel associated with a figure.

    If the event channel doesn't exist yet, it gets created and added to the
    global ``_event_channels`` dict.

    Parameters
    ----------
    fig : %(figure)s
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
            fig._renderer._window_close_connect(delete_event_channel)
        else:
            raise NotImplementedError("This figure type is not support yet.")

    # Now the event channel exists for sure.
    return _event_channels[fig]


@fill_doc
def publish(fig, event):
    """Publish an event to all subscribers of the figure's channel.

    The figure's event channel and all linked event channels are searched for
    subscribers to the given event. Each subscriber had provided a callback
    function when subscribing, so we call that.

    Parameters
    ----------
    fig : %(figure)s
        The figure that publishes the event.
    event : UIEvent
        Event to publish.
    """
    # Compile a list of all event channels that the event should be published
    # on.
    channels = [_get_event_channel(fig)]
    if fig in _event_channel_links:
        linked_channels = [
            _get_event_channel(linked_fig) for linked_fig in _event_channel_links[fig]
        ]
        channels.extend(linked_channels)

    # Publish the event by calling the registered callback functions.
    event.source = fig
    for channel in channels:
        if event.name not in channel:
            channel[event.name] = set()
        for callback in channel[event.name]:
            callback(event=event)


@fill_doc
def subscribe(fig, event_name, callback):
    """Subscribe to an event on a figure's event channel.

    Parameters
    ----------
    fig : %(figure)s
        The figure of which event channel to subscribe.
    event_name : str
        The name of the event to listen for.
    callback : func
        The function that should be called whenever the event is published.
    """
    channel = _get_event_channel(fig)
    if event_name not in channel:
        channel[event_name] = set()
    channel[event_name].add(callback)


@fill_doc
def link(fig1, fig2):
    """Link the event channels of two figures together.

    When event channels are linked, any events that are published on one
    channel are simultaneously published on the other channel. Links are
    bi-directional.

    Parameters
    ----------
    fig1 : %(figure)s
        The first figure whose event channel will be linked to the second.
    fig2 : %(figure)s
        The second figure whose event channel will be linked to the first.
    """
    if fig1 not in _event_channel_links:
        _event_channel_links[fig1] = WeakSet([fig2])
    else:
        _event_channel_links[fig1].add(fig2)
    if fig2 not in _event_channel_links:
        _event_channel_links[fig2] = WeakSet([fig1])
    else:
        _event_channel_links[fig2].add(fig1)


def unlink(fig):
    """Remove all links involving the event channel of the given figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | mne.viz.Brain
        The figure whose event channel should be unlinked from all other event
        channels.
    """
    linked_figs = _event_channel_links.get(fig)
    if linked_figs is not None:
        for linked_fig in linked_figs:
            _event_channel_links[linked_fig].remove(fig)
            if len(_event_channel_links[linked_fig]) == 0:
                del _event_channel_links[linked_fig]
    if fig in _event_channel_links:
        del _event_channel_links[fig]
