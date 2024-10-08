# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import matplotlib.pyplot as plt
import pytest

from mne.datasets import testing
from mne.viz import ui_events

subjects_dir = testing.data_path(download=False) / "subjects"


@pytest.fixture
def event_channels():
    """Fixture that makes sure each test starts with a fresh UI event chans dict."""
    ui_events._event_channels.clear()
    return ui_events._event_channels


@pytest.fixture
def event_channel_links():
    """Fixture that makes sure each test starts with a fresh channel links dict."""
    ui_events._event_channel_links.clear()
    return ui_events._event_channel_links


@pytest.fixture
def disabled_event_channels():
    """Fixture that makes sure each test starts with a fresh disabled channels set."""
    ui_events._disabled_event_channels.clear()
    return ui_events._disabled_event_channels


@testing.requires_testing_data
def test_get_event_channel(event_channels):
    """Test creating and obtaining a figure's UI event channel."""
    # At first, no event channels exist
    assert len(event_channels) == 0

    # Open a figure and get the event channel. This should create it.
    fig = plt.figure()
    ui_events._get_event_channel(fig)
    assert len(event_channels) == 1
    assert fig in event_channels

    # Closing a figure should delete the event channel.
    # During tests, matplotlib does not open an actual window so we need to force the
    # close event.
    fig.canvas.callbacks.process("close_event", None)
    assert len(event_channels) == 0


def test_publish(event_channels):
    """Test publishing UI events."""
    fig = plt.figure()
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))

    # Publishing the event should have created the needed channel.
    assert len(event_channels) == 1
    assert fig in event_channels


def test_subscribe(event_channels):
    """Test subscribing to UI events."""
    callback_calls = list()

    def callback(event):
        """Respond to time change event."""
        callback_calls.append(event)
        assert isinstance(event, ui_events.TimeChange)
        assert event.time == 10.2

    fig = plt.figure()
    ui_events.subscribe(fig, "time_change", callback)

    # Subscribing to the event should have created the needed channel.
    assert "time_change" in ui_events._get_event_channel(fig)

    # Publishing the time change event should call the callback function.
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert callback_calls

    # Publishing a different event should not call the callback function.
    callback_calls.clear()  # Reset
    ui_events.publish(fig, ui_events.FigureClosing())
    assert not callback_calls

    # Test disposing of the event channel, even with subscribers.
    # During tests, matplotlib does not open an actual window so we need to force the
    # close event.
    fig.canvas.callbacks.process("close_event", None)
    assert len(event_channels) == 0


def test_unsubscribe(event_channels):
    """Test unsubscribing from UI events."""
    callback1_calls = list()
    callback2_calls = list()

    def callback1(event):
        """Respond to time change event."""
        callback1_calls.append(event)

    def callback2(event):
        """Respond to time change event."""
        callback2_calls.append(event)

    fig = plt.figure()

    def setup_events():
        """Reset UI event scenario."""
        callback1_calls.clear()
        callback2_calls.clear()
        ui_events.unsubscribe(fig, "all")
        ui_events.subscribe(fig, "figure_closing", callback1)
        ui_events.subscribe(fig, "time_change", callback1)
        ui_events.subscribe(fig, "time_change", callback2)

    # Test unsubscribing from a single event
    setup_events()
    with pytest.warns(RuntimeWarning, match="Cannot unsubscribe"):
        ui_events.unsubscribe(fig, "nonexisting_event")
    ui_events.unsubscribe(fig, "time_change")
    assert "time_change" not in ui_events._get_event_channel(fig)
    assert "figure_closing" in ui_events._get_event_channel(fig)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert not callback1_calls
    assert not callback2_calls
    ui_events.publish(fig, ui_events.FigureClosing())
    assert callback1_calls

    # Test unsubscribing from all events
    setup_events()
    ui_events.unsubscribe(fig, "all")
    assert "time_change" not in ui_events._get_event_channel(fig)
    assert "figure_closing" not in ui_events._get_event_channel(fig)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig, ui_events.FigureClosing())
    assert not callback1_calls
    assert not callback2_calls

    # Test unsubscribing from a list of events
    setup_events()
    ui_events.unsubscribe(fig, ["time_change", "figure_closing"])
    assert "time_change" not in ui_events._get_event_channel(fig)
    assert "figure_closing" not in ui_events._get_event_channel(fig)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig, ui_events.FigureClosing())
    assert not callback1_calls
    assert not callback2_calls

    # Test unsubscribing a specific callback function from a single event
    setup_events()
    with pytest.warns(RuntimeWarning, match="Cannot unsubscribe"):
        ui_events.unsubscribe(fig, "figure_closing", callback2)
    ui_events.unsubscribe(fig, "time_change", callback2)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert callback1_calls
    assert not callback2_calls

    # Test unsubscribing a specific callback function from all events
    setup_events()
    ui_events.unsubscribe(fig, "all", callback2)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert callback1_calls
    assert not callback2_calls

    # Test unsubscribing a specific callback function from a list of events
    setup_events()
    ui_events.unsubscribe(fig, ["time_change", "figure_closing"], callback1)
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig, ui_events.FigureClosing())
    assert not callback1_calls


def test_link(event_channels, event_channel_links):
    """Test linking the event channels of two functions."""
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    callback_calls = list()

    def callback(event):
        """Respond to time change event."""
        callback_calls.append(event)

    # Both figures are subscribed to the time change events.
    ui_events.subscribe(fig1, "time_change", callback)
    ui_events.subscribe(fig2, "time_change", callback)
    ui_events.subscribe(fig3, "time_change", callback)

    # Linking the event channels causes events to be published on both channels.
    ui_events.link(fig1, fig2, fig3)
    assert len(event_channel_links) == 3
    assert fig2 in event_channel_links[fig1]
    assert fig3 in event_channel_links[fig1]
    assert fig1 in event_channel_links[fig2]
    assert fig3 in event_channel_links[fig2]
    assert fig1 in event_channel_links[fig3]
    assert fig2 in event_channel_links[fig3]

    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 3

    callback_calls.clear()
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 3

    callback_calls.clear()
    ui_events.publish(fig3, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 3

    # Closing this should remove the links as well
    fig3.canvas.callbacks.process("close_event", None)

    # Test linking only specific events
    ui_events.link(fig1, fig2, include_events=["time_change"])
    callback_calls.clear()
    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 4  # Called for both figures two times

    ui_events.link(fig1, fig2, include_events=["some_other_event"])
    callback_calls.clear()
    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 2  # Only called for both figures once

    ui_events.link(fig1, fig2, exclude_events=["time_change"])
    callback_calls.clear()
    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert len(callback_calls) == 2  # Only called for both figures once

    # Test cleanup
    fig1.canvas.callbacks.process("close_event", None)
    fig2.canvas.callbacks.process("close_event", None)
    assert len(event_channels) == 0
    assert len(event_channel_links) == 0


def test_unlink(event_channel_links):
    """Test unlinking event channels."""
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ui_events.link(fig1, fig2)
    ui_events.link(fig2, fig3)
    assert len(event_channel_links) == 3

    # Fig1 is involved in two of the 4 links.
    ui_events.unlink(fig1)
    assert len(event_channel_links) == 2
    assert fig1 not in event_channel_links[fig2]
    assert fig1 not in event_channel_links[fig3]
    ui_events.link(fig1, fig2)  # Relink for the next test.

    # Fig2 is involved in all links, unlinking it should clear them all.
    ui_events.unlink(fig2)
    assert len(event_channel_links) == 0


def test_disable_ui_events(event_channels, disabled_event_channels):
    """Test disable_ui_events context manager."""
    callback_calls = list()

    def callback(event):
        """Respond to time change event."""
        callback_calls.append(event)

    fig = plt.figure()
    ui_events.subscribe(fig, "time_change", callback)
    with ui_events.disable_ui_events(fig):
        ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert not callback_calls
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert callback_calls
