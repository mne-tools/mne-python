# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: Simplified BSD
import matplotlib.pyplot as plt
import pytest

from mne.datasets import testing
from mne.viz import ui_events, Brain

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

    # Test different types of figures
    fig = Brain("sample", subjects_dir=subjects_dir, surf="white")
    ui_events._get_event_channel(fig)
    assert len(event_channels) == 1
    assert fig in event_channels
    fig.close()
    assert len(event_channels) == 0

    # TODO: MNEFigure, Figure3D


def test_publish(event_channels):
    """Test publishing UI events."""
    fig = plt.figure()
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))

    # Publishing the event should have created the needed channel.
    assert len(event_channels) == 1
    assert fig in event_channels


def test_subscribe(event_channels):
    """Test subscribing to UI events."""
    global callback_called
    callback_called = False

    def callback(event):
        """Respond to time change event."""
        global callback_called
        callback_called = True
        assert isinstance(event, ui_events.TimeChange)
        assert event.time == 10.2

    fig = plt.figure()
    ui_events.subscribe(fig, "time_change", callback)

    # Subscribing to the event should have created the needed channel.
    assert "time_change" in ui_events._get_event_channel(fig)

    # Publishing the time change event should call the callback function.
    ui_events.publish(fig, ui_events.TimeChange(time=10.2))
    assert callback_called

    # Publishing a different event should not call the callback function.
    callback_called = False  # Reset
    ui_events.publish(fig, ui_events.FigureClosing())
    assert not callback_called

    # Test disposing of the event channel, even with subscribers.
    # During tests, matplotlib does not open an actual window so we need to force the
    # close event.
    fig.canvas.callbacks.process("close_event", None)
    assert len(event_channels) == 0


def test_link(event_channels, event_channel_links):
    """Test linking the event channels of two functions."""
    fig1 = plt.figure()
    fig2 = plt.figure()

    global num_callbacks_called
    num_callbacks_called = 0

    def callback(event):
        """Respond to time change event."""
        global num_callbacks_called
        num_callbacks_called += 1

    # Both figures are subscribed to the time change events.
    ui_events.subscribe(fig1, "time_change", callback)
    ui_events.subscribe(fig2, "time_change", callback)

    # Linking the event channels causes events to be published on both channels.
    ui_events.link(fig1, fig2)
    assert len(event_channel_links) == 2
    assert fig2 in event_channel_links[fig1]
    assert fig1 in event_channel_links[fig2]

    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    assert num_callbacks_called == 2

    num_callbacks_called = 0
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert num_callbacks_called == 2

    # Test linking only specific events
    ui_events.link(fig1, fig2, ["time_change"])
    num_callbacks_called = 0
    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert num_callbacks_called == 4  # Called for both figures two times

    ui_events.link(fig1, fig2, ["some_other_event"])
    num_callbacks_called = 0
    ui_events.publish(fig1, ui_events.TimeChange(time=10.2))
    ui_events.publish(fig2, ui_events.TimeChange(time=10.2))
    assert num_callbacks_called == 2  # Only called for both figures once

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
