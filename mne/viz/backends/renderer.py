"""Core visualization operations."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import importlib
import time
from contextlib import contextmanager
from functools import partial

import numpy as np

from ...utils import (
    _auto_weakref,
    _check_option,
    _validate_type,
    fill_doc,
    get_config,
    logger,
    verbose,
)
from .._3d import _get_3d_option
from ..utils import safe_event
from ._utils import VALID_3D_BACKENDS

MNE_3D_BACKEND = None
MNE_3D_BACKEND_TESTING = False


_backend_name_map = dict(
    pyvistaqt="._qt",
    notebook="._notebook",
)
backend = None


def _reload_backend(backend_name):
    global backend
    backend = importlib.import_module(
        name=_backend_name_map[backend_name], package="mne.viz.backends"
    )
    logger.info(f"Using {backend_name} 3d backend.")


def _get_backend():
    _get_3d_backend()
    return backend


def _get_renderer(*args, **kwargs):
    _get_3d_backend()
    return backend._Renderer(*args, **kwargs)


def _check_3d_backend_name(backend_name):
    _validate_type(backend_name, str, "backend_name")
    backend_name = "pyvistaqt" if backend_name == "pyvista" else backend_name
    _check_option("backend_name", backend_name, VALID_3D_BACKENDS)
    return backend_name


@verbose
def set_3d_backend(backend_name, verbose=None):
    """Set the 3D backend for MNE.

    The backend will be set as specified and operations will use
    that backend.

    Parameters
    ----------
    backend_name : str
        The 3d backend to select. See Notes for the capabilities of each
        backend (``'pyvistaqt'`` and ``'notebook'``).

        .. versionchanged:: 0.24
           The ``'pyvista'`` backend was renamed ``'pyvistaqt'``.
    %(verbose)s

    Returns
    -------
    old_backend_name : str | None
        The old backend that was in use.

    Notes
    -----
    To use PyVista, set ``backend_name`` to ``pyvistaqt`` but the value
    ``pyvista`` is still supported for backward compatibility.

    This table shows the capabilities of each backend ("✓" for full support,
    and "-" for partial support):

    .. table::
       :widths: auto

       +--------------------------------------+-----------+----------+
       | **3D function:**                     | pyvistaqt | notebook |
       +======================================+===========+==========+
       | :func:`plot_vector_source_estimates` | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_source_estimates`        | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_alignment`               | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_sparse_source_estimates` | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`plot_evoked_field`            | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`snapshot_brain_montage`       | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | :func:`link_brains`                  | ✓         |          |
       +--------------------------------------+-----------+----------+
       +--------------------------------------+-----------+----------+
       | **Feature:**                                                |
       +--------------------------------------+-----------+----------+
       | Large data                           | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Opacity/transparency                 | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Support geometric glyph              | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Smooth shading                       | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Subplotting                          | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in Jupyter Notebook      |           | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in JupyterLab            |           | ✓        |
       +--------------------------------------+-----------+----------+
       | Inline plot in Google Colab          |           |          |
       +--------------------------------------+-----------+----------+
       | Toolbar                              | ✓         | ✓        |
       +--------------------------------------+-----------+----------+
    """
    global MNE_3D_BACKEND
    old_backend_name = MNE_3D_BACKEND
    backend_name = _check_3d_backend_name(backend_name)
    if MNE_3D_BACKEND != backend_name:
        _reload_backend(backend_name)
        MNE_3D_BACKEND = backend_name
    return old_backend_name


def get_3d_backend():
    """Return the 3D backend currently used.

    Returns
    -------
    backend_used : str | None
        The 3d backend currently in use. If no backend is found,
        returns ``None``.

        .. versionchanged:: 0.24
           The ``'pyvista'`` backend has been renamed ``'pyvistaqt'``, so
           ``'pyvista'`` is no longer returned by this function.
    """
    try:
        backend = _get_3d_backend()
    except RuntimeError as exc:
        backend = None
        logger.info(str(exc))
    return backend


def _get_3d_backend():
    """Load and return the current 3d backend."""
    global MNE_3D_BACKEND
    if MNE_3D_BACKEND is None:
        MNE_3D_BACKEND = get_config(key="MNE_3D_BACKEND", default=None)
        if MNE_3D_BACKEND is None:  # try them in order
            errors = dict()
            for name in VALID_3D_BACKENDS:
                try:
                    _reload_backend(name)
                except ImportError as exc:
                    errors[name] = str(exc)
                else:
                    MNE_3D_BACKEND = name
                    break
            else:
                raise RuntimeError(
                    "Could not load any valid 3D backend\n"
                    + "\n".join(f"{key}: {val}" for key, val in errors.items())
                    + "\n".join(
                        (
                            "\n\n install pyvistaqt, using pip or conda:",
                            "'pip install pyvistaqt'",
                            "'conda install -c conda-forge pyvistaqt'",
                            "\n or install ipywidgets, "
                            + "if using a notebook backend",
                            "'pip install ipywidgets'",
                            "'conda install -c conda-forge ipywidgets'",
                        )
                    )
                )

        else:
            MNE_3D_BACKEND = _check_3d_backend_name(MNE_3D_BACKEND)
            _reload_backend(MNE_3D_BACKEND)
    MNE_3D_BACKEND = _check_3d_backend_name(MNE_3D_BACKEND)
    return MNE_3D_BACKEND


@contextmanager
def use_3d_backend(backend_name):
    """Create a 3d visualization context using the designated backend.

    See :func:`mne.viz.set_3d_backend` for more details on the available
    3d backends and their capabilities.

    Parameters
    ----------
    backend_name : {'pyvistaqt', 'notebook'}
        The 3d backend to use in the context.
    """
    old_backend = set_3d_backend(backend_name)
    try:
        yield
    finally:
        if old_backend is not None:
            try:
                set_3d_backend(old_backend)
            except Exception:
                pass


@contextmanager
def _use_test_3d_backend(backend_name, interactive=False):
    """Create a testing viz context.

    Parameters
    ----------
    backend_name : str
        The 3d backend to use in the context.
    interactive : bool
        If True, ensure interactive elements are accessible.
    """
    with _actors_invisible():
        with use_3d_backend(backend_name):
            with backend._testing_context(interactive):
                yield


@contextmanager
def _actors_invisible():
    global MNE_3D_BACKEND_TESTING
    orig_testing = MNE_3D_BACKEND_TESTING
    MNE_3D_BACKEND_TESTING = True
    try:
        yield
    finally:
        MNE_3D_BACKEND_TESTING = orig_testing


@fill_doc
def set_3d_view(
    figure,
    azimuth=None,
    elevation=None,
    focalpoint=None,
    distance=None,
    roll=None,
):
    """Configure the view of the given scene.

    Parameters
    ----------
    figure : object
        The scene which is modified.
    %(azimuth)s
    %(elevation)s
    %(focalpoint)s
    %(distance)s
    %(roll)s
    """
    backend._set_3d_view(
        figure=figure,
        azimuth=azimuth,
        elevation=elevation,
        focalpoint=focalpoint,
        distance=distance,
        roll=roll,
    )


@fill_doc
def set_3d_title(figure, title, size=40, *, color="white", position="upper_left"):
    """Configure the title of the given scene.

    Parameters
    ----------
    figure : object
        The scene which is modified.
    title : str
        The title of the scene.
    size : int
        The size of the title.
    color : matplotlib color
        The color of the title.

        .. versionadded:: 1.9
    position : str
        The position to use, e.g., "upper_left". See
        :meth:`pyvista.Plotter.add_text` for details.

        .. versionadded:: 1.9

    Returns
    -------
    text : object
        The text object returned by the given backend.

        .. versionadded:: 1.0
    """
    return backend._set_3d_title(
        figure=figure, title=title, size=size, color=color, position=position
    )


def create_3d_figure(
    size,
    bgcolor=(0, 0, 0),
    smooth_shading=None,
    handle=None,
    *,
    scene=True,
    show=False,
    title="MNE 3D Figure",
):
    """Return an empty figure based on the current 3d backend.

    .. warning:: Proceed with caution when the renderer object is
                 returned (with ``scene=False``) because the _Renderer
                 API is not necessarily stable enough for production,
                 it's still actively in development.

    Parameters
    ----------
    size : tuple
        The dimensions of the 3d figure (width, height).
    bgcolor : tuple
        The color of the background.
    smooth_shading : bool | None
        Whether to enable smooth shading. If ``None``, uses the config value
        ``MNE_3D_OPTION_SMOOTH_SHADING``. Defaults to ``None``.
    handle : int | None
        The figure identifier.
    scene : bool
        If True (default), the returned object is the Figure3D. If False,
        an advanced, undocumented Renderer object is returned (the API is not
        stable or documented, so this is not recommended).
    show : bool
        If True, show the renderer immediately.

        .. versionadded:: 1.0
    title : str
        The window title to use (if applicable).

        .. versionadded:: 1.9

    Returns
    -------
    figure : instance of Figure3D or ``Renderer``
        The requested empty figure or renderer, depending on ``scene``.
    """
    _validate_type(smooth_shading, (bool, None), "smooth_shading")
    if smooth_shading is None:
        smooth_shading = _get_3d_option("smooth_shading")
    renderer = _get_renderer(
        fig=handle,
        size=size,
        bgcolor=bgcolor,
        smooth_shading=smooth_shading,
        show=show,
        name=title,
    )
    if scene:
        return renderer.scene()
    else:
        return renderer


def close_3d_figure(figure):
    """Close the given scene.

    Parameters
    ----------
    figure : object
        The scene which needs to be closed.
    """
    backend._close_3d_figure(figure)


def close_all_3d_figures():
    """Close all the scenes of the current 3d backend."""
    backend._close_all()


def get_brain_class():
    """Return the proper Brain class based on the current 3d backend.

    Returns
    -------
    brain : object
        The Brain class corresponding to the current 3d backend.
    """
    from ...viz._brain import Brain

    return Brain


class _TimeInteraction:
    """Mixin enabling time interaction controls."""

    def _enable_time_interaction(
        self,
        fig,
        current_time_func,
        times,
        init_playback_speed=0.01,
        playback_speed_range=(0.01, 0.1),
    ):
        from ..ui_events import (
            PlaybackSpeed,
            TimeChange,
            publish,
            subscribe,
        )

        self._fig = fig
        self._current_time_func = current_time_func
        self._times = times
        self._init_time = current_time_func()
        self._init_playback_speed = init_playback_speed

        if not hasattr(self, "_dock"):
            self._dock_initialize()

        if not hasattr(self, "_tool_bar") or self._tool_bar is None:
            self._tool_bar_initialize(name="Toolbar")

        if not hasattr(self, "_widgets"):
            self._widgets = dict()

        # Dock widgets
        @_auto_weakref
        def publish_time_change(time_index):
            publish(
                fig,
                TimeChange(time=np.interp(time_index, np.arange(len(times)), times)),
            )

        layout = self._dock_add_group_box("")
        self._widgets["time_slider"] = self._dock_add_slider(
            name="Time (s)",
            value=np.interp(current_time_func(), times, np.arange(len(times))),
            rng=[0, len(times) - 1],
            double=True,
            callback=publish_time_change,
            compact=False,
            layout=layout,
        )
        hlayout = self._dock_add_layout(vertical=False)
        self._widgets["min_time"] = self._dock_add_label("-", layout=hlayout)
        self._dock_add_stretch(hlayout)
        self._widgets["current_time"] = self._dock_add_label(value="x", layout=hlayout)
        self._dock_add_stretch(hlayout)
        self._widgets["max_time"] = self._dock_add_label(value="+", layout=hlayout)
        self._layout_add_widget(layout, hlayout)

        self._widgets["min_time"].set_value(f"{times[0]: .3f}")
        self._widgets["current_time"].set_value(f"{current_time_func(): .3f}")
        self._widgets["max_time"].set_value(f"{times[-1]: .3f}")

        @_auto_weakref
        def publish_playback_speed(speed):
            publish(fig, PlaybackSpeed(speed=speed))

        self._widgets["playback_speed"] = self._dock_add_spin_box(
            name="Speed",
            value=init_playback_speed,
            rng=playback_speed_range,
            callback=publish_playback_speed,
            layout=layout,
        )

        # Tool bar buttons
        self._widgets["reset"] = self._tool_bar_add_button(
            name="reset", desc="Reset", func=self._reset_time
        )
        self._widgets["play"] = self._tool_bar_add_play_button(
            name="play",
            desc="Play/Pause",
            func=self._toggle_playback,
            shortcut=" ",
        )

        # Configure playback
        self._playback = False
        self._playback_initialize(
            func=self._play,
            timeout=17,
            value=np.interp(current_time_func(), times, np.arange(len(times))),
            rng=[0, len(times) - 1],
            time_widget=self._widgets["time_slider"],
            play_widget=self._widgets["play"],
        )

        # Keyboard shortcuts
        @_auto_weakref
        def shift_time(direction):
            amount = self._widgets["playback_speed"].get_value()
            publish(
                self._fig,
                TimeChange(time=self._current_time_func() + direction * amount),
            )

        if self.plotter.iren is not None:
            self.plotter.add_key_event("n", partial(shift_time, direction=1))
            self.plotter.add_key_event("b", partial(shift_time, direction=-1))

        # Subscribe to relevant UI events
        subscribe(fig, "time_change", self._on_time_change)
        subscribe(fig, "playback_speed", self._on_playback_speed)

    def _on_time_change(self, event):
        """Respond to time_change UI event."""
        from ..ui_events import disable_ui_events

        new_time = np.clip(event.time, self._times[0], self._times[-1])
        new_time_idx = np.interp(new_time, self._times, np.arange(len(self._times)))

        with disable_ui_events(self._fig):
            self._widgets["time_slider"].set_value(new_time_idx)
            self._widgets["current_time"].set_value(f"{new_time:.3f}")

    def _on_playback_speed(self, event):
        """Respond to playback_speed UI event."""
        from ..ui_events import disable_ui_events

        with disable_ui_events(self._fig):
            self._widgets["playback_speed"].set_value(event.speed)

    def _toggle_playback(self, value=None):
        """Toggle time playback."""
        from ..ui_events import TimeChange, publish

        if value is None:
            self._playback = not self._playback
        else:
            self._playback = value

        if self._playback:
            self._tool_bar_update_button_icon(name="play", icon_name="pause")
            if self._current_time_func() == self._times[-1]:  # start over
                publish(self._fig, TimeChange(time=self._times[0]))
            self._last_tick = time.time()
        else:
            self._tool_bar_update_button_icon(name="play", icon_name="play")

    def _reset_time(self):
        """Reset time and playback speed to initial values."""
        from ..ui_events import PlaybackSpeed, TimeChange, publish

        publish(self._fig, TimeChange(time=self._init_time))
        publish(self._fig, PlaybackSpeed(speed=self._init_playback_speed))

    @safe_event
    def _play(self):
        if self._playback:
            try:
                self._advance()
            except Exception:
                self._toggle_playback(value=False)
                raise

    def _advance(self):
        from ..ui_events import TimeChange, publish

        this_time = time.time()
        delta = this_time - self._last_tick
        self._last_tick = time.time()
        time_shift = delta * self._widgets["playback_speed"].get_value()
        new_time = min(self._current_time_func() + time_shift, self._times[-1])
        publish(self._fig, TimeChange(time=new_time))
        if new_time == self._times[-1]:
            self._toggle_playback(value=False)
