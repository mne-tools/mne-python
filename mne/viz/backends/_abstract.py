"""ABCs."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import warnings
from abc import ABC, abstractmethod

from ..ui_events import TimeChange, publish
from ..utils import _BlitManager


class Figure3D(ABC):
    """Class that refers to a 3D figure.

    .. note::
        This class should not be instantiated directly via
        ``mne.viz.Figure3D(...)``. Instead, use
        :func:`mne.viz.create_3d_figure`.

    See Also
    --------
    mne.viz.create_3d_figure
    """

    # Here we use _init rather than __init__ so that users are less tempted to
    # instantiate the class directly. It also helps us
    # document the class more easily, as we don't have to say what all the
    # params are in public docs.

    @classmethod
    @abstractmethod
    def _init(
        self,
        fig=None,
        size=(600, 600),
        bgcolor=(0.0, 0.0, 0.0),
        *,
        title="MNE 3D Figure",
        show=False,
        shape=(1, 1),
        splash=False,
    ):
        pass

    @property
    def plotter(self):
        """The native 3D plotting widget.

        Returns
        -------
        plotter : instance of pyvista.Plotter
            The plotter. Useful for interacting with the native 3D library.
        """
        return self._plotter


class _AbstractRenderer(ABC):
    @classmethod
    @abstractmethod
    def __init__(
        self,
        fig=None,
        size=(600, 600),
        bgcolor=(0.0, 0.0, 0.0),
        *,
        name="MNE-Python 3D Figure",
        show=False,
        shape=(1, 1),
        splash=False,
    ):
        """Set up the scene."""
        pass

    @property
    @abstractmethod
    def _kind(self):
        pass

    @classmethod
    @abstractmethod
    def subplot(self, x, y):
        """Set the active subplot."""
        pass

    @classmethod
    @abstractmethod
    def scene(self):
        """Return scene handle."""
        pass

    @classmethod
    @abstractmethod
    def set_interaction(self, interaction):
        """Set interaction mode."""
        pass

    @classmethod
    @abstractmethod
    def legend(self, labels, size=0.1, face="triangle", loc="upper left"):
        """Add a legend to the scene.

        Parameters
        ----------
        labels : list of tuples
            Each entry must contain two strings, (label, color),
            where ``label`` is the name of the item to add, and
            ``color`` is the color of the label to add.
        size : float
            The size of the entire figure window.
        loc : str
            The location of the legend.
        face : str
            Face shape of legend face.  One of the following:

            * None: ``None``
            * Line: ``"-"`` or ``"line"``
            * Triangle: ``"^"`` or ``'triangle'``
            * Circle: ``"o"`` or ``'circle'``
            * Rectangle: ``"r"`` or ``'rectangle'``
        """
        pass

    @classmethod
    @abstractmethod
    def mesh(
        self,
        x,
        y,
        z,
        triangles,
        color,
        opacity=1.0,
        *,
        backface_culling=False,
        scalars=None,
        colormap=None,
        vmin=None,
        vmax=None,
        interpolate_before_map=True,
        representation="surface",
        line_width=1.0,
        normals=None,
        name=None,
        **kwargs,
    ):
        """Add a mesh in the scene.

        Parameters
        ----------
        x : array, shape (n_vertices,)
           The array containing the X component of the vertices.
        y : array, shape (n_vertices,)
           The array containing the Y component of the vertices.
        z : array, shape (n_vertices,)
           The array containing the Z component of the vertices.
        triangles : array, shape (n_polygons, 3)
           The array containing the indices of the polygons.
        color : tuple | str
            The color of the mesh as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        opacity : float
            The opacity of the mesh.
        shading : bool
            If True, enable the mesh shading.
        backface_culling : bool
            If True, enable backface culling on the mesh.
        scalars : ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        vmin : float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used.
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used.
        colormap : str | np.ndarray | matplotlib.colors.Colormap | None
            The colormap to use.
        interpolate_before_map :
            Enabling makes for a smoother scalars display. Default is True.
            When False, OpenGL will interpolate the mapped colors which can
            result is showing colors that are not present in the color map.
        representation : str
            The representation of the mesh: either 'surface' or 'wireframe'.
        line_width : int
            The width of the lines when representation='wireframe'.
        normals : array, shape (n_vertices, 3)
            The array containing the normal of each vertex.
        name : str | None
            The name of the mesh.
        kwargs : args
            The arguments to pass to triangular_mesh

        Returns
        -------
        surface :
            Handle of the mesh in the scene.
        """
        pass

    @classmethod
    @abstractmethod
    def contour(
        self,
        surface,
        scalars,
        contours,
        width=1.0,
        opacity=1.0,
        vmin=None,
        vmax=None,
        colormap=None,
        normalized_colormap=False,
        kind="line",
        color=None,
    ):
        """Add a contour in the scene.

        Parameters
        ----------
        surface : surface object
            The mesh to use as support for contour.
        scalars : ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        contours : int | list
             Specifying a list of values will only give the requested contours.
        width : float
            The width of the lines or radius of the tubes.
        opacity : float
            The opacity of the contour.
        vmin : float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used.
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used.
        colormap : str | np.ndarray | matplotlib.colors.Colormap | None
            The colormap to use.
        normalized_colormap : bool
            Specify if the values of the colormap are between 0 and 1.
        kind : 'line' | 'tube'
            The type of the primitives to use to display the contours.
        color : tuple | str
            The color of the mesh as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        """
        pass

    @classmethod
    @abstractmethod
    def surface(
        self,
        surface,
        color=None,
        opacity=1.0,
        vmin=None,
        vmax=None,
        colormap=None,
        normalized_colormap=False,
        scalars=None,
        backface_culling=False,
        *,
        name=None,
    ):
        """Add a surface in the scene.

        Parameters
        ----------
        surface : surface object
            The information describing the surface.
        color : tuple | str
            The color of the surface as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        opacity : float
            The opacity of the surface.
        vmin : float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used.
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used.
        colormap : str | np.ndarray | matplotlib.colors.Colormap | None
            The colormap to use.
        scalars : ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        backface_culling : bool
            If True, enable backface culling on the surface.
        name : str | None
            Name of the surface.
        """
        pass

    @classmethod
    @abstractmethod
    def sphere(
        self,
        center,
        color,
        scale,
        opacity=1.0,
        resolution=8,
        backface_culling=False,
        radius=None,
    ):
        """Add sphere in the scene.

        Parameters
        ----------
        center : ndarray, shape(n_center, 3)
            The list of centers to use for the sphere(s).
        color : tuple | str
            The color of the sphere as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scale : float
            The scaling applied to the spheres. The given value specifies
            the maximum size in drawing units.
        opacity : float
            The opacity of the sphere(s).
        resolution : int
            The resolution of the sphere created. This is the number
            of divisions along theta and phi.
        backface_culling : bool
            If True, enable backface culling on the sphere(s).
        radius : float | None
            Replace the glyph scaling by a fixed radius value for each
            sphere.
        """
        pass

    @classmethod
    @abstractmethod
    def tube(
        self,
        origin,
        destination,
        radius=0.001,
        color="white",
        scalars=None,
        vmin=None,
        vmax=None,
        colormap="RdBu",
        normalized_colormap=False,
        reverse_lut=False,
        opacity=None,
    ):
        """Add tube in the scene.

        Parameters
        ----------
        origin : array, shape(n_lines, 3)
            The coordinates of the first end of the tube(s).
        destination : array, shape(n_lines, 3)
            The coordinates of the other end of the tube(s).
        radius : float
            The radius of the tube(s).
        color : tuple | str
            The color of the tube as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scalars : array, shape (n_quivers,) | None
            The optional scalar data to use.
        vmin : float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used.
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used.
        colormap : str | np.ndarray | matplotlib.colors.Colormap | None
            The colormap to use.
        normalized_colormap : bool
            Specify if the values of the colormap are between 0 and 1.
        reverse_lut : bool
            If True, reverse the lookup table.
        opacity : float | None
            The opacity of the tube(s). If None, the renderer default is used.

        Returns
        -------
        actor :
            The actor in the scene.
        surface :
            Handle of the tube in the scene.
        """
        pass

    @classmethod
    @abstractmethod
    def quiver3d(
        self,
        x,
        y,
        z,
        u,
        v,
        w,
        color,
        scale,
        mode,
        glyph_height=None,
        glyph_center=None,
        glyph_resolution=None,
        opacity=1.0,
        scale_mode="none",
        scalars=None,
        backface_culling=False,
        colormap=None,
        vmin=None,
        vmax=None,
        line_width=2.0,
        name=None,
    ):
        """Add quiver3d in the scene.

        Parameters
        ----------
        x : array, shape (n_quivers,)
            The X component of the position of the quiver.
        y : array, shape (n_quivers,)
            The Y component of the position of the quiver.
        z : array, shape (n_quivers,)
            The Z component of the position of the quiver.
        u : array, shape (n_quivers,)
            The last X component of the quiver.
        v : array, shape (n_quivers,)
            The last Y component of the quiver.
        w : array, shape (n_quivers,)
            The last Z component of the quiver.
        color : tuple | str
            The color of the quiver as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        scale : float
            The scaling applied to the glyphs. The size of the glyph
            is by default calculated from the inter-glyph spacing.
            The given value specifies the maximum glyph size in drawing units.
        mode : 'arrow', 'cone' or 'cylinder'
            The type of the quiver.
        glyph_height : float
            The height of the glyph used with the quiver.
        glyph_center : tuple
            The center of the glyph used with the quiver: (x, y, z).
        glyph_resolution : float
            The resolution of the glyph used with the quiver.
        opacity : float
            The opacity of the quiver.
        scale_mode : 'vector', 'scalar' or 'none'
            The scaling mode for the glyph.
        scalars : array, shape (n_quivers,) | None
            The optional scalar data to use.
        backface_culling : bool
            If True, enable backface culling on the quiver.
        colormap : str | np.ndarray | matplotlib.colors.Colormap | None
            The colormap to use.
        vmin : float | None
            vmin is used to scale the colormap.
            If None, the min of the data will be used
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        line_width : float
            The width of the 2d arrows.

        Returns
        -------
        actor :
            The actor in the scene.
        surface :
            Handle of the quiver in the scene.
        """
        pass

    @classmethod
    @abstractmethod
    def instanced_mesh(
        self,
        rr,
        tris,
        positions,
        quats,
        colors,
        scales=None,
        opacity=1.0,
        backface_culling=False,
        name=None,
    ):
        """Add one mesh GPU-instanced at multiple positions/orientations.

        Unlike :meth:`quiver3d` or :meth:`sphere`, which bake a merged,
        static glyph mesh, this draws one copy of a single template mesh at
        each of ``n_instances`` locations using per-instance position,
        orientation, and color that can be changed later without rebuilding
        the geometry (see ``mesh`` in "Returns" below).

        Parameters
        ----------
        rr : array, shape (n_vertices, 3)
            The vertices of the template mesh, in the local (object) frame
            shared by all instances.
        tris : array, shape (n_tris, 3)
            The triangles of the template mesh.
        positions : array, shape (n_instances, 3)
            The position of each instance.
        quats : array, shape (n_instances, 3)
            The orientation of each instance as a unit quaternion, given as
            only the ``(x, y, z)`` vector part (``w`` omitted, recoverable
            as ``sqrt(max(1 - x**2 - y**2 - z**2, 0))``) -- the same
            convention used by :func:`~mne.transforms.rot_to_quat` /
            :func:`~mne.transforms.quat_to_rot`.
        colors : array, shape (n_instances, 4)
            The per-instance RGBA color (float values between 0 and 1) to
            use for each instance. Per-instance alpha (the fourth column)
            is respected.
        scales : array, shape (n_instances,) | None
            The per-instance isotropic scale factor applied to the template
            geometry. If ``None`` (the default), no per-instance scaling is
            applied and the size baked into ``rr`` is used as-is (e.g. for
            MEG coils).
        opacity : float
            A uniform opacity multiplier applied on top of the per-instance
            alpha values in ``colors``.
        backface_culling : bool
            If True, enable backface culling on the mesh.
        name : str | None
            Name of the mesh.

        Returns
        -------
        actor :
            The actor in the scene.
        mesh :
            The point cloud (one point per instance) backing the glyph
            mapper. Its per-instance color and orientation arrays can be
            mutated in place (followed by a render update) to recolor or
            re-orient individual instances live, without rebuilding the
            actor or its geometry.
        """
        pass

    @classmethod
    @abstractmethod
    def text2d(
        self,
        x_window,
        y_window,
        text,
        size=14,
        color="white",
        justification=None,
        font_file=None,
    ):
        """Add 2d text in the scene.

        Parameters
        ----------
        x : float
            The X component to use as position of the text in the
            window coordinates system (window_width, window_height).
        y : float
            The Y component to use as position of the text in the
            window coordinates system (window_width, window_height).
        text : str
            The content of the text.
        size : int
            The size of the font.
        color : tuple | str
            The color of the text as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        justification : str | None
            The text justification. Can be 'left', 'center', or 'right'.
        font_file : str | None
            Path to an absolute path of a font file to use for rendering
            the text. FreeType is used for loading, supporting many font
            formats beyond ``.ttf`` and ``.ttc``. This can be helpful for
            non-ASCII glyph coverage.
        """
        pass

    @classmethod
    @abstractmethod
    def text3d(self, x, y, z, text, width, color="white"):
        """Add 2d text in the scene.

        Parameters
        ----------
        x : float
            The X component to use as position of the text.
        y : float
            The Y component to use as position of the text.
        z : float
            The Z component to use as position of the text.
        text : str
            The content of the text.
        width : float
            The width of the text.
        color : tuple | str
            The color of the text as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        """
        pass

    @classmethod
    @abstractmethod
    def scalarbar(self, source, color="white", title=None, n_labels=4, bgcolor=None):
        """Add a scalar bar in the scene.

        Parameters
        ----------
        source
            The object of the scene used for the colormap.
        color : tuple | str
            The color of the label text.
        title : str | None
            The title of the scalar bar.
        n_labels : int | None
            The number of labels to display on the scalar bar.
        bgcolor : tuple | str
            The color of the background when there is transparency.

        Returns
        -------
        actor
            The scalar bar actor.
        tick_actor
            The actor drawing tick marks along the scalar bar.
        """
        pass

    @classmethod
    @abstractmethod
    def show(self):
        """Render the scene."""
        pass

    @classmethod
    @abstractmethod
    def close(self):
        """Close the scene."""
        pass

    @classmethod
    @abstractmethod
    def set_camera(
        self,
        azimuth=None,
        elevation=None,
        distance=None,
        focalpoint=None,
        roll=None,
    ):
        """Configure the camera of the scene.

        Parameters
        ----------
        azimuth : float
            The azimuthal angle of the camera.
        elevation : float
            The zenith angle of the camera.
        distance : float
            The distance to the focal point.
        focalpoint : tuple
            The focal point of the camera: (x, y, z).
        roll : float
            The rotation of the camera along its axis.
        """
        pass

    @classmethod
    @abstractmethod
    def screenshot(self, mode="rgb", filename=None):
        """Take a screenshot of the scene.

        Parameters
        ----------
        mode : str
            Either 'rgb' or 'rgba' for values to return.
            Default is 'rgb'.
        filename : str | None
            If not None, save the figure to the disk.
        """
        pass

    @classmethod
    @abstractmethod
    def project(self, xyz, ch_names):
        """Convert 3d points to a 2d perspective.

        Parameters
        ----------
        xyz : array, shape(n_points, 3)
            The points to project.
        ch_names : array, shape(_n_points,)
            Names of the channels.
        """
        pass

    @classmethod
    @abstractmethod
    def remove_mesh(self, mesh_data):
        """Remove the given mesh from the scene.

        Parameters
        ----------
        mesh_data : tuple | Surface
            The mesh to remove.
        """
        pass


# -------------------
# Widget Abstractions
# -------------------


class _AbstractWidget(ABC):
    @classmethod
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _show(self):
        pass

    @abstractmethod
    def _hide(self):
        pass

    @abstractmethod
    def _set_enabled(self, state):
        pass

    @abstractmethod
    def _is_enabled(self):
        pass

    @abstractmethod
    def _update(self, repaint=True):
        pass

    @abstractmethod
    def _set_style(self, style):
        pass

    @abstractmethod
    def _add_keypress(self, callback):
        pass

    @abstractmethod
    def _trigger_keypress(self, key):
        pass

    @abstractmethod
    def _set_focus(self):
        pass

    @abstractmethod
    def _set_theme(self, theme):
        pass

    @abstractmethod
    def _set_size(self, width=None, height=None):
        pass


class _AbstractLabel(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, center=False, selectable=False):
        pass


class _AbstractText(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value=None, placeholder=None, callback=None):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass


class _AbstractButton(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, callback, icon=None):
        pass

    @abstractmethod
    def _click(self):
        pass


class _AbstractSlider(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, rng, callback, horizontal=True):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractProgressBar(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, count):
        pass

    @abstractmethod
    def _increment(self):
        pass


class _AbstractCheckBox(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, callback):
        pass

    @abstractmethod
    def _set_checked(self, checked):
        pass

    @abstractmethod
    def _get_checked(self):
        pass


class _AbstractSpinBox(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, rng, callback, step=None):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractComboBox(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, items, callback):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractRadioButtons(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, items, callback):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractGroupBox(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, name, items):
        pass


class _AbstractFileButton(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(
        self,
        callback,
        content_filter=None,
        initial_directory=None,
        save=False,
        is_directory=False,
        icon="folder",
        window=None,
    ):
        pass


class _AbstractPlayMenu(_AbstractWidget):
    @classmethod
    @abstractmethod
    def __init__(self, value, rng, callback):
        pass

    @abstractmethod
    def _play(self):
        pass

    @abstractmethod
    def _pause(self):
        pass

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _loop(self):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass


class _AbstractPopup(_AbstractWidget):
    _supported_button_names = ["Ok"]
    # TODO: Add back support for below, file browser takes care of most
    # so no big need currently
    """
    # from QMessageBox.StandardButtons
    ['Ok', 'Open', 'Save', 'Cancel', 'Close', 'Discard', 'Apply',
    'Reset', 'RestoreDefaults', 'Help', 'SaveAll', 'Yes',
    'YesToAll', 'No', 'NoToAll', 'Abort', 'Retry', 'Ignore']
    """

    _supported_icon_names = ["question", "information", "warning", "critical"]

    @abstractmethod
    def __init__(
        self,
        title,
        text,
        info_text=None,
        callback=None,
        icon="Warning",
        buttons=None,
        window=None,
    ):
        pass

    @abstractmethod
    def _click(self, value):
        pass


# -------
# Layouts
# -------


class _AbstractBoxLayout(ABC):
    @abstractmethod
    def _add_widget(self, widget):
        pass


class _AbstractHBoxLayout(_AbstractBoxLayout):
    @abstractmethod
    def __init__(self, height=None, scroll=None):
        pass


class _AbstractVBoxLayout(_AbstractBoxLayout):
    @abstractmethod
    def __init__(self, width=None, scroll=None):
        pass


class _AbstractGridLayout(ABC):
    @abstractmethod
    def __init__(self, height=None, width=None, scroll=None):
        pass

    @abstractmethod
    def _add_widget(self, widget, row=None, col=None):
        pass


class _AbstractAppWindow(ABC):
    def __init__(self, size=None, fullscreen=False):
        pass

    @abstractmethod
    def _set_central_layout(self, central_layout):
        pass

    @abstractmethod
    def _get_size(self):
        pass

    @abstractmethod
    def _clean(self):
        pass

    @abstractmethod
    def _show(self, block=False):
        pass

    @abstractmethod
    def _close(self):
        pass


# -------------------
# Matplotlib Canvases
# -------------------


class _AbstractCanvas(ABC):
    def __init__(self, width=None, height=None, dpi=None):
        """Initialize the matplotlib Canvas."""
        pass

    def show(self):
        """Show the canvas."""
        if self.manager is None:
            self.show()
        else:
            self.manager.show()

    def close(self):
        """Close the canvas."""
        self.close()

    def update(self):
        """Update the canvas."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear(self):
        """Clear internal variables."""
        self.close()
        self.ax.clear()
        self.fig.clear()
        self.manager = None

    @abstractmethod
    def _set_size(self, width=None, height=None):
        pass


# ------------------------------------
# Non-object-based Widget Abstractions
# ------------------------------------
# These are planned to be removed in favor of the simpler, object-
# oriented abstractions above when time allows.


class _AbstractToolBar(ABC):
    @abstractmethod
    def _tool_bar_initialize(self, name="default", window=None):
        pass

    @abstractmethod
    def _tool_bar_add_button(self, name, desc, func, *, icon_name=None, shortcut=None):
        pass

    @abstractmethod
    def _tool_bar_update_button_icon(self, name, icon_name):
        pass

    @abstractmethod
    def _tool_bar_add_text(self, name, value, placeholder):
        pass

    @abstractmethod
    def _tool_bar_add_spacer(self):
        pass

    @abstractmethod
    def _tool_bar_add_file_button(self, name, desc, func, *, shortcut=None):
        pass

    @abstractmethod
    def _tool_bar_add_play_button(self, name, desc, func, *, shortcut=None):
        pass


class _AbstractDock(ABC):
    @abstractmethod
    def _dock_initialize(
        self, window=None, name="Controls", area="left", max_width=None
    ):
        pass

    @abstractmethod
    def _dock_finalize(self):
        pass

    @abstractmethod
    def _dock_show(self):
        pass

    @abstractmethod
    def _dock_hide(self):
        pass

    @abstractmethod
    def _dock_add_stretch(self, layout=None):
        pass

    @abstractmethod
    def _dock_add_layout(self, vertical=True):
        pass

    @abstractmethod
    def _dock_add_label(
        self,
        value,
        *,
        align=False,
        layout=None,
        selectable=False,
        row=None,
        col=None,
    ):
        pass

    @abstractmethod
    def _dock_add_button(
        self,
        name,
        callback,
        *,
        style="pushbutton",
        icon=None,
        tooltip=None,
        layout=None,
    ):
        pass

    @abstractmethod
    def _dock_named_layout(self, name, *, layout=None, compact=True):
        pass

    @abstractmethod
    def _dock_add_slider(
        self,
        name,
        value,
        rng,
        callback,
        *,
        compact=True,
        double=False,
        tooltip=None,
        layout=None,
        row=None,
        col=None,
    ):
        pass

    @abstractmethod
    def _dock_add_check_box(
        self, name, value, callback, *, tooltip=None, layout=None, row=None, col=None
    ):
        pass

    @abstractmethod
    def _dock_add_spin_box(
        self,
        name,
        value,
        rng,
        callback,
        *,
        compact=True,
        double=True,
        step=None,
        tooltip=None,
        layout=None,
    ):
        pass

    @abstractmethod
    def _dock_add_combo_box(
        self, name, value, rng, callback, *, compact=True, tooltip=None, layout=None
    ):
        pass

    @abstractmethod
    def _dock_add_radio_buttons(
        self, value, rng, callback, *, vertical=True, layout=None
    ):
        pass

    @abstractmethod
    def _dock_add_group_box(self, name, *, collapse=None, layout=None):
        pass

    @abstractmethod
    def _dock_add_text(self, name, value, placeholder, *, callback=None, layout=None):
        pass

    @abstractmethod
    def _dock_add_file_button(
        self,
        name,
        desc,
        func,
        *,
        filter_=None,
        initial_directory=None,
        save=False,
        is_directory=False,
        icon=False,
        tooltip=None,
        layout=None,
    ):
        pass


class _AbstractMenuBar(ABC):
    @abstractmethod
    def _menu_initialize(self, window=None):
        pass

    @abstractmethod
    def _menu_add_submenu(self, name, desc):
        pass

    @abstractmethod
    def _menu_add_button(self, menu_name, name, desc, func):
        pass


class _AbstractStatusBar(ABC):
    @abstractmethod
    def _status_bar_initialize(self, window=None):
        pass

    @abstractmethod
    def _status_bar_add_label(self, value, *, stretch=0, on_click=None):
        pass

    @abstractmethod
    def _status_bar_add_progress_bar(self, stretch=0):
        pass

    @abstractmethod
    def _status_bar_update(self):
        pass


class _AbstractPlayback(ABC):
    @abstractmethod
    def _playback_initialize(self, func, timeout, value, rng, time_widget, play_widget):
        pass


class _AbstractKeyPress(ABC):
    @abstractmethod
    def _keypress_initialize(self, widget=None):
        pass

    @abstractmethod
    def _keypress_add(self, shortcut, callback):
        pass

    @abstractmethod
    def _keypress_trigger(self, shortcut):
        pass


class _AbstractDialog(ABC):
    @abstractmethod
    def _dialog_create(
        self,
        title,
        text,
        info_text,
        callback,
        *,
        icon="Warning",
        buttons=(),
        modal=True,
        window=None,
    ):
        pass


class _AbstractLayout(ABC):
    @abstractmethod
    def _layout_initialize(self, max_width):
        pass

    @abstractmethod
    def _layout_add_widget(self, layout, widget, stretch=0, *, row=None, col=None):
        pass

    @abstractmethod
    def _layout_create(self, orientation="vertical"):
        pass


class _AbstractWidgetList(ABC):
    @abstractmethod
    def set_enabled(self, state):
        pass

    @abstractmethod
    def get_value(self, idx):
        pass

    @abstractmethod
    def set_value(self, idx, value):
        pass


class _AbstractWdgt(ABC):
    def __init__(self, widget):
        self._widget = widget

    @property
    def widget(self):
        return self._widget

    @abstractmethod
    def set_value(self, value):
        pass

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def set_range(self, rng):
        pass

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def hide(self):
        pass

    @abstractmethod
    def set_enabled(self, state):
        pass

    @abstractmethod
    def is_enabled(self):
        pass

    @abstractmethod
    def is_visible(self):
        pass

    @abstractmethod
    def update(self, repaint=True):
        pass

    @abstractmethod
    def get_tooltip(self):
        pass

    @abstractmethod
    def set_tooltip(self, tooltip: str):
        pass

    @abstractmethod
    def set_style(self, style):
        pass

    def set_hover_callbacks(self, enter, leave):
        """Call ``enter``/``leave`` when the pointer enters/leaves the widget.

        Hovering is a pointer-only affordance, so backends that have no notion of it
        (e.g. notebooks) simply do nothing here.
        """
        pass

    @abstractmethod
    def set_items(self, items):
        pass


class _AbstractAction(ABC):
    def __init__(self, action):
        self._action = action

    @abstractmethod
    def trigger(self):
        pass

    @abstractmethod
    def set_icon(self):
        pass

    @abstractmethod
    def set_shortcut(self):
        pass


class _AbstractMplInterface(ABC):
    @abstractmethod
    def _mpl_initialize():
        pass


class _AbstractMplCanvas(ABC):
    def __init__(self, width, height, dpi):
        """Initialize the MplCanvas."""
        from matplotlib.figure import Figure

        self._extra_events = ("resize",)
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        self.axes = self.fig.add_subplot(111)
        self.axes.set(xlabel="Time (s)", ylabel="Activation (AU)")
        self.manager = None
        self._blit = _BlitManager(self.fig, draw=self.update_plot)

    def _connect(self):
        for event in ("button_press", "motion_notify") + self._extra_events:
            self.canvas.mpl_connect(event + "_event", getattr(self, "on_" + event))

    def plot(self, x, y, label, update=True, **kwargs):
        """Plot a curve."""
        (line,) = self.axes.plot(x, y, label=label, **kwargs)
        if update:
            self.update_plot()
        return line

    def plot_time_line(self, x, label, update=True, **kwargs):
        """Plot the vertical line."""
        line = self.axes.axvline(x, label=label, **kwargs)
        self.add_blit_artist(line)
        if update:
            self.update_plot()
        return line

    def add_blit_artist(self, artist):
        """Mark an artist as fast-updating, to be drawn by :meth:`update_blit_artists`.

        Parameters
        ----------
        artist : instance of matplotlib.artist.Artist
            The artist to draw separately. Must live in this canvas's axes, and be
            drawn on top of the curves, as blitting draws it over a cached picture
            of the rest of the figure.
        """
        if artist.axes is not self.axes:
            raise RuntimeError(
                f"{artist!r} must be an artist of this canvas's axes to be drawn "
                "separately, got one in " + repr(artist.axes)
            )
        self._blit.add(artist)

    def remove_blit_artist(self, artist):
        """Stop drawing an artist separately, putting it back in the background.

        Parameters
        ----------
        artist : instance of matplotlib.artist.Artist
            The artist to stop drawing separately. Artists that were never added
            are ignored.
        """
        self._blit.remove(artist)

    def update_blit_artists(self):
        """Redraw only the artists added with :meth:`add_blit_artist`.

        This is the fast path taken while the time line moves; any other change
        to the figure needs :meth:`update_plot` instead.
        """
        self._blit.update()

    def update_plot(self):
        """Update the plot."""
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore", "constrained_layout")
            self.canvas.draw()

    def set_color(self, bg_color, fg_color):
        """Set the widget colors."""
        from matplotlib.ticker import AutoMinorLocator

        self.axes.set_facecolor(bg_color)
        self.fig.patch.set_facecolor(bg_color)

        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        for side in ("bottom", "left"):
            spine = self.axes.spines[side]
            spine.set_color(fg_color)
            spine.set_linewidth(2.0)

        self.axes.xaxis.label.set_color(fg_color)
        self.axes.yaxis.label.set_color(fg_color)
        self.axes.xaxis.label.set_fontsize(14)
        self.axes.yaxis.label.set_fontsize(14)

        self.axes.tick_params(
            axis="both",
            colors=fg_color,
            labelsize=13,
            length=6,
            width=1.5,
            direction="out",
        )

        self.axes.xaxis.set_minor_locator(AutoMinorLocator())
        self.axes.yaxis.set_minor_locator(AutoMinorLocator())
        self.axes.tick_params(which="minor", length=3, width=1.0, colors=fg_color)
        self.axes.grid(which="major", color=fg_color, alpha=0.18, linewidth=0.9)
        self.axes.grid(which="minor", color=fg_color, alpha=0.08, linewidth=0.6)
        self.axes.set_axisbelow(True)

    def show(self):
        """Show the canvas."""
        if self.manager is None:
            self.canvas.show()
        else:
            self.manager.show()

    def close(self):
        """Close the canvas."""
        self.canvas.close()

    def clear(self):
        """Clear internal variables."""
        self.close()
        self._blit.close()
        self.axes.clear()
        self.fig.clear()
        self.canvas = None
        self.manager = None

    def on_resize(self, event):
        """Handle resize events."""
        pass


class _AbstractBrainMplCanvas(_AbstractMplCanvas):
    _legend_in_figure = True

    def __init__(self, brain, width, height, dpi):
        """Initialize the MplCanvas."""
        super().__init__(width, height, dpi)
        self.brain = brain
        self._hovered_line = None
        self._trace_base_alpha = {}

    def update_plot(self):
        """Update the plot."""
        if self._legend_in_figure:
            leg = self.axes.legend(
                prop={"family": "monospace", "size": "small"},
                framealpha=0.5,
                handlelength=1.0,
                facecolor=self.brain._bg_color,
            )
            for text in leg.get_texts():
                text.set_color(self.brain._fg_color)
        self.sync_traces()
        super().update_plot()

    def sync_traces(self):
        """Refresh a native trace-list widget; no-op unless a backend provides one."""

    def set_trace_visible(self, line, visible):
        """Toggle one trace's visibility, in the plot and on its 3D glyph."""
        line.set_visible(visible)
        self.brain._set_trace_visible(line, visible)
        self.update_plot()

    def set_trace_highlight(self, line):
        """Highlight one trace (or none), dimming the plot's other traces."""
        if line is not None and not line.get_visible():
            line = None
        if line is self._hovered_line:
            return
        time_line = getattr(self.brain, "time_line", None)
        origlines = [
            origline
            for origline in self.axes.get_lines()
            if origline is not time_line and origline.get_visible()
        ]
        if self._hovered_line is None and line is not None:
            self._trace_base_alpha = {
                origline: origline.get_alpha() for origline in origlines
            }
        self._hovered_line = line
        for origline in origlines:
            if line is None:
                origline.set_alpha(self._trace_base_alpha.get(origline))
            else:
                origline.set_alpha(1.0 if origline is line else 0.25)
        if line is None:
            self._trace_base_alpha = {}
        self.canvas.draw_idle()
        self.brain._set_trace_highlight(line)

    def on_button_press(self, event):
        """Handle button presses."""
        # left click (and maybe drag) in progress in axes
        if event.inaxes != self.axes or event.button != 1:
            return
        publish(self.brain, TimeChange(time=event.xdata))

    on_motion_notify = on_button_press  # for now they can be the same

    def clear(self):
        """Clear internal variables."""
        super().clear()
        self.brain = None
        self._hovered_line = None
        self._trace_base_alpha = {}


class _AbstractWindow(ABC):
    def _window_initialize(self, *, window=None, central_layout=None, fullscreen=False):
        self._icons = dict()
        self._window = None
        self._interactor = None
        self._mplcanvas = None
        self._show_traces = None
        self._separate_canvas = None
        self._interactor_fraction = None

    @abstractmethod
    def _window_load_icons(self):
        pass

    @abstractmethod
    def _window_close_connect(self, func, *, after=True):
        pass

    @abstractmethod
    def _window_close_disconnect(self, after=True):
        pass

    @abstractmethod
    def _window_get_dpi(self):
        pass

    @abstractmethod
    def _window_get_size(self):
        pass

    def _window_get_mplcanvas_size(self, fraction):
        ratio = (1 - fraction) / fraction
        dpi = self._window_get_dpi()
        w, h = self._window_get_size()
        h /= ratio
        return (w / dpi, h / dpi)

    @abstractmethod
    def _window_get_simple_canvas(self, width, height, dpi):
        pass

    @abstractmethod
    def _window_get_help_canvas(self, pairs, mouse_pairs=None):
        """Return a widget listing ``(key, description)`` pairs.

        ``pairs`` are keyboard shortcuts; ``mouse_pairs``, if given, are
        ``(action, description)`` pairs shown in a separate section.
        """
        pass

    @abstractmethod
    def _window_get_mplcanvas(
        self, brain, interactor_fraction, show_traces, separate_canvas
    ):
        pass

    @abstractmethod
    def _window_adjust_mplcanvas_layout(self):
        pass

    @abstractmethod
    def _window_get_cursor(self):
        pass

    @abstractmethod
    def _window_set_cursor(self, cursor):
        pass

    @abstractmethod
    def _window_new_cursor(self, name):
        pass

    def _window_set_enabled(self, enabled):
        """Enable or disable user interaction with the whole window.

        Blocking input is a pointer/keyboard affordance, so backends that have no
        notion of it (e.g. notebooks) simply do nothing here.
        """
        pass

    def _window_settle_layouts(self):
        """Recompute all pending widget layouts of the window, synchronously.

        Qt lays widgets out lazily, when the posted ``LayoutRequest`` events are
        delivered. Calling this before showing a freshly-built window makes it appear
        fully composed, instead of visibly assembling on screen. Backends without
        lazy layouts (e.g. notebooks) do nothing here.
        """
        pass

    def _window_defer(self, callback):
        """Run ``callback`` from the event loop instead of the current call stack.

        Use this to run a slow operation triggered by a widget *after* that widget has
        finished reacting to the interaction (e.g. a combo box closing its popup) — the
        callback runs the next time events are processed. Backends without an event
        loop run the callback immediately.
        """
        callback()

    @abstractmethod
    def _window_ensure_minimum_sizes(self):
        pass

    @abstractmethod
    def _window_set_theme(self, theme):
        pass

    @abstractmethod
    def _window_create(self):
        pass
