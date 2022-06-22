"""ABCs."""

# Authors: Guillaume Favelier <guillaume.favelier@gmail.com
#          Eric Larson <larson.eric.d@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

from abc import ABC, abstractmethod, abstractclassmethod
from contextlib import nullcontext
import warnings

from ..utils import tight_layout


class _AbstractRenderer(ABC):

    @abstractclassmethod
    def __init__(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
                 name=None, show=False, shape=(1, 1), splash=False):
        """Set up the scene."""
        pass

    @property
    @abstractmethod
    def _kind(self):
        pass

    @abstractclassmethod
    def subplot(self, x, y):
        """Set the active subplot."""
        pass

    @abstractclassmethod
    def scene(self):
        """Return scene handle."""
        pass

    @abstractclassmethod
    def set_interaction(self, interaction):
        """Set interaction mode."""
        pass

    @abstractclassmethod
    def legend(self, labels, border=False, size=0.1, face='triangle',
               loc='upper left'):
        """Add a legend to the scene.

        Parameters
        ----------
        labels : list of tuples
            Each entry must contain two strings, (label, color),
            where ``label`` is the name of the item to add, and
            ``color`` is the color of the label to add.
        border : bool
            Controls if there will be a border around the legend.
            The default is False.
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

    @abstractclassmethod
    def mesh(self, x, y, z, triangles, color, opacity=1.0, shading=False,
             backface_culling=False, scalars=None, colormap=None,
             vmin=None, vmax=None, interpolate_before_map=True,
             representation='surface', line_width=1., normals=None,
             polygon_offset=None, **kwargs):
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
            If None, the min of the data will be used
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap :
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
        polygon_offset : float
            If not None, the factor used to resolve coincident topology.
        kwargs : args
            The arguments to pass to triangular_mesh

        Returns
        -------
        surface :
            Handle of the mesh in the scene.
        """
        pass

    @abstractclassmethod
    def contour(self, surface, scalars, contours, width=1.0, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, kind='line', color=None):
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
            If None, the min of the data will be used
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap :
            The colormap to use.
        normalized_colormap : bool
            Specify if the values of the colormap are between 0 and 1.
        kind : 'line' | 'tube'
            The type of the primitives to use to display the contours.
        color :
            The color of the mesh as a tuple (red, green, blue) of float
            values between 0 and 1 or a valid color name (i.e. 'white'
            or 'w').
        """
        pass

    @abstractclassmethod
    def surface(self, surface, color=None, opacity=1.0,
                vmin=None, vmax=None, colormap=None,
                normalized_colormap=False, scalars=None,
                backface_culling=False, polygon_offset=None):
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
            If None, the min of the data will be used
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap :
            The colormap to use.
        scalars : ndarray, shape (n_vertices,)
            The scalar valued associated to the vertices.
        backface_culling : bool
            If True, enable backface culling on the surface.
        polygon_offset : float
            If not None, the factor used to resolve coincident topology.
        """
        pass

    @abstractclassmethod
    def sphere(self, center, color, scale, opacity=1.0,
               resolution=8, backface_culling=False,
               radius=None):
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

    @abstractclassmethod
    def tube(self, origin, destination, radius=0.001, color='white',
             scalars=None, vmin=None, vmax=None, colormap='RdBu',
             normalized_colormap=False, reverse_lut=False):
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
            If None, the min of the data will be used
        vmax : float | None
            vmax is used to scale the colormap.
            If None, the max of the data will be used
        colormap :
            The colormap to use.
        opacity : float
            The opacity of the tube(s).
        backface_culling : bool
            If True, enable backface culling on the tube(s).
        reverse_lut : bool
            If True, reverse the lookup table.

        Returns
        -------
        actor :
            The actor in the scene.
        surface :
            Handle of the tube in the scene.
        """
        pass

    @abstractclassmethod
    def quiver3d(self, x, y, z, u, v, w, color, scale, mode, resolution=8,
                 glyph_height=None, glyph_center=None, glyph_resolution=None,
                 opacity=1.0, scale_mode='none', scalars=None,
                 backface_culling=False, colormap=None, vmin=None, vmax=None,
                 line_width=2., name=None):
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
        resolution : int
            The resolution of the glyph created. Depending on the type of
            glyph, it represents the number of divisions in its geometric
            representation.
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
        colormap :
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

    @abstractclassmethod
    def text2d(self, x_window, y_window, text, size=14, color='white'):
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
        """
        pass

    @abstractclassmethod
    def text3d(self, x, y, z, text, width, color='white'):
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

    @abstractclassmethod
    def scalarbar(self, source, color="white", title=None, n_labels=4,
                  bgcolor=None):
        """Add a scalar bar in the scene.

        Parameters
        ----------
        source :
            The object of the scene used for the colormap.
        color :
            The color of the label text.
        title : str | None
            The title of the scalar bar.
        n_labels : int | None
            The number of labels to display on the scalar bar.
        bgcolor :
            The color of the background when there is transparency.
        """
        pass

    @abstractclassmethod
    def show(self):
        """Render the scene."""
        pass

    @abstractclassmethod
    def close(self):
        """Close the scene."""
        pass

    @abstractclassmethod
    def set_camera(self, azimuth=None, elevation=None, distance=None,
                   focalpoint=None, roll=None, reset_camera=True):
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
        reset_camera : bool
           If True, reset the camera properties beforehand.
        """
        pass

    @abstractclassmethod
    def reset_camera(self):
        """Reset the camera properties."""
        pass

    @abstractclassmethod
    def screenshot(self, mode='rgb', filename=None):
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

    @abstractclassmethod
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

    @abstractclassmethod
    def remove_mesh(self, mesh_data):
        """Remove the given mesh from the scene.

        Parameters
        ----------
        mesh_data : tuple | Surface
            The mesh to remove.
        """
        pass

# -------
# Widgets
# -------


class _AbstractWidget(ABC):

    @abstractclassmethod
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
    def _get_tooltip(self):
        pass

    @abstractmethod
    def _set_tooltip(self, tooltip: str):
        pass

    @abstractmethod
    def _add_keypress(self, callback):
        pass

    @abstractmethod
    def _set_focus(self):
        pass

    @abstractmethod
    def _set_layout(self, layout):
        pass

    @abstractmethod
    def _set_theme(self, theme):
        pass


class _AbstractLabel(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, center=False, selectable=False):
        pass


class _AbstractText(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value=None, placeholder=None, callback=None):
        pass


class _AbstractButton(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, callback):
        pass

    @abstractmethod
    def _click(self):
        pass

    @abstractmethod
    def _set_icon(self, icon):
        pass


class _AbstractSlider(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, rng, callback, horizontal=True):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass

    @abstractmethod
    def _set_range(self, rng):
        pass


class _AbstractProgressBar(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, count):
        pass

    @abstractmethod
    def _increment(self):
        pass


class _AbstractCheckBox(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, callback):
        pass

    @abstractmethod
    def _set_checked(self, checked):
        pass

    @abstractmethod
    def _get_checked(self):
        pass


class _AbstractSpinBox(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, rng, callback, step=None):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractComboBox(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, items, callback):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractRadioButtons(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, items, callback):
        pass

    @abstractmethod
    def _set_value(self, value):
        pass

    @abstractmethod
    def _get_value(self):
        pass


class _AbstractGroupBox(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, name, items):
        pass


class _AbstractFileButton(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, callback, content_filter=None, initial_directory=None,
                 save=False, is_directory=False, window=None):
        pass


class _AbstractPlayMenu(_AbstractWidget):

    @abstractclassmethod
    def __init__(self, value, rng, callback):
        pass


class _AbstractDialog(_AbstractWidget):

    # from QMessageBox.StandardButtons
    supported_button_names = ['Ok']
    # TODO: Add back support for below, file browser takes care of most
    # so no big need currently
    '''
    ['Ok', 'Open', 'Save', 'Cancel', 'Close', 'Discard', 'Apply',
    'Reset', 'RestoreDefaults', 'Help', 'SaveAll', 'Yes',
    'YesToAll', 'No', 'NoToAll', 'Abort', 'Retry', 'Ignore']
    '''

    supported_icon_names = ['question', 'information', 'warning', 'critical']

    @abstractmethod
    def __init__(self, title, text, info_text, callback,
                 icon='Warning', buttons=None, window=None):
        pass


# -------
# Layouts
# -------

class _AbstractBoxLayout(ABC):

    @abstractmethod
    def _add_widget(self, widget):
        pass

    @abstractmethod
    def _add_stretch(self, amount=1):
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


class _AbstractWindow(ABC):

    def __init__(self, size=None, fullscreen=False):
        pass

    @abstractmethod
    def _set_central_layout(self, central_layout):
        pass

    @abstractmethod
    def _get_dpi(self):
        pass

    @abstractmethod
    def _get_size(self):
        pass

    @abstractmethod
    def _get_cursor(self):
        pass

    @abstractmethod
    def _set_cursor(self, cursor):
        pass

    @abstractmethod
    def _new_cursor(self, name):
        pass

    @abstractmethod
    def _close_connect(self, func, *, after=True):
        pass

    @abstractmethod
    def _close_disconnect(self, after=True):
        pass


# -------------------
# Matplotlib Canvases
# -------------------


class _AbstractMplCanvas(ABC):

    def __init__(self, width, height, dpi):
        """Initialize the MplCanvas."""
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


class _AbstractBrainMplCanvas(_AbstractMplCanvas):
    def __init__(self, brain, width, height, dpi):
        """Initialize the MplCanvas."""
        super().__init__(width, height, dpi)
        self.brain = brain
        self.time_func = brain.callbacks["time"]
        self.axes.set(xlabel='Time (sec)', ylabel='Activation (AU)')

    def on_resize(self, event):
        """Handle resize events."""
        tight_layout(fig=self.ax.figure)

    def update_plot(self):
        """Update the plot."""
        leg = self.axes.legend(
            prop={'family': 'monospace', 'size': 'small'},
            framealpha=0.5, handlelength=1.,
            facecolor=self.brain._bg_color)
        for text in leg.get_texts():
            text.set_color(self.brain._fg_color)
        super().update_plot()

    def on_button_press(self, event):
        """Handle button presses."""
        # left click (and maybe drag) in progress in axes
        if (event.inaxes != self.axes or
                event.button != 1):
            return
        self.time_func(
            event.xdata, update_widget=True, time_as_index=False)

    on_motion_notify = on_button_press  # for now they can be the same

    def clear(self):
        """Clear internal variables."""
        super().clear()
        self.brain = None

    def _connect(self):
        for event in ('button_press', 'motion_notify') + self._extra_events:
            self.mpl_connect(
                event + '_event', getattr(self, 'on_' + event))

    def plot(self, x, y, label, update=True, **kwargs):
        """Plot a curve."""
        line, = self.ax.plot(x, y, label=label, **kwargs)
        if update:
            self.update_plot()
        return line

    def plot_time_line(self, x, label, update=True, **kwargs):
        """Plot the vertical line."""
        line = self.ax.axvline(x, label=label, **kwargs)
        if update:
            self.update_plot()
        return line

    def update_plot(self):
        """Update the plot."""
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('ignore', 'constrained_layout')
            self.draw()

    def set_color(self, bg_color, fg_color):
        """Set the widget colors."""
        self.ax.set_facecolor(bg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        self.ax.spines['top'].set_color(fg_color)
        self.ax.spines['bottom'].set_color(fg_color)
        self.ax.spines['left'].set_color(fg_color)
        self.ax.spines['right'].set_color(fg_color)
        self.ax.tick_params(axis='x', colors=fg_color)
        self.ax.tick_params(axis='y', colors=fg_color)
        self.fig.patch.set_facecolor(bg_color)


class Figure3D(ABC):
    """Class that refers to a 3D figure.

    .. note::
        This class is not meant to be instantiated directly, use
        :func:`mne.viz.create_3d_figure` instead.
    """

    # Here we use _init rather than __init__ so that users are less tempted to
    # instantiate the class directly. It also helps us
    # document the class more easily, as we don't have to say what all the
    # params are in public docs.

    @abstractclassmethod
    def _init(self, fig=None, size=(600, 600), bgcolor=(0., 0., 0.),
              name=None, show=False, shape=(1, 1), splash=False):
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


'''
window startup
self._icons = dict()
self._interactor = None
self._mplcanvas = None
self._show_traces = None
self._separate_canvas = None
self._interactor_fraction = None
'''
