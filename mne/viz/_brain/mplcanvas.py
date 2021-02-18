# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD
import warnings
from ..utils import tight_layout
from ...fixes import nullcontext


class MplCanvas(object):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, brain, width, height, dpi):
        from PyQt5 import QtWidgets
        from matplotlib import rc_context
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        if brain.separate_canvas:
            parent = None
        else:
            parent = brain.window
        # prefer constrained layout here but live with tight_layout otherwise
        context = nullcontext
        extra_events = ('resize',)
        try:
            context = rc_context({'figure.constrained_layout.use': True})
            extra_events = ()
        except KeyError:
            pass
        with context:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.axes.set(xlabel='Time (sec)', ylabel='Activation (AU)')
        self.canvas.setParent(parent)
        FigureCanvasQTAgg.setSizePolicy(
            self.canvas,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvasQTAgg.updateGeometry(self.canvas)
        self.brain = brain
        self.time_func = brain.callbacks["time"]
        for event in ('button_press', 'motion_notify') + extra_events:
            self.canvas.mpl_connect(
                event + '_event', getattr(self, 'on_' + event))

    def plot(self, x, y, label, **kwargs):
        """Plot a curve."""
        line, = self.axes.plot(
            x, y, label=label, **kwargs)
        self.update_plot()
        return line

    def plot_time_line(self, x, label, **kwargs):
        """Plot the vertical line."""
        line = self.axes.axvline(x, label=label, **kwargs)
        self.update_plot()
        return line

    def update_plot(self):
        """Update the plot."""
        leg = self.axes.legend(
            prop={'family': 'monospace', 'size': 'small'},
            framealpha=0.5, handlelength=1.,
            facecolor=self.brain._bg_color)
        for text in leg.get_texts():
            text.set_color(self.brain._fg_color)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings('ignore', 'constrained_layout')
            self.canvas.draw()

    def set_color(self, bg_color, fg_color):
        """Set the widget colors."""
        self.axes.set_facecolor(bg_color)
        self.axes.xaxis.label.set_color(fg_color)
        self.axes.yaxis.label.set_color(fg_color)
        self.axes.spines['top'].set_color(fg_color)
        self.axes.spines['bottom'].set_color(fg_color)
        self.axes.spines['left'].set_color(fg_color)
        self.axes.spines['right'].set_color(fg_color)
        self.axes.tick_params(axis='x', colors=fg_color)
        self.axes.tick_params(axis='y', colors=fg_color)
        self.fig.patch.set_facecolor(bg_color)

    def show(self):
        """Show the canvas."""
        self.canvas.show()

    def close(self):
        """Close the canvas."""
        self.canvas.close()

    def on_button_press(self, event):
        """Handle button presses."""
        # left click (and maybe drag) in progress in axes
        if (event.inaxes != self.axes or
                event.button != 1):
            return
        self.time_func(
            event.xdata, update_widget=True, time_as_index=False)

    def clear(self):
        """Clear internal variables."""
        self.close()
        self.axes.clear()
        self.fig.clear()
        self.brain = None
        self.canvas = None

    on_motion_notify = on_button_press  # for now they can be the same

    def on_resize(self, event):
        """Handle resize events."""
        tight_layout(fig=self.axes.figure)
