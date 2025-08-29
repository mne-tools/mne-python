import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

def _line_plot_onselect(xmin, xmax, **kwargs):
    pass

def _plot_evoked(data, show=True):
    picks = np.arange(len(data)) 
    fig, ax = plt.subplots(1, 1, layout="constrained")
    fig.set_size_inches(6.4, 2.5)

   
    ax.plot(data[picks].T, alpha=1.0)
    ax.set_title("Butterfly Plot")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")

    # SpanSelector
    text = ax.annotate(
        "Loading...",
        xy=(0.01, 0.1),
        xycoords="axes fraction",
        fontsize=12,
        color="green",
        zorder=3,
    )
    text.set_visible(False)

    blit = False if plt.get_backend() == "MacOSX" else True
    ax._span_selector = SpanSelector(
        ax,
        _line_plot_onselect,
        "horizontal",
        useblit=blit,
        props=dict(alpha=0.5, facecolor="red"),
    )

    if show:
        plt.show()
    return fig

# Dummy EEG data 
data = np.random.randn(3, 1000)

fig = _plot_evoked(data, show=False)
fig.savefig("test_output.svg")  
