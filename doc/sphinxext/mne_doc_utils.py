"""Doc building utils."""

import gc
import os
import time
import warnings

import numpy as np

import mne
from mne.utils import (
    _assert_no_instances,
    sizeof_fmt,
)
from mne.viz import Brain

_np_print_defaults = np.get_printoptions()


def reset_warnings(gallery_conf, fname):
    """Ensure we are future compatible and ignore silly warnings."""
    # In principle, our examples should produce no warnings.
    # Here we cause warnings to become errors, with a few exceptions.
    # This list should be considered alongside
    # setup.cfg -> [tool:pytest] -> filterwarnings

    # remove tweaks from other module imports or example runs
    warnings.resetwarnings()
    # restrict
    warnings.filterwarnings("error")
    # allow these, but show them
    warnings.filterwarnings("always", '.*non-standard config type: "foo".*')
    warnings.filterwarnings("always", '.*config type: "MNEE_USE_CUUDAA".*')
    warnings.filterwarnings("always", ".*cannot make axes width small.*")
    warnings.filterwarnings("always", ".*Axes that are not compatible.*")
    warnings.filterwarnings("always", ".*FastICA did not converge.*")
    # ECoG BIDS spec violations:
    warnings.filterwarnings("always", ".*Fiducial point nasion not found.*")
    warnings.filterwarnings("always", ".*DigMontage is only a subset of.*")
    warnings.filterwarnings(  # xhemi morph (should probably update sample)
        "always", ".*does not exist, creating it and saving it.*"
    )
    # internal warnings
    warnings.filterwarnings("default", module="sphinx")
    # allow these warnings, but don't show them
    for key in (
        "invalid version and will not be supported",  # pyxdf
        "distutils Version classes are deprecated",  # seaborn and neo
        "is_categorical_dtype is deprecated",  # seaborn
        "`np.object` is a deprecated alias for the builtin `object`",  # pyxdf
        # nilearn, should be fixed in > 0.9.1
        "In future, it will be an error for 'np.bool_' scalars to",
        # sklearn hasn't updated to SciPy's sym_pos dep
        "The 'sym_pos' keyword is deprecated",
        # numba
        "`np.MachAr` is deprecated",
        # joblib hasn't updated to avoid distutils
        "distutils package is deprecated",
        # jupyter
        "Jupyter is migrating its paths to use standard",
        r"Widget\..* is deprecated\.",
        # PyQt6
        "Enum value .* is marked as deprecated",
        # matplotlib PDF output
        "The py23 module has been deprecated",
        # pkg_resources
        "Implementing implicit namespace packages",
        "Deprecated call to `pkg_resources",
        # nilearn
        "pkg_resources is deprecated as an API",
        r"The .* was deprecated in Matplotlib 3\.7",
        # Matplotlib->tz
        r"datetime\.datetime\.utcfromtimestamp",
        # joblib
        r"ast\.Num is deprecated",
        r"Attribute n is deprecated and will be removed in Python 3\.14",
        # numpydoc
        r"ast\.NameConstant is deprecated and will be removed in Python 3\.14",
        # pooch
        r"Python 3\.14 will, by default, filter extracted tar archives.*",
        # seaborn
        r"DataFrameGroupBy\.apply operated on the grouping columns.*",
        # pandas
        r"\nPyarrow will become a required dependency of pandas.*",
        # latexcodec
        r"open_text is deprecated\. Use files.*",
    ):
        warnings.filterwarnings(  # deal with other modules having bad imports
            "ignore", message=".*%s.*" % key, category=DeprecationWarning
        )
    warnings.filterwarnings(
        "ignore",
        message="Matplotlib is currently using agg, which is a non-GUI backend.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*is non-interactive, and thus cannot.*",
    )
    # seaborn
    warnings.filterwarnings(
        "ignore",
        message="The figure layout has changed to tight",
        category=UserWarning,
    )
    # xarray/netcdf4
    warnings.filterwarnings(
        "ignore",
        message=r"numpy\.ndarray size changed, may indicate.*",
        category=RuntimeWarning,
    )
    # qdarkstyle
    warnings.filterwarnings(
        "ignore",
        message=r".*Setting theme=.*6 in qdarkstyle.*",
        category=RuntimeWarning,
    )
    # pandas, via seaborn (examples/time_frequency/time_frequency_erds.py)
    for message in (
        "use_inf_as_na option is deprecated.*",
        r"iteritems is deprecated.*Use \.items instead\.",
        "is_categorical_dtype is deprecated.*",
        "The default of observed=False.*",
        "When grouping with a length-1 list-like.*",
    ):
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=FutureWarning,
        )
    # pandas in 50_epochs_to_data_frame.py
    warnings.filterwarnings(
        "ignore", message=r"invalid value encountered in cast", category=RuntimeWarning
    )
    # xarray _SixMetaPathImporter (?)
    warnings.filterwarnings(
        "ignore", message=r"falling back to find_module", category=ImportWarning
    )
    # Sphinx deps
    warnings.filterwarnings(
        "ignore", message="The str interface for _CascadingStyleSheet.*"
    )
    # mne-qt-browser until > 0.5.2 released
    warnings.filterwarnings(
        "ignore",
        r"mne\.io\.pick.channel_indices_by_type is deprecated.*",
    )

    # In case we use np.set_printoptions in any tutorials, we only
    # want it to affect those:
    np.set_printoptions(**_np_print_defaults)


t0 = time.time()


def reset_modules(gallery_conf, fname, when):
    """Do the reset."""
    import matplotlib.pyplot as plt

    try:
        from pyvista import Plotter  # noqa
    except ImportError:
        Plotter = None  # noqa
    try:
        from pyvistaqt import BackgroundPlotter  # noqa
    except ImportError:
        BackgroundPlotter = None  # noqa
    try:
        from vtkmodules.vtkCommonDataModel import vtkPolyData  # noqa
    except ImportError:
        vtkPolyData = None  # noqa
    try:
        from mne_qt_browser._pg_figure import MNEQtBrowser
    except ImportError:
        MNEQtBrowser = None
    from mne.viz.backends.renderer import backend

    _Renderer = backend._Renderer if backend is not None else None
    reset_warnings(gallery_conf, fname)
    # in case users have interactive mode turned on in matplotlibrc,
    # turn it off here (otherwise the build can be very slow)
    plt.ioff()
    plt.rcParams["animation.embed_limit"] = 40.0
    plt.rcParams["figure.raise_window"] = False
    # https://github.com/sphinx-gallery/sphinx-gallery/pull/1243#issue-2043332860
    plt.rcParams["animation.html"] = "html5"
    # neo holds on to an exception, which in turn holds a stack frame,
    # which will keep alive the global vars during SG execution
    try:
        import neo

        neo.io.stimfitio.STFIO_ERR = None
    except Exception:
        pass
    gc.collect()

    # Agg does not call close_event so let's clean up on our own :(
    # https://github.com/matplotlib/matplotlib/issues/18609
    mne.viz.ui_events._cleanup_agg()
    assert len(mne.viz.ui_events._event_channels) == 0, list(
        mne.viz.ui_events._event_channels
    )

    when = f"mne/conf.py:Resetter.__call__:{when}:{fname}"
    # Support stuff like
    # MNE_SKIP_INSTANCE_ASSERTIONS="Brain,Plotter,BackgroundPlotter,vtkPolyData,_Renderer" make html-memory  # noqa: E501
    # to just test MNEQtBrowser
    skips = os.getenv("MNE_SKIP_INSTANCE_ASSERTIONS", "").lower()
    prefix = ""
    if skips not in ("true", "1", "all"):
        prefix = "Clean "
        skips = skips.split(",")
        if "brain" not in skips:
            _assert_no_instances(Brain, when)  # calls gc.collect()
        if Plotter is not None and "plotter" not in skips:
            _assert_no_instances(Plotter, when)
        if BackgroundPlotter is not None and "backgroundplotter" not in skips:
            _assert_no_instances(BackgroundPlotter, when)
        if vtkPolyData is not None and "vtkpolydata" not in skips:
            _assert_no_instances(vtkPolyData, when)
        if "_renderer" not in skips:
            _assert_no_instances(_Renderer, when)
        if MNEQtBrowser is not None and "mneqtbrowser" not in skips:
            # Ensure any manual fig.close() events get properly handled
            from mne_qt_browser._pg_figure import QApplication

            inst = QApplication.instance()
            if inst is not None:
                for _ in range(2):
                    inst.processEvents()
            _assert_no_instances(MNEQtBrowser, when)
    # This will overwrite some Sphinx printing but it's useful
    # for memory timestamps
    if os.getenv("SG_STAMP_STARTS", "").lower() == "true":
        import psutil

        process = psutil.Process(os.getpid())
        mem = sizeof_fmt(process.memory_info().rss)
        print(f"{prefix}{time.time() - t0:6.1f} s : {mem}".ljust(22))


report_scraper = mne.report._ReportScraper()
mne_qt_browser_scraper = mne.viz._scraper._MNEQtBrowserScraper()
brain_scraper = mne.viz._brain._BrainScraper()
gui_scraper = mne.gui._GUIScraper()
