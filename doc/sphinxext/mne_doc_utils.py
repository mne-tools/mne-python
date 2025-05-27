"""Doc building utils."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import gc
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pyvista
import sphinx.util.logging

import mne
from mne.utils import (
    _assert_no_instances,
    _get_extra_data_path,
    sizeof_fmt,
)
from mne.viz import Brain

sphinx_logger = sphinx.util.logging.getLogger("mne")
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
    # don't error on joblib warning during parallel doc build otherwise we get a
    # cryptic deadlock instead of a nice traceback
    warnings.filterwarnings(
        "always",
        "A worker stopped while some jobs were given to the executor.*",
        category=UserWarning,
    )
    # ignore (DeprecationWarning)
    for key in (
        # nibabel
        "__array__ implementation doesn't accept.*",
        # pybtex (from sphinxcontrib-bibtex)
        "pkg_resources is deprecated as an API.*",
        "\nImplementing implicit namespace packages",
        # latexcodec
        r"open_text is deprecated\. Use files",
    ):
        warnings.filterwarnings(  # deal with other modules having bad imports
            "ignore", message=f".*{key}.*", category=DeprecationWarning
        )
    # ignore (UserWarning)
    for message in (
        # Matplotlib
        ".*is non-interactive, and thus cannot.*",
        # pybtex
        ".*pkg_resources is deprecated as an API.*",
    ):
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=UserWarning,
        )
    # ignore (RuntimeWarning)
    for message in (
        # mne-python config file "corruption" due to doc build parallelization
        ".*The MNE-Python config file.*valid JSON.*",
    ):
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=RuntimeWarning,
        )

    # In case we use np.set_printoptions in any tutorials, we only
    # want it to affect those:
    np.set_printoptions(**_np_print_defaults)


t0 = time.time()


def reset_modules(gallery_conf, fname, when):
    """Do the reset."""
    import matplotlib.pyplot as plt

    mne.viz.set_3d_backend("pyvistaqt")
    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True

    from pyvista import Plotter  # noqa

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

    orig_when = when
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

    if fname == "50_configure_mne.py":
        # This messes with the config, so let's do so in a temp dir
        if orig_when == "before":
            fake_home = Path(_get_extra_data_path()) / "temp"
            fake_home.mkdir(exist_ok=True, parents=True)
            os.environ["_MNE_FAKE_HOME_DIR"] = str(fake_home)
        else:
            assert orig_when == "after"
            to_del = Path(os.environ["_MNE_FAKE_HOME_DIR"])
            try:
                (to_del / "mne-python.json").unlink()
            except Exception:
                pass
            try:
                to_del.rmdir()
            except Exception:
                pass
            del os.environ["_MNE_FAKE_HOME_DIR"]


report_scraper = mne.report._ReportScraper()
mne_qt_browser_scraper = mne.viz._scraper._MNEQtBrowserScraper()
brain_scraper = mne.viz._brain._BrainScraper()
gui_scraper = mne.gui._GUIScraper()
