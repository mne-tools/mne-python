# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextlib import contextmanager

from ..utils import _pl
from .backends._utils import _pixmap_to_ndarray


class _MNEQtBrowserScraper:
    def __repr__(self):
        return "<MNEQtBrowserScraper>"

    def __call__(self, block, block_vars, gallery_conf):
        import mne_qt_browser
        from sphinx_gallery.scrapers import figure_rst

        if gallery_conf["builder_name"] != "html":
            return ""
        img_fnames = list()
        inst = None
        n_plot = 0
        for gui in list(mne_qt_browser._browser_instances):
            try:
                scraped = getattr(gui, "_scraped", False)
            except Exception:  # super __init__ not called, perhaps stale?
                scraped = True
            if scraped:
                continue
            gui._scraped = True  # monkey-patch but it's easy enough
            n_plot += 1
            img_fnames.append(next(block_vars["image_path_iterator"]))
            pixmap, inst = _mne_qt_browser_screenshot(gui, inst)
            pixmap.save(img_fnames[-1])
            # child figures
            for fig in gui.mne.child_figs:
                # For now we only support Selection
                if not hasattr(fig, "channel_fig"):
                    continue
                fig = fig.channel_fig
                img_fnames.append(next(block_vars["image_path_iterator"]))
                fig.savefig(img_fnames[-1])
            gui.close()
            del gui, pixmap
        if not len(img_fnames):
            return ""
        for _ in range(2):
            inst.processEvents()
        return figure_rst(img_fnames, gallery_conf["src_dir"], f"Raw plot{_pl(n_plot)}")


@contextmanager
def _screenshot_mode(browser):
    if need_zen := browser.mne.scrollbars_visible:
        browser._toggle_zenmode()
    try:
        yield
    finally:
        if need_zen:
            browser._toggle_zenmode()


def _mne_qt_browser_screenshot(browser, inst=None, return_type="pixmap"):
    from mne_qt_browser._pg_figure import QApplication

    if getattr(browser, "load_thread", None) is not None:
        if browser.load_thread.isRunning():
            browser.load_thread.wait(30000)
    if inst is None:
        inst = QApplication.instance()
    # processEvents to make sure our progressBar is updated
    with _screenshot_mode(browser):
        for _ in range(2):
            inst.processEvents()
        pixmap = browser.grab()
    assert return_type in ("pixmap", "ndarray")
    if return_type == "ndarray":
        return _pixmap_to_ndarray(pixmap)
    else:
        return pixmap, inst
