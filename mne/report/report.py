"""Generate self-contained HTML reports from MNE objects."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations  # only needed for Python ≤ 3.9

import base64
import copy
import dataclasses
import os
import os.path as op
import re
import time
import warnings
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from shutil import copyfile

import numpy as np

from .. import __version__ as MNE_VERSION
from .._fiff.meas_info import Info, read_info
from .._fiff.pick import _DATA_CH_TYPES_SPLIT
from .._freesurfer import _mri_orientation, _reorient_image
from ..cov import Covariance, read_cov
from ..defaults import _handle_default
from ..epochs import BaseEpochs, read_epochs
from ..event import read_events
from ..evoked import Evoked, read_evokeds
from ..forward import Forward, read_forward_solution
from ..html_templates import _get_html_template
from ..io import BaseRaw, read_raw
from ..io._read_raw import _get_supported as _get_extension_reader_map
from ..minimum_norm import InverseOperator, read_inverse_operator
from ..parallel import parallel_func
from ..preprocessing.ica import read_ica
from ..proj import read_proj
from ..source_estimate import SourceEstimate, read_source_estimate
from ..surface import dig_mri_distances
from ..transforms import Transform, read_trans
from ..utils import (
    _check_ch_locs,
    _check_fname,
    _check_option,
    _ensure_int,
    _import_h5io_funcs,
    _import_nibabel,
    _path_like,
    _pl,
    _safe_input,
    _validate_type,
    _verbose_safe_false,
    fill_doc,
    get_subjects_dir,
    logger,
    sys_info,
    use_log_level,
    verbose,
    warn,
)
from ..utils.spectrum import _split_psd_kwargs
from ..viz import (
    Figure3D,
    _get_plot_ch_type,
    create_3d_figure,
    get_3d_backend,
    plot_alignment,
    plot_compare_evokeds,
    plot_cov,
    plot_events,
    plot_projs_joint,
    plot_projs_topomap,
    set_3d_view,
    use_browser_backend,
)
from ..viz._brain.view import views_dicts
from ..viz._scraper import _mne_qt_browser_screenshot
from ..viz.misc import _get_bem_plotting_surfaces, _plot_mri_contours
from ..viz.utils import _ndarray_to_fig

_BEM_VIEWS = ("axial", "sagittal", "coronal")


# For raw files, we want to support different suffixes + extensions for all
# supported file formats
SUPPORTED_READ_RAW_EXTENSIONS = tuple(_get_extension_reader_map())
RAW_EXTENSIONS = []
for ext in SUPPORTED_READ_RAW_EXTENSIONS:
    RAW_EXTENSIONS.append(f"raw{ext}")
    if ext not in (".bdf", ".edf", ".set", ".vhdr"):  # EEG-only formats
        RAW_EXTENSIONS.append(f"meg{ext}")
    RAW_EXTENSIONS.append(f"eeg{ext}")
    RAW_EXTENSIONS.append(f"ieeg{ext}")
    RAW_EXTENSIONS.append(f"nirs{ext}")

# Processed data will always be in (gzipped) FIFF format
VALID_EXTENSIONS = (
    "sss.fif",
    "sss.fif.gz",
    "eve.fif",
    "eve.fif.gz",
    "cov.fif",
    "cov.fif.gz",
    "proj.fif",
    "prof.fif.gz",
    "trans.fif",
    "trans.fif.gz",
    "fwd.fif",
    "fwd.fif.gz",
    "epo.fif",
    "epo.fif.gz",
    "inv.fif",
    "inv.fif.gz",
    "ave.fif",
    "ave.fif.gz",
    "T1.mgz",
) + tuple(RAW_EXTENSIONS)
del RAW_EXTENSIONS

CONTENT_ORDER = (
    "raw",
    "events",
    "epochs",
    "ssp-projectors",
    "evoked",
    "covariance",
    "coregistration",
    "bem",
    "forward-solution",
    "inverse-operator",
    "source-estimate",
)

html_include_dir = Path(__file__).parent / "js_and_css"
JAVASCRIPT = (html_include_dir / "report.js").read_text(encoding="utf-8")
CSS = (html_include_dir / "report.css").read_text(encoding="utf-8")

MAX_IMG_RES = 100  # in dots per inch
MAX_IMG_WIDTH = 850  # in pixels


def _get_data_ch_types(inst):
    return [ch_type for ch_type in _DATA_CH_TYPES_SPLIT if ch_type in inst]


def _id_sanitize(title):
    """Sanitize title for use as DOM id."""
    _validate_type(title, str, "title")
    # replace any whitespace runs with underscores
    title = re.sub(r"\s+", "_", title)
    # replace any non-alphanumeric (plus dash and underscore) with underscores
    # (this is very greedy but should be safe enough)
    title = re.sub("[^a-zA-Z0-9_-]", "_", title)
    return title


def _renderer(kind):
    return _get_html_template("report", kind).render


###############################################################################
# HTML generation


def _html_header_element(*, lang, include, js, css, title, tags, mne_logo_img):
    return _renderer("header.html.jinja")(
        lang=lang,
        include=include,
        js=js,
        css=css,
        title=title,
        tags=tags,
        mne_logo_img=mne_logo_img,
    )


def _html_footer_element(*, mne_version, date):
    return _renderer("footer.html.jinja")(mne_version=mne_version, date=date)


def _html_toc_element(*, titles, dom_ids, tags):
    return _renderer("toc.html.jinja")(titles=titles, dom_ids=dom_ids, tags=tags)


def _html_forward_sol_element(*, id_, repr_, sensitivity_maps, title, tags):
    return _renderer("forward.html.jinja")(
        id=id_,
        repr=repr_,
        sensitivity_maps=sensitivity_maps,
        tags=tags,
        title=title,
    )


def _html_inverse_operator_element(*, id_, repr_, source_space, title, tags):
    return _renderer("inverse.html.jinja")(
        id=id_, repr=repr_, source_space=source_space, tags=tags, title=title
    )


def _html_slider_element(
    *, id_, images, captions, start_idx, image_format, title, tags, show, klass=""
):
    captions_ = []
    for caption in captions:
        if caption is None:
            caption = ""
        captions_.append(caption)
    del captions

    return _renderer("slider.html.jinja")(
        id=id_,
        images=images,
        captions=captions_,
        tags=tags,
        title=title,
        start_idx=start_idx,
        image_format=image_format,
        klass=klass,
        show="show" if show else "",
    )


def _html_image_element(
    *, id_, img, image_format, caption, show, div_klass, img_klass, title, tags
):
    return _renderer("image.html.jinja")(
        id=id_,
        img=img,
        caption=caption,
        tags=tags,
        title=title,
        image_format=image_format,
        div_klass=div_klass,
        img_klass=img_klass,
        show="show" if show else "",
    )


def _html_code_element(*, id_, code, language, title, tags):
    return _renderer("code.html.jinja")(
        id=id_,
        code=code,
        language=language,
        title=title,
        tags=tags,
    )


def _html_section_element(*, id_, div_klass, htmls, title, tags, show):
    return _renderer("section.html.jinja")(
        id=id_,
        div_klass=div_klass,
        htmls=htmls,
        title=title,
        tags=tags,
        show="show" if show else "",
    )


def _html_bem_element(
    *,
    id_,
    div_klass,
    html_slider_axial,
    html_slider_sagittal,
    html_slider_coronal,
    title,
    tags,
):
    return _renderer("bem.html.jinja")(
        id=id_,
        div_klass=div_klass,
        html_slider_axial=html_slider_axial,
        html_slider_sagittal=html_slider_sagittal,
        html_slider_coronal=html_slider_coronal,
        title=title,
        tags=tags,
    )


def _html_element(*, id_, div_klass, html, title, tags, show):
    return _renderer("html.html.jinja")(
        id=id_,
        div_klass=div_klass,
        html=html,
        title=title,
        tags=tags,
        show="show" if show else "",
    )


@dataclass
class _ContentElement:
    name: str
    section: str | None
    dom_id: str
    tags: tuple[str]
    html: str


def _check_tags(tags) -> tuple[str]:
    # Must be iterable, but not a string
    if isinstance(tags, str):
        tags = (tags,)
    elif isinstance(tags, Sequence | np.ndarray):
        tags = tuple(tags)
    else:
        raise TypeError(
            f"tags must be a string (without spaces or special characters) or "
            f"an array-like object of such strings, but got {type(tags)} "
            f"instead: {tags}"
        )

    # Check for invalid dtypes
    bad_tags = [tag for tag in tags if not isinstance(tag, str)]
    if bad_tags:
        raise TypeError(
            f"All tags must be strings without spaces or special characters, "
            f"but got the following instead: "
            f'{", ".join([str(tag) for tag in bad_tags])}'
        )

    # Check for invalid characters
    invalid_chars = (" ", '"', "\n")  # we'll probably find more :-)
    bad_tags = []
    for tag in tags:
        for invalid_char in invalid_chars:
            if invalid_char in tag:
                bad_tags.append(tag)
                break
    if bad_tags:
        raise ValueError(
            f"The following tags contained invalid characters: "
            f'{", ".join(repr(tag) for tag in bad_tags)}'
        )

    return tags


###############################################################################
# PLOTTING FUNCTIONS


def _constrain_fig_resolution(fig, *, max_width, max_res):
    """Limit the resolution (DPI) of a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure whose DPI to adjust.
    max_width : int | None
        The max. allowed width, in pixels.
    max_res : float | None
        The max. allowed resolution, in DPI.

    Returns
    -------
    Nothing, alters the figure's properties in-place.
    """
    dpi = orig_dpi = fig.get_dpi()
    # Limited by figure width?
    if max_width is not None:
        dpi = min(dpi, max_width / fig.get_size_inches()[0])
    # Limited by resolution?
    if max_res is not None:
        dpi = min(dpi, max_res)
    if orig_dpi != dpi:
        fig.set_dpi(dpi)


def _fig_to_img(
    fig,
    *,
    image_format="png",
    own_figure=True,
    max_width=MAX_IMG_WIDTH,
    max_res=MAX_IMG_RES,
    **mpl_kwargs,
):
    """Plot figure and create a binary image."""
    # fig can be ndarray, mpl Figure, PyVista Figure
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if isinstance(fig, np.ndarray):
        # In this case, we are creating the fig, so we might as well
        # auto-close in all cases
        fig = _ndarray_to_fig(fig)
        if own_figure:
            _constrain_fig_resolution(fig, max_width=max_width, max_res=max_res)
        own_figure = True  # close the figure we just created
    elif isinstance(fig, Figure):
        if own_figure:
            _constrain_fig_resolution(fig, max_width=max_width, max_res=max_res)
    else:
        # Don't attempt a mne_qt_browser import here (it might pull in Qt
        # libraries we don't want), so use a probably good enough class name
        # check instead
        if fig.__class__.__name__ in ("MNEQtBrowser", "PyQtGraphBrowser"):
            img = _mne_qt_browser_screenshot(fig, return_type="ndarray")
        elif isinstance(fig, Figure3D):
            from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING, backend

            backend._check_3d_figure(figure=fig)
            if not MNE_3D_BACKEND_TESTING:
                img = backend._take_3d_screenshot(figure=fig)
            else:  # Testing mode
                img = np.zeros((2, 2, 3))
            if own_figure:
                backend._close_3d_figure(figure=fig)
        else:
            raise TypeError(
                "figure must be an instance of np.ndarray, matplotlib Figure, "
                "mne_qt_browser.figure.MNEQtBrowser, or mne.viz.Figure3D, got "
                f"{type(fig)}"
            )
        fig = _ndarray_to_fig(img)
        if own_figure:
            _constrain_fig_resolution(fig, max_width=max_width, max_res=max_res)
        own_figure = True  # close the fig we just created

    output = BytesIO()
    dpi = fig.get_dpi()
    logger.debug(
        f"Saving figure with dimension {fig.get_size_inches()} inches with "
        f"{dpi} dpi"
    )

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    pil_kwargs = dict()
    if image_format == "webp":
        pil_kwargs.update(lossless=True, method=6)
    elif image_format == "png":
        pil_kwargs.update(optimize=True, compress_level=9)
    if pil_kwargs:
        # matplotlib modifies the passed dict, which is a bug
        mpl_kwargs["pil_kwargs"] = pil_kwargs.copy()

    mpl_format = image_format
    if image_format == "ndarray":
        mpl_format = "png"
    fig.savefig(output, format=mpl_format, dpi=dpi, **mpl_kwargs)

    if own_figure:
        plt.close(fig)

    # Remove alpha
    if image_format not in ("svg", "ndarray"):
        from PIL import Image

        output.seek(0)
        orig = Image.open(output)
        if orig.mode == "RGBA":
            background = Image.new("RGBA", orig.size, (255, 255, 255))
            new = Image.alpha_composite(background, orig).convert("RGB")
            output = BytesIO()
            new.save(output, format=image_format, dpi=(dpi, dpi), **pil_kwargs)

    if image_format == "ndarray":
        output.seek(0)
        output = plt.imread(output, format="png")
    else:
        output = output.getvalue()
        if image_format == "svg":
            output = output.decode("utf-8")
        else:
            output = base64.b64encode(output).decode("ascii")
    return output


def _get_bem_contour_figs_as_arrays(
    *, sl, n_jobs, mri_fname, surfaces, orientation, src, show, show_orientation, width
):
    """Render BEM surface contours on MRI slices.

    Returns
    -------
    list of array
        A list of NumPy arrays that represent the generated Matplotlib figures.
    """
    parallel, p_fun, n_jobs = parallel_func(
        _plot_mri_contours, n_jobs, max_jobs=len(sl), prefer="threads"
    )
    outs = parallel(
        p_fun(
            slices=s,
            mri_fname=mri_fname,
            surfaces=surfaces,
            orientation=orientation,
            src=src,
            show=show,
            show_orientation=show_orientation,
            width=width,
            slices_as_subplots=False,
        )
        for s in np.array_split(sl, n_jobs)
    )
    out = list()
    for o in outs:
        out.extend(o)
    return out


def _iterate_trans_views(function, alpha, **kwargs):
    """Auxiliary function to iterate over views in trans fig."""
    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING

    # TODO: Eventually maybe we should expose the size option?
    size = (80, 80) if MNE_3D_BACKEND_TESTING else (800, 800)
    fig = create_3d_figure(size, bgcolor=(0.5, 0.5, 0.5))
    from ..viz.backends.renderer import backend

    try:
        try:
            return _itv(function, fig, surfaces={"head-dense": alpha}, **kwargs)
        except OSError:
            return _itv(function, fig, surfaces={"head": alpha}, **kwargs)
    finally:
        backend._close_3d_figure(fig)


def _itv(function, fig, *, max_width=MAX_IMG_WIDTH, max_res=MAX_IMG_RES, **kwargs):
    from ..viz.backends.renderer import MNE_3D_BACKEND_TESTING, backend

    function(fig=fig, **kwargs)

    views = ("frontal", "lateral", "medial", "axial", "rostral", "coronal")

    images = []
    for view in views:
        if not MNE_3D_BACKEND_TESTING:
            set_3d_view(fig, **views_dicts["both"][view])
            backend._check_3d_figure(fig)
            im = backend._take_3d_screenshot(figure=fig)
        else:  # Testing mode
            im = np.zeros((2, 2, 3))
        images.append(im)

    images = np.concatenate(
        [np.concatenate(images[:3], axis=1), np.concatenate(images[3:], axis=1)], axis=0
    )

    try:
        dists = dig_mri_distances(
            info=kwargs["info"],
            trans=kwargs["trans"],
            subject=kwargs["subject"],
            subjects_dir=kwargs["subjects_dir"],
            on_defects="ignore",
        )
        caption = (
            f"Average distance from {len(dists)} digitized points to "
            f"head: {1e3 * np.mean(dists):.2f} mm"
        )
    except BaseException as e:
        caption = "Distances could not be calculated from digitized points"
        warn(f"{caption}: {e}")
    img = _fig_to_img(images, image_format="png", max_width=max_width, max_res=max_res)

    return img, caption


def _plot_ica_properties_as_arrays(
    *, ica, inst, picks, n_jobs, max_width=MAX_IMG_WIDTH, max_res=MAX_IMG_RES
):
    """Parallelize ICA component properties plotting, and return arrays.

    Returns
    -------
    outs : list of array
        The properties plots as NumPy arrays.
    """

    def _plot_one_ica_property(*, ica, inst, pick):
        figs = ica.plot_properties(inst=inst, picks=pick, show=False)
        assert len(figs) == 1
        return _fig_to_img(
            figs[0],
            max_width=max_width,
            max_res=max_res,
            image_format="ndarray",
            pad_inches=0,
        )

    parallel, p_fun, n_jobs = parallel_func(
        func=_plot_one_ica_property, n_jobs=n_jobs, max_jobs=len(picks)
    )
    outs = parallel(p_fun(ica=ica, inst=inst, pick=pick) for pick in picks)
    return outs


###############################################################################
# TOC FUNCTIONS


def _endswith(fname: str | Path, suffixes):
    """Aux function to test if file name includes the specified suffixes."""
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    if isinstance(fname, Path):
        fname = fname.name
    for suffix in suffixes:
        for ext in SUPPORTED_READ_RAW_EXTENSIONS:
            if fname.endswith(
                (
                    f"-{suffix}{ext}",
                    f"-{suffix}{ext}",
                    f"_{suffix}{ext}",
                    f"_{suffix}{ext}",
                )
            ):
                return True
    return False


_backward_compat_map = dict(
    img_max_width=MAX_IMG_WIDTH,
    img_max_res=MAX_IMG_RES,
    collapse=(),
)


def open_report(fname, **params):
    """Read a saved report or, if it doesn't exist yet, create a new one.

    The returned report can be used as a context manager, in which case any
    changes to the report are saved when exiting the context block.

    Parameters
    ----------
    fname : path-like
        The file containing the report, stored in the HDF5 format. If the file
        does not exist yet, a new report is created that will be saved to the
        specified file.
    **params : kwargs
        When creating a new report, any named parameters other than ``fname``
        are passed to the ``__init__`` function of the `Report` object. When
        reading an existing report, the parameters are checked with the
        loaded report and an exception is raised when they don't match.

    Returns
    -------
    report : instance of Report
        The report.
    """
    fname = str(_check_fname(fname=fname, overwrite="read", must_exist=False))
    if op.exists(fname):
        # Check **params with the loaded report
        read_hdf5, _ = _import_h5io_funcs()
        state = read_hdf5(fname, title="mnepython")
        for param in params:
            if param not in state:
                if param in _backward_compat_map:
                    state[param] = _backward_compat_map[param]
                else:
                    raise ValueError(f"The loaded report has no attribute {param}")
            if params[param] != state[param]:
                raise ValueError(
                    f"Attribute '{param}' of loaded report ({params[param]}) does not "
                    f"match the given parameter ({state[param]})."
                )
        report = Report()
        report.__setstate__(state)
    else:
        report = Report(**params)
    # Keep track of the filename in case the Report object is used as a context
    # manager.
    report.fname = fname
    return report


###############################################################################
# HTML scan renderer

mne_logo_path = Path(__file__).parents[1] / "icons" / "mne_icon-cropped.png"
mne_logo = base64.b64encode(mne_logo_path.read_bytes()).decode("ascii")

_ALLOWED_IMAGE_FORMATS = ("png", "svg", "webp")


def _check_image_format(rep, image_format):
    """Ensure fmt is valid."""
    if rep is None or image_format is not None:
        allowed = list(_ALLOWED_IMAGE_FORMATS) + ["auto"]
        extra = ""
        _check_option("image_format", image_format, allowed_values=allowed, extra=extra)
    else:
        image_format = rep.image_format
    if image_format == "auto":
        image_format = "webp"
    return image_format


@fill_doc
class Report:
    r"""Object for rendering HTML.

    Parameters
    ----------
    info_fname : None | str
        Name of the file containing the info dictionary.
    %(subjects_dir)s
    subject : str | None
        Subject name.
    title : str
        Title of the report.
    cov_fname : None | str
        Name of the file containing the noise covariance.
    %(baseline_report)s
        Defaults to ``None``, i.e. no baseline correction.
    image_format : 'png' | 'svg' | 'webp' | 'auto'
        Default image format to use (default is ``'auto'``, which will use
        ``'webp'`` if available and ``'png'`` otherwise).
        ``'svg'`` uses vector graphics, so fidelity is higher but can increase
        file size and browser image rendering time as well.

        .. versionadded:: 0.15
        .. versionchanged:: 1.3
           Added support for ``'webp'`` format, removed support for GIF, and
           set the default to ``'auto'``.
    raw_psd : bool | dict
        If True, include PSD plots for raw files. Can be False (default) to
        omit, True to plot, or a dict to pass as ``kwargs`` to
        :meth:`mne.time_frequency.Spectrum.plot`.

        .. versionadded:: 0.17
        .. versionchanged:: 1.4
           kwargs are sent to ``spectrum.plot`` instead of ``raw.plot_psd``.
    projs : bool
        Whether to include topographic plots of SSP projectors, if present in
        the data. Defaults to ``False``.

        .. versionadded:: 0.21
    img_max_width : int | None
        Maximum image width in pixels.

        .. versionadded:: 1.9
    img_max_res : float | None
        Maximum image resolution in dots per inch.

        .. versionadded:: 1.9
    collapse : tuple of str | str
        Tuple of elements to collapse by default. Defaults to an empty tuple.
        For now the only option it can contain is "section".

        .. versionadded:: 1.9
    %(verbose)s

    Attributes
    ----------
    info_fname : None | str
        Name of the file containing the info dictionary.
    %(subjects_dir)s
    subject : str | None
        Subject name.
    title : str
        Title of the report.
    cov_fname : None | str
        Name of the file containing the noise covariance.
    %(baseline_report)s
        Defaults to ``None``, i.e. no baseline correction.
    image_format : str
        Default image format to use.

        .. versionadded:: 0.15
    raw_psd : bool | dict
        If True, include PSD plots for raw files. Can be False (default) to
        omit, True to plot, or a dict to pass as ``kwargs`` to
        :meth:`mne.time_frequency.Spectrum.plot`.

        .. versionadded:: 0.17
        .. versionchanged:: 1.4
           kwargs are sent to ``spectrum.plot`` instead of ``raw.plot_psd``.
    projs : bool
        Whether to include topographic plots of SSP projectors, if present in
        the data. Defaults to ``False``.

        .. versionadded:: 0.21
    %(verbose)s
    html : list of str
        Contains items of html-page.
    include : list of str
        Dictionary containing elements included in head.
    fnames : list of str
        List of file names rendered.
    sections : list of str
        List of sections.
    lang : str
        language setting for the HTML file.
    img_max_width : int | None
        Maximum image width in pixels.

        .. versionadded:: 1.9
    img_max_res : float | None
        Maximum image resolution in dots per inch.

        .. versionadded:: 1.9
    collapse : tuple of str
        Tuple of elements to collapse by default. See above.

        .. versionadded:: 1.9

    Notes
    -----
    See :ref:`tut-report` for an introduction to using ``mne.Report``.

    .. versionadded:: 0.8.0
    """

    @verbose
    def __init__(
        self,
        info_fname=None,
        subjects_dir=None,
        subject=None,
        title=None,
        cov_fname=None,
        baseline=None,
        image_format="auto",
        raw_psd=False,
        projs=False,
        *,
        img_max_width=MAX_IMG_WIDTH,
        img_max_res=MAX_IMG_RES,
        collapse=(),
        verbose=None,
    ):
        self.info_fname = str(info_fname) if info_fname is not None else None
        self.cov_fname = str(cov_fname) if cov_fname is not None else None
        self.baseline = baseline
        if subjects_dir is not None:
            subjects_dir = get_subjects_dir(subjects_dir)
            if subjects_dir is not None:
                subjects_dir = str(subjects_dir)
        self.subjects_dir = subjects_dir
        self.subject = subject
        self.title = title
        self.image_format = _check_image_format(None, image_format)
        self.projs = projs
        # dom_id is mostly for backward compat and testing nowadays
        self._dom_id = 0
        self._dup_limit = 10000  # should be enough duplicates
        self._content = []
        self.include = []
        self.lang = "en-us"  # language setting for the HTML file
        self.img_max_width = img_max_width
        self.img_max_res = img_max_res
        self.collapse = collapse
        if not isinstance(raw_psd, bool) and not isinstance(raw_psd, dict):
            raise TypeError(f"raw_psd must be bool or dict, got {type(raw_psd)}")
        self.raw_psd = raw_psd
        self._init_render()  # Initialize the renderer

        self.fname = None  # The name of the saved report
        self.data_path = None

    @property
    def img_max_width(self):
        return self._img_max_width

    @img_max_width.setter
    def img_max_width(self, value):
        _validate_type(value, ("int-like", None), "img_max_width")
        if value is not None:
            value = int(value)
            if value < 1:
                raise ValueError(f"img_max_width must be at least 1, got {value}")
        self._img_max_width = value

    @property
    def img_max_res(self):
        return self._img_max_res

    @img_max_res.setter
    def img_max_res(self, value):
        _validate_type(value, ("numeric", None), "img_max_res")
        if value is not None:
            value = float(value)
            if value < 1:
                raise ValueError(f"img_max_res must be at least 1, got {value}")
        self._img_max_res = value

    @property
    def collapse(self):
        return self._collapse

    @collapse.setter
    def collapse(self, value):
        _validate_type(value, (list, tuple, str), "collapse")
        if isinstance(value, str):
            value = [value]
        for vi, v in enumerate(value):
            _validate_type(v, str, f"collapse[{vi}]")
            _check_option(f"collapse[{vi}]", v, ("section",))
        self._collapse = tuple(value)

    def __repr__(self):
        """Print useful info about report."""
        htmls, _, titles, _ = self._content_as_html()
        items = self._content
        s = "<Report"
        s += f" | {len(titles)} title{_pl(titles)}"
        s += f" | {len(items)} item{_pl(items)}"
        if self.title is not None:
            s += f" | {self.title}"
        if len(titles) > 0:
            titles = [f" {t}" for t in titles]  # indent
            tr = max(len(s), 50)  # trim to larger of opening str and 50
            titles = [f"{t[:tr - 2]} …" if len(t) > tr else t for t in titles]
            # then trim to the max length of all of these
            tr = max(len(title) for title in titles)
            tr = max(tr, len(s))
            b_to_mb = 1.0 / (1024.0**2)
            content_element_mb = [len(html) * b_to_mb for html in htmls]
            total_mb = f"{sum(content_element_mb):0.1f}"
            content_element_mb = [
                f"{sz:0.1f}".rjust(len(total_mb)) for sz in content_element_mb
            ]
            s = f"{s.ljust(tr + 1)} | {total_mb} MB"
            s += "\n" + "\n".join(
                f"{title[:tr].ljust(tr + 1)} | {sz} MB"
                for title, sz in zip(titles, content_element_mb)
            )
            s += "\n"
        s += ">"
        return s

    def __len__(self):
        """Return the number of files processed by the report.

        Returns
        -------
        n_files : int
            The number of files processed.
        """
        return len(self._content)

    @staticmethod
    def _get_state_params():
        # Which attributes to store in and read from HDF5 files
        return (
            "baseline",
            "cov_fname",
            "include",
            "_content",
            "image_format",
            "info_fname",
            "_dom_id",
            # dup_limit omitted because we never change it from default
            "raw_psd",
            "projs",
            "subjects_dir",
            "subject",
            "title",
            "data_path",
            "lang",
            "fname",
            "img_max_width",
            "img_max_res",
            "collapse",
        )

    def _get_dom_id(self, *, section, title, extra_exclude=None):
        """Get unique ID for content to append to the DOM."""
        _validate_type(title, str, "title")
        _validate_type(section, (str, None), "section")
        if section is not None:
            title = f"{section}-{title}"
        dom_id = _id_sanitize(title)
        del title, section
        # find a new unique ID
        dom_ids = set(c.dom_id for c in self._content)
        if extra_exclude:
            assert isinstance(extra_exclude, set), type(extra_exclude)
            dom_ids.update(extra_exclude)
        if dom_id not in dom_ids:
            return dom_id
        for ii in range(1, self._dup_limit):  # should be enough duplicates...
            dom_id_inc = f"{dom_id}-{ii}"
            if dom_id_inc not in dom_ids:
                return dom_id_inc
        # But let's not fail if the limit is low
        self._dom_id += 1
        return f"global-{self._dom_id}"

    def _validate_topomap_kwargs(self, topomap_kwargs):
        _validate_type(topomap_kwargs, (dict, None), "topomap_kwargs")
        topomap_kwargs = dict() if topomap_kwargs is None else topomap_kwargs
        return topomap_kwargs

    def _validate_input(self, items, captions, tag, comments=None):
        """Validate input."""
        if not isinstance(items, list | tuple):
            items = [items]
        if not isinstance(captions, list | tuple):
            captions = [captions]
        if not isinstance(comments, list | tuple) and comments is not None:
            comments = [comments]
        if comments is not None and len(comments) != len(items):
            raise ValueError(
                f'Number of "comments" and report items must be equal, '
                f"or comments should be None; got "
                f"{len(comments)} and {len(items)}"
            )
        elif captions is not None and len(captions) != len(items):
            raise ValueError(
                f'Number of "captions" and report items must be equal; '
                f"got {len(captions)} and {len(items)}"
            )
        return items, captions, comments

    def copy(self):
        """Return a deepcopy of the report.

        Returns
        -------
        report : instance of Report
            The copied report.
        """
        return copy.deepcopy(self)

    def get_contents(self):
        """Get the content of the report.

        Returns
        -------
        titles : list of str
            The title of each content element.
        tags : list of list of str
            The tags for each content element, one list per element.
        htmls : list of str
            The HTML contents for each element.

        Notes
        -----
        .. versionadded:: 1.7
        """
        htmls, _, titles, tags = self._content_as_html()
        return titles, tags, htmls

    def reorder(self, order):
        """Reorder the report content.

        Parameters
        ----------
        order : array-like of int
            The indices of the new order (as if you were reordering an array).
            For example if there are 4 elements in the report,
            ``order=[3, 0, 1, 2]`` would take the last element and move it to
            the front. In other words, ``elements = [elements[ii] for ii in order]]``.

        Notes
        -----
        .. versionadded:: 1.7
        """
        _validate_type(order, "array-like", "order")
        order = np.array(order)
        if order.dtype.kind != "i" or order.ndim != 1:
            raise ValueError(
                "order must be an array of integers, got "
                f"{order.ndim}D array of dtype {order.dtype}"
            )
        n_elements = len(self._content)
        if not np.array_equal(np.sort(order), np.arange(n_elements)):
            raise ValueError(
                f"order must be a permutation of range({n_elements}), got:\n{order}"
            )
        self._content = [self._content[ii] for ii in order]

    def _content_as_html(self):
        """Generate HTML representations based on the added content & sections.

        Returns
        -------
        htmls : list of str
            The HTML representations of the content.
        dom_ids : list of str
            The DOM IDs corresponding to the HTML representations.
        titles : list of str
            The titles corresponding to the HTML representations.
        tags : list of tuple of str
            The tags corresponding to the HTML representations.
        """
        section_dom_ids = set()
        htmls = []
        dom_ids = []
        titles = []
        tags = []

        content_elements = self._content.copy()  # shallow copy

        # We loop over all content elements and implement special treatment
        # for those that are part of a section: Those sections don't actually
        # exist in `self._content` – we're creating them on-the-fly here!
        for idx, content_element in enumerate(content_elements):
            if content_element.section:
                if content_element.section in titles:
                    # The section and all its child elements have already been
                    # added
                    continue

                # Add all elements belonging to the current section
                section_elements = [
                    el
                    for el in content_elements[idx:]
                    if el.section == content_element.section
                ]
                section_htmls = [el.html for el in section_elements]
                section_tags = tuple(
                    sorted(set([t for el in section_elements for t in el.tags]))
                )
                section_dom_id = self._get_dom_id(
                    section=None,  # root level of document
                    title=content_element.section,
                    extra_exclude=section_dom_ids,
                )
                section_dom_ids.add(section_dom_id)

                # Finally, create the section HTML element.
                section_html = _html_section_element(
                    id_=section_dom_id,
                    htmls=section_htmls,
                    tags=section_tags,
                    title=content_element.section,
                    div_klass="section",
                    show="section" not in self.collapse,
                )
                htmls.append(section_html)
                dom_ids.append(section_dom_id)
                titles.append(content_element.section)
                tags.append(section_tags)
            else:
                # The element is not part of a section, so we can simply
                # append it as-is.
                htmls.append(content_element.html)
                dom_ids.append(content_element.dom_id)
                titles.append(content_element.name)
                tags.append(content_element.tags)

        return htmls, dom_ids, titles, tags

    @property
    def html(self):
        """A list of HTML representations for all content elements."""
        return self._content_as_html()[0]

    @property
    def tags(self):
        """A sorted tuple of all tags currently used in the report."""
        return tuple(sorted(set(sum(self._content_as_html()[3], ()))))

    def add_custom_css(self, css):
        """Add custom CSS to the report.

        Parameters
        ----------
        css : str
            Style definitions to add to the report. The content of this string
            will be embedded between HTML ``<style>`` and ``</style>`` tags.

        Notes
        -----
        .. versionadded:: 0.23
        """
        style = f'\n<style type="text/css">\n{css}\n</style>'
        self.include += style

    def add_custom_js(self, js):
        """Add custom JavaScript to the report.

        Parameters
        ----------
        js : str
            JavaScript code to add to the report. The content of this string
            will be embedded between HTML ``<script>`` and ``</script>`` tags.

        Notes
        -----
        .. versionadded:: 0.23
        """
        script = f'\n<script type="text/javascript">\n{js}\n</script>'
        self.include += script

    @fill_doc
    def add_epochs(
        self,
        epochs,
        title,
        *,
        psd=True,
        projs=None,
        image_kwargs=None,
        topomap_kwargs=None,
        drop_log_ignore=("IGNORED",),
        tags=("epochs",),
        replace=False,
    ):
        """Add `~mne.Epochs` to the report.

        Parameters
        ----------
        epochs : path-like | instance of Epochs
            The epochs to add to the report.
        title : str
            The title to add.
        psd : bool | float
            If a float, the duration of data to use for creation of PSD plots,
            in seconds. PSD will be calculated on as many epochs as required to
            cover at least this duration. Epochs will be picked across the
            entire time range in equally-spaced distance.

            .. note::
              In rare edge cases, we may not be able to create a grid of
              equally-spaced epochs that cover the entire requested time range.
              In these situations, a warning will be emitted, informing you
              about the duration that's actually being used.

            If ``True``, add PSD plots based on all ``epochs``. If ``False``,
            do not add PSD plots.
        %(projs_report)s
        image_kwargs : dict | None
            Keyword arguments to pass to the "epochs image"-generating
            function (:meth:`mne.Epochs.plot_image`).
            Keys are channel types, values are dicts containing kwargs to pass.
            For example, to use the rejection limits per channel type you could pass::

                image_kwargs=dict(
                    grad=dict(vmin=-reject['grad'], vmax=-reject['grad']),
                    mag=dict(vmin=-reject['mag'], vmax=reject['mag']),
                )

            .. versionadded:: 1.7
        %(topomap_kwargs)s
        drop_log_ignore : array-like of str
            The drop reasons to ignore when creating the drop log bar plot.
            All epochs for which a drop reason listed here appears in
            ``epochs.drop_log`` will be excluded from the drop log plot.
        %(tags_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24
        """
        tags = _check_tags(tags)
        add_projs = self.projs if projs is None else projs
        self._add_epochs(
            epochs=epochs,
            psd=psd,
            add_projs=add_projs,
            image_kwargs=image_kwargs,
            topomap_kwargs=topomap_kwargs,
            drop_log_ignore=drop_log_ignore,
            section=title,
            tags=tags,
            image_format=self.image_format,
            replace=replace,
        )

    @fill_doc
    def add_evokeds(
        self,
        evokeds,
        *,
        titles=None,
        noise_cov=None,
        projs=None,
        n_time_points=None,
        tags=("evoked",),
        replace=False,
        topomap_kwargs=None,
        n_jobs=None,
    ):
        """Add `~mne.Evoked` objects to the report.

        Parameters
        ----------
        evokeds : path-like | instance of Evoked | list of Evoked
            The evoked data to add to the report. Multiple `~mne.Evoked`
            objects – as returned from `mne.read_evokeds` – can be passed as
            a list.
        titles : str | list of str | None
            The titles corresponding to the evoked data. If ``None``, the
            content of ``evoked.comment`` from each evoked will be used as
            title.
        noise_cov : path-like | instance of Covariance | None
            A noise covariance matrix. If provided, will be used to whiten
            the ``evokeds``. If ``None``, will fall back to the ``cov_fname``
            provided upon report creation.
        %(projs_report)s
        n_time_points : int | None
            The number of equidistant time points to render. If ``None``,
            will render each `~mne.Evoked` at 21 time points, unless the data
            contains fewer time points, in which case all will be rendered.
        %(tags_report)s
        %(replace_report)s
        %(topomap_kwargs)s
        %(n_jobs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        if isinstance(evokeds, Evoked):
            evokeds = [evokeds]
        elif isinstance(evokeds, list):
            pass
        else:
            evoked_fname = evokeds
            logger.debug(f"Evoked: Reading {evoked_fname}")
            evokeds = read_evokeds(evoked_fname, verbose=False)

        if self.baseline is not None:
            evokeds = [e.copy().apply_baseline(self.baseline) for e in evokeds]

        if titles is None:
            titles = [e.comment for e in evokeds]
        elif isinstance(titles, str):
            titles = [titles]

        if len(evokeds) != len(titles):
            raise ValueError(
                f"Number of evoked objects ({len(evokeds)}) must "
                f"match number of captions ({len(titles)})"
            )

        if noise_cov is None:
            noise_cov = self.cov_fname
        if noise_cov is not None and not isinstance(noise_cov, Covariance):
            noise_cov = read_cov(fname=noise_cov)
        tags = _check_tags(tags)

        add_projs = self.projs if projs is None else projs

        for evoked, title in zip(evokeds, titles):
            self._add_evoked(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=self.image_format,
                add_projs=add_projs,
                n_time_points=n_time_points,
                tags=tags,
                section=title,
                topomap_kwargs=topomap_kwargs,
                n_jobs=n_jobs,
                replace=replace,
            )

    @fill_doc
    def add_raw(
        self,
        raw,
        title,
        *,
        psd=None,
        projs=None,
        butterfly=True,
        scalings=None,
        tags=("raw",),
        replace=False,
        topomap_kwargs=None,
    ):
        """Add `~mne.io.Raw` objects to the report.

        Parameters
        ----------
        raw : path-like | instance of Raw
            The data to add to the report.
        title : str
            The title corresponding to the ``raw`` object.
        psd : bool | None
            Whether to add PSD plots. Overrides the ``raw_psd`` parameter
            passed when initializing the `~mne.Report`. If ``None``, use
            ``raw_psd`` from `~mne.Report` creation.
        %(projs_report)s
        butterfly : bool | int
            Whether to add butterfly plots of the data. Can be useful to
            spot problematic channels. If ``True``, 10 equally-spaced 1-second
            segments will be plotted. If an integer, specifies the number of
            1-second segments to plot. Larger numbers may take a considerable
            amount of time if the data contains many sensors. You can disable
            butterfly plots altogether by passing ``False``.
        %(scalings)s
        %(tags_report)s
        %(replace_report)s
        %(topomap_kwargs)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        if psd is None:
            add_psd = dict() if self.raw_psd is True else self.raw_psd
        elif psd is True:
            add_psd = dict()
        else:
            add_psd = False

        add_projs = self.projs if projs is None else projs

        self._add_raw(
            raw=raw,
            add_psd=add_psd,
            add_projs=add_projs,
            butterfly=butterfly,
            butterfly_scalings=scalings,
            image_format=self.image_format,
            tags=tags,
            topomap_kwargs=topomap_kwargs,
            section=title,
            replace=replace,
        )

    @fill_doc
    def add_stc(
        self,
        stc,
        title,
        *,
        subject=None,
        subjects_dir=None,
        n_time_points=None,
        tags=("source-estimate",),
        replace=False,
        section=None,
        stc_plot_kwargs=None,
    ):
        """Add a `~mne.SourceEstimate` (STC) to the report.

        Parameters
        ----------
        stc : path-like | instance of SourceEstimate
            The `~mne.SourceEstimate` to add to the report.
        title : str
            The title to add.
        subject : str | None
            The name of the FreeSurfer subject the STC belongs to. The name is
            not stored with the STC data and therefore needs to be specified.
            If ``None``, will use the value of ``subject`` passed on report
            creation.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        n_time_points : int | None
            The number of equidistant time points to render. If ``None``,
            will render ``stc`` at 51 time points, unless the data
            contains fewer time points, in which case all will be rendered.
        %(tags_report)s
        %(replace_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(stc_plot_kwargs_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_stc(
            stc=stc,
            title=title,
            tags=tags,
            image_format=self.image_format,
            subject=subject,
            subjects_dir=subjects_dir,
            n_time_points=n_time_points,
            stc_plot_kwargs=stc_plot_kwargs,
            section=section,
            replace=replace,
        )

    @fill_doc
    def add_forward(
        self,
        forward,
        title,
        *,
        subject=None,
        subjects_dir=None,
        tags=("forward-solution",),
        section=None,
        replace=False,
    ):
        """Add a forward solution.

        Parameters
        ----------
        forward : instance of Forward | path-like
            The forward solution to add to the report.
        title : str
            The title corresponding to forward solution.
        subject : str | None
            The name of the FreeSurfer subject ``forward`` belongs to. If
            provided, the sensitivity maps of the forward solution will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        self._add_forward(
            forward=forward,
            subject=subject,
            subjects_dir=subjects_dir,
            title=title,
            image_format=self.image_format,
            section=section,
            tags=tags,
            replace=replace,
        )

    @fill_doc
    def add_inverse_operator(
        self,
        inverse_operator,
        title,
        *,
        subject=None,
        subjects_dir=None,
        trans=None,
        tags=("inverse-operator",),
        section=None,
        replace=False,
    ):
        """Add an inverse operator.

        Parameters
        ----------
        inverse_operator : instance of InverseOperator | path-like
            The inverse operator to add to the report.
        title : str
            The title corresponding to the inverse operator object.
        subject : str | None
            The name of the FreeSurfer subject ``inverse_op`` belongs to. If
            provided, the source space the inverse solution is based on will
            be visualized. If ``None``, will use the value of ``subject``
            passed on report creation. If supplied, also pass ``subjects_dir``
            and ``trans``.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        trans : path-like | instance of Transform | None
            The ``head -> MRI`` transformation for ``subject``.
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)

        if (subject is not None and trans is None) or (
            trans is not None and subject is None
        ):
            raise ValueError("Please pass subject AND trans, or neither.")

        self._add_inverse_operator(
            inverse_operator=inverse_operator,
            subject=subject,
            subjects_dir=subjects_dir,
            trans=trans,
            title=title,
            image_format=self.image_format,
            section=section,
            tags=tags,
            replace=replace,
        )

    @fill_doc
    def add_trans(
        self,
        trans,
        *,
        info,
        title,
        subject=None,
        subjects_dir=None,
        alpha=None,
        tags=("coregistration",),
        section=None,
        coord_frame="mri",
        replace=False,
    ):
        """Add a coregistration visualization to the report.

        Parameters
        ----------
        trans : path-like | instance of Transform
            The ``head -> MRI`` transformation to render.
        info : path-like | instance of Info
            The `~mne.Info` corresponding to ``trans``.
        title : str
            The title to add.
        subject : str | None
            The name of the FreeSurfer subject the ``trans```` belong to. The
            name is not stored with the ``trans`` and therefore needs to be
            specified. If ``None``, will use the value of ``subject`` passed on
            report creation.
        subjects_dir : path-like | None
            The FreeSurfer ``SUBJECTS_DIR``.
        alpha : float | None
            The level of opacity to apply to the head surface. If a float, must
            be between 0 and 1 (inclusive), where 1 means fully opaque. If
            ``None``, will use the MNE-Python default value.
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        coord_frame : 'auto' | 'head' | 'meg' | 'mri'
            Coordinate frame used for plotting. See :func:`mne.viz.plot_alignment`.
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_trans(
            trans=trans,
            info=info,
            subject=subject,
            subjects_dir=subjects_dir,
            alpha=alpha,
            title=title,
            section=section,
            tags=tags,
            coord_frame=coord_frame,
            replace=replace,
        )

    @fill_doc
    def add_covariance(self, cov, *, info, title, tags=("covariance",), replace=False):
        """Add covariance to the report.

        Parameters
        ----------
        cov : path-like | instance of Covariance
            The `~mne.Covariance` to add to the report.
        info : path-like | instance of Info
            The `~mne.Info` corresponding to ``cov``.
        title : str
            The title corresponding to the `~mne.Covariance` object.
        %(tags_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_cov(
            cov=cov,
            info=info,
            image_format=self.image_format,
            section=title,
            tags=tags,
            replace=replace,
        )

    @fill_doc
    def add_events(
        self,
        events,
        title,
        *,
        event_id=None,
        sfreq,
        first_samp=0,
        color=None,
        tags=("events",),
        section=None,
        replace=False,
    ):
        """Add events to the report.

        Parameters
        ----------
        events : path-like | array, shape (n_events, 3)
            An MNE-Python events array.
        title : str
            The title corresponding to the events.
        event_id : dict
            A dictionary mapping event names (keys) to event codes (values).
        sfreq : float
            The sampling frequency used while recording.
        first_samp : int
            The first sample point in the recording. This corresponds to
            ``raw.first_samp`` on files created with Elekta/Neuromag systems.
        color : dict | None
            Dictionary of event_id integers as keys and colors as values. This
            parameter is directly passed to :func:`mne.viz.plot_events`.

            .. versionadded:: 1.8.0
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            first_samp=first_samp,
            color=color,
            title=title,
            section=section,
            image_format=self.image_format,
            tags=tags,
            replace=replace,
        )

    @fill_doc
    def add_projs(
        self,
        *,
        info,
        title,
        projs=None,
        topomap_kwargs=None,
        tags=("ssp",),
        joint=False,
        picks_trace=None,
        section=None,
        replace=False,
    ):
        """Render (SSP) projection vectors.

        Parameters
        ----------
        info : instance of Info | instance of Evoked | path-like
            An `~mne.Info` structure or the path of a file containing one.
        title : str
            The title corresponding to the :class:`~mne.Projection` object.
        projs : iterable of mne.Projection | path-like | None
            The projection vectors to add to the report. Can be the path to a
            file that will be loaded via `mne.read_proj`. If ``None``, the
            projectors are taken from ``info['projs']``.
        %(topomap_kwargs)s
        %(tags_report)s
        joint : bool
            If True (default False), plot the projectors using
            :func:`mne.viz.plot_projs_joint`, otherwise use
            :func:`mne.viz.plot_projs_topomap`. If True, then ``info`` must be an
            instance of :class:`mne.Evoked`.

            .. versionadded:: 1.9
        %(picks_plot_projs_joint_trace)s
            Only used when ``joint=True``.

            .. versionadded:: 1.9
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_projs(
            info=info,
            projs=projs,
            title=title,
            image_format=self.image_format,
            section=section,
            tags=tags,
            topomap_kwargs=topomap_kwargs,
            replace=replace,
            joint=joint,
        )

    def _add_ica_overlay(self, *, ica, inst, image_format, section, tags, replace):
        if isinstance(inst, BaseRaw):
            inst_ = inst
        else:  # Epochs
            inst_ = inst.average()

        fig = ica.plot_overlay(inst=inst_, show=False, on_baseline="reapply")
        del inst_
        self._add_figure(
            fig=fig,
            title="Original and cleaned signal",
            caption=None,
            image_format=image_format,
            section=section,
            tags=tags,
            replace=replace,
            own_figure=True,
        )

    def _add_ica_properties(
        self, *, ica, picks, inst, n_jobs, image_format, section, tags, replace
    ):
        ch_type = _get_plot_ch_type(inst=ica.info, ch_type=None)
        if not _check_ch_locs(info=ica.info, ch_type=ch_type):
            ch_type_name = _handle_default("titles")[ch_type]
            warn(
                f"No {ch_type_name} channel locations found, cannot "
                f"create ICA properties plots"
            )
            return

        if picks is None:
            picks = list(range(ica.n_components_))

        figs = _plot_ica_properties_as_arrays(
            ica=ica,
            inst=inst,
            picks=picks,
            n_jobs=n_jobs,
            max_width=MAX_IMG_WIDTH,
            max_res=MAX_IMG_RES,
        )
        assert len(figs) == len(picks)

        captions = []
        for idx in range(len(figs)):
            caption = f"ICA component {picks[idx]}."
            captions.append(caption)

        title = "ICA component properties"
        # Only render a slider if we have more than 1 component.
        if len(figs) == 1:
            self._add_figure(
                fig=figs[0],
                title=title,
                caption=captions[0],
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )
        else:
            self._add_slider(
                figs=figs,
                imgs=None,
                title=title,
                captions=captions,
                start_idx=0,
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
                own_figure=True,
            )

    def _add_ica_artifact_sources(
        self, *, ica, inst, artifact_type, image_format, section, tags, replace
    ):
        with use_browser_backend("matplotlib"):
            fig = ica.plot_sources(inst=inst, show=False)
        self._add_figure(
            fig=fig,
            title=f"Original and cleaned {artifact_type} epochs",
            caption=None,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_ica_artifact_scores(
        self, *, ica, scores, artifact_type, image_format, section, tags, replace
    ):
        assert artifact_type in ("EOG", "ECG")
        fig = ica.plot_scores(
            scores=scores, title=None, labels=artifact_type.lower(), show=False
        )
        self._add_figure(
            fig=fig,
            title=f"Scores for matching {artifact_type} patterns",
            caption=None,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_ica_components(self, *, ica, picks, image_format, section, tags, replace):
        ch_type = _get_plot_ch_type(inst=ica.info, ch_type=None)
        if not _check_ch_locs(info=ica.info, ch_type=ch_type):
            ch_type_name = _handle_default("titles")[ch_type]
            warn(
                f"No {ch_type_name} channel locations found, cannot "
                f"create ICA component plots"
            )
            return ""

        figs = ica.plot_components(picks=picks, title="", colorbar=True, show=False)
        if not isinstance(figs, list):
            figs = [figs]

        title = "ICA component topographies"
        if len(figs) == 1:
            fig = figs[0]
            self._add_figure(
                fig=fig,
                title=title,
                caption=None,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )
        else:
            self._add_slider(
                figs=figs,
                imgs=None,
                title=title,
                captions=[None] * len(figs),
                start_idx=0,
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

    def _add_ica(
        self,
        *,
        ica,
        inst,
        picks,
        ecg_evoked,
        eog_evoked,
        ecg_scores,
        eog_scores,
        title,
        image_format,
        section,
        tags,
        n_jobs,
        replace,
    ):
        if _path_like(ica):
            ica = read_ica(ica)

        if ica.current_fit == "unfitted":
            raise RuntimeError(
                "ICA must be fitted before it can be added to the report."
            )

        if inst is None:
            pass  # no-op
        elif _path_like(inst):
            # We cannot know which data type to expect, so let's first try to
            # read a Raw, and if that fails, try to load Epochs
            fname = str(inst)  # could e.g. be a Path!
            raw_kwargs = dict(fname=fname, preload=False)
            if fname.endswith((".fif", ".fif.gz")):
                raw_kwargs["allow_maxshield"] = "yes"

            try:
                inst = read_raw(**raw_kwargs)
            except ValueError:
                try:
                    inst = read_epochs(fname)
                except ValueError:
                    raise ValueError(
                        f"The specified file, {fname}, does not seem to "
                        f"contain Raw data or Epochs"
                    )
        elif not inst.preload:
            raise RuntimeError(
                'You passed an object to Report.add_ica() via the "inst" '
                "parameter that was not preloaded. Please preload the data  "
                "via the load_data() method"
            )

        if _path_like(ecg_evoked):
            ecg_evoked = read_evokeds(fname=ecg_evoked, condition=0)

        if _path_like(eog_evoked):
            eog_evoked = read_evokeds(fname=eog_evoked, condition=0)

        # Summary table
        self._add_html_repr(
            inst=ica,
            title="Info",
            tags=tags,
            section=section,
            replace=replace,
            div_klass="ica",
        )

        # Overlay plot
        if inst:
            self._add_ica_overlay(
                ica=ica,
                inst=inst,
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

        # ECG artifact
        if ecg_scores is not None:
            self._add_ica_artifact_scores(
                ica=ica,
                scores=ecg_scores,
                artifact_type="ECG",
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )
        if ecg_evoked:
            self._add_ica_artifact_sources(
                ica=ica,
                inst=ecg_evoked,
                artifact_type="ECG",
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

        # EOG artifact
        if eog_scores is not None:
            self._add_ica_artifact_scores(
                ica=ica,
                scores=eog_scores,
                artifact_type="EOG",
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )
        if eog_evoked:
            self._add_ica_artifact_sources(
                ica=ica,
                inst=eog_evoked,
                artifact_type="EOG",
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

        # Component topography plots
        self._add_ica_components(
            ica=ica,
            picks=picks,
            image_format=image_format,
            section=section,
            tags=tags,
            replace=replace,
        )

        # Properties plots
        if inst:
            self._add_ica_properties(
                ica=ica,
                picks=picks,
                inst=inst,
                n_jobs=n_jobs,
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

    @fill_doc
    def add_ica(
        self,
        ica,
        title,
        *,
        inst,
        picks=None,
        ecg_evoked=None,
        eog_evoked=None,
        ecg_scores=None,
        eog_scores=None,
        n_jobs=None,
        tags=("ica",),
        replace=False,
    ):
        """Add (a fitted) `~mne.preprocessing.ICA` to the report.

        Parameters
        ----------
        ica : path-like | instance of mne.preprocessing.ICA
            The fitted ICA to add.
        title : str
            The title to add.
        inst : path-like | mne.io.Raw | mne.Epochs | None
            The data to use for visualization of the effects of ICA cleaning.
            To only plot the ICA component topographies, explicitly pass
            ``None``.
        %(picks_ica)s This only affects the behavior of the component
            topography and properties plots.
        ecg_evoked, eog_evoked : path-line | mne.Evoked | None
            Evoked signal based on ECG and EOG epochs, respectively. If passed,
            will be used to visualize the effects of artifact rejection.
        ecg_scores, eog_scores : array of float | list of array of float | None
            The scores produced by :meth:`mne.preprocessing.ICA.find_bads_ecg`
            and :meth:`mne.preprocessing.ICA.find_bads_eog`, respectively.
            If passed, will be used to visualize the scoring for each ICA
            component.
        %(n_jobs)s
        %(tags_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        self._add_ica(
            ica=ica,
            inst=inst,
            picks=picks,
            ecg_evoked=ecg_evoked,
            eog_evoked=eog_evoked,
            ecg_scores=ecg_scores,
            eog_scores=eog_scores,
            title=title,
            image_format=self.image_format,
            tags=tags,
            section=title,
            n_jobs=n_jobs,
            replace=replace,
        )

    def remove(self, *, title=None, tags=None, remove_all=False):
        """Remove elements from the report.

        The element to remove is searched for by its title. Optionally, tags
        may be specified as well to narrow down the search to elements that
        have the supplied tags.

        Parameters
        ----------
        title : str
            The title of the element(s) to remove.

            .. versionadded:: 0.24.0
        tags : array-like of str | str | None
             If supplied, restrict the operation to elements with the supplied
             tags.

            .. versionadded:: 0.24.0
        remove_all : bool
            Controls the behavior if multiple elements match the search
            criteria. If ``False`` (default) only the element last added to the
            report will be removed. If ``True``, all matches will be removed.

            .. versionadded:: 0.24.0

        Returns
        -------
        removed_index : int | tuple of int | None
            The indices of the elements that were removed, or ``None`` if no
            element matched the search criteria. A tuple will always be
            returned if ``remove_all`` was set to ``True`` and at least one
            element was removed.

            .. versionchanged:: 0.24.0
               Returns tuple if ``remove_all`` is ``True``.
        """
        remove_idx = []
        for idx, element in enumerate(self._content):
            if element.name == title:
                if tags is not None and not all(t in element.tags for t in tags):
                    continue
                remove_idx.append(idx)

        if not remove_idx:
            remove_idx = None
        elif not remove_all:  # only remove last occurrence
            remove_idx = remove_idx[-1]
            del self._content[remove_idx]
        else:  # remove all occurrences
            remove_idx = tuple(remove_idx)
            self._content = [
                e for idx, e in enumerate(self._content) if idx not in remove_idx
            ]

        return remove_idx

    @fill_doc
    def _add_or_replace(self, *, title, section, tags, html_partial, replace=False):
        """Append HTML content report, or replace it if it already exists.

        Parameters
        ----------
        title : str
            The title entry.
        %(section_report)s
        tags : tuple of str
            The tags associated with the added element.
        html_partial : callable
            Callable that renders a HTML string, called as::

                html_partial(id_=...)
        replace : bool
            Whether to replace existing content if the title and section match.
            If an object is replaced, its dom_id is preserved.
        """
        assert callable(html_partial), type(html_partial)

        # Temporarily set HTML and dom_id to dummy values
        new_content = _ContentElement(
            name=title, section=section, dom_id="", tags=tags, html=""
        )

        append = True
        if replace:
            matches = [
                ii
                for ii, element in enumerate(self._content)
                if (element.name, element.section) == (title, section)
            ]
            if matches:
                dom_id = self._content[matches[-1]].dom_id
                self._content[matches[-1]] = new_content
                append = False
        if append:
            dom_id = self._get_dom_id(section=section, title=title)
            self._content.append(new_content)
        new_content.dom_id = dom_id
        new_content.html = html_partial(id_=dom_id)
        assert isinstance(new_content.html, str), type(new_content.html)

    def _add_code(self, *, code, title, language, section, tags, replace):
        if isinstance(code, Path):
            code = Path(code).read_text()
        html_partial = partial(
            _html_code_element,
            tags=tags,
            title=title,
            code=code,
            language=language,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    @fill_doc
    def add_code(
        self,
        code,
        title,
        *,
        language="python",
        tags=("code",),
        section=None,
        replace=False,
    ):
        """Add a code snippet (e.g., an analysis script) to the report.

        Parameters
        ----------
        code : str | pathlib.Path
            The code to add to the report as a string, or the path to a file
            as a `pathlib.Path` object.

            .. note:: Paths must be passed as `pathlib.Path` object, since
                      strings will be treated as literal code.
        title : str
            The title corresponding to the code.
        language : str
            The programming language of ``code``. This will be used for syntax
            highlighting. Can be ``'auto'`` to try to auto-detect the language.
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        language = language.lower()
        self._add_code(
            code=code,
            title=title,
            language=language,
            section=section,
            tags=tags,
            replace=replace,
        )

    @fill_doc
    def add_sys_info(self, title, *, tags=("mne-sysinfo",), replace=False):
        """Add a MNE-Python system information to the report.

        This is a convenience method that captures the output of
        `mne.sys_info` and adds it to the report.

        Parameters
        ----------
        title : str
            The title to assign.
        %(tags_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        with StringIO() as f:
            sys_info(f)
            info = f.getvalue()
        self.add_code(
            code=info, title=title, language="shell", tags=tags, replace=replace
        )

    def _add_image(
        self,
        *,
        img,
        title,
        caption,
        image_format,
        tags,
        section,
        replace,
    ):
        html_partial = partial(
            _html_image_element,
            img=img,
            div_klass="custom-image",
            img_klass="custom-image",
            title=title,
            caption=caption,
            show=True,
            image_format=image_format,
            tags=tags,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    def _fig_to_img(self, *, fig, image_format, own_figure=True, **mpl_kwargs):
        return _fig_to_img(
            fig,
            image_format=image_format,
            own_figure=own_figure,
            max_width=self.img_max_width,
            max_res=self.img_max_res,
            **mpl_kwargs,
        )

    def _add_figure(
        self,
        *,
        fig,
        title,
        caption,
        image_format,
        tags,
        section,
        replace,
        own_figure,
    ):
        img = self._fig_to_img(
            fig=fig,
            image_format=image_format,
            own_figure=own_figure,
        )
        self._add_image(
            img=img,
            title=title,
            caption=caption,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
        )

    @fill_doc
    def add_figure(
        self,
        fig,
        title,
        *,
        caption=None,
        image_format=None,
        tags=("custom-figure",),
        section=None,
        replace=False,
    ):
        """Add figures to the report.

        Parameters
        ----------
        fig : matplotlib.figure.Figure | Figure3D | array | array-like of matplotlib.figure.Figure | array-like of Figure3D | array-like of array
            One or more figures to add to the report. All figures must be an
            instance of :class:`matplotlib.figure.Figure`,
            :class:`mne.viz.Figure3D`, or :class:`numpy.ndarray`. If
            multiple figures are passed, they will be added as "slides"
            that can be navigated using buttons and a slider element.
        title : str
            The title corresponding to the figure(s).
        caption : str | array-like of str | None
            The caption(s) to add to the figure(s).
        %(image_format_report)s
        %(tags_report)s
        %(section_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """  # noqa E501
        tags = _check_tags(tags)
        if image_format is None:
            image_format = self.image_format

        if hasattr(fig, "__len__") and not isinstance(fig, np.ndarray):
            figs = tuple(fig)
        else:
            figs = (fig,)

        for fig in figs:
            if _path_like(fig):
                raise TypeError(
                    f"It seems you passed a path to `add_figure`. However, "
                    f"only Matplotlib figures, PyVista scenes, and NumPy "
                    f"arrays are accepted. You may want to try `add_image` "
                    f"instead. The provided path was: {fig}"
                )
        del fig

        if isinstance(caption, str):
            captions = (caption,)
        elif caption is None and len(figs) == 1:
            captions = [None]
        elif caption is None and len(figs) > 1:
            captions = [f"Figure {i + 1}" for i in range(len(figs))]
        else:
            captions = tuple(caption)

        del caption

        assert figs
        if len(figs) == 1:
            self._add_figure(
                title=title,
                fig=figs[0],
                caption=captions[0],
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
                own_figure=False,
            )
        else:
            self._add_slider(
                figs=figs,
                imgs=None,
                title=title,
                captions=captions,
                start_idx=0,
                image_format=image_format,
                section=section,
                tags=tags,
                own_figure=False,
                replace=replace,
            )

    @fill_doc
    def add_image(
        self,
        image,
        title,
        *,
        caption=None,
        tags=("custom-image",),
        section=None,
        replace=False,
    ):
        """Add an image (e.g., PNG or JPEG pictures) to the report.

        Parameters
        ----------
        image : path-like
            The image to add.
        title : str
            Title corresponding to the images.
        caption : str | None
            If not ``None``, the caption to add to the image.
        %(tags_report)s
        %(section_report)s
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        image = Path(_check_fname(image, overwrite="read", must_exist=True))
        img_format = Path(image).suffix.lower()[1:]  # omit leading period
        _check_option(
            "Image format",
            value=img_format,
            allowed_values=list(_ALLOWED_IMAGE_FORMATS) + ["gif"],
        )
        img_base64 = base64.b64encode(image.read_bytes()).decode("ascii")
        self._add_image(
            img=img_base64,
            title=title,
            caption=caption,
            image_format=img_format,
            tags=tags,
            section=section,
            replace=replace,
        )

    @fill_doc
    def add_html(
        self, html, title, *, tags=("custom-html",), section=None, replace=False
    ):
        """Add HTML content to the report.

        Parameters
        ----------
        html : str
            The HTML content to add.
        title : str
            The title corresponding to ``html``.
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.3
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        html_partial = partial(
            _html_element,
            html=html,
            title=title,
            tags=tags,
            div_klass="custom-html",
            show=True,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    @fill_doc
    def add_bem(
        self,
        subject,
        title,
        *,
        subjects_dir=None,
        decim=2,
        width=512,
        n_jobs=None,
        tags=("bem",),
        section=None,
        replace=False,
    ):
        """Render a visualization of the boundary element model (BEM) surfaces.

        Parameters
        ----------
        subject : str
            The FreeSurfer subject name.
        title : str
            The title corresponding to the BEM image.
        %(subjects_dir)s
        decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        width : int
            The width of the MRI images (in pixels). Larger values will have
            clearer surface lines, but will create larger HTML files.
            Typically a factor of 2 more than the number of MRI voxels along
            each dimension (typically 512, default) is reasonable.
        %(n_jobs)s
        %(tags_report)s
        %(section_report)s

            .. versionadded:: 1.9
        %(replace_report)s

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        tags = _check_tags(tags)
        width = _ensure_int(width, "width")
        self._add_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            decim=decim,
            n_jobs=n_jobs,
            width=width,
            image_format=self.image_format,
            title=title,
            tags=tags,
            section=section,
            replace=replace,
        )

    def _render_slider(
        self,
        *,
        figs,
        imgs,
        title,
        captions,
        start_idx,
        image_format,
        tags,
        klass,
        own_figure,
    ):
        # This method only exists to make add_bem()'s life easier…
        if figs is not None and imgs is not None:
            raise ValueError("Must only provide either figs or imgs")

        if figs is not None and len(figs) != len(captions):
            raise ValueError(
                f"Number of captions ({len(captions)}) must be equal to the "
                f"number of figures ({len(figs)})"
            )
        elif imgs is not None and len(imgs) != len(captions):
            raise ValueError(
                f"Number of captions ({len(captions)}) must be equal to the "
                f"number of images ({len(imgs)})"
            )
        elif figs:  # figs can be None if imgs is provided
            imgs = [
                self._fig_to_img(
                    fig=fig,
                    image_format=image_format,
                    own_figure=own_figure,
                )
                for fig in figs
            ]

        html_partial = partial(
            _html_slider_element,
            title=title,
            captions=captions,
            tags=tags,
            images=imgs,
            image_format=image_format,
            start_idx=start_idx,
            klass=klass,
            show=True,
        )
        return html_partial

    def _add_slider(
        self,
        *,
        figs,
        imgs,
        title,
        captions,
        start_idx,
        image_format,
        tags,
        section,
        replace,
        klass="",
        own_figure=True,
    ):
        html_partial = self._render_slider(
            figs=figs,
            imgs=imgs,
            title=title,
            captions=captions,
            start_idx=start_idx,
            image_format=image_format,
            tags=tags,
            klass=klass,
            own_figure=own_figure,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    ###########################################################################
    # global rendering functions
    @verbose
    def _init_render(self, verbose=None):
        """Initialize the renderer."""
        inc_fnames = [
            "jquery-3.6.0.min.js",
            "bootstrap.bundle.min.js",
            "bootstrap.min.css",
            "bootstrap-table/bootstrap-table.min.js",
            "bootstrap-table/bootstrap-table.min.css",
            "bootstrap-table/bootstrap-table-copy-rows.min.js",
            "bootstrap-table/bootstrap-table-export.min.js",
            "bootstrap-table/tableExport.min.js",
            "bootstrap-icons/bootstrap-icons.mne.min.css",
            "highlightjs/highlight.min.js",
            "highlightjs/atom-one-dark-reasonable.min.css",
        ]

        include = list()
        for inc_fname in inc_fnames:
            logger.info(f"Embedding : {inc_fname}")
            fname = html_include_dir / inc_fname
            file_content = fname.read_text(encoding="utf-8")

            if inc_fname.endswith(".js"):
                include.append(
                    f'<script type="text/javascript">\n'
                    f"{file_content}\n"
                    f"</script>"
                )
            elif inc_fname.endswith(".css"):
                include.append(f'<style type="text/css">\n{file_content}\n</style>')
        self.include = "".join(include)

    def _iterate_files(
        self,
        *,
        fnames,
        cov,
        sfreq,
        raw_butterfly,
        n_time_points_evokeds,
        n_time_points_stcs,
        on_error,
        stc_plot_kwargs,
        topomap_kwargs,
    ):
        """Parallel process in batch mode."""
        assert self.data_path is not None

        for fname in fnames:
            logger.info(f"Rendering: {fname}")

            title = Path(fname).name
            try:
                if _endswith(fname, ["raw", "sss", "meg", "nirs"]):
                    self.add_raw(
                        raw=fname,
                        title=title,
                        psd=self.raw_psd,
                        projs=self.projs,
                        butterfly=raw_butterfly,
                    )
                elif _endswith(fname, "fwd"):
                    self.add_forward(
                        forward=fname,
                        title=title,
                        subject=self.subject,
                        subjects_dir=self.subjects_dir,
                    )
                elif _endswith(fname, "inv"):
                    # XXX if we pass trans, we can plot the source space, too…
                    self.add_inverse_operator(inverse_operator=fname, title=title)
                elif _endswith(fname, "ave"):
                    evokeds = read_evokeds(fname)
                    titles = [f"{Path(fname).name}: {e.comment}" for e in evokeds]
                    self.add_evokeds(
                        evokeds=fname,
                        titles=titles,
                        noise_cov=cov,
                        n_time_points=n_time_points_evokeds,
                        topomap_kwargs=topomap_kwargs,
                    )
                elif _endswith(fname, "eve"):
                    if self.info_fname is not None:
                        sfreq = read_info(self.info_fname)["sfreq"]
                    else:
                        sfreq = None
                    self.add_events(events=fname, title=title, sfreq=sfreq)
                elif _endswith(fname, "epo"):
                    self.add_epochs(epochs=fname, title=title)
                elif _endswith(fname, "cov") and self.info_fname is not None:
                    self.add_covariance(cov=fname, info=self.info_fname, title=title)
                elif _endswith(fname, "proj") and self.info_fname is not None:
                    self.add_projs(
                        info=self.info_fname,
                        projs=fname,
                        title=title,
                        topomap_kwargs=topomap_kwargs,
                    )
                # XXX TODO We could render ICA components here someday
                # elif _endswith(fname, 'ica') and ica:
                #     pass
                elif (
                    _endswith(fname, "trans")
                    and self.info_fname is not None
                    and self.subjects_dir is not None
                    and self.subject is not None
                ):
                    self.add_trans(
                        trans=fname,
                        info=self.info_fname,
                        subject=self.subject,
                        subjects_dir=self.subjects_dir,
                        title=title,
                    )
                elif (
                    fname.endswith("-lh.stc")
                    or fname.endswith("-rh.stc")
                    and self.info_fname is not None
                    and self.subjects_dir is not None
                    and self.subject is not None
                ):
                    self.add_stc(
                        stc=fname,
                        title=title,
                        subject=self.subject,
                        subjects_dir=self.subjects_dir,
                        n_time_points=n_time_points_stcs,
                        stc_plot_kwargs=stc_plot_kwargs,
                    )
            except Exception as e:
                if on_error == "warn":
                    warn(f'Failed to process file {fname}:\n"{e}"')
                elif on_error == "raise":
                    raise

    @verbose
    def parse_folder(
        self,
        data_path,
        pattern=None,
        n_jobs=None,
        mri_decim=2,
        sort_content=True,
        *,
        on_error="warn",
        image_format=None,
        render_bem=True,
        n_time_points_evokeds=None,
        n_time_points_stcs=None,
        raw_butterfly=True,
        stc_plot_kwargs=None,
        topomap_kwargs=None,
        verbose=None,
    ):
        r"""Render all the files in the folder.

        Parameters
        ----------
        data_path : path-like
            Path to the folder containing data whose HTML report will be created.
        pattern : None | str | list of str
            Filename global pattern(s) to include in the report.
            For example, ``['\*raw.fif', '\*ave.fif']`` will include
            :class:`~mne.io.Raw` as well as :class:`~mne.Evoked` files. If ``None``,
            include all supported file formats.

            .. versionchanged:: 0.23
               Include supported non-FIFF files by default.
        %(n_jobs)s
        mri_decim : int
            Use this decimation factor for generating MRI/BEM images
            (since it can be time consuming).
        sort_content : bool
            If ``True``, sort the content based on tags in the order:
            raw -> events -> epochs -> evoked -> covariance -> coregistration
            -> bem -> forward-solution -> inverse-operator -> source-estimate.

            .. versionadded:: 0.24.0
        on_error : ``'ignore'`` | ``'warn'`` | ``'raise'``
            What to do if a file cannot be rendered. Can be ``'ignore'``, ``'warn'``
            (default), or ``'raise'``.
        %(image_format_report)s

            .. versionadded:: 0.15
        render_bem : bool
            If True (default), try to render the BEM.

            .. versionadded:: 0.16
        n_time_points_evokeds, n_time_points_stcs : int | None
            The number of equidistant time points to render for :class:`~mne.Evoked`
            and :class:`~mne.SourceEstimate` data, respectively. If ``None``,
            will render each :class:`~mne.Evoked` at 21 and each
            :class:`~mne.SourceEstimate` at 51 time points, unless the respective data
            contains fewer time points, in which case all will be rendered.

            .. versionadded:: 0.24.0
        raw_butterfly : bool
            Whether to render butterfly plots for (decimated) :class:`~mne.io.Raw` data.

            .. versionadded:: 0.24.0
        %(stc_plot_kwargs_report)s

            .. versionadded:: 0.24.0
        %(topomap_kwargs)s

            .. versionadded:: 0.24.0
        %(verbose)s
        """
        self.data_path = _check_fname(
            data_path,
            overwrite="read",
            must_exist=True,
            name="data_path",
            need_dir=True,
        )
        image_format = _check_image_format(self, image_format)
        _check_option("on_error", on_error, ["ignore", "warn", "raise"])

        if self.title is None:
            self.title = f"MNE Report for {self.data_path.name[-20:]}"

        if pattern is None:
            pattern = [f"*{ext}" for ext in SUPPORTED_READ_RAW_EXTENSIONS]
        else:
            if not isinstance(pattern, list | tuple):
                pattern = [pattern]
            for elt in pattern:
                _validate_type(elt, str, "pattern")

        # iterate through the possible patterns
        fnames = list()
        for p in pattern:
            for match in self.data_path.rglob(p):
                if match.name.endswith(VALID_EXTENSIONS):
                    fnames.append(match)

        if not fnames and not render_bem:
            raise RuntimeError(f"No matching files found in {self.data_path}.")

        fnames_to_remove = []
        for fname in fnames:
            # For split files, only keep the first one.
            if _endswith(fname, ("raw", "sss", "meg")):
                kwargs = dict(fname=fname, preload=False)
                if fname.name.endswith((".fif", ".fif.gz")):
                    kwargs["allow_maxshield"] = "yes"
                inst = read_raw(**kwargs)

                if len(inst.filenames) > 1:
                    fnames_to_remove.extend(inst.filenames[1:])
            # For STCs, only keep one hemisphere
            elif fname.name.endswith("-lh.stc") or fname.name.endswith("-rh.stc"):
                first_hemi_fname = fname.name
                if first_hemi_fname.endswidth("-lh.stc"):
                    second_hemi_fname = first_hemi_fname.replace("-lh.stc", "-rh.stc")
                else:
                    second_hemi_fname = first_hemi_fname.replace("-rh.stc", "-lh.stc")

                if (
                    fname.parent / second_hemi_fname in fnames
                    and fname.parent / first_hemi_fname not in fnames_to_remove
                ):
                    fnames_to_remove.extend(first_hemi_fname)
            else:
                continue

        fnames_to_remove = list(set(fnames_to_remove))  # Drop duplicates
        for fname in fnames_to_remove:
            if fname in fnames:
                del fnames[fnames.index(fname)]
        del fnames_to_remove

        if self.info_fname is not None:
            info = read_info(self.info_fname, verbose=False)
            sfreq = info["sfreq"]
        else:
            # only warn if relevant
            if any(_endswith(fname, "cov") for fname in fnames):
                warn("`info_fname` not provided. Cannot render -cov.fif(.gz) files.")
            if any(_endswith(fname, "trans") for fname in fnames):
                warn("`info_fname` not provided. Cannot render -trans.fif(.gz) files.")
            if any(_endswith(fname, "proj") for fname in fnames):
                warn("`info_fname` not provided. Cannot render -proj.fif(.gz) files.")
            info, sfreq = None, None

        cov = None
        if self.cov_fname is not None:
            cov = read_cov(self.cov_fname)

        # render plots in parallel; check that n_jobs <= # of files
        logger.info(
            f"Iterating over {len(fnames)} potential files (this may take some "
        )
        parallel, p_fun, n_jobs = parallel_func(
            self._iterate_files, n_jobs, max_jobs=len(fnames)
        )
        parallel(
            p_fun(
                fnames=fname,
                cov=cov,
                sfreq=sfreq,
                raw_butterfly=raw_butterfly,
                n_time_points_evokeds=n_time_points_evokeds,
                n_time_points_stcs=n_time_points_stcs,
                on_error=on_error,
                stc_plot_kwargs=stc_plot_kwargs,
                topomap_kwargs=topomap_kwargs,
            )
            for fname in np.array_split(fnames, n_jobs)
        )

        # Render BEM
        if render_bem:
            if self.subjects_dir is not None and self.subject is not None:
                logger.info("Rendering BEM")
                self.add_bem(
                    subject=self.subject,
                    subjects_dir=self.subjects_dir,
                    title="BEM surfaces",
                    decim=mri_decim,
                    n_jobs=n_jobs,
                )
            else:
                warn(
                    "`subjects_dir` and `subject` not provided. Cannot "
                    "render MRI and -trans.fif(.gz) files."
                )

        if sort_content:
            self._sort(order=CONTENT_ORDER)

    def __getstate__(self):
        """Get the state of the report as a dictionary."""
        state = dict()
        for param_name in self._get_state_params():
            param_val = getattr(self, param_name)

            # Workaround as h5io doesn't support dataclasses
            if param_name == "_content":
                assert all(dataclasses.is_dataclass(val) for val in param_val)
                param_val = [dataclasses.asdict(val) for val in param_val]

            state[param_name] = param_val
        return state

    def __setstate__(self, state):
        """Set the state of the report."""
        for param_name in self._get_state_params():
            param_val = state[param_name]

            # Workaround as h5io doesn't support dataclasses
            if param_name == "_content":
                param_val = [_ContentElement(**val) for val in param_val]
            setattr(self, param_name, param_val)
        return state

    @verbose
    def save(
        self,
        fname=None,
        open_browser=True,
        overwrite=False,
        sort_content=False,
        *,
        verbose=None,
    ):
        """Save the report and optionally open it in browser.

        Parameters
        ----------
        fname : path-like | None
            Output filename. If the name ends with ``.h5`` or ``.hdf5``, the
            report is saved in HDF5 format, so it can later be loaded again
            with :func:`open_report`. For any other suffix, the report will be
            saved in HTML format. If ``None`` and :meth:`Report.parse_folder`
            was **not** called, the report is saved as ``report.html`` in the
            current working directory. If ``None`` and
            :meth:`Report.parse_folder` **was** used, the report is saved as
            ``report.html`` inside the ``data_path`` supplied to
            :meth:`Report.parse_folder`.
        open_browser : bool
            Whether to open the rendered HTML report in the default web browser
            after saving. This is ignored when writing an HDF5 file.
        %(overwrite)s
        sort_content : bool
            If ``True``, sort the content based on tags before saving in the
            order:
            raw -> events -> epochs -> evoked -> covariance -> coregistration
            -> bem -> forward-solution -> inverse-operator -> source-estimate.

            .. versionadded:: 0.24.0
        %(verbose)s

        Returns
        -------
        fname : str
            The file name to which the report was saved.
        """
        if fname is None:
            if self.data_path is None:
                self.data_path = os.getcwd()
                warn(f"`data_path` not provided. Using {self.data_path} instead")
            fname = op.join(self.data_path, "report.html")

        fname = str(_check_fname(fname, overwrite=overwrite, name=fname))
        fname = op.realpath(fname)  # resolve symlinks

        if sort_content:
            self._sort(order=CONTENT_ORDER)

        if not overwrite and op.isfile(fname):
            msg = f"Report already exists at location {fname}. Overwrite it (y/[n])? "
            answer = _safe_input(msg, alt="pass overwrite=True")
            if answer.lower() == "y":
                overwrite = True

        _, ext = op.splitext(fname)
        is_hdf5 = ext.lower() in [".h5", ".hdf5"]

        if overwrite or not op.isfile(fname):
            logger.info(f"Saving report to : {fname}")

            if is_hdf5:
                _, write_hdf5 = _import_h5io_funcs()
                import h5py

                with h5py.File(fname, "a") as f:  # Read/write if exists, else create
                    write_hdf5(f, self.__getstate__(), title="mnepython")
            else:
                # Add header, TOC, and footer.
                header_html = _html_header_element(
                    title=self.title,
                    include=self.include,
                    lang=self.lang,
                    tags=self.tags,
                    js=JAVASCRIPT,
                    css=CSS,
                    mne_logo_img=mne_logo,
                )

                # toc_html = _html_toc_element(content_elements=self._content)
                _, dom_ids, titles, tags = self._content_as_html()
                toc_html = _html_toc_element(titles=titles, dom_ids=dom_ids, tags=tags)

                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("ignore")
                    footer_html = _html_footer_element(
                        mne_version=MNE_VERSION, date=time.strftime("%B %d, %Y")
                    )

                html = [header_html, toc_html, *self.html, footer_html]
                Path(fname).write_text(data="".join(html), encoding="utf-8")

        building_doc = os.getenv("_MNE_BUILDING_DOC", "").lower() == "true"
        if open_browser and not is_hdf5 and not building_doc:
            webbrowser.open_new_tab("file://" + fname)

        if self.fname is None:
            self.fname = fname
        return fname

    def __enter__(self):
        """Do nothing when entering the context block."""
        return self

    def __exit__(self, exception_type, value, traceback):
        """Save the report when leaving the context block."""
        if self.fname is not None:
            self.save(self.fname, open_browser=False, overwrite=True)

    def _sort(self, *, order):
        """Reorder content to reflect "natural" ordering."""
        content_sorted_idx = []

        # First arrange content with known tags in the predefined order
        for tag in order:
            for idx, content in enumerate(self._content):
                if tag in content.tags:
                    content_sorted_idx.append(idx)

        # Now simply append the rest (custom tags)
        self.reorder(
            np.r_[
                content_sorted_idx,
                np.setdiff1d(np.arange(len(self._content)), content_sorted_idx),
            ]
        )

    def _render_one_bem_axis(
        self,
        *,
        mri_fname,
        surfaces,
        image_format,
        orientation,
        decim=2,
        n_jobs=None,
        width=512,
        tags,
    ):
        """Render one axis of bem contours (only PNG)."""
        nib = _import_nibabel("render BEM contours")
        nim = nib.load(mri_fname)
        data = _reorient_image(nim)[0]
        axis = _mri_orientation(orientation)[0]
        n_slices = data.shape[axis]

        sl = np.arange(0, n_slices, decim)
        logger.debug(f"Rendering BEM {orientation} with {len(sl)} slices")
        figs = _get_bem_contour_figs_as_arrays(
            sl=sl,
            n_jobs=n_jobs,
            mri_fname=mri_fname,
            surfaces=surfaces,
            orientation=orientation,
            src=None,
            show=False,
            show_orientation="always",
            width=width,
        )

        # Render the slider
        captions = [f"Slice index: {i * decim}" for i in range(len(figs))]
        start_idx = int(round(len(figs) / 2))
        html_partial = self._render_slider(
            figs=figs,
            imgs=None,
            captions=captions,
            title=orientation,
            image_format=image_format,
            start_idx=start_idx,
            tags=tags,
            klass="bem col-md",
            own_figure=True,
        )
        # If we replace a BEM section this could end up like `axial-1`, but
        # we never refer to it in the TOC so it's probably okay
        html = html_partial(id_=self._get_dom_id(section="bem", title=orientation))
        return html

    def _add_html_repr(self, *, inst, title, tags, section, replace, div_klass):
        html = inst._repr_html_()
        self._add_html_element(
            html=html,
            title=title,
            tags=tags,
            section=section,
            replace=replace,
            div_klass=div_klass,
        )

    def _add_raw_butterfly_segments(
        self,
        *,
        raw: BaseRaw,
        n_segments,
        scalings,
        image_format,
        tags,
        section,
        replace,
    ):
        # Pick n_segments + 2 equally-spaced 1-second time slices, but omit
        # the first and last slice, so we end up with n_segments slices
        n = n_segments + 2
        times = np.linspace(raw.times[0], raw.times[-1], n)[1:-1]
        t_starts = np.array([max(t - 0.5, 0) for t in times])
        t_stops = np.array([min(t + 0.5, raw.times[-1]) for t in times])
        durations = t_stops - t_starts

        # Remove annotations before plotting for better performance.
        # Ensure we later restore raw.annotations even in case of an exception
        orig_annotations = raw.annotations.copy()

        try:
            raw.set_annotations(None)

            # Create the figure once and reuse it for performance reasons
            with use_browser_backend("matplotlib"):
                fig = raw.plot(
                    butterfly=True,
                    show_scrollbars=False,
                    start=t_starts[0],
                    duration=durations[0],
                    scalings=scalings,
                    show=False,
                )
            images = [self._fig_to_img(fig=fig, image_format=image_format)]

            for start, duration in zip(t_starts[1:], durations[1:]):
                fig.mne.t_start = start
                fig.mne.duration = duration
                fig._update_hscroll()
                fig._redraw(annotations=False)
                images.append(self._fig_to_img(fig=fig, image_format=image_format))
        except Exception:
            raise
        finally:
            raw.set_annotations(orig_annotations)

        del orig_annotations

        captions = [f"Segment {i + 1} of {len(images)}" for i in range(len(images))]

        self._add_slider(
            figs=None,
            imgs=images,
            title="Time series",
            captions=captions,
            start_idx=0,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
        )

    def _add_raw(
        self,
        *,
        raw,
        add_psd,
        add_projs,
        butterfly,
        butterfly_scalings,
        image_format,
        tags,
        topomap_kwargs,
        section,
        replace,
    ):
        """Render raw."""
        if isinstance(raw, BaseRaw):
            fname = raw.filenames[0]
        else:
            fname = str(raw)  # could e.g. be a Path!
            kwargs = dict(fname=fname, preload=False)
            if fname.endswith((".fif", ".fif.gz")):
                kwargs["allow_maxshield"] = "yes"
            raw = read_raw(**kwargs)

        # Summary table
        self._add_html_repr(
            inst=raw,
            title="Info",
            tags=tags,
            section=section,
            replace=replace,
            div_klass="raw",
        )

        # Butterfly plot
        if butterfly:
            n_butterfly_segments = 10 if butterfly is True else butterfly
            self._add_raw_butterfly_segments(
                raw=raw,
                scalings=butterfly_scalings,
                n_segments=n_butterfly_segments,
                image_format=image_format,
                tags=tags,
                replace=replace,
                section=section,
            )

        # PSD
        if isinstance(add_psd, dict):
            if raw.info["lowpass"] is not None:
                fmax = raw.info["lowpass"] + 15
                # Must not exceed half the sampling frequency
                if fmax > 0.5 * raw.info["sfreq"]:
                    fmax = np.inf
            else:
                fmax = np.inf

            # shim: convert legacy .plot_psd(...) → .compute_psd(...).plot(...)
            init_kwargs, plot_kwargs = _split_psd_kwargs(kwargs=add_psd)
            init_kwargs.setdefault("fmax", fmax)
            plot_kwargs.setdefault("show", False)
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                fig = raw.compute_psd(**init_kwargs).plot(**plot_kwargs)
            self._add_figure(
                fig=fig,
                title="PSD",
                caption=None,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )

        # SSP projectors
        if add_projs:
            self._add_projs(
                info=raw,
                projs=None,
                title="Projectors",
                image_format=image_format,
                tags=tags,
                topomap_kwargs=topomap_kwargs,
                section=section,
                replace=replace,
            )

    def _add_projs(
        self,
        *,
        info,
        projs,
        title,
        image_format,
        tags,
        section,
        topomap_kwargs,
        replace,
        picks_trace=None,
        joint=False,
    ):
        evoked = None
        if isinstance(info, Info):  # no-op
            pass
        elif hasattr(info, "info"):  # try to get the file name
            if isinstance(info, Evoked):
                evoked = info
            info = info.info
        else:  # read from a file
            info = read_info(info, verbose=False)
        if joint and evoked is None:
            raise ValueError(
                "joint=True requires an evoked instance to be passed as the info"
            )

        if projs is None:
            projs = info["projs"]
        elif not isinstance(projs, list):
            projs = read_proj(projs)

        if not projs:
            raise ValueError("No SSP projectors found")

        if not _check_ch_locs(info=info):
            raise ValueError(
                "The provided data does not contain digitization "
                "information (channel locations). However, this is "
                "required for rendering the projectors."
            )

        topomap_kwargs = self._validate_topomap_kwargs(topomap_kwargs)
        if evoked is None:
            fig = plot_projs_topomap(
                projs=projs,
                info=info,
                colorbar=True,
                vlim="joint",
                show=False,
                **topomap_kwargs,
            )
            # TODO This seems like a bad idea, better to provide a way to set a
            # desired size in plot_projs_topomap, but that uses prepare_trellis...
            # hard to see how (6, 4) could work in all number-of-projs by
            # number-of-channel-types conditions...
            fig.set_size_inches((6, 4))
        else:
            fig = plot_projs_joint(
                projs,
                evoked=evoked,
                picks_trace=picks_trace,
                topomap_kwargs=topomap_kwargs,
                show=False,
            )

        self._add_figure(
            fig=fig,
            title=title,
            caption=None,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_forward(
        self,
        *,
        forward,
        subject,
        subjects_dir,
        title,
        image_format,
        section,
        tags,
        replace,
    ):
        """Render forward solution."""
        if not isinstance(forward, Forward):
            forward = read_forward_solution(forward)

        subject = self.subject if subject is None else subject
        subjects_dir = self.subjects_dir if subjects_dir is None else subjects_dir

        # XXX Todo
        # Render sensitivity maps
        if subject is not None:
            sensitivity_maps_html = ""
        else:
            sensitivity_maps_html = ""

        html_partial = partial(
            _html_forward_sol_element,
            repr_=forward._repr_html_(),
            sensitivity_maps=sensitivity_maps_html,
            title=title,
            tags=tags,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    def _add_inverse_operator(
        self,
        *,
        inverse_operator,
        subject,
        subjects_dir,
        trans,
        title,
        image_format,
        section,
        tags,
        replace,
    ):
        """Render inverse operator."""
        if not isinstance(inverse_operator, InverseOperator):
            inverse_operator = read_inverse_operator(inverse_operator)

        if trans is not None and not isinstance(trans, Transform):
            trans = read_trans(trans)

        subject = self.subject if subject is None else subject
        subjects_dir = self.subjects_dir if subjects_dir is None else subjects_dir

        # XXX Todo Render source space?
        # if subject is not None and trans is not None:
        #     src = inverse_operator['src']

        #     fig = plot_alignment(
        #         subject=subject,
        #         subjects_dir=subjects_dir,
        #         trans=trans,
        #         surfaces='white',
        #         src=src
        #     )
        #     set_3d_view(fig, focalpoint=(0., 0., 0.06))
        #     img = self._fig_to_img(fig=fig, image_format=image_format)

        #     src_img_html = partial(
        #         _html_image_element,
        #         img=img,
        #         div_klass='inverse-operator source-space',
        #         img_klass='inverse-operator source-space',
        #         title='Source space', caption=None, show=True,
        #         image_format=image_format,
        #         tags=tags
        #     )
        # else:
        src_img_html = ""

        html_partial = partial(
            _html_inverse_operator_element,
            repr_=inverse_operator._repr_html_(),
            source_space=src_img_html,
            title=title,
            tags=tags,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    def _add_evoked_joint(
        self, *, evoked, ch_types, image_format, section, tags, topomap_kwargs, replace
    ):
        for ch_type in ch_types:
            if not _check_ch_locs(info=evoked.info, ch_type=ch_type):
                ch_type_name = _handle_default("titles")[ch_type]
                warn(
                    f"No {ch_type_name} channel locations found, cannot "
                    f"create joint plot"
                )
                continue

            with use_log_level(_verbose_safe_false(level="error")):
                fig = (
                    evoked.copy()
                    .pick(ch_type, verbose=False)
                    .plot_joint(
                        ts_args=dict(gfp=True),
                        title=None,
                        show=False,
                        topomap_args=topomap_kwargs,
                    )
                )

            title = f'Time course ({_handle_default("titles")[ch_type]})'
            self._add_figure(
                fig=fig,
                title=title,
                caption=None,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )

    def _plot_one_evoked_topomap_timepoint(
        self, *, evoked, time, ch_types, vmin, vmax, topomap_kwargs
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            1,
            len(ch_types) * 2,
            gridspec_kw={"width_ratios": [8, 0.5] * len(ch_types)},
            figsize=(2.5 * len(ch_types), 2),
            layout="constrained",
        )
        ch_type_ax_map = dict(
            zip(
                ch_types,
                [(ax[i], ax[i + 1]) for i in range(0, 2 * len(ch_types) - 1, 2)],
            )
        )

        for ch_type in ch_types:
            evoked.plot_topomap(
                times=[time],
                ch_type=ch_type,
                vlim=(vmin[ch_type], vmax[ch_type]),
                axes=ch_type_ax_map[ch_type],
                show=False,
                **topomap_kwargs,
            )
            ch_type_ax_map[ch_type][0].set_title(ch_type)

        return self._fig_to_img(
            fig=fig,
            image_format="ndarray",
            pad_inches=0,
        )

    def _add_evoked_topomap_slider(
        self,
        *,
        evoked,
        ch_types,
        n_time_points,
        image_format,
        section,
        tags,
        topomap_kwargs,
        n_jobs,
        replace,
    ):
        if n_time_points is None:
            n_time_points = min(len(evoked.times), 21)
        elif n_time_points > len(evoked.times):
            raise ValueError(
                f"The requested number of time points ({n_time_points}) "
                f"exceeds the time points in the provided Evoked object "
                f"({len(evoked.times)})"
            )

        if n_time_points == 1:  # only a single time point, pick the first one
            times = [evoked.times[0]]
        else:
            times = np.linspace(start=evoked.tmin, stop=evoked.tmax, num=n_time_points)

        t_zero_idx = np.abs(times).argmin()  # index closest to zero

        # global min and max values for each channel type
        scalings = dict(eeg=1e6, grad=1e13, mag=1e15)

        vmax = dict()
        vmin = dict()
        for ch_type in ch_types:
            if not _check_ch_locs(info=evoked.info, ch_type=ch_type):
                ch_type_name = _handle_default("titles")[ch_type]
                warn(
                    f"No {ch_type_name} channel locations found, cannot "
                    f"create topography plots"
                )
                continue

            vmax[ch_type] = (
                np.abs(
                    evoked.copy().pick(ch_type, exclude="bads", verbose=False).data
                ).max()
            ) * scalings[ch_type]
            if ch_type == "grad":
                vmin[ch_type] = 0
            else:
                vmin[ch_type] = -vmax[ch_type]

        if not (vmin and vmax):  # we only had EEG data and no digpoints
            return  # No need to warn here, we did that above
        else:
            topomap_kwargs = self._validate_topomap_kwargs(topomap_kwargs)
            parallel, p_fun, n_jobs = parallel_func(
                func=self._plot_one_evoked_topomap_timepoint,
                n_jobs=n_jobs,
                max_jobs=len(times),
            )
            with use_log_level(_verbose_safe_false(level="error")):
                fig_arrays = parallel(
                    p_fun(
                        evoked=evoked,
                        time=time,
                        ch_types=ch_types,
                        vmin=vmin,
                        vmax=vmax,
                        topomap_kwargs=topomap_kwargs,
                    )
                    for time in times
                )

            captions = [f"Time point: {round(t, 3):0.3f} s" for t in times]
            self._add_slider(
                figs=fig_arrays,
                imgs=None,
                captions=captions,
                title="Topographies",
                image_format=image_format,
                start_idx=t_zero_idx,
                section=section,
                tags=tags,
                replace=replace,
            )

    def _add_evoked_gfp(
        self, *, evoked, ch_types, image_format, section, tags, replace
    ):
        # Make legend labels shorter by removing the multiplicative factors
        pattern = r"\d\.\d* × "
        label = evoked.comment
        if label is None:
            label = ""
        for match in re.findall(pattern=pattern, string=label):
            label = label.replace(match, "")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(ch_types), 1, sharex=True, layout="constrained")
        if len(ch_types) == 1:
            ax = [ax]
        for idx, ch_type in enumerate(ch_types):
            with use_log_level(_verbose_safe_false(level="error")):
                plot_compare_evokeds(
                    evokeds={label: evoked.copy().pick(ch_type, verbose=False)},
                    ci=None,
                    truncate_xaxis=False,
                    truncate_yaxis=False,
                    legend=False,
                    axes=ax[idx],
                    show=False,
                )
            ax[idx].set_title(ch_type)

            # Hide x axis label for all but the last subplot
            if idx < len(ch_types) - 1:
                ax[idx].set_xlabel(None)

        title = "Global field power"
        self._add_figure(
            fig=fig,
            title=title,
            caption=None,
            image_format=image_format,
            section=section,
            tags=tags,
            replace=replace,
            own_figure=True,
        )

    def _add_evoked_whitened(
        self, *, evoked, noise_cov, image_format, section, tags, replace
    ):
        """Render whitened evoked."""
        fig = evoked.plot_white(noise_cov=noise_cov, show=False)
        title = "Whitened"

        self._add_figure(
            fig=fig,
            title=title,
            caption=None,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_evoked(
        self,
        *,
        evoked,
        noise_cov,
        add_projs,
        n_time_points,
        image_format,
        section,
        tags,
        topomap_kwargs,
        n_jobs,
        replace,
    ):
        # Summary table
        self._add_html_repr(
            inst=evoked,
            title="Info",
            tags=tags,
            section=section,
            replace=replace,
            div_klass="evoked",
        )

        # Joint plot
        ch_types = _get_data_ch_types(evoked)
        self._add_evoked_joint(
            evoked=evoked,
            ch_types=ch_types,
            image_format=image_format,
            section=section,
            tags=tags,
            topomap_kwargs=topomap_kwargs,
            replace=replace,
        )

        # Topomaps
        self._add_evoked_topomap_slider(
            evoked=evoked,
            ch_types=ch_types,
            n_time_points=n_time_points,
            image_format=image_format,
            section=section,
            tags=tags,
            topomap_kwargs=topomap_kwargs,
            n_jobs=n_jobs,
            replace=replace,
        )

        # GFP
        self._add_evoked_gfp(
            evoked=evoked,
            ch_types=ch_types,
            image_format=image_format,
            section=section,
            tags=tags,
            replace=replace,
        )

        # Whitened evoked
        if noise_cov is not None:
            self._add_evoked_whitened(
                evoked=evoked,
                noise_cov=noise_cov,
                image_format=image_format,
                section=section,
                tags=tags,
                replace=replace,
            )

        # SSP projectors
        if add_projs:
            self._add_projs(
                info=evoked,
                projs=None,
                title="Projectors",
                image_format=image_format,
                section=section,
                tags=tags,
                topomap_kwargs=topomap_kwargs,
                replace=replace,
            )

        logger.debug("Evoked: done")

    def _add_events(
        self,
        *,
        events,
        event_id,
        color,
        sfreq,
        first_samp,
        title,
        section,
        image_format,
        tags,
        replace,
    ):
        """Render events."""
        if not isinstance(events, np.ndarray):
            events = read_events(filename=events)

        fig = plot_events(
            events=events,
            event_id=event_id,
            sfreq=sfreq,
            first_samp=first_samp,
            color=color,
            show=False,
        )
        self._add_figure(
            fig=fig,
            title=title,
            caption=None,
            image_format=image_format,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_epochs_psd(self, *, epochs, psd, image_format, tags, section, replace):
        epoch_duration = epochs.tmax - epochs.tmin

        if psd is True:  # Entire time range -> all epochs
            epochs_for_psd = epochs  # Avoid creating a copy
        else:  # Only a subset of epochs
            signal_duration = len(epochs) * epoch_duration
            n_epochs_required = int(np.ceil(psd / epoch_duration))
            if n_epochs_required > len(epochs):
                raise ValueError(
                    f"You requested to calculate PSD on a duration of "
                    f"{psd:.3f} s, but all your epochs "
                    f"are only {signal_duration:.1f} s long"
                )
            epochs_idx = np.round(
                np.linspace(start=0, stop=len(epochs) - 1, num=n_epochs_required)
            ).astype(int)
            # Edge case: there might be duplicate indices due to rounding?
            epochs_idx_unique = np.unique(epochs_idx)
            if len(epochs_idx_unique) != len(epochs_idx):
                duration = round(len(epochs_idx_unique) * epoch_duration, 1)
                warn(
                    f"Using {len(epochs_idx_unique)} epochs, only "
                    f"covering {duration:.1f} s of data"
                )
                del duration

            epochs_for_psd = epochs[epochs_idx_unique]

        if epochs.info["lowpass"] is None:
            fmax = np.inf
        else:
            fmax = epochs.info["lowpass"] + 15
            # Must not exceed half the sampling frequency
            if fmax > 0.5 * epochs.info["sfreq"]:
                fmax = np.inf

        fig = epochs_for_psd.compute_psd(fmax=fmax).plot(amplitude=False, show=False)
        duration = round(epoch_duration * len(epochs_for_psd), 1)
        caption = (
            f"PSD calculated from {len(epochs_for_psd)} epochs ({duration:.1f} s)."
        )
        self._add_figure(
            fig=fig,
            image_format=image_format,
            title="PSD",
            caption=caption,
            tags=tags,
            section=section,
            replace=replace,
            own_figure=True,
        )

    def _add_html_element(self, *, html, title, tags, section, replace, div_klass):
        html_partial = partial(
            _html_element,
            div_klass=div_klass,
            tags=tags,
            title=title,
            html=html,
            show=True,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )

    def _add_epochs_metadata(self, *, epochs, section, tags, replace):
        metadata = epochs.metadata.copy()

        # Ensure we have a named index
        if not metadata.index.name:
            metadata.index.name = "Epoch #"

        assert metadata.index.is_unique
        data_id = metadata.index.name  # store for later use
        metadata = metadata.reset_index()  # We want "proper" columns only
        html = _df_bootstrap_table(df=metadata, data_id=data_id)
        self._add_html_element(
            div_klass="epochs",
            tags=tags,
            title="Metadata",
            html=html,
            section=section,
            replace=replace,
        )

    def _add_epochs(
        self,
        *,
        epochs,
        psd,
        add_projs,
        image_kwargs,
        topomap_kwargs,
        drop_log_ignore,
        image_format,
        section,
        tags,
        replace,
    ):
        """Render epochs."""
        if isinstance(epochs, BaseEpochs):
            fname = epochs.filename
        else:
            fname = epochs
            epochs = read_epochs(fname, preload=False)

        # Summary table
        self._add_html_repr(
            inst=epochs,
            title="Info",
            tags=tags,
            section=section,
            replace=replace,
            div_klass="epochs",
        )

        # Metadata table
        if epochs.metadata is not None:
            self._add_epochs_metadata(
                epochs=epochs, tags=tags, section=section, replace=replace
            )

        # ERP/ERF image(s)
        ch_types = _get_data_ch_types(epochs)
        epochs.load_data()

        _validate_type(image_kwargs, (dict, None), "image_kwargs")
        # ensure dict with shallow copy because we will modify it
        image_kwargs = dict() if image_kwargs is None else image_kwargs.copy()

        for ch_type in ch_types:
            with use_log_level(_verbose_safe_false(level="error")):
                figs = (
                    epochs.copy()
                    .pick(ch_type, verbose=False)
                    .plot_image(show=False, **image_kwargs.pop(ch_type, dict()))
                )

            assert len(figs) == 1
            fig = figs[0]
            if ch_type in ("mag", "grad"):
                title_start = "ERF image"
            else:
                assert "eeg" in ch_type
                title_start = "ERP image"

            title = f'{title_start} ({_handle_default("titles")[ch_type]})'

            self._add_figure(
                fig=fig,
                title=title,
                caption=None,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )
        if image_kwargs:
            raise ValueError(
                f"Ensure the keys in image_kwargs map onto channel types plotted in "
                f"epochs.plot_image() of {ch_types}, could not use: "
                f"{list(image_kwargs)}"
            )

        # Drop log
        if epochs._bad_dropped:
            title = "Drop log"

            if epochs.drop_log_stats(ignore=drop_log_ignore) == 0:  # No drops
                self._add_html_element(
                    html="No epochs exceeded the rejection thresholds. "
                    "Nothing was dropped.",
                    div_klass="epochs",
                    title=title,
                    tags=tags,
                    section=section,
                    replace=replace,
                )
            else:
                fig = epochs.plot_drop_log(
                    subject=self.subject, ignore=drop_log_ignore, show=False
                )
                self._add_figure(
                    fig=fig,
                    image_format=image_format,
                    title=title,
                    caption=None,
                    tags=tags,
                    section=section,
                    replace=replace,
                    own_figure=True,
                )

        if psd:
            self._add_epochs_psd(
                epochs=epochs,
                psd=psd,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
            )
        if add_projs:
            self._add_projs(
                info=epochs,
                projs=None,
                title="Projections",
                image_format=image_format,
                tags=tags,
                topomap_kwargs=topomap_kwargs,
                section=section,
                replace=replace,
            )

    def _add_cov(self, *, cov, info, image_format, section, tags, replace):
        """Render covariance matrix & SVD."""
        if not isinstance(cov, Covariance):
            cov = read_cov(cov)
        if not isinstance(info, Info):
            info = read_info(info)

        fig_cov, fig_svd = plot_cov(cov=cov, info=info, show=False, show_svd=True)
        figs = [fig_cov, fig_svd]
        titles = ("Covariance matrix", "Singular values")
        for fig, title in zip(figs, titles):
            self._add_figure(
                fig=fig,
                title=title,
                caption=None,
                image_format=image_format,
                tags=tags,
                section=section,
                replace=replace,
                own_figure=True,
            )

    def _add_trans(
        self,
        *,
        trans,
        info,
        subject,
        subjects_dir,
        alpha,
        title,
        section,
        tags,
        coord_frame,
        replace,
    ):
        """Render trans (only PNG)."""
        if not isinstance(trans, Transform):
            trans = read_trans(trans)
        if not isinstance(info, Info):
            info = read_info(info)

        kwargs = dict(
            info=info,
            trans=trans,
            subject=subject,
            subjects_dir=subjects_dir,
            dig=True,
            meg=["helmet", "sensors"],
            show_axes=True,
            coord_frame=coord_frame,
        )
        img, caption = _iterate_trans_views(
            function=plot_alignment,
            alpha=alpha,
            max_width=self.img_max_width,
            max_res=self.img_max_res,
            **kwargs,
        )
        self._add_image(
            img=img,
            title=title,
            section=section,
            caption=caption,
            image_format="png",
            tags=tags,
            replace=replace,
        )

    def _add_stc(
        self,
        *,
        stc,
        title,
        subject,
        subjects_dir,
        n_time_points,
        image_format,
        section,
        tags,
        stc_plot_kwargs,
        replace,
    ):
        """Render STC."""
        if isinstance(stc, SourceEstimate):
            if subject is None:
                subject = self.subject  # supplied during Report init
                if not subject:
                    subject = stc.subject  # supplied when loading STC
                    if not subject:
                        raise ValueError(
                            "Please specify the subject name, as it  cannot "
                            "be found in stc.subject. You may wish to pass "
                            'the "subject" parameter to read_source_estimate()'
                        )
            else:
                subject = subject
        else:
            fname = stc
            stc = read_source_estimate(fname=fname, subject=subject)

        subjects_dir = self.subjects_dir if subjects_dir is None else subjects_dir

        if n_time_points is None:
            n_time_points = min(len(stc.times), 51)
        elif n_time_points > len(stc.times):
            raise ValueError(
                f"The requested number of time points ({n_time_points}) "
                f"exceeds the time points in the provided STC object "
                f"({len(stc.times)})"
            )
        if n_time_points == 1:  # only a single time point, pick the first one
            times = [stc.times[0]]
        else:
            times = np.linspace(
                start=stc.times[0], stop=stc.times[-1], num=n_time_points
            )
        t_zero_idx = np.abs(times).argmin()  # index of time closest to zero

        # Plot using 3d backend if available, and use Matplotlib
        # otherwise.
        import matplotlib.pyplot as plt

        stc_plot_kwargs = _handle_default("report_stc_plot_kwargs", stc_plot_kwargs)
        stc_plot_kwargs.update(subject=subject, subjects_dir=subjects_dir)

        if get_3d_backend() is not None:
            brain = stc.plot(**stc_plot_kwargs)
            brain._renderer.plotter.subplot(0, 0)
            backend_is_3d = True
        else:
            backend_is_3d = False

        figs = []
        for t in times:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore",
                    message="More than 20 figures have been opened",
                    category=RuntimeWarning,
                )

                if backend_is_3d:
                    brain.set_time(t)
                    fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
                    ax.imshow(brain.screenshot(time_viewer=True, mode="rgb"))
                    ax.axis("off")
                    _constrain_fig_resolution(
                        fig,
                        max_width=stc_plot_kwargs["size"][0],
                        max_res=self.img_max_res,
                    )
                    figs.append(fig)
                    plt.close(fig)
                else:
                    fig_lh = plt.figure(layout="constrained")
                    fig_rh = plt.figure(layout="constrained")

                    brain_lh = stc.plot(
                        views="lat",
                        hemi="lh",
                        initial_time=t,
                        backend="matplotlib",
                        subject=subject,
                        subjects_dir=subjects_dir,
                        figure=fig_lh,
                    )
                    brain_rh = stc.plot(
                        views="lat",
                        hemi="rh",
                        initial_time=t,
                        subject=subject,
                        subjects_dir=subjects_dir,
                        backend="matplotlib",
                        figure=fig_rh,
                    )
                    _constrain_fig_resolution(
                        fig_lh,
                        max_width=stc_plot_kwargs["size"][0],
                        max_res=self.img_max_res,
                    )
                    _constrain_fig_resolution(
                        fig_rh,
                        max_width=stc_plot_kwargs["size"][0],
                        max_res=self.img_max_res,
                    )
                    figs.append(brain_lh)
                    figs.append(brain_rh)
                    plt.close(fig_lh)
                    plt.close(fig_rh)

        if backend_is_3d:
            brain.close()
        else:
            brain_lh.close()
            brain_rh.close()

        captions = [f"Time point: {round(t, 3):0.3f} s" for t in times]
        self._add_slider(
            figs=figs,
            imgs=None,
            captions=captions,
            title=title,
            image_format=image_format,
            start_idx=t_zero_idx,
            section=section,
            tags=tags,
            replace=replace,
            own_figure=False,  # prevent rescaling
        )
        for fig in figs:
            plt.close(fig)

    def _add_bem(
        self,
        *,
        subject,
        subjects_dir,
        decim,
        n_jobs,
        width=512,
        image_format,
        title,
        tags,
        section,
        replace,
    ):
        """Render mri+bem (only PNG)."""
        if subjects_dir is None:
            subjects_dir = self.subjects_dir
        subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=True))

        # Get the MRI filename
        mri_fname = op.join(subjects_dir, subject, "mri", "T1.mgz")
        if not op.isfile(mri_fname):
            warn(f'MRI file "{mri_fname}" does not exist')

        # Get the BEM surface filenames
        bem_path = op.join(subjects_dir, subject, "bem")

        surfaces = _get_bem_plotting_surfaces(bem_path)
        if not surfaces:
            warn("No BEM surfaces found, rendering empty MRI")

        htmls = dict()
        for orientation in _BEM_VIEWS:
            htmls[orientation] = self._render_one_bem_axis(
                mri_fname=mri_fname,
                surfaces=surfaces,
                orientation=orientation,
                decim=decim,
                n_jobs=n_jobs,
                width=width,
                image_format=image_format,
                tags=tags,
            )

        # Special handling to deal with our tests, where we monkey-patch
        # _BEM_VIEWS to save time
        html_slider_axial = htmls["axial"] if "axial" in htmls else ""
        html_slider_sagittal = htmls["sagittal"] if "sagittal" in htmls else ""
        html_slider_coronal = htmls["coronal"] if "coronal" in htmls else ""

        html_partial = partial(
            _html_bem_element,
            div_klass="bem",
            html_slider_axial=html_slider_axial,
            html_slider_sagittal=html_slider_sagittal,
            html_slider_coronal=html_slider_coronal,
            tags=tags,
            title=title,
        )
        self._add_or_replace(
            title=title,
            section=section,
            tags=tags,
            html_partial=html_partial,
            replace=replace,
        )


###############################################################################
# Scraper for sphinx-gallery

_SCRAPER_TEXT = """
.. only:: builder_html

    .. container:: row

        .. raw:: html

            <strong><a href="{0}">The generated HTML document.</a></strong>
            <iframe class="sg_report" sandbox="allow-scripts allow-modals" src="{0}"></iframe>

"""  # noqa: E501
# Adapted from fa-file-code
_FA_FILE_CODE = '<svg class="sg_report" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"><path fill="#dec" d="M149.9 349.1l-.2-.2-32.8-28.9 32.8-28.9c3.6-3.2 4-8.8.8-12.4l-.2-.2-17.4-18.6c-3.4-3.6-9-3.7-12.4-.4l-57.7 54.1c-3.7 3.5-3.7 9.4 0 12.8l57.7 54.1c1.6 1.5 3.8 2.4 6 2.4 2.4 0 4.8-1 6.4-2.8l17.4-18.6c3.3-3.5 3.1-9.1-.4-12.4zm220-251.2L286 14C277 5 264.8-.1 252.1-.1H48C21.5 0 0 21.5 0 48v416c0 26.5 21.5 48 48 48h288c26.5 0 48-21.5 48-48V131.9c0-12.7-5.1-25-14.1-34zM256 51.9l76.1 76.1H256zM336 464H48V48h160v104c0 13.3 10.7 24 24 24h104zM209.6 214c-4.7-1.4-9.5 1.3-10.9 6L144 408.1c-1.4 4.7 1.3 9.6 6 10.9l24.4 7.1c4.7 1.4 9.6-1.4 10.9-6L240 231.9c1.4-4.7-1.3-9.6-6-10.9zm24.5 76.9l.2.2 32.8 28.9-32.8 28.9c-3.6 3.2-4 8.8-.8 12.4l.2.2 17.4 18.6c3.3 3.5 8.9 3.7 12.4.4l57.7-54.1c3.7-3.5 3.7-9.4 0-12.8l-57.7-54.1c-3.5-3.3-9.1-3.2-12.4.4l-17.4 18.6c-3.3 3.5-3.1 9.1.4 12.4z" class=""></path></svg>'  # noqa: E501


class _ReportScraper:
    """Scrape Report outputs.

    Only works properly if conf.py is configured properly and the file
    is written to the same directory as the example script.
    """

    def __repr__(self):
        return "<ReportScraper>"

    def __call__(self, block, block_vars, gallery_conf):
        for report in block_vars["example_globals"].values():
            if (
                isinstance(report, Report)
                and report.fname is not None
                and report.fname.endswith(".html")
                and gallery_conf["builder_name"] == "html"
            ):
                # Thumbnail
                image_path_iterator = block_vars["image_path_iterator"]
                img_fname = next(image_path_iterator)
                img_fname = img_fname.replace(".png", ".svg")
                with open(img_fname, "w") as fid:
                    fid.write(_FA_FILE_CODE)
                # copy HTML file
                html_fname = Path(report.fname).name
                srcdir = Path(gallery_conf["src_dir"])
                outdir = Path(gallery_conf["out_dir"])
                out_dir = outdir / Path(block_vars["target_file"]).parent.relative_to(
                    srcdir
                )
                os.makedirs(out_dir, exist_ok=True)
                out_fname = out_dir / html_fname
                copyfile(report.fname, out_fname)
                assert op.isfile(report.fname)
                # embed links/iframe
                data = _SCRAPER_TEXT.format(html_fname)
                return data
        return ""

    def set_dirs(self, app):
        # Inject something into sphinx_gallery_conf as this gets pickled properly
        # during parallel example generation
        app.config.sphinx_gallery_conf["out_dir"] = app.builder.outdir


def _df_bootstrap_table(*, df, data_id):
    html = df.to_html(
        border=0,
        index=False,
        show_dimensions=True,
        justify="unset",
        float_format=lambda x: f"{x:.3f}",
        classes="table table-hover table-striped table-sm table-responsive small",
        na_rep="",
    )
    htmls = html.split("\n")
    header_pattern = "<th>(.*)</th>"

    for idx, html in enumerate(htmls):
        if "<table" in html:
            htmls[idx] = html.replace(
                "<table",
                "<table "
                'id="mytable" '
                'data-toggle="table" '
                f'data-unique-id="{data_id}" '
                'data-search="true" '  # search / filter
                'data-search-highlight="true" '
                'data-show-columns="true" '  # show/hide columns
                'data-show-toggle="true" '  # allow card view
                'data-show-columns-toggle-all="true" '
                'data-click-to-select="true" '
                'data-show-copy-rows="true" '
                'data-show-export="true" '  # export to a file
                'data-export-types="[csv]" '
                'data-export-options=\'{"fileName": "metadata"}\' '
                'data-icon-size="sm" '
                'data-height="400"',
            )
            continue
        elif "<tr" in html:
            # Add checkbox for row selection
            htmls[idx] = f'{html}\n<th data-field="state" data-checkbox="true"></th>'
            continue

        col_headers = re.findall(pattern=header_pattern, string=html)
        if col_headers:
            # Make columns sortable
            assert len(col_headers) == 1
            col_header = col_headers[0]
            htmls[idx] = html.replace(
                "<th>",
                f'<th data-field="{col_header.lower()}" data-sortable="true">',
            )

    html = "\n".join(htmls)
    return html
