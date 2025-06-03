# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import base64
import glob
import os
import pickle
import re
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from mne import (
    Epochs,
    create_info,
    pick_channels_cov,
    read_cov,
    read_events,
    read_evokeds,
)
from mne._fiff.write import DATE_NONE
from mne.datasets import testing
from mne.epochs import make_metadata
from mne.io import RawArray, read_info, read_raw_fif
from mne.preprocessing import ICA
from mne.report import Report, _ReportScraper, open_report, report
from mne.report import report as report_mod
from mne.report.report import (
    _ALLOWED_IMAGE_FORMATS,
    CONTENT_ORDER,
)
from mne.utils import Bunch, _record_warnings
from mne.utils._testing import assert_object_equal
from mne.viz import plot_alignment

data_dir = testing.data_path(download=False)
subjects_dir = data_dir / "subjects"
sample_meg_dir = data_dir / "MEG" / "sample"
raw_fname = sample_meg_dir / "sample_audvis_trunc_raw.fif"
ms_fname = data_dir / "SSS" / "test_move_anon_raw.fif"
events_fname = sample_meg_dir / "sample_audvis_trunc_raw-eve.fif"
evoked_fname = sample_meg_dir / "sample_audvis_trunc-ave.fif"
cov_fname = sample_meg_dir / "sample_audvis_trunc-cov.fif"
ecg_proj_fname = sample_meg_dir / "sample_audvis_ecg-proj.fif"
eog_proj_fname = sample_meg_dir / "sample_audvis_eog-proj.fif"
fwd_fname = sample_meg_dir / "sample_audvis_trunc-meg-eeg-oct-6-fwd.fif"
trans_fname = sample_meg_dir / "sample_audvis_trunc-trans.fif"
inv_fname = sample_meg_dir / "sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif"
stc_fname = sample_meg_dir / "sample_audvis_trunc-meg"
mri_fname = subjects_dir / "sample" / "mri" / "T1.mgz"
bdf_fname = Path(__file__).parents[2] / "io" / "edf" / "tests" / "data" / "test.bdf"
edf_fname = Path(__file__).parents[2] / "io" / "edf" / "tests" / "data" / "test.edf"
base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
evoked_fname = base_dir / "test-ave.fif"
nirs_fname = (
    data_dir / "SNIRF" / "NIRx" / "NIRSport2" / "1.0.3" / "2021-05-05_001.snirf"
)
stc_plot_kwargs = dict(  # for speed
    smoothing_steps=1, size=(300, 300), views="lat", hemi="lh"
)
topomap_kwargs = dict(res=8, contours=0, sensors=False)


def _get_example_figures():
    """Create two example figures."""
    fig1 = np.zeros((2, 2, 3))
    fig2 = np.ones((2, 2, 3))
    return [fig1, fig2]


@pytest.fixture
def invisible_fig(monkeypatch):
    """Make objects invisible to speed up draws."""
    orig = report._fig_to_img

    def _make_invisible(fig, **kwargs):
        if isinstance(fig, plt.Figure):
            for ax in fig.axes:
                for attr in ("lines", "collections", "patches", "images", "texts"):
                    for item in getattr(ax, attr):
                        item.set_visible(False)
                ax.axis("off")
        return orig(fig, **kwargs)

    monkeypatch.setattr(report, "_fig_to_img", _make_invisible)
    yield


@pytest.mark.slowtest
@testing.requires_testing_data
def test_render_report(renderer_pyvistaqt, tmp_path, invisible_fig):
    """Test rendering *.fif files for mne report."""
    pytest.importorskip("pymatreader")

    raw_fname_new = tmp_path / "temp_raw.fif"
    raw_fname_new_bids = tmp_path / "temp_meg.fif"
    ms_fname_new = tmp_path / "temp_ms_raw.fif"
    event_fname_new = tmp_path / "temp_raw-eve.fif"
    cov_fname_new = tmp_path / "temp_raw-cov.fif"
    proj_fname_new = tmp_path / "temp_ecg-proj.fif"
    fwd_fname_new = tmp_path / "temp_raw-fwd.fif"
    inv_fname_new = tmp_path / "temp_raw-inv.fif"
    nirs_fname_new = tmp_path / "temp_raw-nirs.snirf"
    for a, b in [
        [raw_fname, raw_fname_new],
        [raw_fname, raw_fname_new_bids],
        [ms_fname, ms_fname_new],
        [events_fname, event_fname_new],
        [cov_fname, cov_fname_new],
        [ecg_proj_fname, proj_fname_new],
        [fwd_fname, fwd_fname_new],
        [inv_fname, inv_fname_new],
        [nirs_fname, nirs_fname_new],
    ]:
        shutil.copyfile(a, b)

    # create and add -epo.fif and -ave.fif files
    epochs_fname = tmp_path / "temp-epo.fif"
    evoked_fname = tmp_path / "temp-ave.fif"
    # Speed it up by picking channels
    raw = read_raw_fif(raw_fname_new)
    raw.pick(["MEG 0111", "MEG 0121", "EEG 001", "EEG 002"])
    raw.del_proj()
    raw.set_eeg_reference(projection=True).load_data()
    epochs = Epochs(raw, read_events(events_fname), 1, -0.2, 0.2)
    epochs.save(epochs_fname, overwrite=True)
    # This can take forever, so let's make it fast
    # Also, make sure crop range is wide enough to avoid rendering bug
    evoked = epochs.average()
    with pytest.warns(RuntimeWarning, match="tmax is not in time interval"):
        evoked.crop(0.1, 0.2)
    evoked.save(evoked_fname)

    report = Report(
        info_fname=raw_fname_new,
        subjects_dir=subjects_dir,
        projs=False,
        image_format="png",
    )
    with pytest.warns(RuntimeWarning, match="Cannot render MRI"):
        report.parse_folder(
            data_path=tmp_path,
            on_error="raise",
            n_time_points_evokeds=2,
            raw_butterfly=False,
            stc_plot_kwargs=stc_plot_kwargs,
            topomap_kwargs=topomap_kwargs,
        )
    assert repr(report)

    # Check correct paths and filenames
    fnames = glob.glob(str(tmp_path / "*.fif"))
    fnames.extend(glob.glob(str(tmp_path / "*.snirf")))

    titles = [Path(x).name for x in fnames if not x.endswith("-ave.fif")]
    titles.append(f"{evoked_fname.name}: {evoked.comment}")

    _, _, content_titles, _ = report._content_as_html()
    for title in titles:
        assert title in content_titles
        assert "".join(report.html).find(title) != -1

    assert len(content_titles) == len(fnames)

    # Check saving functionality
    report.data_path = tmp_path
    fname = tmp_path / "report.html"
    report.save(fname=fname, open_browser=False)
    assert fname.is_file()
    html = fname.read_text(encoding="utf-8")
    # Evoked in `evoked_fname`
    assert f"{evoked_fname.name}: {evoked.comment}" in html
    assert "Topographies" in html
    assert "Global field power" in html

    # Check saving same report to new filename
    report.save(fname=tmp_path / "report2.html", open_browser=False)
    assert (tmp_path / "report2.html").is_file()

    # Check overwriting file
    report.save(fname=tmp_path / "report.html", open_browser=False, overwrite=True)
    assert (tmp_path / "report.html").is_file()

    # Check pattern matching with multiple patterns
    pattern = ["*proj.fif", "*eve.fif"]
    with pytest.warns(RuntimeWarning, match="Cannot render MRI"):
        report.parse_folder(data_path=tmp_path, pattern=pattern, raw_butterfly=False)
    assert repr(report)

    fnames = glob.glob(str(tmp_path / "*.raw")) + glob.glob(str(tmp_path / "*.raw"))

    content_names = [element.name for element in report._content]
    for fname in fnames:
        fname = Path(fname)
        assert fname.name in [Path(x).name for x in content_names]
        assert "".join(report.html).find(fname.name) != -1

    with pytest.raises(ValueError, match="Invalid value"):
        Report(image_format="foo")
    with pytest.raises(ValueError, match="Invalid value"):
        Report(image_format=None)

    # ndarray support smoke test
    report.add_figure(fig=np.zeros((2, 3, 3)), title="title")

    with pytest.raises(TypeError, match="It seems you passed a path"):
        report.add_figure(fig="foo", title="title")
    with pytest.raises(TypeError, match=".*MNEQtBrowser.*Figure3D.*got.*"):
        report.add_figure(fig=1.0, title="title")


def test_render_mne_qt_browser(tmp_path, browser_backend):
    """Test adding a mne_qt_browser (and matplotlib) raw plot."""
    report = Report()
    info = create_info(1, 1000.0, "eeg")
    data = np.zeros((1, 1000))
    raw = RawArray(data, info)
    fig = raw.plot()
    name = fig.__class__.__name__
    if browser_backend.name == "matplotlib":
        assert "MNEBrowseFigure" in name
    else:
        assert "MNEQtBrowser" in name or "PyQtGraphBrowser" in name
    report.add_figure(fig, title="raw")


@testing.requires_testing_data
def test_render_report_extra(renderer_pyvistaqt, tmp_path, invisible_fig):
    """Test SVG and projector rendering separately."""
    # ... otherwise things are very slow
    raw_fname_new = tmp_path / "temp_raw.fif"
    shutil.copyfile(raw_fname, raw_fname_new)
    report = Report(
        info_fname=raw_fname_new,
        subjects_dir=subjects_dir,
        projs=True,
        image_format="svg",
    )
    with pytest.warns(RuntimeWarning, match="Cannot render MRI"):
        report.parse_folder(
            data_path=tmp_path,
            on_error="raise",
            n_time_points_evokeds=2,
            raw_butterfly=False,
            stc_plot_kwargs=stc_plot_kwargs,
            topomap_kwargs=topomap_kwargs,
        )
    assert repr(report)
    report.data_path = tmp_path
    fname = tmp_path / "report.html"
    report.save(fname=fname, open_browser=False)
    assert fname.is_file()
    html = fname.read_text(encoding="utf-8")
    # Projectors in Raw.info
    assert "Projectors" in html


def test_add_custom_css(tmp_path):
    """Test adding custom CSS rules to the report."""
    fname = tmp_path / "report.html"
    fig = plt.figure()  # Empty figure

    report = Report()
    report.add_figure(fig=fig, title="Test section")
    custom_css = ".report_custom { color: red; }"
    report.add_custom_css(css=custom_css)

    assert custom_css in report.include
    report.save(fname, open_browser=False)
    html = Path(fname).read_text(encoding="utf-8")
    assert custom_css in html


def test_add_custom_js(tmp_path):
    """Test adding custom JavaScript to the report."""
    fname = tmp_path / "report.html"
    fig = plt.figure()  # Empty figure

    report = Report()
    report.add_figure(fig=fig, title="Test section")
    custom_js = 'function hello() {\n  alert("Hello, report!");\n}'
    report.add_custom_js(js=custom_js)

    assert custom_js in report.include
    report.save(fname, open_browser=False)
    html = Path(fname).read_text(encoding="utf-8")
    assert custom_js in html


@testing.requires_testing_data
def test_render_non_fiff(tmp_path):
    """Test rendering non-FIFF files for mne report."""
    fnames_in = [bdf_fname, edf_fname]
    fnames_out = []
    for fname in fnames_in:
        basename = fname.stem
        ext = fname.suffix
        fname_out = f"{basename}_raw{ext}"
        outpath = tmp_path / fname_out
        shutil.copyfile(fname, outpath)
        fnames_out.append(fname_out)

    report = Report()
    report.parse_folder(
        data_path=tmp_path,
        render_bem=False,
        on_error="raise",
        raw_butterfly=False,
    )

    # Check correct paths and filenames
    _, _, content_titles, _ = report._content_as_html()
    for fname in content_titles:
        assert Path(fname).name in [Path(x).name for x in content_titles]

    assert len(content_titles) == len(fnames_out)

    report.data_path = tmp_path
    fname = tmp_path / "report.html"
    report.save(fname=fname, open_browser=False)
    html = fname.read_text(encoding="utf-8")

    assert "test_raw.bdf" in html
    assert "test_raw.edf" in html


@testing.requires_testing_data
def test_report_raw_psd_and_date(tmp_path):
    """Test report raw PSD and DATE_NONE functionality."""
    with pytest.raises(TypeError, match="dict"):
        Report(raw_psd="foo")

    raw = read_raw_fif(raw_fname).crop(0, 1.0).load_data()
    raw.info["experimenter"] = "mne test"
    raw.info["subject_info"] = dict(id=123, his_id="sample")

    raw_fname_new = tmp_path / "temp_raw.fif"
    raw.save(raw_fname_new)
    report = Report(raw_psd=True)
    report.parse_folder(
        data_path=tmp_path,
        render_bem=False,
        on_error="raise",
        raw_butterfly=False,
    )
    assert isinstance(report.html, list)
    assert "PSD" in "".join(report.html)
    assert "Unknown" not in "".join(report.html)
    assert "UTC" in "".join(report.html)

    # test kwargs passed through to underlying array func
    Report(raw_psd=dict(window="boxcar"))

    # test new anonymize functionality
    report = Report()
    raw.anonymize()
    raw.save(raw_fname_new, overwrite=True)
    report.parse_folder(
        data_path=tmp_path,
        render_bem=False,
        on_error="raise",
        raw_butterfly=False,
    )
    assert isinstance(report.html, list)
    assert "Unknown" not in "".join(report.html)

    # DATE_NONE functionality
    report = Report()
    # old style (pre 0.20) date anonymization
    with raw.info._unlock():
        raw.info["meas_date"] = None
    for key in ("file_id", "meas_id"):
        value = raw.info.get(key)
        if value is not None:
            assert "msecs" not in value
            value["secs"] = DATE_NONE[0]
            value["usecs"] = DATE_NONE[1]
    raw.save(raw_fname_new, overwrite=True)
    report.parse_folder(
        data_path=tmp_path,
        render_bem=False,
        on_error="raise",
        raw_butterfly=False,
    )
    assert isinstance(report.html, list)
    assert "Unknown" in "".join(report.html)


@pytest.mark.slowtest  # slow on Azure
@testing.requires_testing_data
def test_render_add_sections(renderer, tmp_path):
    """Test adding figures/images to section."""
    pytest.importorskip("nibabel")
    try:
        from pyvista.plotting.plotter import _ALL_PLOTTERS
    except Exception:  # PV < 0.40
        from pyvista.plotting.plotting import _ALL_PLOTTERS

    report = Report(subjects_dir=subjects_dir)
    # Check add_figure functionality
    plt.close("all")
    assert len(plt.get_fignums()) == 0
    fig = plt.plot([1, 2], [1, 2])[0].figure
    assert len(plt.get_fignums()) == 1

    report.add_figure(fig=fig, title="evoked response", image_format="svg")
    assert "caption" not in report._content[-1].html
    assert len(plt.get_fignums()) == 1

    report.add_figure(fig=fig, title="evoked with caption", caption="descr")
    assert "caption" in report._content[-1].html
    assert len(plt.get_fignums()) == 1

    # Check add_image with png
    img_fname = tmp_path / "testimage.png"
    fig.savefig(img_fname)
    report.add_image(image=img_fname, title="evoked response")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        report.add_image(image="foobar.xxx", title="H")

    evoked = read_evokeds(evoked_fname, condition="Left Auditory", baseline=(-0.2, 0.0))
    n_before = len(_ALL_PLOTTERS)
    fig = plot_alignment(
        evoked.info, trans_fname, subject="sample", subjects_dir=subjects_dir
    )
    n_after = n_before + 1
    assert n_after == len(_ALL_PLOTTERS)

    report.add_figure(fig=fig, title="random image")
    assert n_after == len(_ALL_PLOTTERS)  # not closed
    assert repr(report)
    fname = tmp_path / "test.html"
    report.save(fname, open_browser=False)

    assert len(report) == 4


@pytest.mark.slowtest
@testing.requires_testing_data
def test_render_mri(renderer, tmp_path):
    """Test rendering MRI for mne report."""
    pytest.importorskip("nibabel")
    trans_fname_new = tmp_path / "temp-trans.fif"
    for a, b in [[trans_fname, trans_fname_new]]:
        shutil.copyfile(a, b)
    report = Report(info_fname=raw_fname, subject="sample", subjects_dir=subjects_dir)
    report.parse_folder(data_path=tmp_path, mri_decim=30, pattern="*")
    fname = tmp_path / "report.html"
    report.save(fname, open_browser=False)
    html = Path(fname).read_text(encoding="utf-8")
    assert 'data-mne-tags=" bem "' in html
    assert repr(report)
    report.add_bem(
        subject="sample",
        title="extra",
        tags=("foo",),
        subjects_dir=subjects_dir,
        decim=30,
    )
    report.save(fname, open_browser=False, overwrite=True)
    html = Path(fname).read_text(encoding="utf-8")
    assert 'data-mne-tags=" bem "' in html
    assert 'data-mne-tags=" foo "' in html


@testing.requires_testing_data
@pytest.mark.parametrize(
    "n_jobs",
    [
        1,
        pytest.param(2, marks=pytest.mark.slowtest),  # 1.5 s locally
    ],
)
@pytest.mark.filterwarnings("ignore:No contour levels were.*:UserWarning")
def test_add_bem_n_jobs(n_jobs, monkeypatch):
    """Test add_bem with n_jobs."""
    pytest.importorskip("nibabel")
    if n_jobs == 1:  # in one case, do at init -- in the other, pass in
        use_subjects_dir = None
    else:
        use_subjects_dir = subjects_dir
    report = Report(subjects_dir=use_subjects_dir, image_format="png")
    # implicitly test that subjects_dir is correctly preserved here
    monkeypatch.setattr(report_mod, "_BEM_VIEWS", ("axial",))
    if use_subjects_dir is not None:
        use_subjects_dir = None
    report.add_bem(
        subject="sample",
        title="sample",
        tags=("sample",),
        decim=15,
        n_jobs=n_jobs,
        subjects_dir=subjects_dir,
    )
    assert len(report.html) == 1
    imgs = np.array(
        [
            plt.imread(BytesIO(base64.b64decode(b)), "png")
            for b in re.findall(r'data:image/png;base64,(\S*)">', report.html[0])
        ]
    )
    assert imgs.ndim == 4  # images, h, w, rgba
    assert len(imgs) == 6
    imgs.shape = (len(imgs), -1)
    norms = np.linalg.norm(imgs, axis=-1)
    # should have down-up-down shape
    corr = np.corrcoef(norms, np.hanning(len(imgs)))[0, 1]
    assert 0.778 < corr < 0.80


@testing.requires_testing_data
def test_render_mri_without_bem(tmp_path):
    """Test rendering MRI without BEM for mne report."""
    pytest.importorskip("nibabel")
    os.mkdir(tmp_path / "sample")
    os.mkdir(tmp_path / "sample" / "mri")
    shutil.copyfile(mri_fname, tmp_path / "sample" / "mri" / "T1.mgz")
    report = Report(info_fname=raw_fname, subject="sample", subjects_dir=tmp_path)
    with pytest.raises(RuntimeError, match="No matching files found"):
        report.parse_folder(tmp_path, render_bem=False)
    with pytest.warns(RuntimeWarning, match="No BEM surfaces found"):
        report.parse_folder(tmp_path, render_bem=True, mri_decim=20)
    assert "BEM surfaces" in [element.name for element in report._content]
    report.save(tmp_path / "report.html", open_browser=False)


@testing.requires_testing_data
def test_add_html():
    """Test adding html str to mne report."""
    pytest.importorskip("nibabel")
    report = Report(info_fname=raw_fname, subject="sample", subjects_dir=subjects_dir)
    html = "<b>MNE-Python is AWESOME</b>"
    report.add_html(html=html, title="html")
    assert html in report.html[0]
    assert repr(report)


@testing.requires_testing_data
def test_multiple_figs(tmp_path):
    """Test adding a slider with a series of figures to a Report."""
    report = Report(info_fname=raw_fname, subject="sample", subjects_dir=subjects_dir)
    figs = _get_example_figures()
    report.add_figure(fig=figs, title="my title")
    assert report._content[0].name == "my title"
    report.save(tmp_path / "report.html", open_browser=False)

    with pytest.raises(ValueError):
        report.add_figure(fig=figs, title="title", caption=["wug"])

    with pytest.raises(
        ValueError, match="Number of captions.*must be equal to.*figures"
    ):
        report.add_figure(fig=figs, title="title", caption="wug")

    # Smoke test that SVG with unicode can be added
    report = Report()
    fig, ax = plt.subplots()
    ax.set_xlabel("µ")
    report.add_figure(fig=[fig] * 2, title="title", image_format="svg")


def test_validate_input():
    """Test Report input validation."""
    report = Report()
    items = ["a", "b", "c"]
    captions = ["Letter A", "Letter B", "Letter C"]
    section = "ABCs"
    comments = [
        "First letter of the alphabet.",
        "Second letter of the alphabet",
        "Third letter of the alphabet",
    ]
    pytest.raises(
        ValueError, report._validate_input, items, captions[:-1], section, comments=None
    )
    pytest.raises(
        ValueError,
        report._validate_input,
        items,
        captions,
        section,
        comments=comments[:-1],
    )
    values = report._validate_input(items, captions, section, comments=None)
    items_new, captions_new, comments_new = values


def test_open_report(tmp_path):
    """Test the open_report function."""
    h5py = pytest.importorskip("h5py")
    h5io = pytest.importorskip("h5io")
    hdf5 = str(tmp_path / "report.h5")

    # Test creating a new report through the open_report function
    fig1 = _get_example_figures()[0]
    with open_report(hdf5, subjects_dir=tmp_path) as report:
        assert report.subjects_dir == str(tmp_path)
        assert report.fname == str(hdf5)
        report.add_figure(fig=fig1, title="evoked response")
    # Exiting the context block should have triggered saving to HDF5
    assert Path(hdf5).exists()

    # Let's add some companion data to the HDF5 file
    with h5py.File(hdf5, "r+") as f:
        h5io.write_hdf5(f, "test", title="companion")
    assert h5io.read_hdf5(hdf5, title="companion") == "test"

    # Load the HDF5 version of the report and check equivalence
    report2 = open_report(hdf5)
    assert report2.fname == str(hdf5)
    assert report2.subjects_dir == report.subjects_dir
    assert report2.html == report.html
    assert report2.__getstate__() == report.__getstate__()
    assert "_fname" not in report2.__getstate__()

    # Check parameters when loading a report
    pytest.raises(ValueError, open_report, hdf5, foo="bar")  # non-existing
    pytest.raises(ValueError, open_report, hdf5, subjects_dir="foo")
    open_report(hdf5, subjects_dir=str(tmp_path))  # This should work

    # Check that the context manager doesn't swallow exceptions
    with pytest.raises(ZeroDivisionError):
        with open_report(hdf5, subjects_dir=str(tmp_path)) as report:
            assert h5io.read_hdf5(hdf5, title="companion") == "test"
            1 / 0

    # Check that our companion data survived
    assert h5io.read_hdf5(hdf5, title="companion") == "test"


def test_remove():
    """Test removing figures from a report."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figure(fig=fig1, title="figure1", tags=("slider",))
    r.add_figure(fig=[fig1, fig2], title="figure1", tags=("othertag",))
    r.add_figure(fig=fig2, title="figure1", tags=("slider",))
    r.add_figure(fig=fig2, title="figure2", tags=("slider",))

    # Test removal by title
    r2 = r.copy()
    removed_index = r2.remove(title="figure1")
    assert removed_index == 2
    assert len(r2.html) == 3
    assert r2.html[0] == r.html[0]
    assert r2.html[1] == r.html[1]
    assert r2.html[2] == r.html[3]

    # Test restricting to section
    r2 = r.copy()
    removed_index = r2.remove(title="figure1", tags=("othertag",))
    assert removed_index == 1
    assert len(r2.html) == 3
    assert r2.html[0] == r.html[0]
    assert r2.html[1] == r.html[2]
    assert r2.html[2] == r.html[3]


@pytest.mark.parametrize("tags", (True, False))  # shouldn't matter
def test_add_or_replace(tags):
    """Test replacing existing figures in a report."""
    # Note that tags don't matter, only titles do!
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figure(fig=fig1, title="duplicate", tags=("foo",) if tags else ())
    r_state = r.__getstate__()
    html = r.html
    r_state_after = r.__getstate__()
    assert_object_equal(r_state, r_state_after)
    html_2 = r.html
    assert html == html_2  # stays the same
    r_state_after = r.__getstate__()
    assert_object_equal(r_state, r_state_after)
    assert ' id="global' not in "\n".join(html)
    assert ' id="duplicate" ' in html[0]
    assert ' id="duplicate-' not in "\n".join(html)
    r.add_figure(fig=fig2, title="duplicate", tags=("foo",) if tags else ())
    html = r.html
    assert ' id="duplicate" ' in html[0]
    assert ' id="duplicate-1" ' in html[1]
    assert ' id="duplicate-2" ' not in "\n".join(html)
    r.add_figure(fig=fig1, title="duplicate", tags=("bar",) if tags else ())
    html = r.html
    assert ' id="duplicate" ' in html[0]
    assert ' id="duplicate-1" ' in html[1]
    assert ' id="duplicate-2" ' in html[2]
    assert ' id="duplicate-3" ' not in "\n".join(html)
    r.add_figure(fig=fig2, title="nonduplicate", tags=("foo",) if tags else ())
    html = r.html
    assert ' id="nonduplicate" ' in html[3]
    # By default, replace=False, so all figures should be there
    assert len(r.html) == 4
    assert len(r._content) == 4

    old_r = r.copy()

    # Replace our last occurrence of title='duplicate'
    r.add_figure(
        fig=fig2,
        title="duplicate",
        tags=("bar",) if tags else (),
        replace=True,
    )
    assert len(r._content) == len(r.html) == 4
    # This figure should have changed
    assert r.html[2] != old_r.html[2]
    # All other figures should be the same
    assert r.html[0] == old_r.html[0]
    assert r.html[1] == old_r.html[1]
    assert r.html[3] == old_r.html[3]
    # same DOM IDs
    html = r.html
    assert ' id="duplicate" ' in html[0]
    assert ' id="duplicate-1" ' in html[1]
    assert ' id="duplicate-2" ' in html[2]
    assert ' id="duplicate-3" ' not in "\n".join(html)
    assert ' id="global' not in "\n".join(html)

    # Now we change our max dup limit and should end up with a `global-`
    r._dup_limit = 2
    r.add_figure(
        fig=fig2,
        title="duplicate",
        replace=True,
    )
    html = r.html
    assert ' id="duplicate" ' in html[0]
    assert ' id="duplicate-1" ' in html[1]
    assert ' id="duplicate-2" ' in html[2]  # dom_id preserved
    assert ' id="global' not in "\n".join(html)
    r.add_figure(
        fig=fig2,
        title="duplicate",
    )  # append, should end up with global-1 ID
    html = r.html
    assert len(html) == 5
    assert ' id="global-1" ' in html[4]

    # And if we add a duplicate in a different section, it gets a different
    # DOM ID
    old_html = html
    section = "<div whatever 😀   etc."
    sec_san = "_div_whatever___etc_"
    r.add_figure(
        fig=fig2,
        title="duplicate",
        section=section,
        replace=True,  # should have no effect
    )
    html = r.html
    assert len(html) == 6
    assert html[:5] == old_html
    assert f' id="{sec_san}" ' in html[5]  # section anchor
    assert f' id="{sec_san}-duplicate" ' in html[5]  # and section-title anchor


def test_add_or_replace_section():
    """Test that sections are respected when adding or replacing."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figure(fig=fig1, title="a", section="A")
    r.add_figure(fig=fig1, title="a", section="B")
    r.add_figure(fig=fig1, title="a", section="C")
    # By default, replace=False, so all figures should be there
    assert len(r.html) == 3
    assert len(r._content) == 3

    old_r = r.copy()
    assert r.html[0] == old_r.html[0]
    assert r.html[1] == old_r.html[1]
    assert r.html[2] == old_r.html[2]

    # Replace our one occurrence of title 'a' in section 'B'
    r.add_figure(fig=fig2, title="a", section="B", replace=True)
    assert len(r._content) == 3
    assert len(r.html) == 3
    assert r.html[0] == old_r.html[0]
    assert r.html[1] != old_r.html[1]
    assert r.html[2] == old_r.html[2]
    r.add_figure(fig=fig1, title="a", section="B", replace=True)
    assert r.html[0] == old_r.html[0]
    assert r.html[1] == old_r.html[1]
    assert r.html[2] == old_r.html[2]
    r.add_figure(fig=fig1, title="a", section="C", replace=True)
    assert r.html[0] == old_r.html[0]
    assert r.html[1] == old_r.html[1]
    assert r.html[2] == old_r.html[2]


def test_scraper(tmp_path):
    """Test report scraping."""
    r = Report()
    fig1, fig2 = _get_example_figures()
    r.add_figure(fig=fig1, title="a")
    r.add_figure(fig=fig2, title="b")
    # Mock a Sphinx + sphinx_gallery config
    srcdir = tmp_path
    outdir = tmp_path / "_build" / "html"
    scraper = _ReportScraper()
    gallery_conf = dict(builder_name="html", src_dir=srcdir)
    app = Bunch(
        builder=Bunch(outdir=outdir),
        config=Bunch(sphinx_gallery_conf=gallery_conf),
    )
    scraper.set_dirs(app)
    img_fname = srcdir / "auto_examples" / "images" / "sg_img.png"
    target_file = srcdir / "auto_examples" / "sg.py"
    os.makedirs(img_fname.parent)
    block_vars = dict(
        image_path_iterator=(img for img in [str(img_fname)]),
        example_globals=dict(a=1),
        target_file=target_file,
    )
    # Nothing yet
    block = None
    rst = scraper(block, block_vars, gallery_conf)
    assert rst == ""
    # Still nothing
    block_vars["example_globals"]["r"] = r
    rst = scraper(block, block_vars, gallery_conf)
    # Once it's saved, add it
    assert rst == ""
    fname = srcdir / "my_html.html"
    r.save(fname, open_browser=False)
    out_html = outdir / "auto_examples" / "my_html.html"
    assert not out_html.is_file()
    rst = scraper(block, block_vars, gallery_conf)
    assert out_html.is_file()
    assert rst.count('"') == 8
    assert "<iframe" in rst
    assert img_fname.with_suffix(".svg").is_file()


@testing.requires_testing_data
@pytest.mark.parametrize(
    "split_naming",
    (
        "neuromag",
        "bids",
    ),
)
def test_split_files(tmp_path, split_naming):
    """Test that in the case of split files, we only parse the first."""
    raw = read_raw_fif(raw_fname)
    split_size = "7MB"  # Should produce 3 files
    buffer_size_sec = 1  # Tiny buffer so it's smaller than the split size
    raw.save(
        tmp_path / "raw_meg.fif",
        split_size=split_size,
        split_naming=split_naming,
        buffer_size_sec=buffer_size_sec,
    )

    report = Report()
    report.parse_folder(tmp_path, render_bem=False, raw_butterfly=False)
    assert len(report._content) == 1


@pytest.mark.slowtest  # ~40 s on Azure Windows
@testing.requires_testing_data
def test_survive_pickle(tmp_path):
    """Testing functionality of Report-Object after pickling."""
    raw_fname_new = tmp_path / "temp_raw.fif"
    shutil.copyfile(raw_fname, raw_fname_new)

    # Pickle report object to simulate multiprocessing with joblib
    report = Report(info_fname=raw_fname_new)
    pickled_report = pickle.dumps(report)
    report = pickle.loads(pickled_report)  # nosec B301

    # Just test if no errors occur
    report.parse_folder(tmp_path, render_bem=False)
    save_name = tmp_path / "report.html"
    report.save(fname=save_name, open_browser=False)


@pytest.mark.slowtest  # ~30 s on Azure Windows
@testing.requires_testing_data
def test_manual_report_2d(tmp_path, invisible_fig):
    """Simulate user manually creating report by adding one file at a time."""
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")

    from sklearn.exceptions import ConvergenceWarning

    r = Report(title="My Report")
    raw = read_raw_fif(raw_fname)
    raw.pick(raw.ch_names[:6]).crop(10, None)
    raw.info.normalize_proj()
    raw_non_preloaded = raw.copy()
    raw.load_data()
    cov = read_cov(cov_fname)
    cov = pick_channels_cov(cov, raw.ch_names)
    events = read_events(events_fname)
    event_id = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
        "face": 5,
        "buttonpress": 32,
    }
    metadata, metadata_events, metadata_event_id = make_metadata(
        events=events, event_id=event_id, tmin=-0.2, tmax=0.5, sfreq=raw.info["sfreq"]
    )
    epochs_without_metadata = Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        baseline=None,
        decim=10,
        verbose="error",
    )
    epochs_with_metadata = Epochs(
        raw=raw,
        events=metadata_events,
        event_id=metadata_event_id,
        baseline=None,
        metadata=metadata,
        decim=10,
        verbose="error",
    )
    evokeds = read_evokeds(evoked_fname)
    evoked = evokeds[0].pick("eeg").decimate(10, verbose="error")

    with pytest.warns(ConvergenceWarning, match="did not converge"):
        ica = ICA(n_components=3, max_iter=1, random_state=42).fit(
            inst=raw.copy().crop(tmax=1)
        )
    ica_ecg_scores = ica_eog_scores = np.array([3, 0, 0])
    ica_ecg_evoked = ica_eog_evoked = epochs_without_metadata.average()

    # Normally, ICA.find_bads_*() assembles the labels_ dict; since we didn't run any
    # of these methods, fill in some fake values manually.
    ica.labels_ = {
        "ecg/0/fake ECG channel": [0],
        "eog/0/fake EOG channel": [1],
    }

    r.add_raw(raw=raw, title="my raw data", tags=("raw",), psd=True, projs=False)
    r.add_raw(raw=raw, title="my raw data 2", psd=False, projs=False, butterfly=1)
    r.add_events(events=events_fname, title="my events", sfreq=raw.info["sfreq"])
    r.add_epochs(
        epochs=epochs_without_metadata,
        title="my epochs",
        tags=("epochs",),
        psd=False,
        projs=False,
        image_kwargs=dict(mag=dict(colorbar=False)),
    )
    with pytest.raises(ValueError, match="map onto channel types"):
        r.add_epochs(epochs=epochs_without_metadata, image_kwargs=dict(a=1), title="a")
    r.add_epochs(
        epochs=epochs_without_metadata, title="my epochs 2", psd=1, projs=False
    )
    r.add_epochs(
        epochs=epochs_without_metadata, title="my epochs 2", psd=True, projs=False
    )
    assert "Metadata" in r.html[-1]
    assert "No metadata set" in r.html[-1]

    # Try with metadata
    r.add_epochs(
        epochs=epochs_with_metadata,
        title="my epochs with metadata",
        psd=False,
        projs=False,
    )
    assert "Metadata" in r.html[-1]
    assert "25 rows × 7 columns" in r.html[-1]

    with pytest.raises(ValueError, match="requested to calculate PSD on a duration"):
        r.add_epochs(
            epochs=epochs_with_metadata, title="my epochs 2", psd=100000000, projs=False
        )

    r.add_evokeds(
        evokeds=evoked,
        noise_cov=cov_fname,
        titles=["my evoked 1"],
        tags=("evoked",),
        projs=False,
        n_time_points=2,
    )
    r.add_projs(
        info=raw_fname, projs=ecg_proj_fname, title="my proj", tags=("ssp", "ecg")
    )
    r.add_ica(ica=ica, title="my ica", inst=None)
    with pytest.raises(RuntimeError, match="not preloaded"):
        r.add_ica(ica=ica, title="ica", inst=raw_non_preloaded)
    r.add_ica(
        ica=ica,
        title="my ica with raw inst",
        inst=raw,
        picks=[2],
        ecg_evoked=ica_ecg_evoked,
        eog_evoked=ica_eog_evoked,
        ecg_scores=ica_ecg_scores,
        eog_scores=ica_eog_scores,
    )
    assert "ICA component 2" in r._content[-1].html
    epochs_baseline = epochs_without_metadata.copy().load_data()
    epochs_baseline.apply_baseline((None, 0))
    r.add_ica(
        ica=ica,
        title="my ica with epochs inst",
        inst=epochs_baseline,
        picks=[0],
    )
    r.add_ica(ica=ica, title="my ica with picks=None", inst=epochs_baseline, picks=None)
    r.add_covariance(cov=cov, info=raw_fname, title="my cov")
    r.add_forward(
        forward=fwd_fname,
        title="my forward",
        subject="sample",
        subjects_dir=subjects_dir,
    )
    r.add_html(html="<strong>Hello</strong>", title="Bold")
    r.add_code(code=__file__, title="my code")
    r.add_sys_info(title="my sysinfo")

    # drop locations (only EEG channels in `evoked`)
    evoked_no_ch_locs = evoked.copy()
    for ch in evoked_no_ch_locs.info["chs"]:
        ch["loc"][:3] = np.nan

    with (
        _record_warnings(),
        pytest.warns(
            RuntimeWarning,
            match="No EEG channel locations found, cannot create joint plot",
        ),
    ):
        r.add_evokeds(
            evokeds=evoked_no_ch_locs,
            titles=["evoked no chan locs"],
            tags=("evoked",),
            projs=False,
            n_time_points=1,
        )
    assert "Time course" not in r._content[-1].html
    assert "Topographies" not in r._content[-1].html
    assert evoked.info["projs"]  # only then the following test makes sense
    assert "Projectors" not in r._content[-1].html
    assert "Global field power" in r._content[-1].html

    # Drop locations from Info used for projs
    info_no_ch_locs = raw.info.copy()
    for ch in info_no_ch_locs["chs"]:
        ch["loc"][:3] = np.nan

    with pytest.raises(ValueError, match="does not contain.*channel locations"):
        r.add_projs(info=info_no_ch_locs, title="Projs no chan locs")

    # Drop locations from ICA
    ica_no_ch_locs = ica.copy()
    for ch in ica_no_ch_locs.info["chs"]:
        ch["loc"][:3] = np.nan

    with (
        _record_warnings(),
        pytest.warns(RuntimeWarning, match="No Magnetometers channel locations"),
    ):
        r.add_ica(
            ica=ica_no_ch_locs, picks=[0], inst=raw.copy().load_data(), title="ICA"
        )
    assert "ICA component properties" not in r._content[-1].html
    assert "ICA component topographies" not in r._content[-1].html
    assert "Original and cleaned signal" in r._content[-1].html

    fname = tmp_path / "report.html"
    r.save(fname=fname, open_browser=False)


def test_report_tweaks(tmp_path, monkeypatch):
    """Test tweaking of report params."""
    r = Report(image_format="png")
    assert r.collapse == ()
    assert r.img_max_width == 850
    assert r.img_max_res == 100

    events = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 3]])
    kwargs = dict(events=events, sfreq=1000.0, title="my events", section="my section")
    with plt.rc_context(rc={"figure.dpi": 200, "figure.figsize": (10, 10)}):
        r.add_events(**kwargs)

    fname = tmp_path / "report.html"
    r.save(fname, open_browser=False)

    html = fname.read_text(encoding="utf-8")
    assert html.count("collapse show") == 2, fname  # section and element

    r.collapse = ["section"]
    r.save(fname, open_browser=False, overwrite=True)
    html = fname.read_text(encoding="utf-8")
    assert html.count("collapse show") == 1, fname  # section collapsed

    # Bad input handling
    with pytest.raises(ValueError):
        r.collapse = "foo"
    with pytest.raises(TypeError):
        r.collapse = 1
    with pytest.raises(TypeError):
        r.img_max_width = "foo"
    with pytest.raises(ValueError):
        r.img_max_width = -1
    with pytest.raises(TypeError):
        r.img_max_res = "foo"
    with pytest.raises(ValueError):
        r.img_max_res = -1

    # Figure out the size of our rendered image (max width 850)
    img_re = re.compile(r'src="data:image/png;base64([^"]+)"')
    imgs = img_re.findall(html)
    assert len(imgs) == 2  # the first is our logo
    img = plt.imread(BytesIO(base64.b64decode(imgs[1].encode("ascii"))))
    assert img.shape == (850, 850, 3)

    # Now let's limit it by max resolution (100 dpi)
    r = Report(image_format="png")
    r.img_max_width = None
    with plt.rc_context(rc={"figure.dpi": 200, "figure.figsize": (10, 10)}):
        r.add_events(**kwargs)
    r.save(fname, open_browser=False, overwrite=True)
    imgs = img_re.findall(fname.read_text(encoding="utf-8"))
    assert len(imgs) == 2
    img = plt.imread(BytesIO(base64.b64decode(imgs[1].encode("ascii"))))
    assert img.shape == (1000, 1000, 3)  # figure.figsize * Report.img_max_res

    # Now let's do unconstrained
    r = Report(image_format="png")
    r.img_max_width = r.img_max_res = None
    with plt.rc_context(rc={"figure.dpi": 200, "figure.figsize": (10, 10)}):
        r.add_events(**kwargs)
    r.save(fname, open_browser=False, overwrite=True)
    imgs = img_re.findall(fname.read_text(encoding="utf-8"))
    assert len(imgs) == 2
    img = plt.imread(BytesIO(base64.b64decode(imgs[1].encode("ascii"))))
    assert img.shape == (2000, 2000, 3)  # figure.figsize * figure.dpi


@pytest.mark.slowtest  # 30 s on Azure
@testing.requires_testing_data
def test_manual_report_3d(tmp_path, renderer):
    """Simulate adding 3D sections."""
    pytest.importorskip("nibabel")
    r = Report(title="My Report")
    info = read_info(raw_fname)
    with info._unlock():
        dig, info["dig"] = info["dig"], []
    add_kwargs = dict(
        trans=trans_fname,
        info=info,
        subject="sample",
        subjects_dir=subjects_dir,
        alpha=0.75,
    )
    with (
        _record_warnings(),
        pytest.warns(RuntimeWarning, match="could not be calculated"),
    ):
        r.add_trans(title="coreg no dig", **add_kwargs)
    with info._unlock():
        info["dig"] = dig
    # TODO: We should probably speed this up. We could expose an arg to allow
    # use of sparse rather than dense head, and also possibly an arg to specify
    # which views to actually show. Both of these could probably be useful to
    # end-users, too.
    bad_add_kwargs = add_kwargs.copy()
    bad_add_kwargs.update(dict(trans="auto", subjects_dir=subjects_dir))
    with pytest.raises(RuntimeError, match="Could not find"):
        r.add_trans(title="my coreg", **bad_add_kwargs)
    add_kwargs.update(trans="fsaverage")  # this is wrong but tests fsaverage code path
    add_kwargs.update(plot_kwargs=dict(dig="fiducials"))  # test additional plot kwargs
    r.add_trans(title="my coreg", **add_kwargs)
    r.add_bem(subject="sample", subjects_dir=subjects_dir, title="my bem", decim=100)
    r.add_inverse_operator(
        inverse_operator=inv_fname,
        title="my inverse",
        subject="sample",
        subjects_dir=subjects_dir,
        trans=trans_fname,
    )
    r.add_stc(
        stc=stc_fname,
        title="my stc",
        subject="sample",
        subjects_dir=subjects_dir,
        n_time_points=2,
        stc_plot_kwargs=stc_plot_kwargs,
    )
    fname = tmp_path / "report.html"
    r.save(fname=fname, open_browser=False)


def test_sorting(tmp_path):
    """Test that automated ordering based on tags works."""
    r = Report()

    titles = ["intelligence >9000", "Pythagoras", "source of truth", "veggies"]
    r.add_code(code="E = m * c**2", title=titles[0], tags=("bem",))
    r.add_code(code="a**2 + b**2 = c**2", title=titles[1], tags=("evoked",))
    r.add_code(code="🧠", title=titles[2], tags=("source-estimate",))
    r.add_code(code="🥦", title=titles[3], tags=("raw",))

    # Check that repeated calls of add_* actually continuously appended to
    # the report
    orig_order = ["bem", "evoked", "source-estimate", "raw"]
    assert [c.tags[0] for c in r._content] == orig_order

    # tags property behavior and get_contents
    assert list(r.tags) == sorted(orig_order)
    titles, tags, htmls = r.get_contents()
    assert set(sum(tags, ())) == set(r.tags)
    assert len(titles) == len(tags) == len(htmls) == len(r._content)
    for title, tag, html in zip(titles, tags, htmls):
        title = title.replace(">", "&gt;")
        assert title in html
        for t in tag:
            assert t in html

    # Now check the actual sorting
    r_sorted = r.copy()
    r_sorted._sort(order=CONTENT_ORDER)
    expected_order = ["raw", "evoked", "bem", "source-estimate"]

    assert r_sorted._content != r._content
    assert [c.tags[0] for c in r_sorted._content] == expected_order
    assert [c.tags[0] for c in r._content] == orig_order

    r.copy().save(fname=tmp_path / "report.html", sort_content=True, open_browser=False)

    # Manual sorting should be the same
    r_sorted = r.copy()
    order = np.argsort([CONTENT_ORDER.index(t) for t in orig_order])
    r_sorted.reorder(order)

    assert r_sorted._content != r._content
    got_order = [c.tags[0] for c in r_sorted._content]
    assert [c.tags[0] for c in r._content] == orig_order  # original unmodified
    assert got_order == expected_order

    with pytest.raises(ValueError, match="order must be a permutation"):
        r.reorder(np.arange(len(r._content) + 1))
    with pytest.raises(ValueError, match="array of integers"):
        r.reorder([1.0])


@pytest.mark.parametrize(
    ("tags", "str_or_array", "wrong_dtype", "invalid_chars"),
    [
        # wrong dtype
        (123, False, True, False),
        ([1, 2, 3], True, True, False),
        (["foo", 1], True, True, False),
        # invalid characters
        (["foo bar"], True, False, True),
        (['foo"'], True, False, True),
        (["foo\n"], True, False, True),
        # all good
        ("foo", True, False, False),
        (["foo"], True, False, False),
        (["foo", "bar"], True, False, False),
        (np.array(["foo", "bar"]), True, False, False),
    ],
)
def test_tags(tags, str_or_array, wrong_dtype, invalid_chars):
    """Test handling of invalid tags."""
    r = Report()

    if not str_or_array:
        with pytest.raises(TypeError, match="must be a string.*or an array.*"):
            r.add_code(code="foo", title="bar", tags=tags)
    elif wrong_dtype:
        with pytest.raises(TypeError, match="tags must be strings"):
            r.add_code(code="foo", title="bar", tags=tags)
    elif invalid_chars:
        with pytest.raises(ValueError, match="contained invalid characters"):
            r.add_code(code="foo", title="bar", tags=tags)
    else:
        r.add_code(code="foo", title="bar", tags=tags)


# These are all the ones we claim to support
@pytest.mark.parametrize("image_format", _ALLOWED_IMAGE_FORMATS)
def test_image_format(image_format):
    """Test image format support."""
    r = Report(image_format=image_format)
    fig1, _ = _get_example_figures()
    r.add_figure(fig1, "fig1")
    assert image_format in r.html[0]


def test_gif(tmp_path):
    """Test that GIFs can be embedded using add_image."""
    pytest.importorskip("PIL")
    from PIL import Image

    sequence = [
        Image.fromarray(frame.astype(np.uint8)) for frame in _get_example_figures()
    ]
    fname = tmp_path / "test.gif"
    sequence[0].save(str(fname), save_all=True, append_images=sequence[1:])
    assert fname.is_file()
    with pytest.raises(ValueError, match="Allowed values"):
        Report(image_format="gif")
    r = Report()
    r.add_image(fname, "fname")
    assert "image/gif" in r.html[0]
    bad_name = fname.with_suffix(".foo")
    bad_name.write_bytes(b"")
    with pytest.raises(ValueError, match="Allowed values"):
        r.add_image(bad_name, "fname")
