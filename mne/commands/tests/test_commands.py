# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import glob
import os
import platform
import shutil
from os import path as op
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import mne
from mne import (
    concatenate_raws,
    read_bem_solution,
    read_bem_surfaces,
    read_source_spaces,
    read_surface,
)
from mne.bem import ConductorModel, convert_flash_mris
from mne.commands import (
    mne_anonymize,
    mne_browse_raw,
    mne_bti2fiff,
    mne_clean_eog_ecg,
    mne_compare_fiff,
    mne_compute_proj_ecg,
    mne_compute_proj_eog,
    mne_coreg,
    mne_flash_bem,
    mne_kit2fiff,
    mne_make_scalp_surfaces,
    mne_prepare_bem_model,
    mne_report,
    mne_setup_forward_model,
    mne_setup_source_space,
    mne_show_fiff,
    mne_show_info,
    mne_surf2bem,
    mne_sys_info,
    mne_watershed_bem,
    mne_what,
)
from mne.datasets import testing
from mne.io import read_info, read_raw_fif, show_fiff
from mne.utils import (
    ArgvSetter,
    _record_warnings,
    _stamp_to_dt,
    requires_freesurfer,
    requires_mne,
)

base_dir = op.join(op.dirname(__file__), "..", "..", "io", "tests", "data")
raw_fname = op.join(base_dir, "test_raw.fif")

testing_path = testing.data_path(download=False)
subjects_dir = op.join(testing_path, "subjects")
bem_model_fname = op.join(
    testing_path, "subjects", "sample", "bem", "sample-320-320-320-bem.fif"
)


def check_usage(module, force_help=False):
    """Ensure we print usage."""
    args = ("--help",) if force_help else ()
    with ArgvSetter(args) as out:
        try:
            module.run()
        except SystemExit:
            pass
        assert "Usage: " in out.stdout.getvalue()


@pytest.mark.slowtest
def test_browse_raw():
    """Test mne browse_raw."""
    check_usage(mne_browse_raw)
    with ArgvSetter(("--raw", raw_fname)):
        with _record_warnings():  # mpl show warning
            mne_browse_raw.run()


def test_what():
    """Test mne browse_raw."""
    check_usage(mne_browse_raw)
    with ArgvSetter((raw_fname,)) as out:
        mne_what.run()
    assert "raw" == out.stdout.getvalue().strip()


def test_bti2fiff():
    """Test mne bti2fiff."""
    check_usage(mne_bti2fiff)


def test_compare_fiff():
    """Test mne compare_fiff."""
    check_usage(mne_compare_fiff)


# should match ".*valid tag.*" but conda-linux intermittently fails for some reason
@pytest.mark.filterwarnings("ignore:Invalid tag.*:RuntimeWarning")
def test_show_fiff(tmp_path):
    """Test mne compare_fiff."""
    if os.getenv("MNE_CI_KIND", "") == "conda" and platform.system() == "Linux":
        pytest.skip("Skipping test on conda-linux due to intermittent failures")
    check_usage(mne_show_fiff)
    with ArgvSetter((raw_fname,)):
        mne_show_fiff.run()
    with ArgvSetter((raw_fname, "--tag=102")):
        mne_show_fiff.run()
    bad_fname = tmp_path / "test_bad_raw.fif"
    with open(bad_fname, "wb") as fout, open(raw_fname, "rb") as fin:
        fout.write(fin.read(100000))
    lines = show_fiff(bad_fname, output=list)
    last_line = lines[-1]
    assert last_line.endswith(">>>>BAD @9015")
    assert "302  = FIFF_EPOCH (734412b >f4)" in last_line


@requires_mne
def test_clean_eog_ecg(tmp_path):
    """Test mne clean_eog_ecg."""
    check_usage(mne_clean_eog_ecg)
    tempdir = str(tmp_path)
    raw = concatenate_raws([read_raw_fif(f) for f in [raw_fname, raw_fname, raw_fname]])
    raw.info["bads"] = ["MEG 2443"]
    use_fname = op.join(tempdir, op.basename(raw_fname))
    raw.save(use_fname)
    with ArgvSetter(("-i", use_fname, "--quiet")):
        mne_clean_eog_ecg.run()
    for key, count in (("proj", 2), ("-eve", 3)):
        fnames = glob.glob(op.join(tempdir, f"*{key}.fif"))
        assert len(fnames) == count


@pytest.mark.slowtest
@pytest.mark.parametrize("fun", (mne_compute_proj_ecg, mne_compute_proj_eog))
def test_compute_proj_exg(tmp_path, fun):
    """Test mne compute_proj_ecg/eog."""
    check_usage(fun)
    tempdir = str(tmp_path)
    use_fname = op.join(tempdir, op.basename(raw_fname))
    bad_fname = op.join(tempdir, "bads.txt")
    with open(bad_fname, "w") as fid:
        fid.write("MEG 2443\n")
    shutil.copyfile(raw_fname, use_fname)
    with ArgvSetter(("-i", use_fname, "--bad=" + bad_fname, "--rej-eeg", "150")):
        with _record_warnings():  # samples, sometimes
            fun.run()
    fnames = glob.glob(op.join(tempdir, "*proj.fif"))
    assert len(fnames) == 1
    fnames = glob.glob(op.join(tempdir, "*-eve.fif"))
    assert len(fnames) == 1


def test_coreg():
    """Test mne coreg."""
    assert hasattr(mne_coreg, "run")


def test_kit2fiff():
    """Test mne kit2fiff."""
    # Can't check
    check_usage(mne_kit2fiff, force_help=True)


@pytest.mark.slowtest
@pytest.mark.ultraslowtest
@testing.requires_testing_data
def test_make_scalp_surfaces(tmp_path, monkeypatch):
    """Test mne make_scalp_surfaces."""
    pytest.importorskip("nibabel")
    pytest.importorskip("pyvista")
    check_usage(mne_make_scalp_surfaces)
    has = "SUBJECTS_DIR" in os.environ
    # Copy necessary files to avoid FreeSurfer call
    tempdir = str(tmp_path)
    surf_path = op.join(subjects_dir, "sample", "surf")
    surf_path_new = op.join(tempdir, "sample", "surf")
    os.mkdir(op.join(tempdir, "sample"))
    os.mkdir(surf_path_new)
    subj_dir = op.join(tempdir, "sample", "bem")
    os.mkdir(subj_dir)

    cmd = ("-s", "sample", "--subjects-dir", tempdir)
    monkeypatch.setattr(
        mne.bem,
        "decimate_surface",
        lambda points, triangles, n_triangles: (points, triangles),
    )
    dense_fname = op.join(subj_dir, "sample-head-dense.fif")
    medium_fname = op.join(subj_dir, "sample-head-medium.fif")
    with ArgvSetter(cmd, disable_stdout=False, disable_stderr=False):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        with pytest.raises(RuntimeError, match="The FreeSurfer environ"):
            mne_make_scalp_surfaces.run()
        shutil.copy(op.join(surf_path, "lh.seghead"), surf_path_new)
        monkeypatch.setenv("FREESURFER_HOME", tempdir)
        mne_make_scalp_surfaces.run()
        assert op.isfile(dense_fname)
        assert op.isfile(medium_fname)
        with pytest.raises(OSError, match="overwrite"):
            mne_make_scalp_surfaces.run()
    # actually check the outputs
    head_py = read_bem_surfaces(dense_fname)
    assert_equal(len(head_py), 1)
    head_py = head_py[0]
    head_c = read_bem_surfaces(
        op.join(subjects_dir, "sample", "bem", "sample-head-dense.fif")
    )[0]
    assert_allclose(head_py["rr"], head_c["rr"])
    if not has:
        assert "SUBJECTS_DIR" not in os.environ


@pytest.mark.slowtest
@testing.requires_testing_data
def test_report(tmp_path):
    """Test mne report."""
    pytest.importorskip("nibabel")
    check_usage(mne_report)
    tempdir = str(tmp_path)
    use_fname = op.join(tempdir, op.basename(raw_fname))
    shutil.copyfile(raw_fname, use_fname)
    with ArgvSetter(
        (
            "-p",
            tempdir,
            "-i",
            use_fname,
            "-d",
            subjects_dir,
            "-s",
            "sample",
            "--no-browser",
            "-m",
            "30",
        )
    ):
        with _record_warnings():  # contour levels
            mne_report.run()
    fnames = glob.glob(op.join(tempdir, "*.html"))
    assert len(fnames) == 1


def test_surf2bem():
    """Test mne surf2bem."""
    check_usage(mne_surf2bem)


@pytest.mark.timeout(900)  # took ~400 s on a local test
@pytest.mark.slowtest
@pytest.mark.ultraslowtest
@requires_freesurfer("mri_watershed")
@testing.requires_testing_data
def test_watershed_bem(tmp_path):
    """Test mne watershed bem."""
    pytest.importorskip("nibabel")
    check_usage(mne_watershed_bem)
    # from T1.mgz
    Mdc = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
    Pxyz_c = np.array([-5.273613, 9.039085, -27.287964])
    # Copy necessary files to tempdir
    tempdir = str(tmp_path)
    mridata_path = op.join(subjects_dir, "sample", "mri")
    subject_path_new = op.join(tempdir, "sample")
    mridata_path_new = op.join(subject_path_new, "mri")
    os.makedirs(mridata_path_new)
    new_fname = op.join(mridata_path_new, "T1.mgz")
    shutil.copyfile(op.join(mridata_path, "T1.mgz"), new_fname)
    old_mode = os.stat(new_fname).st_mode
    os.chmod(new_fname, 0)
    args = ("-d", tempdir, "-s", "sample", "-o")
    with pytest.raises(PermissionError, match=r"read permissions.*T1\.mgz"):
        with ArgvSetter(args):
            mne_watershed_bem.run()
    os.chmod(new_fname, old_mode)
    for s in ("outer_skin", "outer_skull", "inner_skull"):
        assert not op.isfile(op.join(subject_path_new, "bem", f"{s}.surf"))
    with ArgvSetter(args):
        mne_watershed_bem.run()

    kwargs = dict(rtol=1e-5, atol=1e-5)
    for s in ("outer_skin", "outer_skull", "inner_skull"):
        rr, tris, vol_info = read_surface(
            op.join(subject_path_new, "bem", f"{s}.surf"), read_metadata=True
        )
        assert_equal(len(tris), 20480)
        assert_equal(tris.min(), 0)
        assert_equal(rr.shape[0], tris.max() + 1)
        # compare the volume info to the mgz header
        assert_allclose(vol_info["xras"], Mdc[0], **kwargs)
        assert_allclose(vol_info["yras"], Mdc[1], **kwargs)
        assert_allclose(vol_info["zras"], Mdc[2], **kwargs)
        assert_allclose(vol_info["cras"], Pxyz_c, **kwargs)


@pytest.mark.timeout(180)  # took ~70 s locally
@pytest.mark.slowtest
@pytest.mark.ultraslowtest
@requires_freesurfer
@testing.requires_testing_data
def test_flash_bem(tmp_path):
    """Test mne flash_bem."""
    pytest.importorskip("nibabel")
    check_usage(mne_flash_bem, force_help=True)
    # Copy necessary files to tempdir
    tempdir = Path(str(tmp_path))
    mridata_path = Path(subjects_dir) / "sample" / "mri"
    subject_path_new = tempdir / "sample"
    mridata_path_new = subject_path_new / "mri"
    flash_path = mridata_path_new / "flash"
    flash_path.mkdir(parents=True, exist_ok=True)
    bem_path = mridata_path_new / "bem"
    bem_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        op.join(mridata_path, "T1.mgz"), op.join(mridata_path_new, "T1.mgz")
    )
    shutil.copyfile(
        op.join(mridata_path, "brain.mgz"), op.join(mridata_path_new, "brain.mgz")
    )
    # Copy the available mri/flash/mef*.mgz files from the dataset
    for kind in (5, 30):
        in_fname = mridata_path / "flash" / f"mef{kind:02d}.mgz"
        in_fname_echo = flash_path / f"mef{kind:02d}_001.mgz"
        shutil.copyfile(in_fname, flash_path / in_fname_echo.name)
    # Test mne flash_bem with --noconvert option
    # (since there are no DICOM Flash images in dataset)
    for s in ("outer_skin", "outer_skull", "inner_skull"):
        assert not op.isfile(subject_path_new / "bem" / f"{s}.surf")

    # First test without flash30
    with ArgvSetter(
        ("-d", tempdir, "-s", "sample", "-r", "-3"),
        disable_stdout=False,
        disable_stderr=False,
    ):
        mne_flash_bem.run()
    for s in ("outer_skin", "outer_skull", "inner_skull"):
        surf_path = subject_path_new / "bem" / f"{s}.surf"
        assert surf_path.exists()
        surf_path.unlink()  # cleanup
    shutil.rmtree(flash_path / "parameter_maps")  # remove old files

    # Test synthesize flash5 with MEF flash5 and flash30 default locations
    flash5_img = convert_flash_mris(
        subject="sample", subjects_dir=tempdir, unwarp=False
    )
    assert flash5_img == (flash_path / "parameter_maps" / "flash5.mgz")
    assert flash5_img.exists()
    shutil.rmtree(flash_path / "parameter_maps")  # remove old files

    # Test with flash5 and flash30
    shutil.rmtree(flash_path)  # first remove old files
    with ArgvSetter(
        (
            "-d",
            tempdir,
            "-s",
            "sample",
            "-3",
            str(mridata_path / "flash" / "mef30.mgz"),
            "-5",
            str(mridata_path / "flash" / "mef05.mgz"),
        ),
        disable_stdout=False,
        disable_stderr=False,
    ):
        mne_flash_bem.run()

    kwargs = dict(rtol=1e-5, atol=1e-5)
    for s in ("outer_skin", "outer_skull", "inner_skull"):
        rr, tris = read_surface(op.join(subject_path_new, "bem", f"{s}.surf"))
        assert_equal(len(tris), 5120)
        assert_equal(tris.min(), 0)
        assert_equal(rr.shape[0], tris.max() + 1)
        # compare to the testing flash surfaces
        rr_c, tris_c = read_surface(op.join(subjects_dir, "sample", "bem", f"{s}.surf"))
        assert_allclose(rr, rr_c, **kwargs)
        assert_allclose(tris, tris_c, **kwargs)


@testing.requires_testing_data
def test_setup_source_space(tmp_path):
    """Test mne setup_source_space."""
    pytest.importorskip("nibabel")
    check_usage(mne_setup_source_space, force_help=True)
    # Using the sample dataset
    use_fname = op.join(tmp_path, "sources-src.fif")
    # Test  command
    with ArgvSetter(
        (
            "--src",
            use_fname,
            "-d",
            subjects_dir,
            "-s",
            "sample",
            "--morph",
            "sample",
            "--add-dist",
            "False",
            "--ico",
            "3",
            "--verbose",
        )
    ):
        mne_setup_source_space.run()
    src = read_source_spaces(use_fname)
    assert len(src) == 2
    with pytest.raises(Exception):
        with ArgvSetter(
            (
                "--src",
                use_fname,
                "-d",
                subjects_dir,
                "-s",
                "sample",
                "--ico",
                "3",
                "--oct",
                "3",
            )
        ):
            assert mne_setup_source_space.run()
    with pytest.raises(Exception):
        with ArgvSetter(
            (
                "--src",
                use_fname,
                "-d",
                subjects_dir,
                "-s",
                "sample",
                "--ico",
                "3",
                "--spacing",
                "10",
            )
        ):
            assert mne_setup_source_space.run()
    with pytest.raises(Exception):
        with ArgvSetter(
            (
                "--src",
                use_fname,
                "-d",
                subjects_dir,
                "-s",
                "sample",
                "--ico",
                "3",
                "--spacing",
                "10",
                "--oct",
                "3",
            )
        ):
            assert mne_setup_source_space.run()


@pytest.mark.slowtest
@testing.requires_testing_data
def test_setup_forward_model(tmp_path):
    """Test mne setup_forward_model."""
    pytest.importorskip("nibabel")
    check_usage(mne_setup_forward_model, force_help=True)
    # Using the sample dataset
    use_fname = op.join(tmp_path, "model-bem.fif")
    # Test  command
    with ArgvSetter(
        (
            "--model",
            use_fname,
            "-d",
            subjects_dir,
            "--homog",
            "-s",
            "sample",
            "--ico",
            "3",
            "--verbose",
        )
    ):
        mne_setup_forward_model.run()
    model = read_bem_surfaces(use_fname)
    assert len(model) == 1
    sol_fname = op.splitext(use_fname)[0] + "-sol.fif"
    read_bem_solution(sol_fname)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_mne_prepare_bem_model(tmp_path):
    """Test mne setup_source_space."""
    check_usage(mne_prepare_bem_model, force_help=True)
    # Using the sample dataset
    bem_solution_fname = op.join(tmp_path, "bem_solution-bem-sol.fif")
    # Test  command
    with ArgvSetter(
        ("--bem", bem_model_fname, "--sol", bem_solution_fname, "--verbose")
    ):
        mne_prepare_bem_model.run()
    bem_solution = read_bem_solution(bem_solution_fname)
    assert isinstance(bem_solution, ConductorModel)


def test_show_info():
    """Test mne show_info."""
    check_usage(mne_show_info)
    with ArgvSetter((raw_fname,)):
        mne_show_info.run()


def test_sys_info():
    """Test mne show_info."""
    check_usage(mne_sys_info, force_help=True)
    with ArgvSetter((raw_fname,)):
        with pytest.raises(SystemExit, match="1"):
            mne_sys_info.run()
    with ArgvSetter(("--no-check-version",)) as out:
        mne_sys_info.run()
    assert "numpy" in out.stdout.getvalue()


def test_anonymize(tmp_path):
    """Test mne anonymize."""
    check_usage(mne_anonymize)
    out_fname = op.join(tmp_path, "anon_test_raw.fif")
    with ArgvSetter(("-f", raw_fname, "-o", out_fname)):
        mne_anonymize.run()
    info = read_info(out_fname)
    assert op.exists(out_fname)
    assert info["meas_date"] == _stamp_to_dt((946684800, 0))
