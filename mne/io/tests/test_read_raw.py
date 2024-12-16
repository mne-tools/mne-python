"""Test generic read_raw function."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path
from shutil import copyfile

import pytest

from mne.datasets import testing
from mne.io import read_raw
from mne.io._read_raw import _get_readers, _get_supported, split_name_ext

base = Path(__file__).parents[1]
test_base = Path(testing.data_path(download=False))


@pytest.mark.parametrize("fname", ["x.xxx", "x"])
def test_read_raw_unsupported_single(fname):
    """Test handling of unsupported file types."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_raw(fname)


@pytest.mark.parametrize("fname", ["x.bin"])
def test_read_raw_unsupported_multi(fname, tmp_path):
    """Test handling of supported file types but with bad data."""
    fname = tmp_path / fname
    fname.write_text("")
    with pytest.raises(RuntimeError, match="Could not read.*using any"):
        read_raw(fname)


@pytest.mark.parametrize("fname", ["x.vmrk", "y.amrk"])
def test_read_raw_suggested(fname):
    """Test handling of unsupported file types with suggested alternatives."""
    with pytest.raises(ValueError, match="Try reading"):
        read_raw(fname)


_testing_mark = testing._pytest_mark()


@pytest.mark.parametrize(
    "fname",
    [
        base / "tests/data/test_raw.fif",
        base / "tests/data/test_raw.fif.gz",
        base / "edf/tests/data/test.edf",
        pytest.param(
            base / "edf/tests/data/test.bdf",
            marks=(
                _testing_mark,
                pytest.mark.filterwarnings("ignore:Channels contain different"),
            ),
        ),
        base / "brainvision/tests/data/test.vhdr",
        base / "kit/tests/data/test.sqd",
        pytest.param(test_base / "KIT" / "data_berlin.con", marks=_testing_mark),
        pytest.param(
            test_base
            / "ARTEMIS123"
            / "Artemis_Data_2017-04-14-10h-38m-59s_Phantom_1k_HPI_1s.bin",
            marks=_testing_mark,
        ),
        pytest.param(
            test_base / "FIL" / "sub-noise_ses-001_task-noise220622_run-001_meg.bin",
            marks=(
                _testing_mark,
                pytest.mark.filterwarnings("ignore:.*problems later!:RuntimeWarning"),
            ),
        ),
    ],
)
def test_read_raw_supported(fname):
    """Test supported file types."""
    read_raw(fname)
    read_raw(fname, verbose=False)
    raw = read_raw(fname, preload=True)
    assert "data loaded" in str(raw)


def test_split_name_ext():
    """Test file name extension splitting."""
    # test known extensions
    for ext in _get_readers():
        assert split_name_ext(f"test{ext}")[1] == ext

    # test unsupported extensions
    for ext in ("this.is.not.supported", "a.b.c.d.e", "fif.gz.xyz"):
        assert split_name_ext(f"test{ext}")[1] is None


def test_read_raw_multiple_dots(tmp_path):
    """Test if file names with multiple dots work correctly."""
    src = base / "edf/tests/data/test.edf"
    dst = tmp_path / "test.this.file.edf"
    copyfile(src, dst)
    read_raw(dst)


reader_excluded_from_read_raw = {
    "read_raw_bti",
    "read_raw_hitachi",
    "read_raw_neuralynx",
}


def test_all_reader_documented():
    """Test that all the readers in the documentation are accepted by read_raw."""
    readers = _get_supported()
    # flatten the dictionaries and retrieve the function names
    functions = [foo.__name__ for value in readers.values() for foo in value.values()]
    # read documentation .rst source file
    doc_folder = Path(__file__).parents[3] / "doc"
    if not doc_folder.exists():
        pytest.skip("Documentation folder not found.")
    doc_file = doc_folder / "api" / "reading_raw_data.rst"
    doc = doc_file.read_text("utf-8")
    reader_lines = [
        line.strip() for line in doc.split("\n") if line.strip().startswith("read_raw_")
    ]
    reader_lines = [
        elt for elt in reader_lines if elt not in reader_excluded_from_read_raw
    ]
    missing_from_read_raw = set(reader_lines) - set(functions)
    missing_from_doc = set(functions) - set(reader_lines)
    if len(missing_from_doc) != 0 or len(missing_from_read_raw) != 0:
        raise AssertionError(
            "Functions missing from documentation:\n\t"
            + "\n\t".join(missing_from_doc)
            + "\n\nFunctions missing from read_raw:\n\t"
            + "\n\t".join(missing_from_read_raw)
        )
    if sorted(reader_lines) != list(reader_lines):
        raise AssertionError(
            "Functions in documentation are not sorted. Expected order:\n\t"
            + "\n\t".join(sorted(reader_lines))
        )


def test_all_reader_documented_in_docstring():
    """Test that all the readers are documented in read_raw docstring."""
    readers = _get_supported()
    # flatten the dictionaries and retrieve the function names
    functions = [foo.__name__ for value in readers.values() for foo in value.values()]
    doc = read_raw.__doc__.split("Parameters")[0]
    documented = [elt.strip().split("`")[0] for elt in doc.split("mne.io.")[1:]]
    missing_from_docstring = set(functions) - set(documented)
    if len(missing_from_docstring) != 0:
        raise AssertionError(
            "Functions missing from docstring:\n\t"
            + "\n\t".join(missing_from_docstring)
        )
    if sorted(documented) != documented:
        raise AssertionError("Functions in docstring are not sorted.")
