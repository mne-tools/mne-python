"""Testing functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect
import os
import sys
import tempfile
import traceback
from functools import wraps
from shutil import rmtree
from unittest import SkipTest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg

from ._logging import ClosingStringIO, warn
from .check import check_version
from .misc import run_subprocess
from .numerics import object_diff


def _explain_exception(start=-1, stop=None, prefix="> "):
    """Explain an exception."""
    # start=-1 means "only the most recent caller"
    etype, value, tb = sys.exc_info()
    string = traceback.format_list(traceback.extract_tb(tb)[start:stop])
    string = "".join(string).split("\n") + traceback.format_exception_only(etype, value)
    string = ":\n" + prefix + ("\n" + prefix).join(string)
    return string


class _TempDir(str):
    """Create and auto-destroy temp dir.

    This is designed to be used with testing modules. Instances should be
    defined inside test functions. Instances defined at module level can not
    guarantee proper destruction of the temporary directory.

    When used at module level, the current use of the __del__() method for
    cleanup can fail because the rmtree function may be cleaned up before this
    object (an alternative could be using the atexit module instead).
    """

    def __new__(self):  # noqa: D105
        new = str.__new__(self, tempfile.mkdtemp(prefix="tmp_mne_tempdir_"))
        return new

    def __init__(self):
        self._path = self.__str__()

    def __del__(self):  # noqa: D105
        rmtree(self._path, ignore_errors=True)


def requires_mne(func):
    """Decorate a function as requiring MNE."""
    return requires_mne_mark()(func)


def requires_mne_mark():
    """Mark pytest tests that require MNE-C."""
    import pytest

    return pytest.mark.skipif(not has_mne_c(), reason="Requires MNE-C")


def requires_openmeeg_mark():
    """Mark pytest tests that require OpenMEEG."""
    import pytest

    return pytest.mark.skipif(
        not check_version("openmeeg", "2.5.6"), reason="Requires OpenMEEG >= 2.5.6"
    )


def requires_freesurfer(arg):
    """Require Freesurfer."""
    import pytest

    reason = "Requires Freesurfer"
    if isinstance(arg, str):
        # Calling as  @requires_freesurfer('progname'): return decorator
        # after checking for progname existence
        reason += f" command: {arg}"
        try:
            run_subprocess([arg, "--version"])
        except Exception:
            skip = True
        else:
            skip = False
        return pytest.mark.skipif(skip, reason=reason)
    else:
        # Calling directly as @requires_freesurfer: return decorated function
        # and just check env var existence
        return pytest.mark.skipif(not has_freesurfer(), reason="Requires Freesurfer")(
            arg
        )


def requires_good_network(func):
    import pytest

    return pytest.mark.skipif(
        int(os.environ.get("MNE_SKIP_NETWORK_TESTS", 0)),
        reason="MNE_SKIP_NETWORK_TESTS is set",
    )(func)


def run_command_if_main():
    """Run a given command if it's __main__."""
    local_vars = inspect.currentframe().f_back.f_locals
    if local_vars.get("__name__", "") == "__main__":
        local_vars["run"]()


class ArgvSetter:
    """Temporarily set sys.argv."""

    def __init__(self, args=(), disable_stdout=True, disable_stderr=True):
        self.argv = list(("python",) + args)
        self.stdout = ClosingStringIO() if disable_stdout else sys.stdout
        self.stderr = ClosingStringIO() if disable_stderr else sys.stderr

    def __enter__(self):  # noqa: D105
        self.orig_argv = sys.argv
        sys.argv = self.argv
        self.orig_stdout = sys.stdout
        sys.stdout = self.stdout
        self.orig_stderr = sys.stderr
        sys.stderr = self.stderr
        return self

    def __exit__(self, *args):  # noqa: D105
        sys.argv = self.orig_argv
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr


def has_mne_c():
    """Check for MNE-C."""
    return "MNE_ROOT" in os.environ


def has_freesurfer():
    """Check for Freesurfer."""
    return "FREESURFER_HOME" in os.environ


def buggy_mkl_svd(function):
    """Decorate tests that make calls to SVD and intermittently fail."""

    @wraps(function)
    def dec(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except np.linalg.LinAlgError as exp:
            if "SVD did not converge" in str(exp):
                msg = "Intel MKL SVD convergence error detected, skipping test"
                warn(msg)
                raise SkipTest(msg)
            raise

    return dec


def assert_and_remove_boundary_annot(annotations, n=1):
    """Assert that there are boundary annotations and remove them."""
    from ..io import BaseRaw

    if isinstance(annotations, BaseRaw):  # allow either input
        annotations = annotations.annotations
    for key in ("EDGE", "BAD"):
        idx = np.where(annotations.description == f"{key} boundary")[0]
        assert len(idx) == n
        annotations.delete(idx)


def assert_object_equal(a, b, *, err_msg="Object mismatch", allclose=False):
    """Assert two objects are equal."""
    d = object_diff(a, b, allclose=allclose)
    assert d == "", f"{err_msg}\n{d}"


def _raw_annot(meas_date, orig_time):
    from .._fiff.meas_info import create_info
    from ..annotations import Annotations, _handle_meas_date
    from ..io import RawArray

    info = create_info(ch_names=10, sfreq=10.0)
    raw = RawArray(data=np.empty((10, 10)), info=info, first_samp=10)
    if meas_date is not None:
        meas_date = _handle_meas_date(meas_date)
    with raw.info._unlock(check_after=True):
        raw.info["meas_date"] = meas_date
    annot = Annotations([0.5], [0.2], ["dummy"], orig_time)
    raw.set_annotations(annotations=annot)
    return raw


def _get_data(x, ch_idx):
    """Get the (n_ch, n_times) data array."""
    from ..evoked import Evoked
    from ..io import BaseRaw

    if isinstance(x, BaseRaw):
        return x[ch_idx][0]
    elif isinstance(x, Evoked):
        return x.data[ch_idx]


def _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind="MEG"):
    """Check the SNR of a set of channels."""
    actual_data = _get_data(actual, picks)
    desired_data = _get_data(desired, picks)
    bench_rms = np.sqrt(np.mean(desired_data * desired_data, axis=1))
    error = actual_data - desired_data
    error_rms = np.sqrt(np.mean(error * error, axis=1))
    np.clip(error_rms, 1e-60, np.inf, out=error_rms)  # avoid division by zero
    snrs = bench_rms / error_rms
    # min tol
    snr = snrs.min()
    bad_count = (snrs < min_tol).sum()
    msg = f" ({msg})" if msg != "" else msg
    assert bad_count == 0, (
        f"SNR (worst {snr:0.2f}) < {min_tol:0.2f} "
        f"for {bad_count}/{len(picks)} channels{msg}"
    )
    # median tol
    snr = np.median(snrs)
    assert snr >= med_tol, f"{kind} SNR median {snr:0.2f} < {med_tol:0.2f}{msg}"


def assert_meg_snr(
    actual, desired, min_tol, med_tol=500.0, chpi_med_tol=500.0, msg=None
):
    """Assert channel SNR of a certain level.

    Mostly useful for operations like Maxwell filtering that modify
    MEG channels while leaving EEG and others intact.
    """
    from .._fiff.pick import pick_types

    picks = pick_types(desired.info, meg=True, exclude=[])
    picks_desired = pick_types(desired.info, meg=True, exclude=[])
    assert_array_equal(picks, picks_desired, err_msg="MEG pick mismatch")
    chpis = pick_types(actual.info, meg=False, chpi=True, exclude=[])
    chpis_desired = pick_types(desired.info, meg=False, chpi=True, exclude=[])
    if chpi_med_tol is not None:
        assert_array_equal(chpis, chpis_desired, err_msg="cHPI pick mismatch")
    others = np.setdiff1d(
        np.arange(len(actual.ch_names)), np.concatenate([picks, chpis])
    )
    others_desired = np.setdiff1d(
        np.arange(len(desired.ch_names)), np.concatenate([picks_desired, chpis_desired])
    )
    assert_array_equal(others, others_desired, err_msg="Other pick mismatch")
    if len(others) > 0:  # if non-MEG channels present
        assert_allclose(
            _get_data(actual, others),
            _get_data(desired, others),
            atol=1e-11,
            rtol=1e-5,
            err_msg="non-MEG channel mismatch",
        )
    _check_snr(actual, desired, picks, min_tol, med_tol, msg, kind="MEG")
    if chpi_med_tol is not None and len(chpis) > 0:
        _check_snr(actual, desired, chpis, 0.0, chpi_med_tol, msg, kind="cHPI")


def assert_snr(actual, desired, tol):
    """Assert actual and desired arrays are within some SNR tolerance."""
    with np.errstate(divide="ignore"):  # allow infinite
        snr = linalg.norm(desired, ord="fro") / linalg.norm(desired - actual, ord="fro")
    assert snr >= tol, f"{snr} < {tol}"


def assert_stcs_equal(stc1, stc2):
    """Check that two STC are equal."""
    assert_allclose(stc1.times, stc2.times)
    assert_allclose(stc1.data, stc2.data)
    assert_array_equal(stc1.vertices[0], stc2.vertices[0])
    assert_array_equal(stc1.vertices[1], stc2.vertices[1])
    assert_allclose(stc1.tmin, stc2.tmin)
    assert_allclose(stc1.tstep, stc2.tstep)


def _dig_sort_key(dig):
    """Sort dig keys."""
    return (dig["kind"], dig["ident"])


def assert_dig_allclose(info_py, info_bin, limit=None):
    """Assert dig allclose."""
    from .._fiff.constants import FIFF
    from .._fiff.meas_info import Info
    from ..bem import fit_sphere_to_headshape
    from ..channels.montage import DigMontage

    # test dig positions
    dig_py, dig_bin = info_py, info_bin
    if isinstance(dig_py, Info):
        assert isinstance(dig_bin, Info)
        dig_py, dig_bin = dig_py["dig"], dig_bin["dig"]
    else:
        assert isinstance(dig_bin, DigMontage)
        assert isinstance(dig_py, DigMontage)
        dig_py, dig_bin = dig_py.dig, dig_bin.dig
        info_py = info_bin = None
    assert isinstance(dig_py, list)
    assert isinstance(dig_bin, list)
    dig_py = sorted(dig_py, key=_dig_sort_key)
    dig_bin = sorted(dig_bin, key=_dig_sort_key)
    assert len(dig_py) == len(dig_bin)
    for ii, (d_py, d_bin) in enumerate(zip(dig_py[:limit], dig_bin[:limit])):
        for key in ("ident", "kind", "coord_frame"):
            assert d_py[key] == d_bin[key], key
        assert_allclose(
            d_py["r"],
            d_bin["r"],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Failure on {ii}:\n{d_py['r']}\n{d_bin['r']}",
        )
    if any(d["kind"] == FIFF.FIFFV_POINT_EXTRA for d in dig_py) and info_py is not None:
        r_bin, o_head_bin, o_dev_bin = fit_sphere_to_headshape(
            info_bin, units="m", verbose="error"
        )
        r_py, o_head_py, o_dev_py = fit_sphere_to_headshape(
            info_py, units="m", verbose="error"
        )
        assert_allclose(r_py, r_bin, atol=1e-6)
        assert_allclose(o_dev_py, o_dev_bin, rtol=1e-5, atol=1e-6)
        assert_allclose(o_head_py, o_head_bin, rtol=1e-5, atol=1e-6)


def _click_ch_name(fig, ch_index=0, button=1):
    """Click on a channel name in a raw/epochs/ICA browse-style plot."""
    from ..viz.utils import _fake_click

    fig.canvas.draw()
    text = fig.mne.ax_main.get_yticklabels()[ch_index]
    bbox = text.get_window_extent()
    x = bbox.intervalx.mean()
    y = bbox.intervaly.mean()
    _fake_click(fig, fig.mne.ax_main, (x, y), xform="pix", button=button)


def _get_suptitle(fig):
    """Get fig suptitle (shim for matplotlib < 3.8.0)."""
    # TODO: obsolete when minimum MPL version is 3.8
    if check_version("matplotlib", "3.8"):
        return fig.get_suptitle()
    else:
        # unreliable hack; should work in most tests as we rarely use `sup_{x,y}label`
        return fig.texts[0].get_text()
