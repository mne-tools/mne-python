"""The config functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import atexit
import json
import multiprocessing
import os
import os.path as op
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from functools import partial
from importlib import import_module
from pathlib import Path

from .check import _validate_type, _check_qt_version, _check_option, _check_fname
from .docs import fill_doc
from .misc import _pl
from ._logging import warn, logger


_temp_home_dir = None


def set_cache_dir(cache_dir):
    """Set the directory to be used for temporary file storage.

    This directory is used by joblib to store memmapped arrays,
    which reduces memory requirements and speeds up parallel
    computation.

    Parameters
    ----------
    cache_dir : str or None
        Directory to use for temporary file storage. None disables
        temporary file storage.
    """
    if cache_dir is not None and not op.exists(cache_dir):
        raise OSError("Directory %s does not exist" % cache_dir)

    set_config("MNE_CACHE_DIR", cache_dir, set_env=False)


def set_memmap_min_size(memmap_min_size):
    """Set the minimum size for memmaping of arrays for parallel processing.

    Parameters
    ----------
    memmap_min_size : str or None
        Threshold on the minimum size of arrays that triggers automated memory
        mapping for parallel processing, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays.
    """
    _validate_type(memmap_min_size, (str, None), "memmap_min_size")
    if memmap_min_size is not None:
        if memmap_min_size[-1] not in ["K", "M", "G"]:
            raise ValueError(
                "The size has to be given in kilo-, mega-, or "
                f"gigabytes, e.g., 100K, 500M, 1G, got {repr(memmap_min_size)}"
            )

    set_config("MNE_MEMMAP_MIN_SIZE", memmap_min_size, set_env=False)


# List the known configuration values
_known_config_types = {
    "MNE_3D_OPTION_ANTIALIAS": (
        "bool, whether to use full-screen antialiasing in 3D plots"
    ),
    "MNE_3D_OPTION_DEPTH_PEELING": "bool, whether to use depth peeling in 3D plots",
    "MNE_3D_OPTION_MULTI_SAMPLES": (
        "int, number of samples to use for full-screen antialiasing"
    ),
    "MNE_3D_OPTION_SMOOTH_SHADING": ("bool, whether to use smooth shading in 3D plots"),
    "MNE_3D_OPTION_THEME": ("str, the color theme (light or dark) to use for 3D plots"),
    "MNE_BROWSE_RAW_SIZE": (
        "tuple, width and height of the raw browser window (in inches)"
    ),
    "MNE_BROWSER_BACKEND": (
        "str, the backend to use for the MNE Browse Raw window (qt or matplotlib)"
    ),
    "MNE_BROWSER_OVERVIEW_MODE": (
        "str, the overview mode to use in the MNE Browse Raw window )"
        "(see mne.viz.plot_raw for valid options)"
    ),
    "MNE_BROWSER_PRECOMPUTE": (
        "bool, whether to precompute raw data in the MNE Browse Raw window"
    ),
    "MNE_BROWSER_THEME": "str, the color theme (light or dark) to use for the browser",
    "MNE_BROWSER_USE_OPENGL": (
        "bool, whether to use OpenGL for rendering in the MNE Browse Raw window"
    ),
    "MNE_CACHE_DIR": "str, path to the cache directory for parallel execution",
    "MNE_COREG_ADVANCED_RENDERING": (
        "bool, whether to use advanced OpenGL rendering in mne coreg"
    ),
    "MNE_COREG_COPY_ANNOT": (
        "bool, whether to copy the annotation files during warping"
    ),
    "MNE_COREG_FULLSCREEN": "bool, whether to use full-screen mode in mne coreg",
    "MNE_COREG_GUESS_MRI_SUBJECT": (
        "bool, whether to guess the MRI subject in mne coreg"
    ),
    "MNE_COREG_HEAD_HIGH_RES": (
        "bool, whether to use high-res head surface in mne coreg"
    ),
    "MNE_COREG_HEAD_OPACITY": ("bool, the head surface opacity to use in mne coreg"),
    "MNE_COREG_HEAD_INSIDE": (
        "bool, whether to add an opaque inner scalp head surface to help "
        "occlude points behind the head in mne coreg"
    ),
    "MNE_COREG_INTERACTION": (
        "str, interaction style in mne coreg (trackball or terrain)"
    ),
    "MNE_COREG_MARK_INSIDE": (
        "bool, whether to mark points inside the head surface in mne coreg"
    ),
    "MNE_COREG_PREPARE_BEM": (
        "bool, whether to prepare the BEM solution after warping in mne coreg"
    ),
    "MNE_COREG_ORIENT_TO_SURFACE": (
        "bool, whether to orient the digitization markers to the head surface "
        "in mne coreg"
    ),
    "MNE_COREG_SCALE_LABELS": (
        "bool, whether to scale the MRI labels during warping in mne coreg"
    ),
    "MNE_COREG_SCALE_BY_DISTANCE": (
        "bool, whether to scale the digitization markers by their distance from "
        "the scalp in mne coreg"
    ),
    "MNE_COREG_SCENE_SCALE": (
        "float, the scale factor of the 3D scene in mne coreg (default 0.16)"
    ),
    "MNE_COREG_WINDOW_HEIGHT": "int, window height for mne coreg",
    "MNE_COREG_WINDOW_WIDTH": "int, window width for mne coreg",
    "MNE_COREG_SUBJECTS_DIR": "str, path to the subjects directory for mne coreg",
    "MNE_CUDA_DEVICE": "int, CUDA device to use for GPU processing",
    "MNE_DATA": "str, default data directory",
    "MNE_DATASETS_BRAINSTORM_PATH": "str, path for brainstorm data",
    "MNE_DATASETS_EEGBCI_PATH": "str, path for EEGBCI data",
    "MNE_DATASETS_EPILEPSY_ECOG_PATH": "str, path for epilepsy_ecog data",
    "MNE_DATASETS_HF_SEF_PATH": "str, path for HF_SEF data",
    "MNE_DATASETS_MEGSIM_PATH": "str, path for MEGSIM data",
    "MNE_DATASETS_MISC_PATH": "str, path for misc data",
    "MNE_DATASETS_MTRF_PATH": "str, path for MTRF data",
    "MNE_DATASETS_SAMPLE_PATH": "str, path for sample data",
    "MNE_DATASETS_SOMATO_PATH": "str, path for somato data",
    "MNE_DATASETS_MULTIMODAL_PATH": "str, path for multimodal data",
    "MNE_DATASETS_FNIRS_MOTOR_PATH": "str, path for fnirs_motor data",
    "MNE_DATASETS_OPM_PATH": "str, path for OPM data",
    "MNE_DATASETS_SPM_FACE_DATASETS_TESTS": "str, path for spm_face data",
    "MNE_DATASETS_SPM_FACE_PATH": "str, path for spm_face data",
    "MNE_DATASETS_TESTING_PATH": "str, path for testing data",
    "MNE_DATASETS_VISUAL_92_CATEGORIES_PATH": "str, path for visual_92_categories data",
    "MNE_DATASETS_KILOWORD_PATH": "str, path for kiloword data",
    "MNE_DATASETS_FIELDTRIP_CMC_PATH": "str, path for fieldtrip_cmc data",
    "MNE_DATASETS_PHANTOM_4DBTI_PATH": "str, path for phantom_4dbti data",
    "MNE_DATASETS_LIMO_PATH": "str, path for limo data",
    "MNE_DATASETS_REFMEG_NOISE_PATH": "str, path for refmeg_noise data",
    "MNE_DATASETS_SSVEP_PATH": "str, path for ssvep data",
    "MNE_DATASETS_ERP_CORE_PATH": "str, path for erp_core data",
    "MNE_FORCE_SERIAL": "bool, force serial rather than parallel execution",
    "MNE_LOGGING_LEVEL": (
        "str or int, controls the level of verbosity of any function "
        "decorated with @verbose. See "
        "https://mne.tools/stable/auto_tutorials/intro/50_configure_mne.html#logging"
    ),
    "MNE_MEMMAP_MIN_SIZE": (
        "str, threshold on the minimum size of arrays passed to the workers that "
        "triggers automated memory mapping, e.g., 1M or 0.5G"
    ),
    "MNE_REPR_HTML": (
        "bool, represent some of our objects with rich HTML in a notebook "
        "environment"
    ),
    "MNE_SKIP_NETWORK_TESTS": (
        "bool, used in a test decorator (@requires_good_network) to skip "
        "tests that include large downloads"
    ),
    "MNE_SKIP_TESTING_DATASET_TESTS": (
        "bool, used in test decorators (@requires_spm_data, "
        "@requires_bstraw_data) to skip tests that require specific datasets"
    ),
    "MNE_STIM_CHANNEL": "string, the default channel name for mne.find_events",
    "MNE_TQDM": (
        'str, either "tqdm", "tqdm.auto", or "off". Controls presence/absence '
        "of progress bars"
    ),
    "MNE_USE_CUDA": "bool, use GPU for filtering/resampling",
    "MNE_USE_NUMBA": (
        "bool, use Numba just-in-time compiler for some of our intensive "
        "computations"
    ),
    "SUBJECTS_DIR": "path-like, directory of freesurfer MRI files for each subject",
}

# These allow for partial matches, e.g. 'MNE_STIM_CHANNEL_1' is okay key
_known_config_wildcards = (
    "MNE_STIM_CHANNEL",  # can have multiple stim channels
    "MNE_DATASETS_FNIRS",  # mne-nirs
    "MNE_NIRS",  # mne-nirs
    "MNE_KIT2FIFF",  # mne-kit-gui
)


def _load_config(config_path, raise_error=False):
    """Safely load a config file."""
    with open(config_path, "r") as fid:
        try:
            config = json.load(fid)
        except ValueError:
            # No JSON object could be decoded --> corrupt file?
            msg = (
                "The MNE-Python config file (%s) is not a valid JSON "
                "file and might be corrupted" % config_path
            )
            if raise_error:
                raise RuntimeError(msg)
            warn(msg)
            config = dict()
    return config


def get_config_path(home_dir=None):
    r"""Get path to standard mne-python config file.

    Parameters
    ----------
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.

    Returns
    -------
    config_path : str
        The path to the mne-python configuration file. On windows, this
        will be '%USERPROFILE%\.mne\mne-python.json'. On every other
        system, this will be ~/.mne/mne-python.json.
    """
    val = op.join(_get_extra_data_path(home_dir=home_dir), "mne-python.json")
    return val


def get_config(key=None, default=None, raise_error=False, home_dir=None, use_env=True):
    """Read MNE-Python preferences from environment or config file.

    Parameters
    ----------
    key : None | str
        The preference key to look for. The os environment is searched first,
        then the mne-python config file is parsed.
        If None, all the config parameters present in environment variables or
        the path are returned. If key is an empty string, a list of all valid
        keys (but not values) is returned.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    use_env : bool
        If True, consider env vars, if available.
        If False, only use MNE-Python configuration file values.

        .. versionadded:: 0.18

    Returns
    -------
    value : dict | str | None
        The preference key value.

    See Also
    --------
    set_config
    """
    _validate_type(key, (str, type(None)), "key", "string or None")

    if key == "":
        # These are str->str (immutable) so we should just copy the dict
        # itself, no need for deepcopy
        return _known_config_types.copy()

    # first, check to see if key is in env
    if use_env and key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in mne-python config file
    config_path = get_config_path(home_dir=home_dir)
    if not op.isfile(config_path):
        config = {}
    else:
        config = _load_config(config_path)

    if key is None:
        # update config with environment variables
        if use_env:
            env_keys = set(config).union(_known_config_types).intersection(os.environ)
            config.update({key: os.environ[key] for key in env_keys})
        return config
    elif raise_error is True and key not in config:
        loc_env = "the environment or in the " if use_env else ""
        meth_env = (
            ('either os.environ["%s"] = VALUE for a temporary ' "solution, or " % key)
            if use_env
            else ""
        )
        extra_env = (
            " You can also set the environment variable before " "running python."
            if use_env
            else ""
        )
        meth_file = (
            'mne.utils.set_config("%s", VALUE, set_env=True) '
            "for a permanent one" % key
        )
        raise KeyError(
            'Key "%s" not found in %s'
            "the mne-python config file (%s). "
            "Try %s%s.%s" % (key, loc_env, config_path, meth_env, meth_file, extra_env)
        )
    else:
        return config.get(key, default)


def set_config(key, value, home_dir=None, set_env=True):
    """Set a MNE-Python preference key in the config file and environment.

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    set_env : bool
        If True (default), update :data:`os.environ` in addition to
        updating the MNE-Python config file.

    See Also
    --------
    get_config
    """
    _validate_type(key, "str", "key")
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    _validate_type(value, (str, "path-like", type(None)), "value")
    if value is not None:
        value = str(value)

    if key not in _known_config_types and not any(
        key.startswith(k) for k in _known_config_wildcards
    ):
        warn('Setting non-standard config type: "%s"' % key)

    # Read all previous values
    config_path = get_config_path(home_dir=home_dir)
    if op.isfile(config_path):
        config = _load_config(config_path, raise_error=True)
    else:
        config = dict()
        logger.info(
            "Attempting to create new mne-python configuration "
            "file:\n%s" % config_path
        )
    if value is None:
        config.pop(key, None)
        if set_env and key in os.environ:
            del os.environ[key]
    else:
        config[key] = value
        if set_env:
            os.environ[key] = value

    # Write all values. This may fail if the default directory is not
    # writeable.
    directory = op.dirname(config_path)
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, "w") as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


def _get_extra_data_path(home_dir=None):
    """Get path to extra data (config, tables, etc.)."""
    global _temp_home_dir
    if home_dir is None:
        home_dir = os.environ.get("_MNE_FAKE_HOME_DIR")
    if home_dir is None:
        # this has been checked on OSX64, Linux64, and Win32
        if "nt" == os.name.lower():
            APPDATA_DIR = os.getenv("APPDATA")
            USERPROFILE_DIR = os.getenv("USERPROFILE")
            if APPDATA_DIR is not None and op.isdir(
                op.join(APPDATA_DIR, ".mne")
            ):  # backward-compat
                home_dir = APPDATA_DIR
            elif USERPROFILE_DIR is not None:
                home_dir = USERPROFILE_DIR
            else:
                raise FileNotFoundError(
                    "The USERPROFILE environment variable is not set, cannot "
                    "determine the location of the MNE-Python configuration "
                    "folder"
                )
            del APPDATA_DIR, USERPROFILE_DIR
        else:
            # This is a more robust way of getting the user's home folder on
            # Linux platforms (not sure about OSX, Unix or BSD) than checking
            # the HOME environment variable. If the user is running some sort
            # of script that isn't launched via the command line (e.g. a script
            # launched via Upstart) then the HOME environment variable will
            # not be set.
            if os.getenv("MNE_DONTWRITE_HOME", "") == "true":
                if _temp_home_dir is None:
                    _temp_home_dir = tempfile.mkdtemp()
                    atexit.register(
                        partial(shutil.rmtree, _temp_home_dir, ignore_errors=True)
                    )
                home_dir = _temp_home_dir
            else:
                home_dir = os.path.expanduser("~")

        if home_dir is None:
            raise ValueError(
                "mne-python config file path could "
                "not be determined, please report this "
                "error to mne-python developers"
            )

    return op.join(home_dir, ".mne")


def get_subjects_dir(subjects_dir=None, raise_error=False):
    """Safely use subjects_dir input to return SUBJECTS_DIR.

    Parameters
    ----------
    subjects_dir : path-like | None
        If a value is provided, return subjects_dir. Otherwise, look for
        SUBJECTS_DIR config and return the result.
    raise_error : bool
        If True, raise a KeyError if no value for SUBJECTS_DIR can be found
        (instead of returning None).

    Returns
    -------
    value : Path | None
        The SUBJECTS_DIR value.
    """
    if subjects_dir is None:
        subjects_dir = get_config("SUBJECTS_DIR", raise_error=raise_error)
    if subjects_dir is not None:
        subjects_dir = _check_fname(
            fname=subjects_dir,
            overwrite="read",
            must_exist=True,
            need_dir=True,
            name="subjects_dir",
        )
    return subjects_dir


@fill_doc
def _get_stim_channel(stim_channel, info, raise_error=True):
    """Determine the appropriate stim_channel.

    First, 'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2', etc.
    are read. If these are not found, it will fall back to 'STI 014' if
    present, then fall back to the first channel of type 'stim', if present.

    Parameters
    ----------
    stim_channel : str | list of str | None
        The stim channel selected by the user.
    %(info_not_none)s

    Returns
    -------
    stim_channel : str | list of str
        The name of the stim channel(s) to use
    """
    if stim_channel is not None:
        if not isinstance(stim_channel, list):
            _validate_type(stim_channel, "str", "Stim channel")
            stim_channel = [stim_channel]
        for channel in stim_channel:
            _validate_type(channel, "str", "Each provided stim channel")
        return stim_channel

    stim_channel = list()
    ch_count = 0
    ch = get_config("MNE_STIM_CHANNEL")
    while ch is not None and ch in info["ch_names"]:
        stim_channel.append(ch)
        ch_count += 1
        ch = get_config("MNE_STIM_CHANNEL_%d" % ch_count)
    if ch_count > 0:
        return stim_channel

    if "STI101" in info["ch_names"]:  # combination channel for newer systems
        return ["STI101"]
    if "STI 014" in info["ch_names"]:  # for older systems
        return ["STI 014"]

    from ..io.pick import pick_types

    stim_channel = pick_types(info, meg=False, ref_meg=False, stim=True)
    if len(stim_channel) > 0:
        stim_channel = [info["ch_names"][ch_] for ch_ in stim_channel]
    elif raise_error:
        raise ValueError(
            "No stim channels found. Consider specifying them "
            "manually using the 'stim_channel' parameter."
        )
    return stim_channel


def _get_root_dir():
    """Get as close to the repo root as possible."""
    root_dir = Path(__file__).parent.parent.expanduser().absolute()
    up_dir = root_dir.parent
    if (up_dir / "setup.py").is_file() and all(
        (up_dir / x).is_dir() for x in ("mne", "examples", "doc")
    ):
        root_dir = up_dir
    return root_dir


def _get_numpy_libs():
    bad_lib = "unknown linalg bindings"
    try:
        from threadpoolctl import threadpool_info
    except Exception as exc:
        return bad_lib + f" (threadpoolctl module not found: {exc})"
    pools = threadpool_info()
    rename = dict(
        openblas="OpenBLAS",
        mkl="MKL",
    )
    for pool in pools:
        if pool["internal_api"] in ("openblas", "mkl"):
            return (
                f'{rename[pool["internal_api"]]} '
                f'{pool["version"]} with '
                f'{pool["num_threads"]} thread{_pl(pool["num_threads"])}'
            )
    return bad_lib


_gpu_cmd = """\
from pyvista import GPUInfo; \
gi = GPUInfo(); \
print(gi.version); \
print(gi.renderer)"""


def _get_gpu_info():
    # Once https://github.com/pyvista/pyvista/pull/2250 is merged and PyVista
    # does a release, we can triage based on version > 0.33.2
    proc = subprocess.run(
        [sys.executable, "-c", _gpu_cmd], check=False, capture_output=True
    )
    out = proc.stdout.decode().strip().replace("\r", "").split("\n")
    if proc.returncode or len(out) != 2:
        return None, None
    return out


def sys_info(fid=None, show_paths=False, *, dependencies="user", unicode=True):
    """Print system information.

    This function prints system information useful when triaging bugs.

    Parameters
    ----------
    fid : file-like | None
        The file to write to. Will be passed to :func:`print()`. Can be None to
        use :data:`sys.stdout`.
    show_paths : bool
        If True, print paths for each module.
    dependencies : 'user' | 'developer'
        Show dependencies relevant for users (default) or for developers
        (i.e., output includes additional dependencies).
    unicode : bool
        Include Unicode symbols in output.

        .. versionadded:: 0.24
    """
    _validate_type(dependencies, str)
    _check_option("dependencies", dependencies, ("user", "developer"))
    ljust = 24 if dependencies == "developer" else 21
    platform_str = platform.platform()
    if platform.system() == "Darwin" and sys.version_info[:2] < (3, 8):
        # platform.platform() in Python < 3.8 doesn't call
        # platform.mac_ver() if we're on Darwin, so we don't get a nice macOS
        # version number. Therefore, let's do this manually here.
        macos_ver = platform.mac_ver()[0]
        macos_architecture = re.findall("Darwin-.*?-(.*)", platform_str)
        if macos_architecture:
            macos_architecture = macos_architecture[0]
            platform_str = f"macOS-{macos_ver}-{macos_architecture}"
        del macos_ver, macos_architecture

    out = partial(print, end="", file=fid)
    out("Platform".ljust(ljust) + platform_str + "\n")
    out("Python".ljust(ljust) + str(sys.version).replace("\n", " ") + "\n")
    out("Executable".ljust(ljust) + sys.executable + "\n")
    out("CPU".ljust(ljust) + f"{platform.processor()} ")
    out(f"({multiprocessing.cpu_count()} cores)\n")
    out("Memory".ljust(ljust))
    try:
        import psutil
    except ImportError:
        out('Unavailable (requires "psutil" package)')
    else:
        out(f"{psutil.virtual_memory().total / float(2 ** 30):0.1f} GB\n")
    out("\n")
    ljust -= 3  # account for +/- symbols
    libs = _get_numpy_libs()
    unavailable = []
    use_mod_names = (
        "# Core",
        "mne",
        "numpy",
        "scipy",
        "matplotlib",
        "pooch",
        "jinja2",
        "",
        "# Numerical (optional)",
        "sklearn",
        "numba",
        "nibabel",
        "nilearn",
        "dipy",
        "openmeeg",
        "cupy",
        "pandas",
        "",
        "# Visualization (optional)",
        "pyvista",
        "pyvistaqt",
        "ipyvtklink",
        "vtk",
        "qtpy",
        "ipympl",
        "pyqtgraph",
        "mne-qt-browser",
        "",
        "# Ecosystem (optional)",
        "mne-bids",
        "mne-nirs",
        "mne-features",
        "mne-connectivity",
        "mne-icalabel",
        "mne-bids-pipeline",
        "",
    )
    if dependencies == "developer":
        use_mod_names += (
            "# Testing",
            "pytest",
            "nbclient",
            "numpydoc",
            "flake8",
            "pydocstyle",
            "",
            "# Documentation",
            "sphinx",
            "sphinx-gallery",
            "pydata-sphinx-theme",
            "",
        )
    try:
        unicode = unicode and (sys.stdout.encoding.lower().startswith("utf"))
    except Exception:  # in case someone overrides sys.stdout in an unsafe way
        unicode = False
    for mi, mod_name in enumerate(use_mod_names):
        # upcoming break
        if mod_name == "":  # break
            if unavailable:
                out("└☐ " if unicode else " - ")
                out("unavailable".ljust(ljust))
                out(f"{', '.join(unavailable)}\n")
                unavailable = []
            if mi != len(use_mod_names) - 1:
                out("\n")
            continue
        elif mod_name.startswith("# "):  # header
            mod_name = mod_name.replace("# ", "")
            out(f"{mod_name}\n")
            continue
        pre = "├"
        last = use_mod_names[mi + 1] == "" and not unavailable
        if last:
            pre = "└"
        try:
            mod = import_module(mod_name.replace("-", "_"))
        except Exception:
            unavailable.append(mod_name)
        else:
            out(f"{pre}☑ " if unicode else " + ")
            out(f"{mod_name}".ljust(ljust))
            if mod_name == "vtk":
                vtk_version = mod.vtkVersion()
                # 9.0 dev has VersionFull but 9.0 doesn't
                for attr in ("GetVTKVersionFull", "GetVTKVersion"):
                    if hasattr(vtk_version, attr):
                        version = getattr(vtk_version, attr)()
                        if version != "":
                            out(version)
                            break
                else:
                    out("unknown")
            else:
                out(mod.__version__.lstrip("v"))
            if mod_name == "numpy":
                out(f" ({libs})")
            elif mod_name == "qtpy":
                version, api = _check_qt_version(return_api=True)
                out(f" ({api}={version})")
            elif mod_name == "matplotlib":
                out(f" (backend={mod.get_backend()})")
            elif mod_name == "pyvista":
                version, renderer = _get_gpu_info()
                if version is None:
                    out(" (OpenGL unavailable)")
                else:
                    out(f" (OpenGL {version} via {renderer})")
            if show_paths:
                if last:
                    pre = "   "
                elif unicode:
                    pre = "│  "
                else:
                    pre = " | "
                out(f'\n{pre}{" " * ljust}{op.dirname(mod.__file__)}')
            out("\n")
