"""Stage the data the JupyterLite notebooks read, and the MNE wheel they install.

The setup cell fetches files over HTTP, so ``conf.py`` serves this subset of
the datasets at the docs root (``/mne_data/...``), copied from ``~/mne_data``.
A badged notebook has to find everything it reads here; ``JUPYTERLITE_EXCLUDE``
in ``conf.py`` lists the ones that do not.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import shutil
from pathlib import Path

from build_lite_wheel import build_wheel, find_wheels
from mne_doc_utils import sphinx_logger

import mne

MNE_DATA = Path(os.path.expanduser("~/mne_data"))
# Refuse anything past this rather than bloat the deploy; the largest file that
# has to be served (sample_audvis_raw.fif) is 128 MB.
MAX_FILE_MB = 150

# MNE-sample-data files the badged notebooks read. Anything not here is a 404
# in the browser, so this list follows the tutorials, not the dataset.
SAMPLE_FILES = [
    "version.txt",
    "MEG/sample/sample_audvis_raw.fif",
    "MEG/sample/sample_audvis_filt-0-40_raw.fif",
    "MEG/sample/sample_audvis_raw-eve.fif",
    "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif",
    "MEG/sample/sample_audvis_ecg-proj.fif",
    "MEG/sample/sample_audvis-ave.fif",
    "MEG/sample/sample_audvis-cov.fif",
    "MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif",
    "MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif",
    "MEG/sample/sample_audvis-meg-oct-6-fwd.fif",
    "MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif",
    "MEG/sample/ernoise_raw.fif",
    "MEG/sample/sample_audvis-no-filter-ave.fif",
    "MEG/sample/sample_audvis_raw-trans.fif",
    "MEG/sample/sample_audvis-shrunk-cov.fif",
    "MEG/sample/sample_audvis-meg-lh.stc",
    "MEG/sample/sample_audvis-meg-rh.stc",
    "MEG/sample/sample_audvis-meg-eeg-lh.stc",
    "MEG/sample/sample_audvis-meg-eeg-rh.stc",
    "MEG/sample/sample_audvis_ecg-eve.fif",
    "subjects/sample/mri/T1.mgz",
    "subjects/sample/mri/aseg.mgz",
    # read_talxfm builds this path itself; plot_alignment estimates the MRI
    # fiducials from it
    "subjects/sample/mri/transforms/talairach.xfm",
    "subjects/sample/bem/sample-oct-6-src.fif",
    # head and skull surfaces for plot_alignment, in the order MNE tries them
    # so the browser draws the same one the rendered docs do; there is no
    # sample-head-dense.fif, and lh.seghead is the dense fallback. The .surf
    # files are symlinks into bem/flash/, which copy2 follows.
    "subjects/sample/bem/outer_skin.surf",
    "subjects/sample/bem/outer_skull.surf",
    "subjects/sample/bem/inner_skull.surf",
    "subjects/sample/bem/sample-head.fif",
    "subjects/sample/surf/lh.seghead",
    # the single-layer BEM solution; the three-layer one is 237 MB, so the
    # notebooks needing it are excluded instead
    "subjects/sample/bem/sample-5120-bem-sol.fif",
    # the fsaverage source space ships inside MNE-sample-data
    "subjects/fsaverage/bem/fsaverage-ico-5-src.fif",
    "subjects/sample/surf/rh.pial",
    "subjects/sample/surf/lh.pial",
    "subjects/sample/surf/rh.white",
    "subjects/sample/surf/lh.white",
    "subjects/sample/surf/rh.inflated",
    "subjects/sample/surf/lh.inflated",
    "subjects/sample/surf/rh.curv",
    "subjects/sample/surf/lh.curv",
    # setup_source_space reads surf/{hemi}.sphere by a path it builds itself
    "subjects/sample/surf/lh.sphere",
    "subjects/sample/surf/rh.sphere",
    "subjects/sample/label/lh.aparc.annot",
    "subjects/sample/label/rh.aparc.annot",
    # the auditory/visual ROIs, whose names about nine notebooks build with an
    # f-string, so a scan of the tutorial text never sees them
    "MEG/sample/labels/Aud-lh.label",
    "MEG/sample/labels/Aud-rh.label",
    "MEG/sample/labels/Vis-lh.label",
    "MEG/sample/labels/Vis-rh.label",
]

# (dataset folder, files): each used by one or two notebooks that read only a
# couple of files out of it. A full docs build downloads all of these datasets.
DATASET_FILES = {
    "MNE-sample-data": SAMPLE_FILES,
    "ssvep-example-data": [
        f"sub-02/ses-01/eeg/sub-02_ses-01_task-ssvep_eeg{s}"
        for s in (".vhdr", ".eeg", ".vmrk")
    ],
    "MNE-misc-data": [
        "xdf/sub-P001_ses-S004_task-Default_run-001_eeg_a2.xdf",
        "movement/simulated_quats.pos",
        "movement/simulated_movement_raw.fif",
        "movement/simulated_stationary_raw.fif",
        "eyetracking/eyelink/px_textpage_ws.asc",
        "eyetracking/eyelink/HREF_textpage_ws.asc",
    ],
    "MNE-eyelink-data": [
        "freeviewing/sub-01_task-freeview_eyetrack.asc",
        "freeviewing/stim/naturalistic.png",
    ],
    "MNE-kiloword-data": ["kword_metadata-epo.fif"],
    "mTRF_1.5": ["speech_data.mat"],
    # exactly the runs tools/circleci_download.sh fetches (subject 1 runs
    # 3/6/10/14, run 3 for subjects 2-4); notebooks wanting runs 1 or 2 are
    # excluded instead
    "MNE-eegbci-data": [
        f"files/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.edf"
        for s, r in ((1, 3), (1, 6), (1, 10), (1, 14), (2, 3), (3, 3), (4, 3))
    ],
}
# whole folders, for readers handed a directory rather than a file
# (read_raw_nirx); a manifest is left for the setup cell
DATASET_TREES = [("MNE-fNIRS-motor-data", "Participant-1")]
# somato, ERP-CORE, refmeg_noise and the rest of the testing and eyelink
# datasets are deliberately not served; see the size tiers in conf.py


def _copy(src, dst):
    """Copy ``src`` to ``dst`` unless a same-sized copy is already there."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def stage_lite_data(dst_base):
    """Copy the served data subset under ``dst_base`` and build the MNE wheel."""
    dst_base = Path(dst_base)
    n_copied = n_missing = 0
    for folder, rels in DATASET_FILES.items():
        for rel in rels:
            src = MNE_DATA / folder / rel
            if not src.exists():
                sphinx_logger.info(f"[JupyterLite]   MISSING {folder}/{rel}")
                n_missing += 1
            elif src.stat().st_size / 1e6 > MAX_FILE_MB:
                sphinx_logger.info(f"[JupyterLite]   SKIPPED {folder}/{rel} (too big)")
            else:
                n_copied += _copy(src, dst_base / folder / rel)
    for folder, rel_dir in DATASET_TREES:
        src_dir = MNE_DATA / folder / rel_dir
        if not src_dir.is_dir():
            sphinx_logger.info(f"[JupyterLite]   MISSING {folder}/{rel_dir}")
            n_missing += 1
            continue
        # zero-byte members do not survive the artifact upload, so listing
        # them would only yield 404s
        names = [
            str(f.relative_to(src_dir))
            for f in sorted(src_dir.rglob("*"))
            if f.is_file() and 0 < f.stat().st_size / 1e6 <= MAX_FILE_MB
        ]
        for name in names:
            n_copied += _copy(src_dir / name, dst_base / folder / rel_dir / name)
        (dst_base / folder / rel_dir / "_lite_manifest.txt").write_text(
            "\n".join(names)
        )
    # The logging tutorial reads a KIT file that lives inside the package under
    # mne/io/kit/tests/, which the wheel leaves out, so serve it and let the
    # setup cell stage it back into the path the tutorial builds.
    kit = Path(mne.__file__).parent / "io" / "kit" / "tests" / "data" / "test.sqd"
    n_copied += _copy(kit, dst_base / "MNE-kit-testdata" / "test.sqd")
    sphinx_logger.info(
        f"[JupyterLite] Served data: {n_copied} files copied, {n_missing} missing"
    )
    # the development wheel, so the browser installs this MNE rather than the
    # PyPI release: doc/sphinxext/build_lite_wheel.py puts it in doc/pypi, where
    # the piplite addon indexes it; `make html` runs that first, so this is
    # only a fallback for a bare sphinx-build
    wheels = find_wheels() or build_wheel()
    sphinx_logger.info(f"[JupyterLite] MNE wheel for the browser kernel: {wheels}")
