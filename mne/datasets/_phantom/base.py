# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ...utils import _check_option, _validate_type, get_subjects_dir, verbose
from ..utils import _manifest_check_download

PHANTOM_MANIFEST_PATH = Path(__file__).parent


@verbose
def fetch_phantom(kind, subjects_dir=None, *, verbose=None):
    """Fetch and update a phantom subject.

    Parameters
    ----------
    kind : str
        The kind of phantom to fetch. Can only be ``'otaniemi'`` (default).
    %(subjects_dir)s
    %(verbose)s

    Returns
    -------
    subject_dir : pathlib.Path
        The resulting phantom subject directory.

    See Also
    --------
    mne.dipole.get_phantom_dipoles

    Notes
    -----
    This function is designed to provide a head surface and T1.mgz for
    the 32-dipole Otaniemi phantom. The VectorView/TRIUX phantom has the same
    basic outside geometry, but different internal dipole positions.

    Unlike most FreeSurfer subjects, the Otaniemi phantom scan was aligned
    to the "head" coordinate frame, so an identity head<->MRI :term:`trans`
    is appropriate.

    .. versionadded:: 0.24
    """
    phantoms = dict(
        otaniemi=dict(
            url="https://osf.io/j5czy/download?version=1",
            hash="42d17db5b1db3e30327ffb4cf2649de8",
        ),
    )
    _validate_type(kind, str, "kind")
    _check_option("kind", kind, list(phantoms))
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subject = f"phantom_{kind}"
    subject_dir = subjects_dir / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    _manifest_check_download(
        manifest_path=PHANTOM_MANIFEST_PATH / f"{subject}.txt",
        destination=subjects_dir,
        url=phantoms[kind]["url"],
        hash_=phantoms[kind]["hash"],
    )
    return subject_dir
