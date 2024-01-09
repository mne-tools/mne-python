from __future__ import annotations

from typing import TYPE_CHECKING

from mne.utils import _validate_type

if TYPE_CHECKING:
    from re import Pattern
    from typing import Optional

    import numpy as np
    from numpy.typing import DTypeLike, NDArray

    from .. import Info

    ScalarIntType: tuple[DTypeLike, ...] = (np.int8, np.int16, np.int32, np.int64)


# fmt: off
def pick_ch_names_to_idx(
    ch_names: list[str] | tuple[str] | set[str],
    picks: Optional[list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice],  # noqa: E501
    exclude: list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice,  # noqa: E501
) -> NDArray[np.int32]:
    """Pick on a list-like of channels with validation.

    Replaces:
    - pick_channels
    - pick_channel_regexp
    """
    _validate_type(ch_names, (list, tuple, set), "ch_names")
    ch_names = list(ch_names) if isinstance(ch_names, (set, tuple)) else ch_names
    exclude = _ensure_int_array_pick_exclude_with_ch_names(ch_names, exclude, "exclude")
    if picks is None or picks == "all":
        picks = np.arange(len(ch_names))
    else:
        picks = _ensure_int_array_pick_exclude_with_ch_names(ch_names, picks, "picks")
    return np.setdiff1d(picks, exclude, assume_unique=True).astype(np.int32)


def _ensure_int_array_pick_exclude_with_ch_names(
    ch_names: list[str],
    var: list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice,  # noqa: E501
    var_name: str
) -> NDArray[np.int32]:
    pass


def pick_info_to_idx(
    info: Info,
    picks: Optional[list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice],  # noqa: E501
    exclude: list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice,  # noqa: E501
) -> NDArray[np.int32]:
    """Pick on an info with validation.

    Replaces:
    - pick_channels
    - pick_channels_regexp
    - pick_types
    """
    _validate_type(info, Info, "info")
    if exclude == "bads":
        exclude = info["bads"]
    else:
        exclude = _ensure_int_array_pick_exclude_with_info(info, exclude, "exclude")
    if picks is None or picks == "all":
        picks = np.arange(len(info["ch_names"]))
    elif picks == "data":
        return _pick_data_to_idx(info, exclude)
    else:
        picks = _ensure_int_array_pick_exclude_with_info(info, picks, "picks")
    return np.setdiff1d(picks, exclude, assume_unique=True).astype(np.int32)


def _pick_data_to_idx(info: Info, exclude: NDArray[np.int32]):
    """Pick all data channels without validation."""
    pass


def _ensure_int_array_pick_exclude_with_info(
    info: Info,
    var: list[str | int] | tuple[str | int] | set[str | int] | NDArray[+ScalarIntType] | str | int | Pattern | slice,  # noqa: E501
    var_name: str
) -> NDArray[np.int32]:
    pass
# fmt: on
