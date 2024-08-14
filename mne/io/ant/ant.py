import importlib

from ...utils import verbose


@verbose
def read_raw_ant(
    fname,
    eog=None,
    misc=r"BIP\d+",
    bipolars=None,
    impedance_annotation="impedance",
    *,
    verbose=None,
):
    r"""Reader for Raw ANT files in .cnt format.

    Parameters
    ----------
    fname : str | Path
        Path to the ANT raw file to load. The file should have the extension ``.cnt``.
    eog : str | None
        Regex pattern to find EOG channel labels. If None, no EOG channels are
        automatically detected.
    misc : str | None
        Regex pattern to find miscellaneous channels. If None, no miscellaneous channels
        are automatically detected. The default pattern ``"BIP\d+"`` will mark all
        bipolar channels as ``misc``.

        .. note::

            A bipolar channel might actually contain ECG, EOG or other signal types
            which might have a dedicated channel type in MNE-Python. In this case, use
            :meth:`mne.io.Raw.set_channel_types` to change the channel type of the
            channel.
    bipolars : list of str | tuple of str | None
        The list of channels to treat as bipolar EEG channels. Each element should be
        a string of the form ``'anode-cathode'`` or in ANT terminology as ``'label-
        reference'``. If None, all channels are interpreted as ``'eeg'`` channels
        referenced to the same reference electrode. Bipolar channels are treated
        as EEG channels with a special coil type in MNE-Python, see also
        :func:`mne.set_bipolar_reference`

        .. warning::

            Do not provide auxiliary channels in this argument, provide them in the
            ``eog`` and ``misc`` arguments.
    impedance_annotation : str
        The string to use for impedance annotations. Defaults to "impedance", however,
        the impedance measurement might mark the end of a segment and the beginning of
        a new segment, in which case a discontinuity similar to what
        :func:`mne.concatenate_raws` produces is present. In this case, it's better to
        include a `BAD_xxx` annotation to mark the discontinuity.

        .. note::

            Note that the impedance annotation will likely have a duration of ``0``.
            If the measurement marks a discontinuity, the duration should be modified to
            cover the discontinuity in its entirety.
    %(verbose)s

    Returns
    -------
    RawANT
        The ANT raw object containing the channel information, data and relevant
        :class:`~mne.Annotations`.
    """
    if importlib.util.find_spec("antio") is None:
        raise ImportError(
            "Missing optional dependency 'antio'. Use pip or conda to install 'antio'."
        )

    from antio.io import read_raw_ant as _read_raw_ant

    return _read_raw_ant(
        fname=fname,
        eog=eog,
        misc=misc,
        bipolars=bipolars,
        impedance_annotation=impedance_annotation,
        verbose=verbose,
    )
