"""
.. _ex-io-ant-impedances:

======================================
Getting impedances from ANT Neuro .cnt
======================================

The ``.cnt`` file format from ANT Neuro stores impedance information in the form of
triggers. The function :func:`mne.io.read_raw_ant` reads this information and mark the
time-segment during which an impedance measurement was performed as
:class:`~mne.Annotations` with the description set in the argument
``impedance_annotation``. However, it doesn't extract the impedance values themselves.
To do so, use the function ``antio.parser.read_triggers``.
"""

from antio import read_cnt
from antio.parser import read_triggers

from mne.datasets import testing
from mne.io import read_raw_ant

fname = testing.data_path() / "antio" / "CA_208" / "test_CA_208.cnt"
cnt = read_cnt(fname)
_, _, _, impedances, _ = read_triggers(cnt)

raw = read_raw_ant(fname)
impedances = [{ch: imp[k] for k, ch in enumerate(raw.ch_names)} for imp in impedances]
print(impedances[0])  # impendance measurement at the beginning of the recording
