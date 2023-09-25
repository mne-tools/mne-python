#%%

from ..utils import _validate_type

from ..io import BaseRaw
from ..epochs import Epochs
from ..evoked import Evoked
from ..time_frequency.spectrum import BaseSpectrum

#%%
def unify_bad_channels(
    insts,
):
    # first check that each object is mne object
    inst_type = type(insts[0])
    valid_types = (BaseRaw, Epochs, Evoked, BaseSpectrum)
    for inst in insts:
        _validate_type(inst, valid_types , "instance type")
        if type(inst) != inst_type:
            raise ValueError("all insts must be the same type")
    # then interate through the objects to get ch names as set

    all_bads = dict()
    for inst in insts:
        all_bads.update(dict.fromkeys(inst.info["bads"]))
    all_bads = list(all_bads)


    new_instances = []

    for inst in insts:
        inst.info["bads"] = all_bads
        new_instances.append(inst)

    return new_instances

