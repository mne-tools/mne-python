# %%
def unifying_bads(
    list_instances,
):
    common_bad_channels = []
    # first check that each object is mne object
    inst_types = set(type(insts[0]))
    valid_types = (Raw, Epochs, Evoked, Spectrum, EpochsSpectrum)
    for inst in insts:
        _validate_type(inst, valid_types, "instance type")
        if type(inst) not in inst_types:
            raise ValueError("all insts must be the same type")
    # then interate through the objects to get ch names as set
    ch_set_1 = list_instances[0].info["bads"]
    common_bad_channels.extend(ch_set_1)
    all_bads = dict()
    for inst in insts:
        all_bads.update(dict.fromkeys(inst.info["bads"]))
    all_bads = list(all_bads)
    for inst in list_instances[1:]:
        ch_set_2 = set(inst.info["bads"])
        set_of_bads = set(common_bad_channels)
        new_bads = ch_set_2.difference(set_of_bads)

        if len(new_bads) > 1:
            common_bad_channels.extend(list(new_bads))

    new_instances = []

    for inst in list_instances:
        inst.info["bads"] = common_bad_channels
        new_instances.append(inst)

    return new_instances
