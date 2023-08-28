"""Functions for fetching remote datasets.

See :ref:`datasets` for more information.
"""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "fieldtrip_cmc",
        "brainstorm",
        "visual_92_categories",
        "kiloword",
        "eegbci",
        "hf_sef",
        "misc",
        "mtrf",
        "sample",
        "somato",
        "multimodal",
        "fnirs_motor",
        "opm",
        "spm_face",
        "testing",
        "_fake",
        "phantom_4dbti",
        "sleep_physionet",
        "limo",
        "refmeg_noise",
        "ssvep",
        "erp_core",
        "epilepsy_ecog",
        "eyelink",
        "ucl_opm_auditory",
    ],
    submod_attrs={
        "_fetch": ["fetch_dataset"],
        "_fsaverage.base": ["fetch_fsaverage"],
        "_infant.base": ["fetch_infant_template"],
        "_phantom.base": ["fetch_phantom"],
        "utils": [
            "_download_all_example_data",
            "fetch_hcp_mmp_parcellation",
            "fetch_aparc_sub_parcellation",
            "has_dataset",
        ],
    },
)
