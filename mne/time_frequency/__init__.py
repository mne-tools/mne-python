"""Time frequency analysis tools."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_stft": [
            "istft",
            "stft",
            "stftfreq",
        ],
        "_stockwell": [
            "tfr_array_stockwell",
            "tfr_stockwell",
        ],
        "ar": ["fit_iir_model_raw"],
        "csd": [
            "CrossSpectralDensity",
            "csd_array_fourier",
            "csd_array_morlet",
            "csd_array_multitaper",
            "csd_fourier",
            "csd_morlet",
            "csd_multitaper",
            "csd_tfr",
            "pick_channels_csd",
            "read_csd",
        ],
        "multitaper": [
            "dpss_windows",
            "psd_array_multitaper",
            "tfr_array_multitaper",
        ],
        "psd": ["psd_array_welch"],
        "spectrum": [
            "EpochsSpectrum",
            "EpochsSpectrumArray",
            "Spectrum",
            "SpectrumArray",
            "read_spectrum",
        ],
        "tfr": [
            "_BaseTFR",
            "AverageTFR",
            "EpochsTFR",
            "fwhm",
            "morlet",
            "read_tfrs",
            "tfr_array_morlet",
            "tfr_morlet",
            "tfr_multitaper",
            "write_tfrs",
        ],
    },
)
