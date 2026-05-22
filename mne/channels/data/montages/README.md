The idealized spherical montages (`spherical_1005`, `spherical_1010`, `spherical_1020`) are taken from the [eeg-positions](https://github.com/sappelhoff/eeg_positions) package, which provides these pre-computed locations as TSV files:

- https://github.com/sappelhoff/eeg_positions/blob/main/data/Nz-T10-Iz-T9/standard_1005_3D.tsv
- https://github.com/sappelhoff/eeg_positions/blob/main/data/Nz-T10-Iz-T9/standard_1010_3D.tsv
- https://github.com/sappelhoff/eeg_positions/blob/main/data/Nz-T10-Iz-T9/standard_1020_3D.tsv

The `colin27_*` montages are also based on the 10–05 system, but the idealized locations were fit on the Colin27 head model (a high-resolution MRI scan of a single subject), similar to the description in this tutorial:

https://www.fieldtriptoolbox.org/tutorial/sensor/opm_helmet_design/

They are available in [FieldTrip](https://github.com/fieldtrip/fieldtrip):

- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_1020.elc
- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_1005.elc
- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_alphabetic.elc
- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_postfixed.elc
- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_prefixed.elc
- https://github.com/fieldtrip/fieldtrip/blob/release/template/electrode/standard_primed.elc

The two `easycap-M1.txt` and `easycap-M10.txt` montages as well as the `GSN-HydroCel-*` montages are also taken from FieldTrip (https://github.com/fieldtrip/fieldtrip/tree/release/template/electrode).
