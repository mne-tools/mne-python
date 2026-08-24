"""Replicate the decoding portion of the Time-GAL ERP example with MNE.

Run from the repository root, supplying the directory containing the original
``Condition_Pleasant.mat`` and ``Condition_Unpleasant.mat`` files::

    python reference_implementation/tutorial_gal_decoding.py \
        /path/to/q56ns-osfstorage-archive \
        --reference-result /path/to/resultsTimeGAL_IAPS_ERP.mat

The data are already CSD-transformed by the original study. This tutorial does
not compute CSD, source estimates, temporal correlations, or Time-GAL masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .gal_decoding import (
        compare_with_matlab,
        compute_gal_scores,
        create_hydrocel_info,
        legacy_gal_inference,
        load_erp_data,
        load_matlab_gal_result,
    )
except ImportError:
    from gal_decoding import (
        compare_with_matlab,
        compute_gal_scores,
        create_hydrocel_info,
        legacy_gal_inference,
        load_erp_data,
        load_matlab_gal_result,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=Path,
        help=(
            "Directory containing Condition_Pleasant.mat and "
            "Condition_Unpleasant.mat."
        ),
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        help="Optional MATLAB resultsTimeGAL_IAPS_ERP.mat file for a parity check.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Workers used to fit the sensor-specific LDA estimators.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory in which to save GAL figures as PNG files.",
    )
    return parser.parse_args()


def _plot_results(scores, inference):
    import mne

    mean_scores = scores.mean(axis=0)
    diagonal = np.diag(mean_scores)
    chance = 0.5
    limit = np.max(np.abs(mean_scores - chance))

    fig, ax = plt.subplots(layout="constrained")
    image = ax.imshow(
        mean_scores,
        origin="lower",
        cmap="RdBu_r",
        vmin=chance - limit,
        vmax=chance + limit,
    )
    ax.contour(inference.positive, levels=[0.5], colors="black", linewidths=0.8)
    ax.contour(inference.negative, levels=[0.5], colors="crimson", linewidths=0.8)
    ax.set(
        xlabel="Test sensor",
        ylabel="Training sensor",
        title="Generalization across location (held-out-subject accuracy)",
    )
    fig.colorbar(image, ax=ax, label="Accuracy")

    topo_fig, topo_ax = plt.subplots(layout="constrained")
    image, _ = mne.viz.plot_topomap(
        diagonal,
        create_hydrocel_info(),
        axes=topo_ax,
        show=False,
        cmap="Reds",
    )
    topo_ax.set_title("Within-sensor GAL accuracy")
    topo_fig.colorbar(image, ax=topo_ax, label="Accuracy")
    return fig, topo_fig


def main():
    """Run the original-data GAL decoding tutorial."""
    args = _parse_args()
    gal_data = load_erp_data(args.data_dir)
    scores = compute_gal_scores(
        gal_data.X, gal_data.y, gal_data.groups, n_jobs=args.n_jobs
    )
    inference = legacy_gal_inference(scores)
    print(f"GAL score shape: {scores.shape}")
    print(f"Legacy corrected alpha: {inference.alpha_corrected:.6f}")
    print(f"Above-chance cells: {inference.positive.sum()}")
    print(f"Below-chance cells: {inference.negative.sum()}")

    if args.reference_result is not None:
        reference = load_matlab_gal_result(args.reference_result)
        try:
            report = compare_with_matlab(scores, inference, reference, gal_data.groups)
        except ValueError as err:
            print(f"MATLAB parity check unavailable: {err}")
        else:
            print(f"Score agreement: {report.scores_match}")
            print(f"Positive-mask agreement: {report.positive_masks_match}")
            print(f"Negative-mask agreement: {report.negative_masks_match}")
            print(f"Maximum score difference: {report.maximum_score_difference:.6f}")

    figures = _plot_results(scores, inference)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for figure, name in zip(figures, ("gal_matrix", "gal_topography"), strict=True):
            figure.savefig(args.output_dir / f"{name}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
