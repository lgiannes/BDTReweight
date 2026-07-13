#!/usr/bin/env python3
"""
Script to plot the dispersion relation of a model from a NUISANCE flat tree.
This creates a 2D plot of pmiss_preFSI (magnitude) vs Emiss_preFSI.

The NUISANCE flat tree has branches:
- pmiss_preFSI_fX, pmiss_preFSI_fY, pmiss_preFSI_fZ
The magnitude of this vector is computed and plotted on the y-axis.
"""

import sys
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust path to your BDTReweight installation
sys.path.insert(0, '/Users/lorenzo/Minerva/fork_reweighting/')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree


def plot_dispersion_relation(input_file: str,
                              emiss_bins: np.ndarray = np.linspace(0, 1.0, 50),
                              pmiss_bins: np.ndarray = np.linspace(0, 1.0, 50),
                              mask: np.ndarray = None,
                              save_fig: str | None = None,
                              title: str = "Dispersion Relation: $p_{miss}^{preFSI}$ vs $E_{miss}^{preFSI}$") -> tuple:
    """
    Compute and plot the dispersion relation from a NUISANCE flat tree.

    Parameters
    ----------
    input_file : str
        Path to the NUISANCE flat tree ROOT file.
    emiss_bins : np.ndarray
        Bin edges for E_miss (GeV) on the x-axis.
    pmiss_bins : np.ndarray
        Bin edges for p_miss magnitude (GeV) on the y-axis.
    mask : np.ndarray, optional
        Boolean mask to select events (e.g., CCQELike events).
    save_fig : str | None
        If provided, save the produced figure to this path.
    title : str
        Title for the plot.

    Returns
    -------
    tuple
        (histogram_2d, emiss_edges, pmiss_edges, counts_2d)
    """

    # Load tree
    print("Loading NUISANCE flat tree...", end='', flush=True)
    tree = NuisanceFlatTree(input_file)
    print(" Done!")

    # Get the missing energy and missing momentum components
    print("Extracting branches...", end='', flush=True)
    emiss = tree.get_event_variable('Emiss_preFSI', mask=mask)
    pmiss_x = tree.get_event_variable('pmiss_preFSI_fX', mask=mask)
    pmiss_y = tree.get_event_variable('pmiss_preFSI_fY', mask=mask)
    pmiss_z = tree.get_event_variable('pmiss_preFSI_fZ', mask=mask)
    print(" Done!")

    # Convert awkward arrays to numpy and handle None/NaN values
    emiss = ak.fill_none(emiss, np.nan)
    emiss = ak.to_numpy(emiss)

    pmiss_x = ak.fill_none(pmiss_x, np.nan)
    pmiss_x = ak.to_numpy(pmiss_x)

    pmiss_y = ak.fill_none(pmiss_y, np.nan)
    pmiss_y = ak.to_numpy(pmiss_y)

    pmiss_z = ak.fill_none(pmiss_z, np.nan)
    pmiss_z = ak.to_numpy(pmiss_z)

    # Compute magnitude of missing momentum vector
    pmiss_mag = np.sqrt(pmiss_x**2 + pmiss_y**2 + pmiss_z**2)

    # Remove entries with NaN values
    valid = np.isfinite(emiss) & np.isfinite(pmiss_mag)
    if not np.any(valid):
        raise RuntimeError('No valid events found after NaN filtering.')

    emiss_vals = emiss[valid]
    pmiss_vals = pmiss_mag[valid]

    print(f"Number of valid events: {np.sum(valid)}")
    print(f"E_miss range: [{np.min(emiss_vals):.3f}, {np.max(emiss_vals):.3f}] GeV")
    print(f"p_miss range: [{np.min(pmiss_vals):.3f}, {np.max(pmiss_vals):.3f}] GeV")

    # Create 2D histogram
    hist_2d, emiss_edges, pmiss_edges = np.histogram2d(
        emiss_vals, pmiss_vals,
        bins=[emiss_bins, pmiss_bins]
    )
    counts_2d, _, _ = np.histogram2d(
        emiss_vals, pmiss_vals,
        bins=[emiss_bins, pmiss_bins]
    )

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))

    # Use pcolormesh for a smooth heatmap
    pcm = ax.pcolormesh(emiss_edges, pmiss_edges, hist_2d.T,
                        shading='auto', cmap='viridis')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Event Count')

    ax.set_xlabel(r'$E_{miss}^{preFSI}$ (GeV)', fontsize=12)
    ax.set_ylabel(r'$p_{miss}^{preFSI}$ magnitude (GeV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if save_fig:
        fig.savefig(save_fig, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_fig}')
    else:
        plt.show()

    return hist_2d, emiss_edges, pmiss_edges, counts_2d


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Plot dispersion relation from NUISANCE flat tree"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the NUISANCE flat tree ROOT file'
    )
    parser.add_argument(
        '--emiss-bins',
        type=int,
        default=50,
        help='Number of bins for E_miss (default: 50)'
    )
    parser.add_argument(
        '--pmiss-bins',
        type=int,
        default=50,
        help='Number of bins for p_miss (default: 50)'
    )
    parser.add_argument(
        '--emiss-range',
        type=float,
        nargs=2,
        default=[0, 1.0],
        help='Range for E_miss bins (default: 0 1.0)'
    )
    parser.add_argument(
        '--pmiss-range',
        type=float,
        nargs=2,
        default=[0, 1.0],
        help='Range for p_miss bins (default: 0 1.0)'
    )
    parser.add_argument(
        '--ccqelike',
        action='store_true',
        help='Apply CCQELike selection'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for figure (if None, display plot)'
    )

    args = parser.parse_args()

    # Create bin arrays
    emiss_bins = np.linspace(args.emiss_range[0], args.emiss_range[1], args.emiss_bins + 1)
    pmiss_bins = np.linspace(args.pmiss_range[0], args.pmiss_range[1], args.pmiss_bins + 1)

    # Load tree and apply mask if needed
    tree = NuisanceFlatTree(args.input_file)
    mask = tree.get_mask_flagCCQELike() if args.ccqelike else None

    # Plot dispersion relation
    plot_dispersion_relation(
        args.input_file,
        emiss_bins=emiss_bins,
        pmiss_bins=pmiss_bins,
        mask=mask,
        save_fig=args.output
    )


if __name__ == '__main__':
    main()

