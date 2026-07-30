"""
Created by Lorenzo Giannessi on 2026-04-01
Take as input a target nuisance flat tree, and a source tree in the suitable format, and three reweighter models for QE, 2p2h and Oth.
Plot the distribution of psi-prime in bins of recoil and pT for the target, source and reweighted source, and the mean of psi-prime vs recoil and pT for the three samples.
Also take an additional argument to specify the number of events to be processed in the source, and take the same number of events from the target to compare with

"""


import os
import sys
# Change this path to your working directory where BDTReweight is installed:
sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
# sys.path.append('/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/')

from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance, draw_source_target_distributions_and_ratio
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
from BDTReweight.utilities import particle_variable_to_latex, diff_xsec_latex_wrt_variable
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import pathlib
import re
import joblib
import ROOT
import pickle
import argparse


plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
})

MUON_MASS_GEV = 0.1056583745
NUCLEON_MASS_GEV = 0.939565
S_RE_GEV = 0.028
K_F_GEV = 0.228
E_SHIFT_GEV = 0.020

MUON_PT_BIN_EDGES_GEV = np.array([
    0.0, 0.075, 0.15, 0.25, 0.325, 0.4, 0.475, 0.55,
    0.7, 0.85, 1.0, 1.25, 1.75, 2.5
], dtype=float)

RECOIL_BIN_EDGES_MEV = np.array([
    0.0, 20.0, 40.0, 80.0, 120.0, 160.0,
    240.0, 320.0, 400.0, 600.0, 800.0, 1400.0
], dtype=float)

PSI_PRIME_BIN_EDGES = np.array([
    -10.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
], dtype=float)

# LaTeX axis labels shared by every plotting function below.
PSI_PRIME_LATEX = r'$\psi^{\prime}_{\rm vis}$'
LATEX_LABELS = {
    'psi_prime': PSI_PRIME_LATEX,
    'muon_pt_gev': r'$p_T^{\mu}$ [GeV]',
    'recoil_mev': r'$\sum T_p$ [MeV]',
}
PROCESS_TITLES = {'QE': 'QE', '2p2h': '2p2h', 'Oth': 'Oth', 'QE+2p2h': 'QE + 2p2h', 'all': 'All processes'}

# Default per-topology configuration; values copied from the 0p0n defaults below.
CATEGORY_CONFIGS = {
    '0p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime', 'weight'],
    },
    '0pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==0', 'neutron': '>=1'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz', 'leading_neutron_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': [
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz','psi_prime'
        ],
        'particle_names': ['leading_neutron','total_proton'],
        'drawing_variables': [
            'leading_neutron_px', 'leading_neutron_py', 'leading_neutron_pz',
            'total_proton_px','total_proton_py','total_proton_pz',
            'total_proton_KE','leading_muon_py','leading_muon_pz', 'weight'
        ],
    },
    '1p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==1', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '1pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==1', 'neutron': '>=1'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '2p0n': {
        'particle_counts': {'muon': '==1', 'proton': '==2', 'neutron': '==0'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    '2pNn': {
        'particle_counts': {'muon': '==1', 'proton': '==2', 'neutron': '>=1'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
    'others': {
        'particle_counts': {'muon': '==1', 'proton': '>=2', 'neutron': '>=1'},
        'variable_exprs': [
            'Enu_true', 'Q2', 'q0', 'q3', 'W',
            'leading_muon_px', 'leading_muon_py', 'leading_muon_pz', 'leading_muon_KE',
            'total_proton_px', 'total_proton_py', 'total_proton_pz', 'total_proton_KE',
        ],
        'reweight_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'psi_prime'],
        'particle_names': ['total_proton'],
        'drawing_variables': ['total_proton_KE', 'leading_muon_py', 'leading_muon_pz', 'weight'],
    },
}


def compute_psi_prime(q0, q3_mag, k_f=K_F_GEV, e_shift=E_SHIFT_GEV):
    q0 = np.asarray(q0, dtype=float)
    q3_mag = np.asarray(q3_mag, dtype=float)

    eta_f = k_f / NUCLEON_MASS_GEV
    kappa = q3_mag / (2.0 * NUCLEON_MASS_GEV)
    lambda_var = (q0 - e_shift) / (2.0 * NUCLEON_MASS_GEV)
    tau = kappa * kappa - lambda_var * lambda_var

    normalizing_inner = np.sqrt(1.0 + eta_f * eta_f) - 1.0
    if normalizing_inner <= 0.0:
        return np.full_like(q0, np.nan, dtype=float)
    normalizing_factor = 1.0 / np.sqrt(normalizing_inner)

    tau_term = tau + tau * tau
    sqrt_tau_term = np.sqrt(np.clip(tau_term, 0.0, None))
    denominator_sq = (1.0 + lambda_var) * tau + kappa * sqrt_tau_term

    valid = (tau_term >= 0.0) & (denominator_sq > 0.0)
    psi_prime = np.full_like(q0, np.nan, dtype=float)
    denominator = np.sqrt(np.clip(denominator_sq, 0.0, None))
    psi_prime[valid] = ((lambda_var - tau) / denominator * normalizing_factor)[valid]
    return psi_prime


def get_psi_prime_from_fs_kinematics(recoil_gev, muon_px_beam, muon_py_beam, muon_pz_beam):
    recoil_gev = np.asarray(recoil_gev, dtype=float)
    muon_px_beam = np.asarray(muon_px_beam, dtype=float)
    muon_py_beam = np.asarray(muon_py_beam, dtype=float)
    muon_pz_beam = np.asarray(muon_pz_beam, dtype=float)

    muon_e = np.sqrt(
        muon_px_beam * muon_px_beam
        + muon_py_beam * muon_py_beam
        + muon_pz_beam * muon_pz_beam
        + MUON_MASS_GEV * MUON_MASS_GEV
    )
    q0 = recoil_gev + S_RE_GEV
    qx = -muon_px_beam
    qy = -muon_py_beam
    q3 = muon_e - muon_pz_beam + recoil_gev + S_RE_GEV
    q_mag = np.sqrt(qx * qx + qy * qy + q3 * q3)

    return compute_psi_prime(q0, q_mag)


def _format_bin_edge(value):
    return f"{value:g}".replace('.', 'p')


def _hist_density_mean(values, weights, bin_edges):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid):
        return np.nan
    values = values[valid]
    weights = weights[valid]

    counts, edges = np.histogram(values, bins=np.asarray(bin_edges, dtype=float), weights=weights)
    bin_widths = np.diff(edges)
    updated_bin_content = counts / bin_widths
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    norm = np.sum(updated_bin_content)
    if norm <= 0.0:
        return np.nan
    return np.sum(updated_bin_content * bin_centers) / norm


def _set_signed_log_yaxis(ax, values, linthresh=1e-3):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.set_ylim(-1.0, 1.0)
        return

    max_abs = np.max(np.abs(finite))
    if max_abs <= 0.0:
        ax.set_yscale('symlog', linthresh=linthresh)
        ax.set_ylim(-1.0, 1.0)
        return

    effective_linthresh = min(linthresh, 0.1 * max_abs)
    effective_linthresh = max(effective_linthresh, 1e-12)
    ax.set_yscale('symlog', linthresh=effective_linthresh)
    ax.set_ylim(-1.2 * max_abs, 1.2 * max_abs)


def save_mean_vs_slice_plot(
        x_centers,
        source_means,
        target_means,
        reweighted_means,
        x_label,
        slice_name,
        unit,
        process,
        category,
        output_dir,
):
    fig, (ax_main, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=200,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
        sharex=True,
    )

    source_means = np.asarray(source_means, dtype=float)
    target_means = np.asarray(target_means, dtype=float)
    reweighted_means = np.asarray(reweighted_means, dtype=float)

    ax_main.plot(x_centers, source_means, 'o-', label='Source', color='tab:green')
    ax_main.plot(x_centers, target_means, 'o-', label='Target', color='tab:red')
    ax_main.plot(x_centers, reweighted_means, 'o-', label='Source (Reweighted)', color='tab:blue')
    ax_main.set_ylabel(rf'Mean {PSI_PRIME_LATEX}')
    ax_main.legend(loc='best')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(
        rf"Mean {PSI_PRIME_LATEX} vs {x_label} — {PROCESS_TITLES.get(process, process)}, category {category}",
    )

    diff_reweighted_source = reweighted_means - target_means
    ax_diff.plot(
        x_centers,
        diff_reweighted_source,
        'o-',
        color='tab:purple',
        label='Reweighted - Target',
    )
    ax_diff.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax_diff.axhline(0.015, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_diff.axhline(-0.015, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_diff.set_xlabel(f'{x_label} [{unit}]')
    ax_diff.set_ylabel(r'$\Delta$ mean')
    _set_signed_log_yaxis(ax_diff, diff_reweighted_source)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc='best', fontsize=8)

    output_name = f"MeanPsiPrime_vs_{slice_name}_{process}_{category}.png"
    fig.savefig(f"{output_dir}{output_name}", bbox_inches='tight')
    print(f"Saved mean-vs-{slice_name} figure to {output_name}")
    plt.close(fig)


def plot_psiprime_grid(
        source_df,
        target_df,
        source_weights,
        target_weights,
        new_source_weights,
        slice_values_source,
        slice_values_target,
        bin_edges,
        grid_shape,
        pics_folder_name,
        process,
        category,
        slice_label,
        slice_latex,
        unit,
):
    """Draw psi-prime distributions (Source, Target, Source Reweighted) in
    every slice of `bin_edges`, one subplot per slice, arranged on a single
    canvas of shape `grid_shape` (n_rows, n_cols)."""
    n_bins = len(bin_edges) - 1
    n_rows, n_cols = grid_shape
    if n_rows * n_cols < n_bins:
        raise ValueError(f"Grid shape {grid_shape} cannot hold {n_bins} slices")

    binning = PSI_PRIME_BIN_EDGES
    bin_widths = np.diff(binning)

    slice_values_source = np.asarray(slice_values_source, dtype=float)
    slice_values_target = np.asarray(slice_values_target, dtype=float)
    source_psi_all = source_df['psi_prime'].to_numpy()
    target_psi_all = target_df['psi_prime'].to_numpy()
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    new_source_weights = np.asarray(new_source_weights, dtype=float)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.3 * n_cols, 3.4 * n_rows),
        dpi=200,
        squeeze=False,
    )

    legend_handles_labels = None

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        if i >= n_bins:
            ax.axis('off')
            continue

        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            source_mask = (slice_values_source >= low) & (slice_values_source <= high)
            target_mask = (slice_values_target >= low) & (slice_values_target <= high)
        else:
            source_mask = (slice_values_source >= low) & (slice_values_source < high)
            target_mask = (slice_values_target >= low) & (slice_values_target < high)

        if not np.any(source_mask) or not np.any(target_mask):
            ax.axis('off')
            continue

        bin_contents_source = np.histogram(source_psi_all[source_mask], bins=binning, weights=source_weights[source_mask])[0] / bin_widths
        bin_contents_target = np.histogram(target_psi_all[target_mask], bins=binning, weights=target_weights[target_mask])[0] / bin_widths
        bin_contents_reweighted = np.histogram(source_psi_all[source_mask], bins=binning, weights=new_source_weights[source_mask])[0] / bin_widths

        ax.step(binning, np.append(bin_contents_source, 0), where='post', color='tab:green', label='Source')
        ax.step(binning, np.append(bin_contents_target, 0), where='post', color='tab:red', label='Target')
        ax.step(binning, np.append(bin_contents_reweighted, 0), where='post', color='tab:blue', linestyle='--', label='Source (Reweighted)')

        ax.set_title(f"{slice_latex} $\\in$ [{low:g}, {high:g}] {unit}")
        ax.grid(alpha=0.25)
        ax.set_xlim(binning[0], binning[-1])
        if legend_handles_labels is None:
            legend_handles_labels = ax.get_legend_handles_labels()

        if col == 0:
            ax.set_ylabel(rf'd$N$/d{PSI_PRIME_LATEX}')
        ax.set_xlabel(PSI_PRIME_LATEX)

    fig.tight_layout(rect=[0, 0, 1, 0.91], h_pad=1.5)
    fig.suptitle(
        rf"{PSI_PRIME_LATEX} distributions in {slice_latex} slices"
        rf" — {PROCESS_TITLES.get(process, process)}, category {category}",
        y=0.98,
    )
    if legend_handles_labels is not None:
        fig.legend(*legend_handles_labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)

    output_name = f"PsiPrimeGrid_{slice_label}_{process}_{category}.png"
    fig.savefig(f"{pics_folder_name}{output_name}")
    print(f"Saved psi-prime grid plot to {output_name}")
    plt.close(fig)


def analyze_process_slice(
        source_df,
        target_df,
        slice_col,
        bin_edges,
        grid_shape,
        slice_label,
        slice_latex,
        unit,
        process,
        category,
        output_folder,
):
    """Compute mean-psi-prime-vs-slice + per-slice psi-prime distributions
    for one process (or 'all') and save both the summary line plot and the
    grid of per-slice histograms."""
    output_folder = output_folder.rstrip('/') + '/'
    slice_values_source = source_df[slice_col].to_numpy()
    slice_values_target = target_df[slice_col].to_numpy()
    source_psi = source_df['psi_prime'].to_numpy()
    target_psi = target_df['psi_prime'].to_numpy()
    source_weights = np.ones(len(source_df))
    target_weights = target_df['weight'].to_numpy()
    reweighted_weights = source_df['weight'].to_numpy()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    centers_valid, means_source, means_target, means_reweighted = [], [], [], []
    errs_source, errs_target, errs_reweighted = [], [], []

    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            source_mask = (slice_values_source >= low) & (slice_values_source <= high)
            target_mask = (slice_values_target >= low) & (slice_values_target <= high)
        else:
            source_mask = (slice_values_source >= low) & (slice_values_source < high)
            target_mask = (slice_values_target >= low) & (slice_values_target < high)

        n_source = int(np.sum(source_mask))
        n_target = int(np.sum(target_mask))
        if n_source == 0 or n_target == 0:
            print(f"Skipping {slice_label} slice [{low:g}, {high:g}] {unit} ({process}): "
                  f"source events={n_source}, target events={n_target}")
            continue

        centers_valid.append(bin_centers[i])
        means_source.append(np.average(source_psi[source_mask]))
        errs_source.append(np.std(source_psi[source_mask]) / np.sqrt(n_source))
        means_target.append(np.average(target_psi[target_mask], weights=target_weights[target_mask]))
        errs_target.append(np.std(target_psi[target_mask]) / np.sqrt(n_target))
        means_reweighted.append(np.average(source_psi[source_mask], weights=reweighted_weights[source_mask]))
        errs_reweighted.append(np.std(source_psi[source_mask]) / np.sqrt(n_source))

        print(f"{slice_label} slice [{low:g}, {high:g}] {unit} ({process}): "
              f"source_mean={means_source[-1]:.3f}, target_mean={means_target[-1]:.3f}, "
              f"reweighted_mean={means_reweighted[-1]:.3f}")

    if len(centers_valid) > 0:
        save_mean_vs_slice_plot(
            x_centers=centers_valid,
            source_means=means_source,
            target_means=means_target,
            reweighted_means=means_reweighted,
            x_label=slice_latex,
            slice_name=slice_label,
            unit=unit,
            process=process,
            category=category,
            output_dir=output_folder,
        )

    plot_psiprime_grid(
        source_df=source_df,
        target_df=target_df,
        source_weights=source_weights,
        target_weights=target_weights,
        new_source_weights=reweighted_weights,
        slice_values_source=slice_values_source,
        slice_values_target=slice_values_target,
        bin_edges=bin_edges,
        grid_shape=grid_shape,
        pics_folder_name=output_folder,
        process=process,
        category=category,
        slice_label=slice_label,
        slice_latex=slice_latex,
        unit=unit,
    )


def validate_overall_scale(tree_source, reweighters, source_ccqelike_xsec, target_ccqelike_xsec, tol=0.02):
    """
    Validate the overall cross-section normalization carried by the per-event weights,
    separately from the shape validation done by the distribution plots.

    The reweighted source predicts a total cross section
        sigma_rw = sigma_source * mean(w),
    so a correctly-normalized weight set must satisfy mean(w) == s, with
        s = sigma_target / sigma_source   (the CCQE-like cross-section ratio).

    Two levels are reported:
      * per process X: mean(w_X) must equal the xsec_scale_factor baked into that
        reweighter's pickle. Since w = r * xsec_scale_factor * norm_factor and
        norm_factor forces mean(r) -> 1, this checks the density-ratio normalization
        is still intact on this sample (norm_factor present, no NaNs, right pickle).
      * global: the composition-weighted mean(w) must equal the independently-computed
        global ratio s. This cross-checks the per-process baked factors against the
        global CCQE-like cross-section ratio (they should telescope to s).

    IMPORTANT: these source events are the same sample the reweighters were trained on,
    so this is a consistency / regression check, NOT a test of BDT generalization. It
    also does NOT prove s equals the true sigma_target/sigma_source -- that depends on
    the two extracted cross sections being correct (units, bin, selection, hadd factor),
    which no weight-closure test can verify.
    """
    s = target_ccqelike_xsec / source_ccqelike_xsec if source_ccqelike_xsec > 0 else np.nan
    weights = tree_source['weight'].to_numpy()
    processes = tree_source['process'].to_numpy()

    n_nan = int(np.sum(~np.isfinite(weights)))

    print("\n" + "=" * 80)
    print("OVERALL SCALE VALIDATION (normalization, independent of shape)")
    print("-" * 80)
    print(f"sigma_source (CCQE-like)       : {source_ccqelike_xsec:.4e} cm^2")
    print(f"sigma_target (CCQE-like)       : {target_ccqelike_xsec:.4e} cm^2")
    print(f"s = sigma_target / sigma_source: {s:.4f}")
    if n_nan:
        print(f"WARNING: {n_nan} non-finite weights present")
    print("-" * 80)
    print(f"{'process':>8} | {'n_events':>9} | {'mean(w)':>9} | {'baked s_X':>9} | {'ratio':>7} | status")
    for process in ['QE', '2p2h', 'Oth']:
        mask = processes == process
        n = int(np.sum(mask))
        if n == 0:
            continue
        mean_w = float(np.mean(weights[mask]))
        baked = float(reweighters[process].xsec_scale_factor)
        ratio = mean_w / baked if baked != 0 else np.nan
        status = "OK" if np.isfinite(ratio) and abs(ratio - 1.0) < tol else "MISMATCH"
        print(f"{process:>8} | {n:>9d} | {mean_w:>9.4f} | {baked:>9.4f} | {ratio:>7.3f} | {status}")

    n_all = len(weights)
    mean_w_all = float(np.mean(weights))
    total_w = float(np.sum(weights))
    ratio_global = mean_w_all / s if (np.isfinite(s) and s > 0) else np.nan
    status_global = "OK" if np.isfinite(ratio_global) and abs(ratio_global - 1.0) < tol else "MISMATCH"
    print("-" * 80)
    print(f"{'GLOBAL':>8} | {n_all:>9d} | {mean_w_all:>9.4f} | {s:>9.4f} | {ratio_global:>7.3f} | {status_global}")
    print(f"total weight = {total_w:.1f}   (expected s*N = {s * n_all:.1f})")
    print(f"predicted sigma_reweighted = sigma_source * mean(w) = {source_ccqelike_xsec * mean_w_all:.4e} cm^2"
          f"   (target = {target_ccqelike_xsec:.4e} cm^2)")
    print("=" * 80 + "\n")


def main():
    p = argparse.ArgumentParser(description="Test training of reweighter models on a single category and process.")
    p.add_argument('--source-file', type=str, required=True, help="Path to source ROOT file (e.g., ROOT tree).")
    p.add_argument('--target-file', type=str, required=True, help="Path to target ROOT file (e.g., NUISANCE flat tree).")
    p.add_argument('--reweighter-folder', type=str, required=True, help="Path to folder containing trained reweighter models (e.g., 'reweighters/').")
    """
    The structure is the following:
    reweighters/
        2p2h/GBReweighterModel_0p0n.pkl
        Oth/GBReweighterModel_0p0n.pkl
        QE/GBReweighterModel_0p0n.pkl
    """
    p.add_argument('--max-events', type=int, default=100000, help="Maximum number of events to process from source and target.")
    p.add_argument('--category', type=str, default='0p0n', choices=CATEGORY_CONFIGS.keys(), help="Interaction category to test (e.g., '0p0n').")
    p.add_argument('--output-folder', type=str, default=None,
                   help="Directory for the output plots. Defaults to '<reweighter-folder>/test_plots'.")
    # These must match the values used at training time (train_by_reaction_config.py /
    # its yaml), so the target xsec -- and hence the scale factor s -- recomputed here
    # equals the one baked into the pickles.
    p.add_argument('--hadd_n_files', type=int, default=1,
                   help="Number of NUISANCE flat trees hadd'd into the target file; the target xsec is divided by this (default 1).")
    p.add_argument('--A_source', type=float, default=1.0,
                   help="Per-nucleon basis of the source xsec (default 1.0). Use 12 for a carbon source vs a CH target.")
    p.add_argument('--A_target', type=float, default=1.0,
                   help="Per-nucleon basis of the target xsec (default 1.0). Use 13 for a polystyrene (CH) target. Target xsec is scaled by A_target/A_source.")

    args = p.parse_args()

    category = args.category
    source_path = args.source_file
    target_path = args.target_file

    # open source file
    tree_source_train = uproot.open(source_path)['EventKinematics_truth'].arrays(library='pd')
    source_max_events = args.max_events
    tree_source = tree_source_train.iloc[:source_max_events]
    tree_source = tree_source.rename(columns={'muon_px':'leading_muon_px', 'muon_py':'leading_muon_py', 'muon_pz':'leading_muon_pz',
                                              'sum_p_px':'total_proton_px', 'sum_p_py':'total_proton_py', 'sum_p_pz':'total_proton_pz', 'sum_Tp':'total_proton_KE', 'leading_n_px':'leading_neutron_px',
                                              'leading_n_py':'leading_neutron_py', 'leading_n_pz':'leading_neutron_pz', 'leading_p_px':'leading_proton_px', 'leading_p_py':'leading_proton_py',
                                              'leading_p_pz':'leading_proton_pz', 'subleading_p_px':'subleading_proton_px', 'subleading_p_py':'subleading_proton_py', 'subleading_p_pz':'subleading_proton_pz'}
                                                 )
    # open target file
    tree_target = NuisanceFlatTree(target_path, max_events=args.max_events)
    variable_exprs = CATEGORY_CONFIGS[category]['variable_exprs']
    reweight_variables = CATEGORY_CONFIGS[category]['reweight_variables']
    particle_names = CATEGORY_CONFIGS[category]['particle_names']
    target_df = create_dataframe_from_nuisance(tree_target, variable_exprs=variable_exprs)
    target_df = transform_momentum_to_reaction_frame(target_df, selector_lepton='leading_muon', particle_names=particle_names)
    print(f"Loaded source tree (first {len(tree_source)} events) from {source_path}.")
    print(f"Loaded target tree (first {len(tree_target._flattree_vars)} events) from {target_path}.")


    # open reweighters and load from pickle file the reweighter for the 0p0n category and QE process
    reweighter_folder = args.reweighter_folder
    category = '0p0n'
    reweighters = {}
    for process in ['QE', '2p2h', 'Oth']:
        reweighter_path = f"{reweighter_folder}/{process}/GBReweighterModel_{category}.pkl"
        reweighter = Reweighter.load_from_pickle(reweighter_path)
        reweighters[process] = reweighter
        print(f"Loaded reweighter for process {process} and category {category} from {reweighter_path}")

    # define the training variables
    feature_names = CATEGORY_CONFIGS[category]['reweight_variables']
    drawing_feature_names = ['recoil_mev', 'muon_pt_gev', 'psi_prime']
    drawing_feature_binnings = {
        'recoil_mev': RECOIL_BIN_EDGES_MEV,
        'muon_pt_gev': MUON_PT_BIN_EDGES_GEV,
        'psi_prime': PSI_PRIME_BIN_EDGES,
    }
    print(f"Using reweighting variables: {feature_names}")
    # load all derived variables in the source and target trees, and check that there are no missing features
    tree_source['recoil_gev'] = np.nan_to_num(tree_source['total_proton_KE'].to_numpy(), nan=0.0)
    tree_source['psi_prime'] = get_psi_prime_from_fs_kinematics(
        recoil_gev=tree_source['recoil_gev'].to_numpy(),
        muon_px_beam=np.zeros_like(tree_source['leading_muon_py'].to_numpy()),
        muon_py_beam=tree_source['leading_muon_py'].to_numpy(),
        muon_pz_beam=tree_source['leading_muon_pz'].to_numpy(),
    )
    tree_source['recoil_mev'] = tree_source['recoil_gev'] * 1000.0
    tree_source['muon_pt_gev'] = np.abs(tree_source['leading_muon_py'])

    target_df['recoil_gev'] = np.nan_to_num(target_df['total_proton_KE'].to_numpy(), nan=0.0)
    target_df['psi_prime'] = get_psi_prime_from_fs_kinematics(
        recoil_gev=target_df['recoil_gev'].to_numpy(),
        muon_px_beam=np.zeros_like(target_df['leading_muon_py'].to_numpy()),
        muon_py_beam=target_df['leading_muon_py'].to_numpy(),
        muon_pz_beam=target_df['leading_muon_pz'].to_numpy(),
    )
    target_df['recoil_mev'] = target_df['recoil_gev'] * 1000.0
    target_df['muon_pt_gev'] = np.abs(target_df['leading_muon_py'])

    # total xsections. Mirror train_by_reaction_config.py exactly so the scale
    # factor s recomputed here matches the one baked into the pickles:
    # sigma = sum(fScaleFactor * InputWeight) / hadd_n_files * (A_target / A_source).
    # get_total_xsec() alone would be wrong for hadd'd (N>1) or weighted (InputWeight!=1)
    # targets. Unweighted single-file targets reduce to get_total_xsec().
    _target_fscale = np.asarray(tree_target._flattree_vars['fScaleFactor'], dtype=float)
    _target_inwgt = np.asarray(tree_target._flattree_vars['InputWeight'], dtype=float)
    target_ccqelike_xsec = float(np.sum(_target_fscale * _target_inwgt)) / args.hadd_n_files
    if args.hadd_n_files != 1:
        print(f"hadd correction: divided target xsec by hadd_n_files={args.hadd_n_files}")
    nucleon_basis_correction = args.A_target / args.A_source
    if nucleon_basis_correction != 1.0:
        print(f"Per-nucleon basis correction A_target/A_source = {args.A_target:g}/{args.A_source:g} "
              f"= {nucleon_basis_correction:.4f} applied to target xsec "
              f"({target_ccqelike_xsec:.3e} -> {target_ccqelike_xsec * nucleon_basis_correction:.3e})")
        target_ccqelike_xsec *= nucleon_basis_correction

    source_file = ROOT.TFile(source_path)
    h_xsec_ccqelike = ROOT.TH1D(source_file.Get('h_eventRate_qelike_cross_section'))
    source_ccqelike_xsec = h_xsec_ccqelike.GetBinContent(1)
    print(f"Source CCQE-like cross-section: {source_ccqelike_xsec:.3e} cm^2")
    print(f"Target CCQE-like cross-section: {target_ccqelike_xsec:.3e} cm^2")
    xsec_scale_factor = target_ccqelike_xsec / source_ccqelike_xsec if source_ccqelike_xsec > 0 else 1.0
    print(f"Cross-section scale factor: {xsec_scale_factor:.3f}")

    target_df['weight'] = xsec_scale_factor

    # check that all features are in the source_events dataframe
    missing_features = [var for var in reweight_variables if var not in tree_source.columns]
    if missing_features:
        print(f"Error: Missing features in source_events for process {process}: {missing_features}")
        exit(1)

    # DEBUG: print the first 50 rows of target and source leading_muon_px and leading_muon_py
    # for i in range(50):
    #     print(f"Source event {i}: leading_muon_px=0.000, leading_muon_py={tree_source['leading_muon_py'].iloc[i]:.3f}, psi_prime={tree_source['psi_prime'].iloc[i]:.3f}")
    #     print(f"Target event {i}: leading_muon_px={target_df['leading_muon_px'].iloc[i]:.3f}, leading_muon_py={target_df['leading_muon_py'].iloc[i]:.3f}, psi_prime={target_df['psi_prime'].iloc[i]:.3f}")

    all_weights = []
    for i in range(min(len(tree_source), len(tree_source))):
        features = [tree_source[var].iloc[i] for var in feature_names]

        mode = tree_source['reactionCode'].iloc[i]
        if mode == 1:
            process = 'QE'
        elif mode == 2:
            process = '2p2h'
        else:
            process = 'Oth'

        weight = reweighters[process].predict_weight_single_event(features)
        if i<500:
            print(f"Event {i}: process={process}, weight={weight:.3f} features: ", end='')
            for val in features:
                print(f", {val:.7f}", end='')
            print()

        all_weights.append(weight)

    # now I have the weights for all events in the source tree, I can apply them to the source tree and compare with the target tree
    tree_source['weight'] = all_weights

    # process labels, used to split the psi-prime slice plots per process
    tree_source['process'] = np.select(
        [tree_source['reactionCode'] == 1, tree_source['reactionCode'] == 2],
        ['QE', '2p2h'],
        default='Oth',
    )
    target_mode = tree_target.get_mode()
    target_df['process'] = np.select(
        [target_mode == 1, target_mode == 2],
        ['QE', '2p2h'],
        default='Oth',
    )

    # Normalization (overall scale) validation, kept separate from the shape plots below.
    validate_overall_scale(
        tree_source=tree_source,
        reweighters=reweighters,
        source_ccqelike_xsec=source_ccqelike_xsec,
        target_ccqelike_xsec=target_ccqelike_xsec,
    )

    # DEBUG: check lengths of source and target trees and weights
    print(f"Source tree length: {len(tree_source)}, Target tree length: {len(target_df)}, Weights length: {len(all_weights)}")
    if len(tree_source) != len(all_weights):
        print("Error: Length of weights does not match length of source tree.")
        exit(1)
    length = min(len(tree_source), len(target_df))

    # plot the three features used for reweighting, comparing source, target and reweighted source distributions
    n_vars = len(drawing_feature_names)
    fig, axes = plt.subplots(2, n_vars, figsize=(5*n_vars, 10), dpi=200)


    for var in drawing_feature_names:
        binning = drawing_feature_binnings[var]
        bin_centers = 0.5 * (binning[:-1] + binning[1:])
        bin_widths = np.diff(binning)
        # print(f"bin centers for variable {var}: {bin_centers}")
        # print(f"bin widths for variable {var}: {bin_widths}")
        bin_contents_source, _ = np.histogram(tree_source[var], bins=binning, density=False)
        bin_errors_source = np.sqrt(bin_contents_source)
        bin_contents_target, _ = np.histogram(target_df[var], bins=binning, density=False, weights=target_df['weight'])
        bin_errors_target = np.sqrt(bin_contents_target)
        bin_contents_reweighted_source, _ = np.histogram(tree_source[var], bins=binning, weights=tree_source['weight'], density=False)
        bin_errors_reweighted_source = np.sqrt(np.histogram(tree_source[var], bins=binning, weights=tree_source['weight']**2)[0])

        # Normalize by bin width
        bin_contents_source_norm = bin_contents_source / bin_widths
        bin_contents_source_norm = np.concatenate(( bin_contents_source_norm, [0]))
        bin_errors_source_norm = bin_errors_source / bin_widths
        bin_errors_source_norm = np.concatenate(( bin_errors_source_norm, [0]))
        bin_contents_target_norm = bin_contents_target / bin_widths
        bin_contents_target_norm = np.concatenate(( bin_contents_target_norm, [0]))
        bin_errors_target_norm = bin_errors_target / bin_widths
        bin_errors_target_norm = np.concatenate(( bin_errors_target_norm, [0]))
        bin_contents_reweighted_source_norm = bin_contents_reweighted_source / bin_widths
        bin_contents_reweighted_source_norm = np.concatenate(( bin_contents_reweighted_source_norm, [0]))
        bin_errors_reweighted_source_norm = bin_errors_reweighted_source / bin_widths
        bin_errors_reweighted_source_norm = np.concatenate(( bin_errors_reweighted_source_norm, [0]))

        bin_centers = np.concatenate(( bin_centers, [binning[-1]]))

        ax_top = axes[0, drawing_feature_names.index(var)]
        ax_top.step(binning, bin_contents_source_norm, where='post', label='Source', color='tab:green')
        ax_top.errorbar(bin_centers, bin_contents_source_norm, yerr=bin_errors_source_norm, fmt='none', ecolor='tab:green', alpha=0.5)
        ax_top.step(binning, bin_contents_target_norm, where='post', label='Target', color='tab:red')
        ax_top.errorbar(bin_centers, bin_contents_target_norm, yerr=bin_errors_target_norm, fmt='none', ecolor='tab:red', alpha=0.5)
        ax_top.step(binning, bin_contents_reweighted_source_norm, where='post', label='Source (Reweighted)', color='tab:blue', linestyle='--')
        ax_top.errorbar(bin_centers, bin_contents_reweighted_source_norm, yerr=bin_errors_reweighted_source_norm, fmt='none', ecolor='tab:blue', alpha=0.5)

        var_label = LATEX_LABELS.get(var, var)
        ax_top.set_xlabel(var_label)
        ax_top.set_title(f"Distribution of {var_label} for category {category}")
        ax_top.legend()
        ax_bottom = axes[1, drawing_feature_names.index(var)]
        # compute difference between reweighted source and target distributions
        diff = ( bin_contents_reweighted_source_norm - bin_contents_target_norm ) / np.where(bin_contents_target_norm > 0, bin_contents_target_norm, 1.0)
        diff_errors = np.sqrt(bin_errors_reweighted_source_norm**2 + bin_errors_target_norm**2) / np.where(bin_contents_target_norm > 0, bin_contents_target_norm, 1.0)
        ax_bottom.step(binning, diff, where='post', label='Reweighted Source - Target', color='tab:purple')
        ax_bottom.errorbar(bin_centers, diff, yerr=diff_errors, fmt='none', ecolor='tab:purple', alpha=0.5)
        ax_bottom.axhline(0.0, color='black', linestyle='--', linewidth=1)
        ax_bottom.set_xlabel(var_label)
        ax_bottom.set_ylabel('Relative Difference rew - target')
        _set_signed_log_yaxis(ax_bottom, diff)
        # ax_bottom.set_ylim(-0.1, 0.1)
        ax_bottom.set_title(f"(Reweighted Source - Target) / Target")
        if (var == 'psi_prime'):
            ax_top.set_xlim(-5.0, 10.0)
            ax_bottom.set_xlim(-5.0, 10.0)
        ax_bottom.legend()

        # Set shared x-limits for top and bottom plots
        x_min = binning[0]
        x_max = binning[-1]
        ax_top.set_xlim(x_min, x_max)
        ax_bottom.set_xlim(x_min, x_max)

    output_folder = args.output_folder if args.output_folder else f"{reweighter_folder}/test_plots"
    os.makedirs(output_folder, exist_ok=True)
    output_name = f"{output_folder}/Distribution_all_vars_{category}.png"
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Saved combined distribution plot to {output_name}")
    plt.close()

    # time to perform the check of each slice in recoil and muon pt, and plot the mean of psi-prime in each slice
    # (and the full psi-prime distribution grid) for source, target and reweighted source, both per-process and
    # for all processes combined.
    for process in ['QE', '2p2h', 'Oth', 'QE+2p2h', 'all']:
        if process == 'all':
            source_proc_df = tree_source
            target_proc_df = target_df
        elif process == 'QE+2p2h':
            source_proc_df = tree_source[tree_source['process'].isin(['QE', '2p2h'])]
            target_proc_df = target_df[target_df['process'].isin(['QE', '2p2h'])]
        else:
            source_proc_df = tree_source[tree_source['process'] == process]
            target_proc_df = target_df[target_df['process'] == process]

        print(f"\n=== Analyzing SumTp (recoil) slices — process: {process} ===")
        analyze_process_slice(
            source_df=source_proc_df,
            target_df=target_proc_df,
            slice_col='recoil_mev',
            bin_edges=RECOIL_BIN_EDGES_MEV,
            grid_shape=(4, 3),
            slice_label='SumTp',
            slice_latex=LATEX_LABELS['recoil_mev'],
            unit='MeV',
            process=process,
            category=category,
            output_folder=output_folder,
        )

        print(f"\n=== Analyzing muon pT slices — process: {process} ===")
        analyze_process_slice(
            source_df=source_proc_df,
            target_df=target_proc_df,
            slice_col='muon_pt_gev',
            bin_edges=MUON_PT_BIN_EDGES_GEV,
            grid_shape=(5, 3),
            slice_label='MuonPT',
            slice_latex=LATEX_LABELS['muon_pt_gev'],
            unit='GeV',
            process=process,
            category=category,
            output_folder=output_folder,
        )

if __name__ == "__main__":
    main()


