"""
Created by Lorenzo Giannessi on 2026-04-01
Take as input a target nuisance flat tree, and a source tree in the suitable format, and three reweighter models for QE, 2p2h and Oth.
Plot the distribution of psi-prime in bins of recoil and pT for the target, source and reweighted source, and the mean of psi-prime vs recoil and pT for the three samples.
Also take an additional argument to specify the number of events to be processed in the source, and take the same number of events from the target to compare with

"""


import os
import sys
# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/')

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
    ax_main.set_ylabel(r'Mean $\psi^\prime$')
    ax_main.legend(loc='best')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(
        f"Mean $\\psi^\\prime$ vs {slice_name} ({unit}). Process: {process}, category: {category}",
        fontsize=12,
    )

    diff_target_source = target_means - source_means
    diff_reweighted_source = reweighted_means - target_means
    # ax_diff.plot(x_centers, diff_target_source, 'o-', color='tab:orange', label='Target - Source')
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
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc='best', fontsize=8)

    output_name = f"mean_vs_{slice_name}_{process}_{category}.png"
    fig.savefig(f"{output_dir}{output_name}", bbox_inches='tight')
    print(f"Saved mean-vs-{slice_name} figure to {output_name}")
    plt.close(fig)


def save_psi_prime_slice_plot(
        source_df,
        target_df,
        source_weights,
        target_weights,
        new_source_weights,
        source_mask,
        target_mask,
        pics_folder_name,
        process,
        category,
        slice_type,
        bin_index,
        low,
        high,
        unit,
):
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    n_source = int(np.sum(source_mask))
    n_target = int(np.sum(target_mask))

    if n_source == 0 or n_target == 0:
        print(
            f"Skipping {slice_type} slice [{low:g}, {high:g}] {unit}: "
            f"source events={n_source}, target events={n_target}"
        )
        return

    source_slice = source_df.iloc[source_mask]
    target_slice = target_df.iloc[target_mask]
    source_weights_slice = np.asarray(source_weights, dtype=float)[source_mask]
    target_weights_slice = np.asarray(target_weights, dtype=float)[target_mask]
    new_source_weights_slice = np.asarray(new_source_weights, dtype=float)[source_mask]

    fig = draw_source_target_distributions_and_ratio(
        source_slice,
        target_slice,
        variables=['psi_prime'],
        source_weights=source_weights_slice,
        target_weights=target_weights_slice * float(len(source_train_p)) / len(target_train_p),
        new_source_weights=new_source_weights_slice,
        legends=['Source', 'Source (Reweighted)', 'Target'],
        variable_bins={'psi_prime': PSI_PRIME_BIN_EDGES},
    )

    fig.suptitle(
        f"Psi-prime. {slice_type} bin {bin_index}: [{low:g}, {high:g}] {unit}. "
        f"Process: {process}, category: {category}",
        fontsize=16,
    )
    output_name = f"PsiPrime_{slice_type}Slice_bin{bin_index}_{process}_{category}.png"
    fig.savefig(f"{pics_folder_name}{output_name}")
    print(f"Saved psi-prime slice plot to {output_name}")
    plt.close()


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

    # total xsections
    target_ccqelike_xsec = tree_target.get_total_xsec()
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
        # print(f"Event {i}: process={process}, features={features}, weight={weight:.3f}")
        all_weights.append(weight)

    # now I have the weights for all events in the source tree, I can apply them to the source tree and compare with the target tree
    tree_source['weight'] = all_weights

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
        print(f"bin centers for variable {var}: {bin_centers}")
        bin_widths = np.diff(binning)
        print(f"bin widths for variable {var}: {bin_widths}")
        bin_contents_source, _ = np.histogram(tree_source[var], bins=binning, density=False)
        bin_errors_source = np.sqrt(bin_contents_source)
        bin_contents_target, _ = np.histogram(target_df[var], bins=binning, density=False, weights=target_df['weight'])
        bin_errors_target = np.sqrt(bin_contents_target)
        bin_contents_reweighted_source, _ = np.histogram(tree_source[var], bins=binning, weights=tree_source['weight'], density=False)
        bin_errors_reweighted_source = np.sqrt(np.histogram(tree_source[var], bins=binning, weights=tree_source['weight']**2)[0])

        # Normalize by bin width
        bin_contents_source_norm = bin_contents_source / bin_widths
        bin_errors_source_norm = bin_errors_source / bin_widths
        bin_contents_target_norm = bin_contents_target / bin_widths
        bin_errors_target_norm = bin_errors_target / bin_widths
        bin_contents_reweighted_source_norm = bin_contents_reweighted_source / bin_widths
        bin_errors_reweighted_source_norm = bin_errors_reweighted_source / bin_widths

        ax_top = axes[0, drawing_feature_names.index(var)]
        ax_top.step(bin_centers, bin_contents_source_norm, where='mid', label='Source', color='tab:green')
        ax_top.errorbar(bin_centers - bin_widths/4., bin_contents_source_norm, yerr=bin_errors_source_norm, fmt='none', ecolor='tab:green', alpha=0.5)
        ax_top.step(bin_centers, bin_contents_target_norm, where='mid', label='Target', color='tab:red')
        ax_top.errorbar(bin_centers, bin_contents_target_norm, yerr=bin_errors_target_norm, fmt='none', ecolor='tab:red', alpha=0.5)
        ax_top.step(bin_centers, bin_contents_reweighted_source_norm, where='mid', label='Source (Reweighted)', color='tab:blue', linestyle='--')
        ax_top.errorbar(bin_centers + bin_widths/4., bin_contents_reweighted_source_norm, yerr=bin_errors_reweighted_source_norm, fmt='none', ecolor='tab:blue', alpha=0.5)

        ax_top.set_xlabel(var)
        ax_top.set_title(f"Distribution of {var} for category {category}")
        ax_top.legend()
        ax_bottom = axes[1, drawing_feature_names.index(var)]
        # compute difference between reweighted source and target distributions
        diff = ( bin_contents_reweighted_source_norm - bin_contents_target_norm ) / np.where(bin_contents_target_norm > 0, bin_contents_target_norm, 1.0)
        diff_errors = np.sqrt(bin_errors_reweighted_source_norm**2 + bin_errors_target_norm**2) / np.where(bin_contents_target_norm > 0, bin_contents_target_norm, 1.0)
        ax_bottom.step(bin_centers, diff, where='mid', label='Reweighted Source - Target', color='tab:purple')
        ax_bottom.errorbar(bin_centers, diff, yerr=diff_errors, fmt='none', ecolor='tab:purple', alpha=0.5)
        ax_bottom.axhline(0.0, color='black', linestyle='--', linewidth=1)
        ax_bottom.set_xlabel(var)
        ax_bottom.set_ylabel('Relative Difference rew - target')
        ax_bottom.set_ylim(-0.1, 0.1)
        ax_bottom.set_title(f"(Reweighted Source - Target) / Target")
        ax_bottom.legend()

        # Set shared x-limits for top and bottom plots
        x_min = binning[0]
        x_max = binning[-1]
        ax_top.set_xlim(x_min, x_max)
        ax_bottom.set_xlim(x_min, x_max)

    output_folder = f"{reweighter_folder}/test_plots"
    os.makedirs(output_folder, exist_ok=True)
    output_name = f"{output_folder}/Distribution_all_vars_{category}.png"
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Saved combined distribution plot to {output_name}")
    plt.close()






if __name__ == "__main__":
    main()


