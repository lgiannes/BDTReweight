import os
import sys
# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/')

from BDTReweight.analysis import transform_momentum_to_reaction_frame, create_dataframe_from_nuisance, draw_source_target_distributions_and_ratio
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree
from BDTReweight.reweighter import Reweighter
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import pathlib
import re
import pickle
import argparse
import yaml
from tqdm import tqdm


MUON_MASS_GEV = 0.1056583745
NUCLEON_MASS_GEV = 0.939565
S_RE_GEV = 0.028
K_F_GEV = 0.228
E_SHIFT_GEV = 0.020

# Variables computed by add_derived_columns() (rather than pulled straight from
# variable_exprs), and the raw variable_exprs entries each one needs to be computable.
DERIVED_VARIABLE_DEPS = {
    'muon_pt_gev': ['leading_muon_py'],
    'recoil_gev': ['total_proton_KE'],
    'recoil_mev': ['total_proton_KE'],
    'psi_prime': ['leading_muon_py', 'leading_muon_pz', 'total_proton_KE'],
}
DERIVED_VARIABLES = set(DERIVED_VARIABLE_DEPS)

# The reaction-frame transform is always called with this selector lepton (see main()).
REACTION_FRAME_SELECTOR_LEPTON = 'leading_muon'

# The psi_prime slice diagnostics and the reaction-frame transform always run
# (independent of reweight_variables / drawing_variables), so their dependencies
# must always be present in variable_exprs.
ALWAYS_REQUIRED_BASE_VARIABLES = sorted(set(DERIVED_VARIABLE_DEPS['psi_prime']) | {
    f'{REACTION_FRAME_SELECTOR_LEPTON}_px', f'{REACTION_FRAME_SELECTOR_LEPTON}_py', f'{REACTION_FRAME_SELECTOR_LEPTON}_pz',
})


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # topology_codes keys come back as ints from YAML already; make sure of it
    cfg['topology_codes'] = {int(k): v for k, v in cfg['topology_codes'].items()}

    for key in ('muon_pt_gev', 'recoil_mev', 'psi_prime'):
        cfg['binning'][key] = np.array(cfg['binning'][key], dtype=float)

    return cfg


def apply_rule(values, rule):
    """Evaluate a python comparison rule (e.g. '==1', '>2') against an array."""
    values = np.asarray(values)
    return eval(f'values {rule}')


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
    psi_prime_bin_edges,
    n_source_train,
    n_target_train,
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
        target_weights=target_weights_slice * float(n_source_train) / n_target_train,
        new_source_weights=new_source_weights_slice,
        legends=['Source', 'Source (Reweighted)', 'Target'],
        variable_bins={'psi_prime': psi_prime_bin_edges},
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


def produce_slice_diagnostics(
    source_df,
    target_df,
    source_weights_np,
    target_weights_np,
    predicted_weights,
    slice_variable,
    bin_edges,
    unit,
    slice_type,
    process_pics_folder,
    process,
    category,
    psi_prime_bin_edges,
    n_source_train,
    n_target_train,
):
    """Produce per-slice psi_prime comparison plots plus a mean-vs-slice summary plot."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_source = []
    mean_target = []
    mean_reweighted = []

    source_slice_var = source_df[slice_variable].to_numpy()
    target_slice_var = target_df[slice_variable].to_numpy()
    source_psi_prime = source_df['psi_prime'].to_numpy()
    target_psi_prime = target_df['psi_prime'].to_numpy()

    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        is_last_bin = (i == len(bin_edges) - 2)
        if is_last_bin:
            source_mask = (source_slice_var >= low) & (source_slice_var <= high)
            target_mask = (target_slice_var >= low) & (target_slice_var <= high)
        else:
            source_mask = (source_slice_var >= low) & (source_slice_var < high)
            target_mask = (target_slice_var >= low) & (target_slice_var < high)

        save_psi_prime_slice_plot(
            source_df=source_df,
            target_df=target_df,
            source_weights=source_weights_np,
            target_weights=target_weights_np,
            new_source_weights=predicted_weights,
            source_mask=source_mask,
            target_mask=target_mask,
            pics_folder_name=process_pics_folder,
            process=process,
            category=category,
            slice_type=slice_type,
            bin_index=i,
            low=low,
            high=high,
            unit=unit,
            psi_prime_bin_edges=psi_prime_bin_edges,
            n_source_train=n_source_train,
            n_target_train=n_target_train,
        )

        mean_source.append(_hist_density_mean(source_psi_prime[source_mask], source_weights_np[source_mask], psi_prime_bin_edges))
        mean_target.append(_hist_density_mean(target_psi_prime[target_mask], target_weights_np[target_mask], psi_prime_bin_edges))
        mean_reweighted.append(_hist_density_mean(source_psi_prime[source_mask], predicted_weights[source_mask], psi_prime_bin_edges))

    save_mean_vs_slice_plot(
        x_centers=centers,
        source_means=mean_source,
        target_means=mean_target,
        reweighted_means=mean_reweighted,
        x_label=slice_variable,
        slice_name=slice_type,
        unit=unit,
        process=process,
        category=category,
        output_dir=process_pics_folder,
    )


def add_derived_columns(df, psi_prime_bin_edges=None):
    """Add muon_pt_gev, recoil_gev/mev and psi_prime columns to a source/target dataframe in place."""
    muon_py = df['leading_muon_py'].to_numpy()
    muon_pz = df['leading_muon_pz'].to_numpy()
    muon_px = np.zeros_like(muon_py)

    df['muon_pt_gev'] = np.abs(muon_py)
    df['recoil_gev'] = np.nan_to_num(df['total_proton_KE'].to_numpy(), nan=0.0)
    df['recoil_mev'] = 1000.0 * df['recoil_gev']
    df['psi_prime'] = get_psi_prime_from_fs_kinematics(
        recoil_gev=df['recoil_gev'].to_numpy(),
        muon_px_beam=muon_px,
        muon_py_beam=muon_py,
        muon_pz_beam=muon_pz,
    )
    return df


def validate_variable_definitions(category_name, category_cfg):
    """
    Check that every variable listed in reweight_variables / drawing_variables is
    something the code actually knows how to compute: either a raw variable_expr
    pulled from the flat trees, a derived variable (see DERIVED_VARIABLE_DEPS), or
    'weight' (only valid for drawing, since it's not a real source-sample column).
    Raises ValueError with all problems found, rather than failing later with a
    confusing KeyError deep inside training or plotting.
    """
    variable_exprs = set(category_cfg['variable_exprs'])
    known_for_training = variable_exprs | DERIVED_VARIABLES
    known_for_drawing = known_for_training | {'weight'}

    errors = []

    missing_base = [v for v in ALWAYS_REQUIRED_BASE_VARIABLES if v not in variable_exprs]
    if missing_base:
        errors.append(
            f"variable_exprs is missing {missing_base}, which are required to compute "
            f"the psi_prime diagnostic and the reaction-frame transform that always run."
        )

    missing_particle_components = []
    for particle_name in category_cfg['particle_names']:
        for component in ('px', 'py', 'pz'):
            var = f'{particle_name}_{component}'
            if var not in variable_exprs:
                missing_particle_components.append(var)
    if missing_particle_components:
        errors.append(
            f"particle_names {category_cfg['particle_names']} require {missing_particle_components} "
            f"in variable_exprs (needed to rotate momenta into the reaction frame)."
        )

    unknown_reweight = [v for v in category_cfg['reweight_variables'] if v not in known_for_training]
    if unknown_reweight:
        errors.append(
            f"reweight_variables {unknown_reweight} have no definition. Add them to "
            f"variable_exprs, or use one of the derived variables {sorted(DERIVED_VARIABLES)}."
        )

    unknown_drawing = [v for v in category_cfg['drawing_variables'] if v not in known_for_drawing]
    if unknown_drawing:
        errors.append(
            f"drawing_variables {unknown_drawing} have no definition. Add them to "
            f"variable_exprs, use one of the derived variables {sorted(DERIVED_VARIABLES)}, or 'weight'."
        )

    for derived_var in DERIVED_VARIABLES:
        if derived_var in category_cfg['reweight_variables'] or derived_var in category_cfg['drawing_variables']:
            missing_deps = [d for d in DERIVED_VARIABLE_DEPS[derived_var] if d not in variable_exprs]
            if missing_deps:
                errors.append(
                    f"'{derived_var}' is used but its dependencies {missing_deps} are "
                    f"missing from variable_exprs."
                )

    if errors:
        raise ValueError(
            f"Invalid variable configuration for category '{category_name}':\n- " + "\n- ".join(errors)
        )


def main():
    p = argparse.ArgumentParser(description='Train BDT reweighter by reaction channel, driven by a YAML config.')
    p.add_argument('--config', '-c', type=str, required=True, help='Path to the YAML configuration file.')
    p.add_argument('--source_path', '-s', type=str, help='Path to the source model ROOT file.')
    p.add_argument('--target_path', '-t', type=str, help='Path to the target model ROOT file.')
    p.add_argument('--module_path', '-m', type=str, help='Path to the BDTReweight module.')
    p.add_argument('--model_name', type=str, help='Identifier of the target model.')
    p.add_argument('--build_tree_of_weights', action='store_true', help='Activate building a ROOT TTree with the reweighting weights.')
    p.add_argument('--shape_only', action='store_true', help='Only reweight shape, do not change total cross section')
    p.add_argument('--max_events', type=int, default=None, help='Maximum number of events to use for training (for both source and target).')
    p.add_argument('--plots_dir', type=str, default=None, help='Full output directory for plots. If set, this path is used directly.')
    p.add_argument('--category', type=str, default='0p0n', help='Reaction category to train on (e.g. 0p0n, 1p0n, etc.), as defined in the config file.')
    args = p.parse_args()

    if args.module_path:
        sys.path.append(args.module_path)

    cfg = load_config(args.config)

    category = args.category
    if category not in cfg['categories']:
        raise ValueError(f"Unknown category '{category}'. Available: {list(cfg['categories'].keys())}")

    category_cfg = cfg['categories'][category]
    validate_variable_definitions(category, category_cfg)

    particle_counts = category_cfg['particle_counts']
    variable_exprs = category_cfg['variable_exprs']
    reweight_variables = category_cfg['reweight_variables']
    particle_names = category_cfg['particle_names']
    drawing_variables = category_cfg['drawing_variables']

    ke_thresholds = cfg['ke_thresholds']
    processes = cfg['processes']
    process_names = [proc['name'] for proc in processes]
    binning = cfg['binning']
    output_cfg = cfg['output']

    target_path = args.target_path
    source_path = args.source_path

    if args.model_name:
        target_model_name = args.model_name
    else:
        target_model_name = pathlib.Path(target_path).stem
        target_model_name = re.search(r'MINERvAflux_([^_]+)_', target_model_name).group(1)
        if target_model_name is None:
            print("CAN'T IDENTIFY TARGET MODEL NAME! ABORT!")
            exit()

    print(f'Reweighting to target model: {target_model_name}')

    tree_source_train = uproot.open(source_path)['EventKinematics_truth'].arrays(library='pd')
    if args.max_events is not None:
        source_max_events = int(1.5 * args.max_events)
        print(f"Limiting number of events to {source_max_events} for source model.")
        tree_source_train = tree_source_train.iloc[:source_max_events]

    tree_source_train['topology'] = tree_source_train['topology'].map(cfg['topology_codes'])
    tree_source_train = tree_source_train.rename(columns={
        'muon_px': 'leading_muon_px', 'muon_py': 'leading_muon_py', 'muon_pz': 'leading_muon_pz',
        'sum_p_px': 'total_proton_px', 'sum_p_py': 'total_proton_py', 'sum_p_pz': 'total_proton_pz',
        'sum_Tp': 'total_proton_KE', 'leading_n_px': 'leading_neutron_px',
        'leading_n_py': 'leading_neutron_py', 'leading_n_pz': 'leading_neutron_pz',
        'leading_p_px': 'leading_proton_px', 'leading_p_py': 'leading_proton_py',
        'leading_p_pz': 'leading_proton_pz', 'subleading_p_px': 'subleading_proton_px',
        'subleading_p_py': 'subleading_proton_py', 'subleading_p_pz': 'subleading_proton_pz',
    })

    if args.plots_dir is not None:
        plot_root = pathlib.Path(args.plots_dir).expanduser().resolve()
    else:
        plot_root = pathlib.Path(args.module_path) / "pics" / target_model_name
    plot_root.mkdir(parents=True, exist_ok=True)
    pics_folder_name = str(plot_root) + "/"

    plt.figure()
    plt.hist(
        tree_source_train[tree_source_train['topology'] == category]['total_proton_KE'],
        bins=300, label='source model', alpha=0.5, range=(0.001, 2.),
        weights=tree_source_train[tree_source_train['topology'] == category]['init_wgt'],
    )
    plt.xlabel(r'$\sum T_{p}$ [GeV]')
    plt.ylabel('counts')
    plt.savefig(f'{pics_folder_name}sum_Tp_source_model_{category}.png')
    print(f"Saved sum_Tp_source_model_{category}.png")
    plt.close()

    source_train = {}
    for topology in cfg['topology_codes'].values():
        source_train[topology] = tree_source_train[tree_source_train['topology'] == topology].copy()

    # Load the target tree to compute the total cross section.
    tree_target_train = NuisanceFlatTree(target_path)
    target_is_from_hadded = False
    target_ccqelike_xsec = tree_target_train.get_total_xsec()
    if target_is_from_hadded:
        target_ccqelike_xsec /= 10

    # Per-nucleon basis correction. NUISANCE reports the target cross section per
    # total nucleon (incl. hydrogen for a composite CH target), while the source is
    # normalized per A_source nucleon. Rescale the target xsec by A_target/A_source
    # onto the source's per-nucleon basis so the total-xsec ratio s is consistent.
    # For a carbon source vs a polystyrene (CH) target: A_source=12, A_target=13.
    A_source = float(cfg.get('A_source', 1.0))
    A_target = float(cfg.get('A_target', 1.0))
    nucleon_basis_correction = A_target / A_source
    if nucleon_basis_correction != 1.0:
        print(f"Per-nucleon basis correction A_target/A_source = {A_target:g}/{A_source:g} "
              f"= {nucleon_basis_correction:.4f} applied to target xsec "
              f"({target_ccqelike_xsec:.3e} -> {target_ccqelike_xsec * nucleon_basis_correction:.3e})")
        target_ccqelike_xsec *= nucleon_basis_correction

    if args.max_events is not None:
        print(f"Limiting number of events to {args.max_events} for target tree.")
        tree_target_train = NuisanceFlatTree(target_path, max_events=args.max_events)

    # Quick overview plot: event counts per category for source and target before any further processing.
    category_order = list(cfg['categories'].keys())
    source_counts = [int(np.sum(tree_source_train['topology'] == cat)) for cat in category_order]
    target_counts = []
    for cat in category_order:
        pc = cfg['categories'][cat]['particle_counts']
        mask = np.asarray(tree_target_train.get_mask_topology(particle_counts=pc, KE_thresholds=ke_thresholds), dtype=bool)
        target_counts.append(int(np.sum(mask)))

    x = np.arange(len(category_order))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.bar(x - width / 2, source_counts, width, label='Source', color='tab:green', alpha=0.7)
    ax.bar(x + width / 2, target_counts, width, label='Target', color='tab:blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(category_order, rotation=30, ha='right')
    ax.set_ylabel('Events')
    ax.set_title('Event counts per category (pre-selection)')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    for i, count in enumerate(source_counts):
        ax.text(x[i] - width / 2, count, f"{count}", ha='center', va='bottom', fontsize=8, color='tab:green')
    for i, count in enumerate(target_counts):
        ax.text(x[i] + width / 2, count, f"{count}", ha='center', va='bottom', fontsize=8, color='tab:blue')
    category_plot_name = f"{pics_folder_name}category_counts_source_target.png"
    fig.tight_layout()
    fig.savefig(category_plot_name)
    print(f"Saved category count plot to {category_plot_name}")
    plt.close(fig)

    # GENIEv3 has a bug with events with zero proton KE. Remove them.
    if target_model_name.startswith('GENIEv3'):
        target_rows_before = tree_target_train.get_n_entries()
        positive_recoil_mask = np.asarray(tree_target_train.get_mask_positive_recoil_energy(), dtype=bool)
        tree_target_train.update_tree_with_mask(positive_recoil_mask)
        removed_zero_ke = target_rows_before - tree_target_train.get_n_entries()
        print("Because this is model GENIEv3, we remove events with zero proton kinetic energy cause they look ill-defined in the target flat tree.")
        print(f"Removed {removed_zero_ke} target events with zero proton kinetic energy")

    # extract cross section from source model file
    import ROOT
    source_file = ROOT.TFile(source_path)
    h_xsec_ccqelike = ROOT.TH1D(source_file.Get('h_eventRate_qelike_cross_section'))
    source_ccqelike_xsec = h_xsec_ccqelike.GetBinContent(1)
    h_xsec_total = ROOT.TH1D(source_file.Get('h_eventRate_mc_cross_section'))
    source_total_xsec = h_xsec_total.GetBinContent(1)
    print(f"Total xsec from source model: {source_total_xsec * 1e38:.2f} x 10^-38 cm^2")
    print(f"Total CCQELike xsec from source model: {source_ccqelike_xsec * 1e38:.2f} x 10^-38 cm^2")
    print(f"Total CCQELike xsec from target model: {target_ccqelike_xsec * 1e38:.2f} x 10^-38 cm^2")

    scale_source_train = 1.0
    scale_target_train = target_ccqelike_xsec / source_ccqelike_xsec
    if args.shape_only:
        print('Ignoring total cross section and modifying only shape')
        scale_target_train = 1.0

    source_total = len(source_train[category])
    print("Number of events:")
    print(f"SOURCE: True QE events:      {np.sum(source_train[category]['reactionCode']==1)} ({np.sum(source_train[category]['reactionCode']==1)/source_total*100:.2f} %)")
    print(f"SOURCE: True 2p2h events:    {np.sum(source_train[category]['reactionCode']==2)} ({np.sum(source_train[category]['reactionCode']==2)/source_total*100:.2f} %)")
    print(f"SOURCE: True RES+DIS events: {np.sum(source_train[category]['reactionCode']>2)} ({np.sum(source_train[category]['reactionCode']>2)/source_total*100:.2f} %)")
    target_total = len(tree_target_train._flattree_vars)
    print(f"TARGET: True QE events:      {np.sum(tree_target_train.get_mode()==1)} ({np.sum(tree_target_train.get_mode()==1)/target_total*100:.2f} %)")
    print(f"TARGET: True 2p2h events:    {np.sum(tree_target_train.get_mode()==2)} ({np.sum(tree_target_train.get_mode()==2)/target_total*100:.2f} %)")
    print(f"TARGET: True RES+DIS events: {np.sum(tree_target_train.get_mode()>2)} ({np.sum(tree_target_train.get_mode()>2)/target_total*100:.2f} %)")

    print(f"Scale target/source: {scale_target_train:.2f} = {target_ccqelike_xsec:.2e} / {source_ccqelike_xsec:.2e} ")

    # Per-process cross section ratios (target/source), computed generically over the
    # configured processes rather than a hard-coded QE/2p2h/Oth split.
    source_reaction_code = source_train[category]['reactionCode'].to_numpy()
    source_init_wgt = source_train[category]['init_wgt'].to_numpy()
    target_mode_all = tree_target_train.get_mode()

    source_total_event_rate = scale_source_train * np.sum(source_init_wgt)
    target_total_event_rate = scale_target_train * float(source_total / target_total) * target_total

    xsec_ratio = {}
    print("Cross section ratios (target/source):")
    for proc in processes:
        proc_name = proc['name']
        proc_source_mask = apply_rule(source_reaction_code, proc['reaction_code_rule'])
        proc_target_mask = apply_rule(target_mode_all, proc['mode_rule'])

        source_event_rate = scale_source_train * np.sum(source_init_wgt[proc_source_mask])
        target_event_rate = scale_target_train * float(source_total / target_total) * np.sum(proc_target_mask)

        percent_source = source_event_rate / source_total_event_rate * 100 if source_total_event_rate > 0 else np.nan
        percent_target = target_event_rate / target_total_event_rate * 100 if target_total_event_rate > 0 else np.nan

        xsec_ratio[proc_name] = scale_target_train * percent_target / percent_source if percent_source > 0 else np.nan
        print(f"|  {proc_name}: {xsec_ratio[proc_name]:.2f} (source {percent_source:.2f}%, target {percent_target:.2f}%)")

    print(f"Training on variables: {', '.join(reweight_variables)}")

    dict_to_tree = {}

    # Accumulated across the per-process loop below to produce a combined
    # "all processes" diagnostic view after it finishes.
    all_processes_source_test_list = []
    all_processes_target_test_list = []
    all_processes_weights_list = []

    for proc in processes:
        process = proc['name']
        process_pics_folder = f'{pics_folder_name}{process}/'
        os.makedirs(process_pics_folder, exist_ok=True)

        target_mask = np.asarray(tree_target_train.get_mask_topology(particle_counts=particle_counts, KE_thresholds=ke_thresholds), dtype=bool)
        print(f"\nReweighting process: {process}")
        source_mask = apply_rule(source_train[category]['reactionCode'].to_numpy(), proc['reaction_code_rule'])
        target_mask &= apply_rule(tree_target_train.get_mode(), proc['mode_rule'])

        target_train_cat = create_dataframe_from_nuisance(tree_target_train, variable_exprs=variable_exprs, mask=target_mask)
        target_train_cat = transform_momentum_to_reaction_frame(target_train_cat, selector_lepton='leading_muon', particle_names=particle_names)
        target_train_cat['weight'] = scale_target_train

        n_negative_ke = np.sum(target_train_cat['total_proton_KE'] < 0)
        if n_negative_ke > 0:
            print(f"Warning: found {n_negative_ke} events with negative total_proton_KE in target_train for category {category}. These events will be dropped.")
            target_train_cat = target_train_cat[target_train_cat['total_proton_KE'] >= 0]

        source_train_p = source_train[category][source_mask].copy()
        target_train_p = target_train_cat.copy()

        add_derived_columns(source_train_p)
        add_derived_columns(target_train_p)

        source_test_p = source_train_p.iloc[np.arange(0, int(len(source_train_p) / 10), 1)].copy()
        target_test_p = target_train_p.copy()

        print(f"Source sample shape: {source_train_p[reweight_variables].shape}")
        print(f"Target sample shape: {target_train_p[reweight_variables].shape}")

        print("Fitting reweighter...")
        reweighter = Reweighter(n_estimators=100, learning_rate=0.4, max_depth=4, min_samples_leaf=30, gb_args={'subsample': 1.0})
        reweighter.fit(original=source_train_p[reweight_variables], target=target_train_p[reweight_variables])
        reweighter.set_weight_normalization_factor(original=source_train_p[reweight_variables])
        reweighter.set_xsec_scale_factor(xsec_ratio[process])

        print(f"Set cross-section scale factor in reweighter to {reweighter.xsec_scale_factor:.2f}")
        print(f"Set weight normalization factor in reweighter to {reweighter.norm_factor:.2f}")

        print("Saving model ...", end='')
        output_model_path = pathlib.Path(target_path).parent / 'BDTreweight_outputs'
        output_model_path.mkdir(parents=True, exist_ok=True)
        pickle_output_file = output_model_path / target_model_name / process / output_cfg['model_pickle_template'].format(category=category)
        os.makedirs(pickle_output_file.parent, exist_ok=True)
        pickle.dump(reweighter, open(pickle_output_file, 'wb'), protocol=4)
        print(f" Done. Pickle saved to {pickle_output_file}")

        # Legacy bulk-predicted weights, kept only to populate the exported
        # ROOT weight tree below (dict_to_tree) -- unrelated to diagnostics.
        all_weights = reweighter.predict_matched_total_weights(
            source_train_p[reweight_variables],
            target_weight=target_train_p['weight'] * float(len(source_train_p)) / len(target_train_p),
        )

        # Diagnostics use predict_weight_single_event exclusively: this is the
        # per-event call the production C++ binding actually makes, so it's
        # what needs to be validated here, not the bulk/vectorized predictor.
        print(f"Computing per-event weights via predict_weight_single_event for {len(source_test_p)} test events (this is what the production C++ binding calls)...")
        source_test_features = source_test_p[reweight_variables].to_numpy()
        single_event_weights = np.array([
            reweighter.predict_weight_single_event(feature_row)
            for feature_row in tqdm(source_test_features, desc=f"Computing weights for {process}", leave=False)
        ])

        target_n_events = np.sum(target_test_p['weight'])
        source_n_events_before = np.sum(source_test_p['init_wgt'])
        source_n_events_after = np.sum(single_event_weights)
        print(f"Target n. events: {target_n_events}")

        fig = draw_source_target_distributions_and_ratio(
            source_test_p, target_test_p,
            variables=drawing_variables,
            source_weights=source_test_p['init_wgt'],
            target_weights=target_test_p['weight'] * float(len(source_test_p)) / len(target_test_p),
            new_source_weights=single_event_weights,
            legends=['Source', 'Source (Reweighted)', 'Target'],
        )
        fig.suptitle(f'Reweighting Result for process: {process} in category: {category}', fontsize=16)
        fig.savefig(f'{process_pics_folder}ReweightingResult_{process}_{category}.png')
        print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}.png")
        plt.close()

        fig = draw_source_target_distributions_and_ratio(
            source_test_p, target_test_p,
            variables=drawing_variables,
            source_weights=source_test_p['init_wgt'],
            target_weights=target_test_p['weight'] * float(len(source_test_p)) / len(target_test_p),
            new_source_weights=single_event_weights,
            legends=['Source', 'Source (Reweighted)', 'Target'],
            shape_only=True,
        )
        fig.suptitle(f'Shape only. Process: {process} in category: {category}', fontsize=16)
        fig.savefig(f'{process_pics_folder}ReweightingResult_{process}_{category}_Shape.png')
        print(f"Saved reweighting result figure to ReweightingResult_{process}_{category}_Shape.png")
        plt.close()

        source_weights_np = source_test_p['init_wgt'].to_numpy()
        target_weights_np = target_test_p['weight'].to_numpy()
        print(f"Source weights average: {np.mean(source_weights_np):.4e}, length: {len(source_weights_np)}")
        print(f"Target weights average: {np.mean(target_weights_np):.4e}, length: {len(target_weights_np)}")

        print("Producing per-process psi-prime plots in muon pT slices...")
        produce_slice_diagnostics(
            source_test_p, target_test_p, source_weights_np, target_weights_np, single_event_weights,
            slice_variable='muon_pt_gev', bin_edges=binning['muon_pt_gev'], unit='GeV', slice_type='pt',
            process_pics_folder=process_pics_folder, process=process, category=category,
            psi_prime_bin_edges=binning['psi_prime'],
            n_source_train=len(source_test_p), n_target_train=len(target_test_p),
        )

        print("Producing per-process psi-prime plots in recoil slices...")
        produce_slice_diagnostics(
            source_test_p, target_test_p, source_weights_np, target_weights_np, single_event_weights,
            slice_variable='recoil_mev', bin_edges=binning['recoil_mev'], unit='MeV', slice_type='recoil',
            process_pics_folder=process_pics_folder, process=process, category=category,
            psi_prime_bin_edges=binning['psi_prime'],
            n_source_train=len(source_test_p), n_target_train=len(target_test_p),
        )

        dict_process = {
            'eventID': source_train_p['eventID'],
            'originalTreeEntry': source_train_p['originalTreeEntry'],
            'init_wgt': source_train_p['init_wgt'],
            'weight': all_weights,
        }
        for key in dict_process.keys():
            dict_to_tree.setdefault(key, [])
            dict_to_tree[key].extend(dict_process[key])

        print(f"Total event rate before reweighting for process {process}: {source_n_events_before:.2f}")
        print(f"Total event rate after reweighting for process {process}: {source_n_events_after:.2f}")

        all_processes_source_test_list.append(source_test_p)
        all_processes_target_test_list.append(target_test_p)
        all_processes_weights_list.append(single_event_weights)

    # ========================================================================
    # WEIGHTS FROM predict_weight_single_event FOR ALL PROCESSES COMBINED
    # ========================================================================
    print("\n" + "=" * 80)
    print("WEIGHTS FROM predict_weight_single_event FOR ALL PROCESSES COMBINED")
    print("=" * 80)

    all_processes_source_test = pd.concat(all_processes_source_test_list, ignore_index=True) if all_processes_source_test_list else pd.DataFrame()
    all_processes_target_test = pd.concat(all_processes_target_test_list, ignore_index=True) if all_processes_target_test_list else pd.DataFrame()
    all_predicted_weights_array = np.concatenate(all_processes_weights_list) if all_processes_weights_list else np.array([])

    all_process_pics_folder = f'{pics_folder_name}all_processes/'
    os.makedirs(all_process_pics_folder, exist_ok=True)

    print(f"Size of all_processes_source_test: {len(all_processes_source_test)}, size of all_processes_target_test: {len(all_processes_target_test)}")

    print("Producing all-processes psi-prime plots in muon pT slices...")
    produce_slice_diagnostics(
        all_processes_source_test, all_processes_target_test,
        np.ones(len(all_processes_source_test)), np.ones(len(all_processes_target_test)), all_predicted_weights_array,
        slice_variable='muon_pt_gev', bin_edges=binning['muon_pt_gev'], unit='GeV', slice_type='pt',
        process_pics_folder=all_process_pics_folder, process='all', category=category,
        psi_prime_bin_edges=binning['psi_prime'],
        n_source_train=len(all_processes_source_test), n_target_train=len(all_processes_target_test),
    )

    print("Producing all-processes psi-prime plots in recoil slices...")
    produce_slice_diagnostics(
        all_processes_source_test, all_processes_target_test,
        np.ones(len(all_processes_source_test)), np.ones(len(all_processes_target_test)), all_predicted_weights_array,
        slice_variable='recoil_mev', bin_edges=binning['recoil_mev'], unit='MeV', slice_type='recoil',
        process_pics_folder=all_process_pics_folder, process='all', category=category,
        psi_prime_bin_edges=binning['psi_prime'],
        n_source_train=len(all_processes_source_test), n_target_train=len(all_processes_target_test),
    )

    global_avg_predicted = np.mean(all_predicted_weights_array) if len(all_predicted_weights_array) else np.nan
    global_total_predicted = np.sum(all_predicted_weights_array)
    print("-" * 80)
    print(f"Global       : n_events={len(all_predicted_weights_array):6d}, avg_weight={global_avg_predicted:.3f}, total_weight={global_total_predicted:.3f} [xsec ratio: {scale_target_train:.2f}]")
    print("=" * 80 + "\n")

    # sort dict_to_tree entries by originalTreeEntry
    sorted_indices = np.argsort(dict_to_tree['originalTreeEntry'])
    for key in dict_to_tree.keys():
        dict_to_tree[key] = np.array(dict_to_tree[key])[sorted_indices]

    output_folder = pathlib.Path(target_path).parent / 'BDTreweight_outputs'
    output_folder.mkdir(parents=True, exist_ok=True)
    source_basename = pathlib.Path(source_path).stem
    match = re.search(r'minervame..', source_basename)
    playlist_name = match.group(0) if match else 'unknownPlaylist'

    if args.build_tree_of_weights:
        output_root_file = output_folder / output_cfg['weight_tree_filename_template'].format(
            playlist=playlist_name, model=target_model_name, category=category
        )
        with uproot.recreate(output_root_file) as f_out:
            f_out.mktree(output_cfg['weight_tree_name'], dict_to_tree)

        f_in = uproot.open(output_root_file)
        tree_in = f_in[output_cfg['weight_tree_name']]
        tree_in.show()
        print("")
        print(tree_in.arrays(library='pd'))

        print(f"Produced weights saved to {output_root_file}")


if __name__ == '__main__':
    main()
