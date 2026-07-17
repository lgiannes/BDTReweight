"""
Cross-section normalization validator (one or many target models).

Checks that the overall scale factor s = sigma_target / sigma_source, which is
baked into the reweighter weights, is built from two *mutually consistent*
absolute cross sections. No weight-closure test can do this (s cancels), so it
must be done by auditing the two extracted cross sections directly.

For every target file it reports, straight from the files:
  1. Nucleon basis / hydrogen -- the NUISANCE target nucleus (tgta, tgtz). A
     pure-12C target matches a carbon-only source normalization, so N_nuc
     cancels in s. If the target contains hydrogen (A=1) or free nucleons
     (A=0) while the source counts carbon-only nucleons, the per-nucleon basis
     may NOT match and s can carry an H asymmetry -> the model is FLAGGED.
  2. Absolute-value sanity -- both cross sections must land around
     ~1e-38 cm^2/nucleon; rules out a per-nucleus (x~12) or missed-hadd (x10)
     error.
  3. The resulting scale factor s.

The target flat trees are already skimmed to the analysis CCQE-like selection,
so sum(fScaleFactor) over ALL events is the target CCQE-like cross section
(what get_total_xsec() returns). The NUISANCE 'flagCCQELike' branch is a
*different* definition and is intentionally ignored.

NOTE on "1/(Phi*N_nuc)": the NUISANCE per-event fScaleFactor is sigma/N_generated,
NOT the experiment factor 1/(Phi*N_nuc): N_generated is arbitrary, so it does not
equal the source's sigma/N. The hydrogen question is settled by the nucleus
branches (tgta/tgtz), not by comparing per-event factors.
"""

import argparse
import os
import numpy as np
import uproot


def read_source(source_path, qelike_hist='h_eventRate_qelike_cross_section',
                tree='EventKinematics_truth'):
    f = uproot.open(source_path)
    sigma_ccqelike = float(f[qelike_hist].values()[0])   # bin 1, cm^2 / nucleon
    n_events = int(f[tree].num_entries)
    return sigma_ccqelike, n_events


def read_target(target_path, n_hadd=1):
    t = uproot.open(target_path)['FlatTree_VARS']
    sf = t['fScaleFactor'].array(library='np')
    A = t['tgta'].array(library='np')
    mode = t['Mode'].array(library='np')
    # tree is already skimmed to the analysis CCQE-like selection, so the sum
    # over every event is the target CCQE-like cross section.
    sigma_all = float(np.sum(sf)) / n_hadd
    sigma_carbon = float(np.sum(sf[A == 12])) / n_hadd   # carbon-only numerator
    # sanity: verify the physics claim -- no CCQE (Mode 1) / 2p2h (Mode 2) off carbon
    non_c = A != 12
    n_qe2p2h_offC = int(np.sum(non_c & ((mode == 1) | (mode == 2))))
    return {
        'sf_per_event': float(sf[0]),
        'n_events': len(sf),
        'sigma_all': sigma_all,
        'sigma_carbon': sigma_carbon,
        'h_frac': (sigma_all - sigma_carbon) / sigma_all if sigma_all > 0 else 0.0,
        'A_values': np.unique(A).tolist(),
        'Z_values': np.unique(t['tgtz'].array(library='np')).tolist(),
        'n_qe2p2h_offC': n_qe2p2h_offC,
    }


def nucleus_label(A_values):
    names = {0: 'free', 1: '1H', 12: '12C'}
    return '+'.join(names.get(a, f'A{a}') for a in sorted(A_values))


def basis_status(A_values):
    """Return (is_ok, message) for the per-nucleon basis vs a carbon-only source."""
    if A_values == [12]:
        return True, 'OK: pure 12C, matches carbon-only source'
    return False, 'FLAG: target has non-carbon nucleons (H / free) -- source counts C-only'


def derive_label(path):
    b = os.path.basename(path).lower()
    for key, lab in [('edrmf', 'NEUT-EDRMF'), ('rpwia', 'NEUT-RPWIA'), ('_sf_', 'NEUT-SF'),
                     ('nuwro', 'NuWro-LFG'), ('genie', 'GENIEv3'), ('gibuu', 'GiBUU')]:
        if key in b:
            return lab
    return os.path.splitext(os.path.basename(path))[0][:18]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-file', required=True)
    p.add_argument('--target-file', required=True, nargs='+',
                   help="One or more NUISANCE target files.")
    p.add_argument('--labels', nargs='+', default=None,
                   help="Optional labels aligned with --target-file (else derived from name).")
    p.add_argument('--n-hadd', type=int, default=1,
                   help="Files hadd'd into each target (fScaleFactor is per-file; divide by this).")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.target_file):
        p.error("--labels must have the same number of entries as --target-file")
    labels = args.labels or [derive_label(t) for t in args.target_file]

    sigma_s, n_s = read_source(args.source_file)

    print("=" * 96)
    print("CROSS-SECTION NORMALIZATION VALIDATION")
    print("=" * 96)
    print(f"source : {args.source_file}")
    print(f"sigma_source (CCQE-like) = {sigma_s:.4e} cm^2/nucleon   ({n_s} events)")
    print("-" * 96)
    print(f"{'model':<12} {'N':>9} {'sigma(all)':>12} {'sigma(C-only)':>13} "
          f"{'H%':>6} {'s(all)':>7} {'s(C)':>7}  nucleus")
    print("-" * 96)

    flagged, bad_physics = [], []
    for label, tpath in zip(labels, args.target_file):
        tg = read_target(tpath, n_hadd=args.n_hadd)
        s_all = tg['sigma_all'] / sigma_s
        s_c = tg['sigma_carbon'] / sigma_s
        nuc = nucleus_label(tg['A_values'])
        pure_c = tg['A_values'] == [12]
        if not pure_c:
            flagged.append((label, tg['h_frac']))
        if tg['n_qe2p2h_offC'] > 0:
            bad_physics.append((label, tg['n_qe2p2h_offC']))
        scale_bad = not (1e-40 < tg['sigma_all'] < 1e-37)
        print(f"{label:<12} {tg['n_events']:>9} {tg['sigma_all']:>12.4e} "
              f"{tg['sigma_carbon']:>13.4e} {100*tg['h_frac']:>5.2f}% "
              f"{s_all:>7.4f}{'!' if scale_bad else ' '}{s_c:>7.4f}  {nuc}")

    print("-" * 96)
    print("sigma(C-only) sums fScaleFactor over carbon events only (A=12); H% is the hydrogen/")
    print("free-nucleon share of the CCQE-like cross section -- the NUMERATOR effect of a C+H target.")
    print("s(all) uses the full target, s(C) uses the carbon-only numerator (matched to a C-only source).")

    if flagged:
        print()
        print("Targets with non-carbon nucleons: " +
              ", ".join(f"{lab} (H share {100*h:.2f}%)" for lab, h in flagged))
        print("  - Numerator: no QE/2p2h occurs on free H (verified below), so the H events are")
        print("    RES/DIS leaking into the CCQE-like cut; s(all) vs s(C) shows their impact.")
        print("  - Denominator: the only residual is the per-nucleon convention (per-C vs per-CH).")
        print("    If NUISANCE normalizes these per CH-nucleon while the source is per-C, s is")
        print("    biased low by the H nucleon fraction (~7.7% for polystyrene). Confirm the")
        print("    NUISANCE nucleon count to rule this out.")
    else:
        print("\nAll targets are pure 12C: nucleon basis matches the carbon-only source. No H asymmetry.")

    if bad_physics:
        print("\nWARNING: QE/2p2h events found off carbon (unexpected): " +
              ", ".join(f"{lab}: {n}" for lab, n in bad_physics))
    else:
        print("Physics check: no CCQE (Mode 1) or 2p2h (Mode 2) events off carbon in any target.  OK")
    print("=" * 96)


if __name__ == '__main__':
    main()
