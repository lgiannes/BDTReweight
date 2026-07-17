# BDTReweight
MINERvA's boosted decision tree reweight of Monte Carlo neutrino interaction events based on hep_ml.reweight and NUISANCE flat tree event record format. 
Requirements:
* python3.11
* hep_ml
* uproot
* awkward
* numpy
* pandas
* matplotlib

## Running a training

Training is driven by `train_by_reaction_config.py`, which reweights a
"source" MC sample (`EventKinematics_truth` tree) to match a "target"
NUISANCE flat tree (`FlatTree_VARS`) sample, separately per final-state
topology category (0p0n, 0pNn, 1p0n, ...) and per physics process (QE, 2p2h,
Oth, ...), as defined in a YAML config (e.g. `train_by_reaction_config.yaml`).

Example invocation (see `run_train_by_reaction_config_local.sh` for a working
local example):

```bash
module_path="/path/to/fork_reweighting/"     # parent directory of this BDTReweight repo
config="${module_path}/BDTReweight/train_by_reaction_config.yaml"

PYTHONPATH=${module_path}:${PYTHONPATH} \
python3 ${module_path}/BDTReweight/train_by_reaction_config.py \
    --config $config \
    --source_path /path/to/ReweightSourceCCQELike_....root \
    --target_path /path/to/neut_..._NUISFLAT_CCQELike.root \
    --module_path $module_path \
    --model_name NEUT-SF
```

`--module_path` must be the directory that *contains* the `BDTReweight`
package (i.e. `fork_reweighting/`, not `fork_reweighting/BDTReweight/`), and
should also be on `PYTHONPATH` so `from BDTReweight...` imports resolve.

Useful optional flags:
* `--category` — restrict training to a single topology category (default `0p0n`).
* `--max_events` — cap the number of events read from source/target, for a quick test run.
* `--build_tree_of_weights` — also export a ROOT tree of per-event weights for the target sample.
* `--shape_only` — train reweighters using only shape information (ignore overall normalization).
* `--plots_dir` — override the default diagnostic-plots output location.

### Output

A training run produces, per process (QE, 2p2h, Oth, ... as defined by
`processes` in the YAML config) and per topology category:

* **Trained model pickle** — `GBReweighterModel_<category>.pkl`, written to
  `<target_path's parent dir>/BDTreweight_outputs/<model_name>/<process>/`.
  This is the artifact consumed downstream (e.g. by the
  [C++ usage example](usage_example_cpp/README.md) or production analysis
  code via `predict_weight_single_event`).
* **Diagnostic plots**, written to `<module_path>/pics/<model_name>/<process>/`
  (or `--plots_dir` if given), including:
  * `ReweightingResult_<process>_<category>.png` / `..._Shape.png` — source
    vs. target vs. reweighted-source distributions for each reweight variable.
  * `PsiPrimeSlice_bin<N>_<process>_<category>.png` (and `recoilSlice`
    variants) — sliced comparisons in bins of `pT`/recoil.
  * `mean_vs_pt_...png` / `mean_vs_recoil_...png` — mean reweighted-vs-target
    trends across those slices.
  * `sum_Tp_source_model_<category>.png`, `category_counts_source_target.png`
    — sanity-check plots of the input samples themselves.
  * The same plots aggregated across all processes live under an
    `all_processes/` subfolder.
* **Optional weight tree** (only with `--build_tree_of_weights`) — a ROOT
  file under the same `BDTreweight_outputs/` folder containing per-event
  weights for the target sample, named per the `output.weight_tree_filename_template`
  / `weight_tree_name` entries in the YAML config.

## Validating a trained model (`test_training.py`)

`test_training.py` does **not** train anything: it loads the trained
reweighter pickles for one topology category and applies them to a source
sample, producing diagnostic plots and a normalization audit that compare
**source**, **target**, and **reweighted source** distributions of
`psi_prime`. It exercises the finished models exactly as the production C++
binding does (through `predict_weight_single_event`), reading the first
`--max-events` events from the source and the *same number* from the target so
the two are compared on an equal footing — which is what makes the overall
normalization directly comparable between the two curves.

Example invocation (see `run_test_training.sh`):

```bash
module_path="/path/to/fork_reweighting/"
PYTHONPATH=${module_path}:${PYTHONPATH} \
python3 ${module_path}/BDTReweight/test_training.py \
    --source-file /path/to/ReweightSourceCCQELike_....root \
    --target-file /path/to/neut_..._NUISFLAT_CCQELike.root \
    --reweighter-folder /path/to/BDTreweight_outputs/NEUT-SF \
    --max-events 100000
```

Arguments:
* `--source-file` — source MC ROOT file (`EventKinematics_truth` tree).
* `--target-file` — target NUISANCE flat tree (`FlatTree_VARS`).
* `--reweighter-folder` — folder holding the trained pickles, with layout
  `<process>/GBReweighterModel_<category>.pkl` for `QE`, `2p2h`, `Oth`.
* `--max-events` — events read from source (and the matching count from
  target); default `100000`.
* `--category` — topology category to test (default `0p0n`).
* `--output-folder` — where to write plots (default
  `<reweighter-folder>/test_plots`).

Outputs (in `--output-folder`), for each process group **QE**, **2p2h**,
**Oth**, **QE+2p2h** (Oth excluded), and **all** processes combined:

* `PsiPrimeGrid_MuonPT_<group>_<category>.png` — `psi_prime` distributions in a
  5×3 grid of muon-pT slices (source / target / reweighted source overlaid).
* `PsiPrimeGrid_SumTp_<group>_<category>.png` — the same in a 4×3 grid of
  ΣT_p (recoil) slices.
* `MeanPsiPrime_vs_MuonPT_<group>_<category>.png` /
  `MeanPsiPrime_vs_SumTp_<group>_<category>.png` — mean `psi_prime` vs. the
  slice variable, with a reweighted-minus-target residual panel.
* `Distribution_all_vars_<category>.png` — combined source/target/reweighted
  distributions of the reweight variables.

It also prints an **overall-scale validation** table: per process, the mean
per-event weight `mean(w)` versus the `xsec_scale_factor` baked into each
pickle, and globally `mean(w)` versus the independently computed cross-section
ratio `s = sigma_target / sigma_source`, plus the predicted reweighted cross
section. This confirms the weights carry the intended normalization — a
consistency/regression check, not a proof that `s` itself is correct (that
depends on the two input cross sections being extracted on a matching
per-nucleon basis).
