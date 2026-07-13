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
