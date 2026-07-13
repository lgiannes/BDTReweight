# C++ usage example

A minimal, standalone example of calling a trained BDT reweighter from C++,
modeled on production code's `reweighter_QE->GetWeight(mc, verbose)` pattern
(e.g. `LightSelection.cxx`) but stripped of the systematics/error-band/
HyperDimLinearizer machinery and the MAT/CCQENuUtils software stack.

It embeds a Python interpreter (via pybind11) inside a ROOT macro, loads a
trained `GBReweighterModel_0p0n.pkl`, reweights the 0p0n/QE source sample
event-by-event with the exact call production code makes
(`predict_weight_single_event`), and compares the result against the target
distribution for the two BDT training variables (total proton KE, leading
muon `py`) plus the derived `psi_prime` variable.

## Files

- `BDTReweighter.h` — the C++ interface (`ReweighterUtils::CCQELikeBDTReweighter`).
  Embeds Python, imports `BDTReweight_api.py`, and exposes `GetWeight0p0n(...)`.
- `BDTReweight_api.py` — thin Python wrapper that loads the pickled
  `Reweighter` models and calls `predict_weight_single_event()`
  (see `../reweighter.py`).
- `SetupPybind11.C` — one-time, interpreted ROOT setup script that registers
  the pybind11/Python include and link flags with ACLiC. Must be run
  **before** compiling anything that includes `BDTReweighter.h`.
- `CompareSourceToTarget.C` — the usage example itself: reads the source and
  target ROOT files, reweights the source, draws the three comparison plots.
- `compare_total_proton_KE.png`, `compare_leading_muon_py.png`,
  `compare_psi_prime.png` — example output, kept in the repo for reference.

## Requirements

- ROOT built with PyROOT/ACLiC support (any recent ROOT 6.x works; this was
  developed against a standard macOS Homebrew ROOT install).
- A `python3` on `PATH` with:
  - [`pybind11`](https://pybind11.readthedocs.io/) installed (needed at
    compile time by `SetupPybind11.C`, which shells out to
    `python3 -c "import pybind11; print(pybind11.get_include())"`).
  - `hep_ml` and the rest of this repo's Python dependencies (see the main
    [README.md](../README.md)) installed, since `BDTReweight_api.py`
    unpickles a `BDTReweight.reweighter.Reweighter` object, which requires
    `hep_ml` to be importable.
  - This must be the **same** `python3` (same environment) that trained and
    pickled the reweighter models, or unpickling can fail.
- A trained model directory containing `GBReweighterModel_0p0n.pkl`,
  produced by running `train_by_reaction_config.py` (see the main
  [README.md](../README.md) for how to produce one).
- The source (`ReweightSourceCCQELike_*.root`, `EventKinematics_truth` tree)
  and target (NUISANCE flat tree, `FlatTree_VARS`) ROOT files used to train
  that model, or any compatible files in the same format.

## Before running

Edit the hardcoded paths at the top of `CompareSourceToTarget()` in
`CompareSourceToTarget.C` to point at your own files:

```cpp
const TString source_file_path = "/path/to/ReweightSourceCCQELike_....root";
const TString target_file_path = "/path/to/neut_..._NUISFLAT_CCQELike.root";
const TString model_path = "/path/to/BDTreweight_outputs/<model_name>/QE";
```

`model_path` should be the directory holding `GBReweighterModel_0p0n.pkl`
for one process (e.g. `.../BDTreweight_outputs/NEUT-SF/QE`).

## Running

From this directory:

```
root -l
root [0] .x SetupPybind11.C
root [1] .x CompareSourceToTarget.C+
```

The two-step dance is required: ACLiC needs the pybind11/Python include and
link flags registered (by `SetupPybind11.C`) **before** it compiles anything
that `#include`s `BDTReweighter.h` (i.e. before the `.C+` compile of
`CompareSourceToTarget.C`). Running `SetupPybind11.C` also preloads the ROOT
graphics libraries (`libTree`, `libHist`, `libGraf`, `libGpad`) that ACLiC
needs to link `TH1D`/`TCanvas`/`TLegend` symbols, which aren't loaded by
default in batch mode.

On success, the macro prints progress for the source and target event loops
and writes three PNGs to the current directory:

- `compare_total_proton_KE.png`
- `compare_leading_muon_py.png`
- `compare_psi_prime.png`

Each plot overlays the raw source, BDT-reweighted source, and target
distributions (all normalized to unit area), so you can visually check that
reweighting pulls the source distribution towards the target.

## TODO

**Segfault fixed via `TH1::AddDirectory(kFALSE)` — worth double-checking if
this macro is ever restructured.** During development, `CompareSourceToTarget.C`
segfaulted partway through the target-event loop (after `source_file->Close()`
had already run). Root cause: ROOT's default `TH1::AddDirectory(kTRUE)`
behavior attaches every `new TH1D(...)` created while a `TFile` is open to
that file's current directory; calling `TFile::Close()` on that file then
deletes those directory-owned histograms, leaving dangling pointers that
corrupt the heap the next time they're filled/read. The six source-side
histograms in this macro are created while `source_file` is open, and
`source_file->Close()` is called before the macro is done using them — this
is exactly the failure mode.

The fix is the `TH1::AddDirectory(kFALSE);` call at the very top of
`CompareSourceToTarget()`, which makes all histograms created afterwards
independent of `gDirectory`/any open `TFile`. If you add new histograms, new
`TFile::Open`/`Close()` calls, or split this logic into helper functions
across multiple files, make sure `TH1::AddDirectory(kFALSE)` still runs
before the first `new TH1D(...)` (or set it again locally) — it's a global,
mutable interpreter/process setting, not scoped to this function.
