# Loads pickled BDT reweighter models and exposes per-topology
# predict_weight_* functions. This is the exact interface production C++
# analysis code calls (via BDTReweighter.h, which embeds this module with
# pybind11) -- each predict_weight_* below calls straight into
# Reweighter.predict_weight_single_event() (see ../reweighter.py).
#
# Adapted from the production BDTReweight_api.py used by
# CCQELikeBDTReweighter_L.h: the sys.path.append() of the BDTReweight
# package location is resolved relative to this file instead of hardcoded
# to a specific EOS path, so this usage example is portable. Override with
# the BDTREWEIGHT_PACKAGE_PATH environment variable if this file is moved
# out of the repository.

import gc
import os
import pickle
import sys

# This file lives at <repo_root>/usage_example_cpp/BDTReweight_api.py, and
# unpickling a Reweighter requires "BDTReweight" (i.e. <repo_root>, since
# the repo directory itself is named BDTReweight) to be importable as a
# package, matching how the pickles were created in train_by_reaction_config.py.
_package_path = os.environ.get(
    'BDTREWEIGHT_PACKAGE_PATH',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.append(_package_path)

# Normalization factors, one per topology category, applied on top of the
# reweighter's own prediction. Only 0p0n is trained/enabled by default in
# this codebase, so every other category is switched off here; flip these
# on once reweighters for the other categories exist.
CATEGORIES = ("0p0n", "0pNn", "1p0n", "1pNn", "2p0n", "2pNn", "others")
NORMALIZATIONS = {"0p0n": 1, "0pNn": 0, "1p0n": 0, "1pNn": 0, "2p0n": 0, "2pNn": 0, "others": 0}


class BDTReweightAPI:
    """
    Loads the GBReweighterModel_<category>.pkl files found under
    base_path (one process' worth of trained reweighters, e.g.
    ".../BDTreweight_outputs/<model_name>/QE") and exposes a
    predict_weight_<category>(features) method for each topology.
    """

    def __init__(self, base_path):
        self._reweighters = {}
        self._load_all(base_path)

    def _load_all(self, base_path):
        sys.path.append(f"{base_path}/")
        gc.disable()

        for category in CATEGORIES:
            pickle_path = f"{base_path}/GBReweighterModel_{category}.pkl"
            print(f"Trying to load reweighter from {pickle_path}...", end='  ')
            try:
                with open(pickle_path, 'rb') as f:
                    self._reweighters[category] = pickle.load(f)
                print("Success.")
            except Exception as category_not_found:
                self._reweighters[category] = None
                print(category_not_found)

        if self._reweighters["0p0n"] is None:
            print("ERROR: Reweighter file for 0p0n not found or failed to load.")

        loaded = [c for c, rw in self._reweighters.items() if rw is not None]
        if not loaded:
            print("Warning: No reweighters were loaded successfully.")
            return

        print(f"Loaded reweighters for categories: {loaded}")
        for category in loaded:
            rw = self._reweighters[category]
            print(f"  {category}: norm factor: {rw.norm_factor}, xsec scale factor: {rw.xsec_scale_factor}")

    def _predict(self, category, features):
        rw = self._reweighters.get(category)
        if rw is None:
            print(f"ERROR: no model for category {category}. Returning w=1.0.")
            return 1.0
        return float(rw.predict_weight_single_event(features) * NORMALIZATIONS[category])

    def predict_weight_0p0n(self, features):
        return self._predict("0p0n", features)

    def predict_weight_0pNn(self, features):
        return self._predict("0pNn", features)

    def predict_weight_1p0n(self, features):
        return self._predict("1p0n", features)

    def predict_weight_1pNn(self, features):
        return self._predict("1pNn", features)

    def predict_weight_2p0n(self, features):
        return self._predict("2p0n", features)

    def predict_weight_2pNn(self, features):
        return self._predict("2pNn", features)

    def predict_weight_others(self, features):
        return self._predict("others", features)
