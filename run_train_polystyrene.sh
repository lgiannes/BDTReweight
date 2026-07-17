#!/bin/bash
# Retrain the two composite-target (CH / polystyrene) models with the per-nucleon
# basis correction (A_source=12, A_target=13 -> target xsec x 13/12), writing to
# BDTreweight_outputs/<model>_polystyrene and pics/<model>_polystyrene.
set -u

source /opt/root_install/bin/thisroot.sh
module_path="/Users/lorenzo/Minerva/fork_reweighting"
export PYTHONPATH="${module_path}:${PYTHONPATH}"

config="${module_path}/BDTReweight/train_by_reaction_config_polystyrene.yaml"
src="/Users/lorenzo/cernbox/MINERVA_MC/source/ReweightSourceCCQELike_ABCDEFGLMNOP.root"
tdir="/Users/lorenzo/cernbox/MINERVA_MC/target"

train_one () {
    local model="$1" tfile="$2"
    echo "=================================================================="
    echo "RETRAIN (polystyrene): ${model}"
    echo "=================================================================="
    python3 "${module_path}/BDTReweight/train_by_reaction_config.py" \
        --config "${config}" \
        --source_path "${src}" \
        --target_path "${tdir}/${tfile}" \
        --module_path "${module_path}" \
        --model_name "${model}"
    echo "  exit=$?"
}

train_one NuWro-LFG_polystyrene "flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
train_one GENIEv3_polystyrene   "flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"

echo "ALL POLYSTYRENE RETRAINING DONE"
