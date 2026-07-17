#!/bin/bash
# Run test_training on every target model (canonical name-matched reweighters),
# saving each model's plots under pics/TEST_TRAINING/<model>/.
set -u

source /opt/root_install/bin/thisroot.sh
module_path="/Users/lorenzo/Minerva/fork_reweighting"
export PYTHONPATH="${module_path}:${PYTHONPATH}"

src="/Users/lorenzo/cernbox/MINERVA_MC/source/ReweightSourceCCQELike_ABCDEFGLMNOP.root"
tdir="/Users/lorenzo/cernbox/MINERVA_MC/target"
rdir="${tdir}/BDTreweight_outputs"
pics="${module_path}/pics/TEST_TRAINING"
maxev=100000

# model | target file | reweighter folder
run_one () {
    local model="$1" tfile="$2" rfolder="$3"
    local out="${pics}/${model}"
    mkdir -p "${out}"
    echo "=================================================================="
    echo "MODEL: ${model}   ->  ${out}"
    echo "=================================================================="
    python3 "${module_path}/BDTReweight/test_training.py" \
        --source-file "${src}" \
        --target-file "${tdir}/${tfile}" \
        --reweighter-folder "${rfolder}" \
        --output-folder "${out}" \
        --max-events "${maxev}" > "${out}/run.log" 2>&1
    echo "  exit=$?  (log: ${out}/run.log)"
}

run_one NEUT-EDRMF "neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root" "${rdir}/NEUT-EDRMF"
run_one NEUT-SF    "neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"    "${rdir}/NEUT-SF"
run_one NuWro-LFG  "flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"           "${rdir}/NuWro-LFG"
run_one GENIEv3    "flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"         "${rdir}/GENIEv3"

echo "ALL DONE"
