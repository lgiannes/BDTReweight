#!/bin/bash

# Run test_training.py on the cluster to validate an already-trained model.
# It loads the per-process reweighter pickles produced by
# train_by_reaction_config.py (under <target_folder>/BDTreweight_outputs/<model>/)
# and produces the psi-prime grid + mean-vs-slice validation plots.
#
# Usage: $0 <target_model_id> [category]   (category defaults to 0p0n)

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <target_model_id> [category]"
    exit 1
fi

# check that the setup is correct
if [ -z "$MINERVAEXE" ]; then
  source ${MINERVA}/setup_CCQENu.sh
fi

source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_ABCDEFGLMNOP.root"

target_NEUTSF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
target_NEUTEDRMF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"
target_NuWro_LFG="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
target_GENIEv3="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"
target_RPWIA="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_RPWIA_all_NUISFLAT_CCQELike.root"
target_EDRMF_EbMinus10="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_all_EbMinus10_NUISFLAT_CCQELike.root"
target_EDRMF_newflux="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_newFlux_20files_NUISFLAT_CCQELike.root"
target_SF_newflux="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_newFlux_40files_NUISFLAT_CCQELike.root"

target_model_id="$1"
category="${2:-0p0n}"

case "$target_model_id" in
  "NEUT-SF")            target=${target_NEUTSF} ;;
  "NEUT-EDRMF")        target=${target_NEUTEDRMF} ;;
  "NuWro-LFG")         target=${target_NuWro_LFG} ;;
  "GENIEv3")           target=${target_GENIEv3} ;;
  "NEUT-RPWIA")        target=${target_RPWIA} ;;
  "NEUT-EDRMF-EbM10")  target=${target_EDRMF_EbMinus10} ;;
  "NEUT-EDRMF-newflux") target=${target_EDRMF_newflux} ;;
  "NEUT-SF-newflux")   target=${target_SF_newflux} ;;
  *)
    echo "Error: Unknown target model ID '$target_model_id'. Please use 'NEUT-SF', 'NEUT-EDRMF', 'NuWro-LFG', 'GENIEv3', 'NEUT-RPWIA', 'NEUT-EDRMF-EbM10', 'NEUT-EDRMF-newflux', or 'NEUT-SF-newflux'."
    exit 1
    ;;
esac

target_folder=$(dirname "$target")

# Reweighter pickles written by train_by_reaction_config.py live here, with one
# QE/ 2p2h/ Oth/ subfolder each holding GBReweighterModel_<category>.pkl.
reweighter_folder="${target_folder}/BDTreweight_outputs/${target_model_id}"

python3 ${MINERVA}/BDTReweight/test_training.py \
                    --source-file $source \
                    --target-file $target \
                    --reweighter-folder ${reweighter_folder} \
                    --category ${category} \
                    --output-folder ${reweighter_folder}/test_plots # --max-events 100000
