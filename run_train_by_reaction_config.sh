#!/bin/bash

# Config-driven counterpart of run_train_by_reaction.sh: same targets and CLI,
# but it calls train_by_reaction_config.py (the maintained version) with a YAML
# config. Per-target hadd correction is passed on the command line via
# --hadd_n_files, so a single config file serves every target.

# require one argument for the target model ID
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <target_model_id>"
    exit 1
fi

# check that the setup is correct
if [ -z "$MINERVAEXE" ]; then
  source ${MINERVA}/setup_CCQENu.sh
fi

# The script imports `from BDTReweight...` at module load, before --module_path
# is applied, so the PARENT of BDTReweight/ (i.e. ${MINERVA}) must be on PYTHONPATH.
export PYTHONPATH="${MINERVA}:${PYTHONPATH}"

# YAML config that drives the training (categories, processes, binning, A_source/
# A_target, default hadd_n_files). The default file has A_source=A_target=1 and
# hadd_n_files=1, which is correct for these CH/carbon MC targets.
config="${MINERVA}/BDTReweight/train_by_reaction_config.yaml"

source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_ABCDEFGLMNOP.root"
# source="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_minervame1A.root"

target_NEUTSF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
target_NEUTEDRMF="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"

target_NuWro_LFG="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
target_GENIEv3="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"

target_RPWIA="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_RPWIA_all_NUISFLAT_CCQELike.root"
target_EDRMF_EbMinus10="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_all_EbMinus10_NUISFLAT_CCQELike.root"
target_EDRMF_newflux="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_EDRMF_newFlux_20files_NUISFLAT_CCQELike.root"
target_SF_newflux="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/TargetsForReweighting/neut_MINERvAflux_SF_newFlux_40files_NUISFLAT_CCQELike.root"

# set the target model ID from the command line argument
target_model_id="$1"

# Number of NUISANCE flat trees hadd'd into the target file. Divides out the
# xsec inflation from merging. Single-file targets = 1; the *_newFlux_20files_*
# targets are hadd'd from 20 files.
hadd_n_files=1

# select the target file based on the target model ID
case "$target_model_id" in
  "NEUT-SF")
    target=${target_NEUTSF}
    ;;
  "NEUT-EDRMF")
    target=${target_NEUTEDRMF}
    ;;
  "NuWro-LFG")
    target=${target_NuWro_LFG}
    ;;
  "GENIEv3")
    target=${target_GENIEv3}
    ;;
  "NEUT-RPWIA")
    target=${target_RPWIA}
    ;;
  "NEUT-EDRMF-EbM10")
    target=${target_EDRMF_EbMinus10}
    ;;
  "NEUT-EDRMF-newflux")
    target=${target_EDRMF_newflux}
    hadd_n_files=20
    ;;
  "NEUT-SF-newflux")
    target=${target_SF_newflux}
    hadd_n_files=40   # same 20-file production; drop to 1 if it is a single file
    ;;
  *)
    echo "Error: Unknown target model ID '$target_model_id'. Please use 'NEUT-SF', 'NEUT-EDRMF', 'NuWro-LFG', 'GENIEv3', 'NEUT-RPWIA', 'NEUT-EDRMF-EbM10', 'NEUT-EDRMF-newflux', or 'NEUT-SF-newflux'."
    exit 1
    ;;
esac

target_folder=$(dirname "$target")

python3 ${MINERVA}/BDTReweight/train_by_reaction_config.py \
                    --config $config \
                    --source_path $source \
                    --target_path $target \
                    --module_path ${MINERVA} \
                    --plots_dir ${target_folder}/plots_${target_model_id} \
                    --hadd_n_files ${hadd_n_files} \
                    --model_name ${target_model_id} # --max_events 1000
