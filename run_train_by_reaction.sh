#!/bin/bash

# check that the setup is correct
if [ -z "$MINERVAEXE" ]; then
  echo "Error: MINERVAEXE is not set. Please source the setup script."
  exit 1
fi

python3 ${MINERVA}/BDTReweight/train_by_reaction.py \
                    --source_path /eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/SourcesForReweighting/ReweightSourceCCQELike_minervame1M.root \
                    --target_path /eos/user/l/lgiannes/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root \
                    --module_path ${MINERVA}