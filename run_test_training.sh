#!/bin/bash

source="/Users/lorenzo/cernbox/MINERVA_MC/source//ReweightSourceCCQELike_ABCDEFGLMNOP.root"

#target="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root"
#reweighter_folder="/Users/lorenzo/cernbox/MINERVA_MC/target/BDTreweight_outputs/NEUT-SF-enhanced/"

target="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_CCQELike.root"
reweighter_folder="/Users/lorenzo/cernbox/MINERVA_MC/target/BDTreweight_outputs/TEST-EDRMF/"

#target="/Users/lorenzo/cernbox/MINERVA_MC/target/flat_NuWro_CH_LFG_v2109_50M_CCQELike.root"
#reweighter_folder="/Users/lorenzo/cernbox/MINERVA_MC/target/BDTreweight_outputs/NuWro-LFG/"


#target="/Users/lorenzo/cernbox/MINERVA_MC/target/flat_GENIE_G18_10b_02_11a_50M_CCQELike.root"
#reweighter_folder="/Users/lorenzo/cernbox/MINERVA_MC/target/BDTreweight_outputs/GENIEv3/"


module_path="/Users/lorenzo/Minerva/fork_reweighting/"

PYTHONPATH=${module_path}:${PYTHONPATH}

python3 ${module_path}/BDTReweight/test_training.py \
                    --source-file $source \
                    --target-file $target \
                    --reweighter-folder $reweighter_folder \
                    --max-events 100000
