#!/bin/zsh

sample="CCQELike"
input="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT.root"
output="/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_EDRMF_nu_all_NUISFLAT_${sample}.root"

python3 target_selection.py --input_file $input --output_file $output --sample $sample