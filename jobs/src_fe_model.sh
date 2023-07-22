#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=15:00:00
#PBS -l mem=192GB
#PBS -l ncpus=4
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v V,EXP

###############################################################################
# Run iron model
# V,EXP,LON,R
# To submit: qsub -v V=0,EXP=0,LON=220,R=0 src_fe_model.sh
# To submit: qsub -v V=0,EXP=0 src_fe_model.sh
###############################################################################

ECHO=/bin/echo
$ECHO "version=0, scenario=$EXP, longitude=$LON, index=$R."
module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04

#python3 /g/data/e14/as3189/stellema/felx/scripts/format_fe_model_sources.py -s $EXP -x $LON -v $V -r $R

for R in {0..7}; do
    for LON in 165 190 220 250; do
        python3 /g/data/e14/as3189/stellema/felx/scripts/format_fe_model_sources.py -s $EXP -x $LON -v $V -r $R &
    done
done

wait
