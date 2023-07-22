#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=06:00:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v V,EXP,LON

###############################################################################
# Run iron model
# V,EXP,LON,R
# To submit: qsub -v V=0,EXP=0,LON=220 src_fe_model1.sh
###############################################################################

ECHO=/bin/echo
$ECHO "version=0, scenario=$EXP, longitude=$LON, index=0-7."
module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04

for R in {0..7}; do
    python3 /g/data/e14/as3189/stellema/felx/scripts/format_fe_model_sources.py -s $EXP -x $LON -v $V -r $R
done

