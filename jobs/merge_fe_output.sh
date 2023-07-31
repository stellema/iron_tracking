#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=03:00:00
#PBS -l mem=26GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v V,EXP,LON,R

###############################################################################
# Run iron model
# To submit: qsub -v V=4,EXP=0,LON=220,R=0 merge_fe_output.sh
###############################################################################

ECHO=/bin/echo
module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
$ECHO "version=$V, scenario=$EXP, longitude=$LON, index=$R."
python3 /g/data/e14/as3189/stellema/felx/scripts/fe_model.py -s $EXP -x $LON -v $V -r $R -f 'save'

#for R in {0..3}; do
#    $ECHO "version=$V, scenario=$EXP, longitude=$LON, index=$R."
#    python3 /g/data/e14/as3189/stellema/felx/scripts/fe_model.py -s $EXP -x $LON -v $V -r $R -f 'save'
#done
