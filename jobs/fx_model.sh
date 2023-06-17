#!/bin/bash
#PBS -P e14
#PBS -q normalsr
#PBS -l walltime=48:00:00
#PBS -l mem=312GB
#PBS -l ncpus=65
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,LON,R

###############################################################################
# Run iron model
# To submit: qsub -v EXP=0,LON=220,R=0 fx_model.sh
###############################################################################

ECHO=/bin/echo
$ECHO "version=0, scenario=$EXP, longitude=$LON, index=$R."
module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/fe_model.py -s $EXP -x $LON -v 0 -r $R -f 'fix'
