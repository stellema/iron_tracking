#!/bin/bash
#PBS -P e14
#PBS -q normalsr
#PBS -l walltime=40:00:00
#PBS -l mem=482GB
#PBS -l ncpus=52
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v V,EXP,LON,R

###############################################################################
# Run iron model
# To submit:
# qsub -v V=4,EXP=0,LON=165,R=0 fe_model.sh
# qsub -v V=4,EXP=0,LON=190,R=0 fe_model.sh
# qsub -v V=4,EXP=0,LON=220,R=0 fe_model.sh
# qsub -v V=4,EXP=0,LON=250,R=0 fe_model.sh
###############################################################################

ECHO=/bin/echo
$ECHO "version=$V, scenario=$EXP, longitude=$LON, index=$R."
module use /g/data3/hh5/public/modules
module load conda/analysis3-23.01
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/fe_model.py -s $EXP -x $LON -v $V -r $R -f 'run'
