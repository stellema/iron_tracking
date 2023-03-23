#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,LON,R,VAR,N

###############################################################################
# Save biogeochemical fields at particle positions (single CPU version - NO LONGER IMPLEMENTED).
# To submit: qsub -v EXP=0,LON=250,R=0,VAR='temp',N=0 felx_var_bgc_fields.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Save BGC fields at EUC particle positions for: exp=$EXP, lon=$LON, R=$R, variable=$VAR & n=$N."

module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/particle_BGC_fields.py -e $EXP -x $LON -v 0 -r $R -func 'bgc_fields_var' -n $N -var $VAR
