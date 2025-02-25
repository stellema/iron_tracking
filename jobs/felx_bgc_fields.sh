#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=06:00:00
#PBS -l mem=98GB
#PBS -l ncpus=16
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,LON,R,V

###############################################################################
# Save biogeochemical fields at particle positions.
# To submit: qsub -v EXP=0,LON=250,R=0,V=0 felx_bgc_fields.sh
# To submit: qsub -v EXP=0,LON=165,R=0,V=0 felx_bgc_fields.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Save BGC fields at EUC particle positions for: exp=$EXP, lon=$LON, R=$R, V_index=$V & NCPUS=$PBS_NCPUS."

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/particle_BGC_fields.py -e $EXP -x $LON -v 0 -r $R -func 'bgc_fields' -iv $V
