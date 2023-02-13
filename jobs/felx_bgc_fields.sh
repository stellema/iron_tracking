#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=10GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,LON,R

###############################################################################
# Save biogeochemical fields at particle positions.
# To submit: qsub -v EXP=0,LON=165,R=0 felx_bgc_fields.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Save BGC fields at EUC particle positions for exp $EXP at lon $LON."

module use /g/data3/hh5/public/modules
module load conda/analysis3
python3 /g/data/e14/as3189/stellema/felx/scripts/particle_BGC_fields.py -e $EXP -x $LON -v 0 -r $R
