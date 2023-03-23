#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=4:00:00
#PBS -l mem=190GB
#PBS -l ncpus=4
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP

###############################################################################
# Save biogeochemical fields prereq files.
# To submit: qsub -v EXP=0 felx_prereq.sh
# 190 GB for RCP & 172 for hist (particle dataset)
###############################################################################

ECHO=/bin/echo
$ECHO "Save pre-req files for saving BGC fields at EUC particle positions: exp=$EXP."

module use /g/data3/hh5/public/modules
module load conda/analysis3-unstable
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/particle_BGC_fields.py -e $EXP -func 'prereq_files'
