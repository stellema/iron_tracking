#!/bin/bash
#PBS -P e14
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=192GB
#PBS -l ncpus=48
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON

###############################################################################
# Run iron model
# To submit: qsub -v LON=220 optimize_felx.sh
###############################################################################

ECHO=/bin/echo

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/optimise_iron_model_params.py -x $LON
