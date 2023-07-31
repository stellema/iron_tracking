#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=10:00:00
#PBS -l mem=8GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v LON

###############################################################################
# Run iron model
# To submit: qsub -v LON=6 optimize_felx_plot.sh
###############################################################################

ECHO=/bin/echo

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
mpiexec -n $PBS_NCPUS python3 /g/data/e14/as3189/stellema/felx/scripts/optimise_iron_model_params.py -x $LON -f 'plot'
