#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l mem=3GB
#PBS -l ncpus=6
#PBS -l storage=gdata/e14
#PBS -l wd

###############################################################################
# Uncompress OFAM3 data files.
###############################################################################

cd /g/data/e14/as3189/OFAM/trop_pac
# Historical 2000-2012
for var in "phy" "zoo" "det" "no3" "temp" "fe" "w" "u" "v"; do
  gzip -7v ocean_"$var"_*.nc &
done

wait
