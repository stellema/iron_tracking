#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=3GB
#PBS -l ncpus=6
#PBS -l storage=gdata/e14
#PBS -l wd

###############################################################################
# Uncompress OFAM3 data files.
###############################################################################

cd /g/data/e14/as3189/OFAM/trop_pac
# Historical 2000-2012
for var in "phy" "zoo" "det" "no3" "temp" "fe"; do
  gunzip -v ocean_"$var"_2*.nc.gz &
done

wait
