#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=3GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14
#PBS -l wd

###############################################################################
# Compressparticle BGC tmp files.
###############################################################################

cd /g/data/e14/as3189/stellema/felx/data/felx

gzip -9v /g/data/e14/as3189/stellema/felx/data/felx/felx_bgc_hist*_tmp.nc

for LON in 165 190 220 250
do
    for R in {0..7}
    do
        gzip -9 /g/data/e14/as3189/stellema/felx/data/felx/tmp_felx_bgc_hist_"$LON"_v0_0"$R"/*.nc
        #tar -zcvf /g/data/e14/as3189/stellema/felx/data/felx/tmp_felx_bgc_hist_"$LON"_v0_0"$R".tar.gz /g/data/e14/as3189/stellema/felx/data/felx/tmp_felx_bgc_hist_"$LON"_v0_0"$R"
    done
done
