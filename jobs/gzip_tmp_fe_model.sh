#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=24:00:00
#PBS -l mem=3GB
#PBS -l ncpus=1
#PBS -l storage=gdata/e14
#PBS -l wd

###############################################################################
# Compress fe_model particle datasets
###############################################################################

cd /g/data/e14/as3189/stellema/felx/data/fe_model
#tar -zcvf ./felx_bgc_tmp.tar.gz ./felx_bgc_tmp
## Zip BGC file tmps (precalculated stuff for actual file)
#gzip -9v /g/data/e14/as3189/stellema/felx/data/fe_model/felx_bgc_hist*_tmp.nc

## Compress felx_bgc split up tmp netcdf files (not needed - all files deleted)
#for LON in 165 190 220 250; do
#    for R in {0..7}; do
#        gzip -9 /g/data/e14/as3189/stellema/felx/data/fe_model/tmp_felx_bgc_hist_"$LON"_v0_0"$R"/*.nc
#        #tar -zcvf /g/data/e14/as3189/stellema/felx/data/fe_model/tmp_felx_bgc_hist_"$LON"_v0_0"$R".tar.gz /g/data/e14/as3189/stellema/felx/data/fe_model/tmp_felx_bgc_hist_"$LON"_v0_0"$R"
#    done
#done

cd v3
tar -zcvf ./tmp.tar.gz ./tmp

