#!/bin/bash
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=06:30:00
#PBS -l mem=42GB
#PBS -l ncpus=1
#PBS -l storage=gdata/hh5+gdata/e14
#PBS -l wd
#PBS -m ae
#PBS -M astellemas@gmail.com
#PBS -v EXP,LON

###############################################################################
# Save biogeochemical fields at particle positions from tmp subset files.
# To submit: qsub -v EXP=0,LON=250 felx_ds.sh
###############################################################################

ECHO=/bin/echo
$ECHO "Merge and save particle BGC fields from subsets: exp=$EXP, lon=$LON."

module use /g/data3/hh5/public/modules
module load conda/analysis3-22.04
if [[ $EXP -eq 0 ]]
then
    EXPSTR="hist"
else
    EXPSTR="rcp"
fi

for R in {0..7}
do
    $ECHO "Saving: exp=$EXP, lon=$LON, $R."
    python3 /g/data/e14/as3189/stellema/felx/scripts/particle_BGC_fields.py -e $EXP -x $LON -v 0 -r $R -func 'save_files'
    #tar -zcvf /g/data/e14/as3189/stellema/felx/data/felx/tmp_felx_bgc_"$EXPSTR"_"$LON"_v0_0"$R".tar.gz /g/data/e14/as3189/stellema/felx/data/felx/tmp_felx_bgc_"$EXPSTR"_"$LON"_v0_0"$R"
    #gzip -9v /g/data/e14/as3189/stellema/felx/data/felx/felx_bgc_"$EXPSTR"_"$LON"_v0_0"$R"_tmp.nc
done
