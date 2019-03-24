#!/bin/bash

PS_METHOD=$1

today=`date +%Y%m%d_%H%M%S`
stat="pansharpenning_stats_"$today".csv"

if [ $# -lt 1 ];then
	echo "Method input is missing."
fi

if [ "$2" == "--multi-thread" ];then
/usr/bin/python perf_main_latest.py \
	--ms-file data/spot6/GEBZE/S6_GEBZE_MS.tiff \
	--pan-file data/spot6/GEBZE/S6_GEBZE_PAN.tiff \
	--xml-file data/spot6/GEBZE/S6_GEBZE_MS.XML \
	--filter ideal_lpf \
	--cutoff-freq 0.125 \
	--ps-method $PS_METHOD \
	--out-file data/spot6/GEBZE/ \
	--stat-file $stat \
	--run-times 10 \
	--multi-thread
else
/usr/bin/python perf_main_latest.py \
        --ms-file data/spot6/GEBZE/S6_GEBZE_MS.tiff \
        --pan-file data/spot6/GEBZE/S6_GEBZE_PAN.tiff \
        --xml-file data/spot6/GEBZE/S6_GEBZE_MS.XML \
        --filter ideal_lpf \
        --cutoff-freq 0.125 \
        --ps-method $PS_METHOD \
        --out-file data/spot6/GEBZE/ \
	--stat-file $stat \
        --run-times 10 
fi

