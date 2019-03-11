#!/bin/bash

PS_METHOD=$1

if [ $# -lt 1 ];then
	echo "Method input is missing."
fi

if [ "$2" == "--multi-thread" ];then
/usr/bin/python perf_main_v3.py \
	--ms-file data/spot6/GEBZE/S6_GEBZE_MS.tiff \
	--pan-file data/spot6/GEBZE/S6_GEBZE_PAN.tiff \
	--xml-file data/spot6/GEBZE/S6_GEBZE_MS.XML \
	--filter ideal_lpf \
	--cutoff-freq 0.125 \
	--ps-method $PS_METHOD \
	--out-file data/spot6/GEBZE/out.tiff \
	--run-times 10 \
	--multi-thread
else
/usr/bin/python perf_main_v3.py \
        --ms-file data/spot6/GEBZE/S6_GEBZE_MS.tiff \
        --pan-file data/spot6/GEBZE/S6_GEBZE_PAN.tiff \
        --xml-file data/spot6/GEBZE/S6_GEBZE_MS.XML \
        --filter ideal_lpf \
        --cutoff-freq 0.125 \
        --ps-method $PS_METHOD \
        --out-file data/spot6/GEBZE/out.tiff \
        --run-times 10 
fi

