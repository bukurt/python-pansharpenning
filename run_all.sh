#!/bin/bash

#if [ $# -lt 5 ];then
#	echo "Parameter missing"
#	echo -e "Usage:\e[33m $0 \e[93m ms-path pan-path xml-path out-file-path stat-file \033[0m"
#	exit 1
#fi

#ms=$1
#pan=$2
#xml=$3
#out=$4
#stat=$5
stat='pansharpenning_stats.csv'

declare -a DATAS=(
"data/spot6/GEBZE/S6_GEBZE_MS.tiff" "data/spot6/GEBZE/S6_GEBZE_PAN.tiff" "data/spot6/GEBZE/S6_GEBZE_MS.XML" "data/spot6/GEBZE/OUT.tiff"
"data/spot6/IST_ORMAN/S6_IST_ORMAN_MS.tiff" "data/spot6/IST_ORMAN/S6_IST_ORMAN_PAN.tiff" "data/spot6/IST_ORMAN/S6_IST_ORMAN_MS.XML" "data/spot6/IST_ORMAN/OUT.tiff"
"data/spot6/URFA/S6_URFA_MS.tiff" "data/spot6/URFA/S6_URFA_PAN.tiff" "data/spot6/URFA/S6_URFA_MS.XML" "data/spot6/URFA/OUT.tiff"
"data/spot6/IST_ORMAN2/S6_IST_ORMAN2_MS.tiff" "data/spot6/IST_ORMAN2/S6_IST_ORMAN2_PAN.tiff" "data/spot6/IST_ORMAN2/S6_IST_ORMAN2_MS.XML" "data/spot6/IST_ORMAN2/OUT.tiff"
)

declare -a PS_METHODS=("fft" "ihs" "ihs_fft" "lab" "lab_fft" "brovey,hfm")
declare -a FILTERS=("ideal_lpf" "hamming" "hanning" "lbtw" "gauss_low")

data_len=${#DATAS[@]}
for (( i=0; i<${data_len}+1; i=i+4 )); do
  ms=${DATAS[$i]}
  pan=${DATAS[$i+1]}
  xml=${DATAS[$i+2]}
  out=${DATAS[$i+3]}
 for method in ${PS_METHODS[@]}; do
   for filter in ${FILTERS[@]}; do
     echo -e "\e[93mFile: $out\033[0m"
     echo -e "\e[93mSingle Thread-$method-$filter\033[0m"
     /usr/bin/python perf_main_latest.py \
 	--ms-file $ms \
 	--pan-file $pan \
 	--xml-file $xml \
 	--filter $filter \
 	--cutoff-freq 0.125 \
 	--ps-method $method \
 	--out-file $out \
 	--stat-file $stat \
 	--run-times 10 \
     echo -e "\e[93mMulti  Thread-$method-$filter\033[0m"
     /usr/bin/python perf_main_latest.py \
 	--ms-file $ms \
 	--pan-file $pan \
 	--xml-file $xml \
 	--filter $filter \
 	--cutoff-freq 0.125 \
 	--ps-method $method \
 	--out-file $out \
 	--stat-file $stat \
 	--run-times 10 \
 	--multi-thread
   done
 done
done
