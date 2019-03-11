#!/bin/bash

/usr/bin/python perf_main_v2.py \
	--ms-file data/spot6/GEBZE/S6_GEBZE_MS.tiff \
	--pan-file data/spot6/GEBZE/S6_GEBZE_PAN.tiff \
	--xml-file data/spot6/GEBZE/S6_GEBZE_MS.XML \
	--filter ideal_lpf \
	--cutoff-freq 0.125 \
	--ps-method fft \
	--out-file data/spot6/GEBZE/ot.tiff \
	--run-times 10 \
	--histogram-match \
