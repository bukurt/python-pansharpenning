#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:14:09 2019

@author: burak
"""
from functions import load_test_data
from functions import test_fft_ifft_single
import time

start = time.time()
pan = load_test_data(dset='ms')
pan2 = test_fft_ifft_single(pan)
print "## ", str(time.time() - start), " seconds."
