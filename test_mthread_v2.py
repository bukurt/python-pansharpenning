# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import threading
from threading import Thread
import numpy as np
from osgeo import gdal
import cv2
import time
from sys import getsizeof
import matplotlib.pyplot as plt
# params

INTERPOLATION = cv2.INTER_LINEAR
gdal.UseExceptions()
path_im_ms = './data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = './data/spot6/GEBZE/S6_GEBZE_PAN.tiff'

start = time.time()
ds_ms = gdal.Open(path_im_ms, gdal.GA_ReadOnly)
ds_pan = gdal.Open(path_im_pan, gdal.GA_ReadOnly)
x_pan = ds_pan.RasterXSize/4
y_pan = ds_pan.RasterYSize/4
x_ms = ds_ms.RasterXSize/4
y_ms = ds_ms.RasterYSize/4
np_pan_orj = ds_pan.GetRasterBand(1).ReadAsArray()
np_ms1_orj = ds_ms.GetRasterBand(1).ReadAsArray()
np_ms2_orj = ds_ms.GetRasterBand(2).ReadAsArray()
np_ms3_orj = ds_ms.GetRasterBand(3).ReadAsArray()
np_pan = cv2.resize(np_pan_orj, dsize=(ds_pan.RasterXSize/4,
                                       ds_pan.RasterYSize/4), interpolation=INTERPOLATION)
np_ms1 = cv2.resize(np_ms1_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION)
np_ms2 = cv2.resize(np_ms2_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION)
np_ms3 = cv2.resize(np_ms3_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION)
del np_pan_orj, np_ms1_orj, np_ms2_orj, np_ms3_orj
end1 = time.time() - start
print "## ", str(end1), " seconds for load data"

def npfft2(np_in):
    return np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())

def npfft2_v2(np_in, np_out, xy):
    np_out[xy[0]:xy[1], xy[2]:xy[3]] = np.fft.fft2(np_in)

def npifft2_v2(np_in, np_out, xy):
    np_out[xy[0]:xy[1], xy[2]:xy[3]] = np.fft.ifft2(np_in)

# single core
fft1 = np.empty((np.shape(np_pan)), dtype='complex64')
start = time.time()
fft1 = npfft2(np_pan)
print "## ", str(time.time() - start), " seconds for fftw"

        start = time.time()
        for i in (np_pan, np_pan):
            th = Thread(target=npfft2, args=(i, ))
            th.start()
        th.join()
        print "## ", str(time.time() - start), " seconds for fftw"

for thread in threading.enumerate():
    print("Thread name is %s." % thread.getName())

# Multi-thread FFT on PAN
np_pan_fft = np.zeros((np.shape(np_pan)), dtype='complex64')
num_part = 2
threads = [None] * num_part**2
cntr = 0
start = time.time()
for i in range(num_part):
    for j in range(num_part):
        x_from = (i)*(x_pan/num_part)
        x_to = (i+1)*(x_pan/num_part) - 1
        y_from = (j)*(x_pan/num_part)
        y_to = (j+1)*(x_pan/num_part) - 1
        xy_range = (x_from,x_to,y_from,y_to)
        threads[cntr] = Thread(target=npfft2_v2, args=(np_pan[x_from:x_to,y_from:y_to], np_pan_fft, xy_range))
        threads[cntr].start()
        cntr += 1
# do some other stuff
for i in range(len(threads)):
    threads[i].join()
print "## ", str(time.time() - start), " seconds for fftw"


# Reverse multi-thread FFT on PAN
np_pan_ifft = np.zeros((np.shape(np_pan)), dtype='float32')
cntr = 0
start = time.time()
for i in range(num_part):
    for j in range(num_part):
        x_from = (i)*(x_pan/num_part)
        x_to = (i+1)*(x_pan/num_part) - 1
        y_from = (j)*(x_pan/num_part)
        y_to = (j+1)*(x_pan/num_part) - 1
        xy_range = (x_from,x_to,y_from,y_to)
        threads[cntr] = Thread(target=npifft2_v2, args=(np_pan_fft[x_from:x_to,y_from:y_to], np_pan_ifft, xy_range))
        threads[cntr].start()
        cntr += 1
# do some other stuff
for i in range(len(threads)):
    threads[i].join()
print "## ", str(time.time() - start), " seconds for fftw"




npfft2_v2(np_pan[x_from:x_to,y_from:y_to], np_pan_fft, xy_range)