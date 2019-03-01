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
# params

INTERPOLATION = cv2.INTER_LINEAR
gdal.UseExceptions()
path_im_ms = '../data/spot6/GEBZE/S6_GEBZE_MS.tiff'
path_im_pan = '../data/spot6/GEBZE/S6_GEBZE_PAN.tiff'

start = time.time()
ds_ms = gdal.Open(path_im_ms)
ds_pan = gdal.Open(path_im_pan)
np_pan_orj = np.array(ds_pan.GetRasterBand(1).ReadAsArray())
np_ms1_orj = np.array(ds_ms.GetRasterBand(1).ReadAsArray())
np_ms2_orj = np.array(ds_ms.GetRasterBand(2).ReadAsArray())
np_ms3_orj = np.array(ds_ms.GetRasterBand(3).ReadAsArray())
np_pan = cv2.resize(np_pan_orj, dsize=(ds_pan.RasterXSize/4,
                                       ds_pan.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms1 = cv2.resize(np_ms1_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms2 = cv2.resize(np_ms2_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
np_ms3 = cv2.resize(np_ms3_orj, dsize=(ds_ms.RasterXSize/4,
                                       ds_ms.RasterYSize/4), interpolation=INTERPOLATION).astype('complex64')
del np_pan_orj, np_ms1_orj, np_ms2_orj, np_ms3_orj
end1 = time.time() - start
print "## ", str(end1), " seconds for load data"

def npfft2(np_in):
    return np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())

def npfft2_v2(np_in, np_out, index):
    np_out[index] = np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())


#Creating only four threads for now
fft1 = np.empty((np.shape(np_pan_orj)), dtype='complex64')
start = time.time()
fft1 = npfft2(np_pan_orj)
print "## ", str(time.time() - start), " seconds for fftw"

start = time.time()
for i in (np_pan_orj, np_pan_orj):
    th = Thread(target=npfft2, args=(i, ))
    th.start()
th.join()
print "## ", str(time.time() - start), " seconds for fftw"

for thread in threading.enumerate():
    print("Thread name is %s." % thread.getName())

############
threads = [None] * 4
for i in range(len(threads)):
    threads[i] = Thread(target=foo, args=(np_pan_orj[0:i], results, i))
    threads[i].start()

# do some other stuff

for i in range(len(threads)):
    threads[i].join()



