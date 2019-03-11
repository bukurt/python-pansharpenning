# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
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
# path_im_ms = './data/spot6/URFA2/S6_URFA_MS.jp2'
# path_im_pan = './data/spot6/URFA2/S6_URFA_PAN.JP2'
start = time.time()
ds_ms = gdal.Open(path_im_ms, gdal.GA_ReadOnly)
ds_pan = gdal.Open(path_im_pan, gdal.GA_ReadOnly)
x_pan = ds_pan.RasterXSize/4
y_pan = ds_pan.RasterYSize/4
x_ms = ds_ms.RasterXSize/4
y_ms = ds_ms.RasterYSize/4
# np_pan = ds_pan.GetRasterBand(1).ReadAsArray()
# np_pan = np.concatenate([np_pan, np_pan], axis=0)
# np_pan = np.concatenate([np_pan, np_pan], axis=1)
# x_pan,y_pan = np.shape(np_pan)
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
print "## ", str(end1), " seconds"

def npfft2(np_in):
    return np.fft.fft2(np_in)
    #print("FINISH %s " % threading.current_thread())

def np_fft2_v2(np_in,):
    return np.fft.fft2(np_in)

def np_ifft2_v2(np_in,):
    return np.fft.ifft2(np_in)

def np_ifft2_v2(np_in, np_out, xy):
    np_out[xy[0]:xy[1], xy[2]:xy[3]] = np.fft.ifft2(np_in)

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

# single core
fft1 = np.empty((np.shape(np_pan)), dtype='complex64')
start0 = time.time()
fft1 = npfft2(np_pan)
print "## ", str(time.time() - start0), " seconds: FFT"
start1 = time.time()
pan_single = np.fft.ifft2(fft1)
print "## ", str(time.time() - start1), " seconds: IFFT"
print "## ", str(time.time() - start0), " seconds: Total"



if __name__ == '__main__':
    q = Queue()
    np_pan_f1, np_pan_f2, np_pan_f3, np_pan_f4 = split(np_pan, x_pan/2, y_pan/2)
    np_pans = split(np_pan, x_pan/2, y_pan/2)
    # p1 = Process(target=np.fft.fft2, args=(np_pan_f1,))
    processes = [Process(target=np_fft2_v2, args=(x, q)) for x in np_pans]
    start = time.time()
    # Run processes
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print "## ", str(time.time() - start), " seconds for fftw"

pool = Pool(processes=2)
start0 = time.time()
results = pool.map(np_fft2_v2, np_pans)
print "## ", str(time.time() - start0), " seconds: FFT"
start1 = time.time()
pan_multi = pool.map(np.fft.ifft2, results)
print "## ", str(time.time() - start1), " seconds: FFT"
print "## ", str(time.time() - start0), " seconds: Total"
print(results)



