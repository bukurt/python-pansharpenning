#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:00:02 2019

@author: burak
"""
import time
import threading
from threading import Thread

def sleepMe(i):
    print("Thread %s going to sleep for 5 seconds." % threading.current_thread())
    time.sleep(1)
    print("Thread %s is awake now.\n" % threading.current_thread())

#Creating only four threads for now
for i in range(4):
    th = Thread(target=sleepMe, args=(i, ))
    th.start()


for thread in threading.enumerate():
    print("Thread name is %s." % thread.getName())

