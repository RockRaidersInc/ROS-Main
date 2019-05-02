#! /usr/bin/python
from __future__ import print_function
import cv2
import numpy as np
import logging
import sys


def imshow(img, title=None, as_float=False, scale=None):
    # Quick imshow helper function for debugging 
    if scale is not None:
        img = cv2.resize(img, (0,0), fx=scale, fy=scale) 
    if not as_float:
        img = img.astype(np.uint8)
    if title is None:
        title = 'Quick imshow'
    cv2.namedWindow(title)
    cv2.moveWindow(title, 40,30)
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        print("exiting (q was pressed)", file=sys.stderr)
        sys.exit()

    cv2.destroyAllWindows()
    
def cs(*args, **kwarg):
    # Quick checksum for debugging
    for arg in args:
        print('Checksum: {}'.format(np.sum(arg)))
    for key, arg in kwarg.items():
        print('Checksum ({}): {}'.format(key,np.sum(arg)))
        
def s(*arrays,**kw_arrays):
    # Quick check shape for debugging
    for array in arrays:
        print('Shape: {}'.format(array.shape))
    for key, array in kw_arrays.items():
        print('Shape ({}): {}'.format(key,array.shape))
        
def d(*args,**kwargs):
    # Quick logging debug
    for arg in args:
        logging.debug(arg)
    for keyword, arg in kwargs.items():
        logging.debug('{}| {}'.format(keyword, arg)) 