from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import os

'''
Binary PCa Classification in mpMRI
Script:         Redundancy Check (Patch Dimensions)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         11/02/2020

'''

split           = 'images/train'
target_path     = '../../../../dataset/numpy/'+split
target_shape    = (4,16,64,64,3)

# Create Directory List
files = []
for r, d, f in os.walk(target_path):
    for file in f:
        if '.npy' in file:
            files.append(file)

# Verify Patch Shape
error_list = []
for f in files:
	img = np.load(target_path+'/'+str(f))
	if (img.shape!=target_shape):
		img = img[:4,:,:,:,:]
		np.save(target_path+'/'+str(f),img)
		error_list.append(str(f).split('.npy')[0])
		print('ERROR!')

# Error Log
np.save('../../../../dataset/error.npy', np.array(error_list))
