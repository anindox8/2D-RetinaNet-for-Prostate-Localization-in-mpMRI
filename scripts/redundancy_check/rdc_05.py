from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

'''
Binary PCa Classification in mpMRI
Script:         Redundancy Check (Tumor 2D Slice Counter)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         10/03/2020

'''

partition      = 'validation/'
partition_path = './dataset/numpy/patch_112bo/labels/' + partition   
benign_scans   = 0
tumor_scans    = 0

# For Each Weights Checkpoint
for f in os.listdir(partition_path):
    if '.npy' in f:
        label = np.load(partition_path+f)
        for e in range(label.shape[0]):
            if (np.any(label[e]==3).astype(np.int32)==0): benign_scans += 1
            else:                                         tumor_scans  += 1
# Display
print('Benign Scans: '+str(benign_scans))
print('Tumor Scans: '+str(tumor_scans))
