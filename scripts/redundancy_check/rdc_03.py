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
Script:         Redundancy Check (Tumor Voxels per Diseased Cases)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         27/02/2020

'''

# Master Directories
split          = 'validation'   # 'train'/'validation'/'test'
labels_path    = './dataset/numpy/patch_64bo/labels/'+split
save_path      = './dataset/tumor_voxels_per_scan'
patches        = 8

# Create Directory List
files = []
for r, d, f in os.walk(labels_path):
    for file in f:
        if '.npy' in file:
            files.append(file)

subject_id_list    = []
subject_lbl_list   = []
patch_num_list     = []
patch_lbl_list     = []
tumor_vox_list     = []

# For Each Annotation
for f in files:
    # I/O Directories -----------------------------------------------------------------------------------------------------------------------------
    lbl         = np.load(labels_path+'/'+str(f))
    subject_id  = str(f).split('.npy')[0]                                   # Scan ID
    subject_lbl = np.any(lbl==3).astype(np.int32)                           # Scan Label
    
    # For Each Patch
    for e in range(patches):
        patch_num = e                                                       # Patch Number
        patch_lbl = np.any(lbl[e,:,:,:]==3).astype(np.int32)                # Patch Label
        tumor_vox = np.count_nonzero(lbl[e,:,:,:]==3)                       # Number of Tumor Voxels
        
        # Populate Lists
        subject_id_list.append(subject_id)  
        subject_lbl_list.append(subject_lbl) 
        patch_num_list.append(patch_num)   
        patch_lbl_list.append(patch_lbl)   
        tumor_vox_list.append(tumor_vox)     

# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(subject_id_list,
                                      subject_lbl_list,
                                      patch_num_list,
                                      patch_lbl_list,
                                      tumor_vox_list)),
columns      =  ['subject_id', 'subject_lbl', 'patch_num', 'patch_lbl', 'tumor_voxels'])
CSVdata_name =  save_path+'.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)
