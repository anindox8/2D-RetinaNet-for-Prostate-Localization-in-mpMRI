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
Script:         Redundancy Check (All Segmentation Dimensions)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         18/02/2020

'''

# Resample Images to Target Resolution Spacing [Ref:SimpleITK]
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    
    out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                 int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                 int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(itk_image)

# Determine 3D Bounding Box
def bbox_3D(img):
    z = np.any(img, axis=(1,2))
    x = np.any(img, axis=(0,2))
    y = np.any(img, axis=(0,1))
    zmin, zmax = np.where(z)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    return zmin,zmax,xmin,xmax,ymin,ymax


# Master Directories
split          = 'train'   # 'train'/'validation'/'test'
annot_path     = '../../../../dataset/annotations/'+split
MRI_path       = '../../../../../matin/Data/MHDs'
zonal_seg_path = '../../../../dataset/zonal_segmentations'
save_path      = '../../../../dataset/prostate_segmentation_shapes'

# Patch Parameters
patch_res      = [0.5, 0.5, 3.6]

# Create Directory List
files = []
for r, d, f in os.walk(annot_path):
    for file in f:
        if '.nii' in file:
            files.append(file)

subject_id_list           = []
prostate_zshape_vox_list  = []
prostate_xshape_vox_list  = []
prostate_yshape_vox_list  = []
prostate_zshape_mm_list   = []
prostate_xshape_mm_list   = []
prostate_yshape_mm_list   = []

for f in files:
    # I/O Directories -----------------------------------------------------------------------------------------------------------------------------
    subject_id = str(f).split('.nii')[0]
    mask_io    = zonal_seg_path+'/'+subject_id+'_zones.nii.gz'        

    # Loading Zonal Segmentations ----------------------------------------------------------------------------------------------------------- 
    if os.path.exists(mask_io): mask = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(mask_io),out_spacing=patch_res,is_label=True)).astype(np.uint8))
    else:                       mask = np.zeros(shape=(192,192,192),dtype=np.uint8)

    # Binarize Segmentation Mask
    try:
        zmin,zmax,xmin,xmax,ymin,ymax = bbox_3D(mask)
    except:
        zmin,zmax,xmin,xmax,ymin,ymax = 1,0,1,0,1,0

    # Populate Lists
    subject_id_list.append(subject_id)           
    prostate_zshape_vox_list.append(zmax-zmin)  
    prostate_xshape_vox_list.append(xmax-xmin)  
    prostate_yshape_vox_list.append(ymax-ymin)  
    prostate_zshape_mm_list.append((zmax-zmin)*patch_res[2])   
    prostate_xshape_mm_list.append((xmax-xmin)*patch_res[0])   
    prostate_yshape_mm_list.append((ymax-ymin)*patch_res[1])   

# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(subject_id_list,
                                      prostate_zshape_vox_list,
                                      prostate_xshape_vox_list,
                                      prostate_yshape_vox_list,
                                      prostate_zshape_mm_list,
                                      prostate_xshape_mm_list,
                                      prostate_yshape_mm_list)),
columns      =  ['subject_id', 'length[z]-voxels', 'width[x]-voxels', 'height[y]-voxels', 'length[z]-mm', 'width[x]-mm', 'height[y]-mm'])
CSVdata_name =  save_path+'.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)
