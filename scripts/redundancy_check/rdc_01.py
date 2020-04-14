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
Script:         Redundancy Check (All Original Scan Dimensions)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         17/02/2020

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



 # Master Directories
split          = 'train'   # 'train'/'validation'/'test'
annot_path     = '../../../../dataset/annotations/'+split
MRI_path       = '../../../../../matin/Data/MHDs'
save_path      = '../../../../dataset/prostate_mpMRI_shapes'

# Patch Parameters
patch_res      = [0.5, 0.5, 3.6]

# Create Directory List
files = []
for r, d, f in os.walk(annot_path):
    for file in f:
        if '.nii' in file:
            files.append(file)

subject_id_list = []
T2W_shape_list  = []
ADC_shape_list  = []
DWI_shape_list  = []

for f in files:
    # I/O Directories -----------------------------------------------------------------------------------------------------------------------------
    subject_id = str(f).split('.nii')[0]        

    # Search All 3 Directories for Target MRI Scans
    if (os.path.exists(MRI_path+'/Detection2018/'+subject_id+'_t2w.mhd')): img_T2W_io = MRI_path+'/Detection2018/'+subject_id+'_t2w.mhd'
    elif os.path.exists(MRI_path+'/Detection2017/'+subject_id+'_t2w.mhd'): img_T2W_io = MRI_path+'/Detection2017/'+subject_id+'_t2w.mhd'
    else:                                                                  img_T2W_io = MRI_path+'/Detection2016/'+subject_id+'_t2w.mhd'        
    if (os.path.exists(MRI_path+'/Detection2018/'+subject_id+'_adc.mhd')): img_ADC_io = MRI_path+'/Detection2018/'+subject_id+'_adc.mhd'
    elif os.path.exists(MRI_path+'/Detection2017/'+subject_id+'_adc.mhd'): img_ADC_io = MRI_path+'/Detection2017/'+subject_id+'_adc.mhd'
    else:                                                                  img_ADC_io = MRI_path+'/Detection2016/'+subject_id+'_adc.mhd'
    if (os.path.exists(MRI_path+'/Detection2018/'+subject_id+'_hbv.mhd')): img_DWI_io = MRI_path+'/Detection2018/'+subject_id+'_hbv.mhd'
    elif os.path.exists(MRI_path+'/Detection2017/'+subject_id+'_hbv.mhd'): img_DWI_io = MRI_path+'/Detection2017/'+subject_id+'_hbv.mhd'
    else:                                                                  img_DWI_io = MRI_path+'/Detection2016/'+subject_id+'_hbv.mhd'

    # Preprocessing mpMRI and Annotations ---------------------------------------------------------------------------------------------------------
    # Load Data
    img_T2W    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_T2W_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))
    img_ADC    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_ADC_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))
    img_DWI    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_DWI_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))

    # Populate Lists
    subject_id_list.append(subject_id)
    T2W_shape_list.append(img_T2W.shape)
    ADC_shape_list.append(img_ADC.shape)
    DWI_shape_list.append(img_DWI.shape)

# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(subject_id_list,T2W_shape_list,ADC_shape_list,DWI_shape_list)),
columns      =  ['subject_id', 'T2W_image_shape', 'ADC_image_shape', 'DWI_image_shape'])
CSVdata_name =  save_path+'.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)
