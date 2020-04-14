from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import SimpleITK as sitk
import os
import numpy as np
import scipy.ndimage
import time
import os
import cv2
from skimage.measure import regionprops


'''
Prostate Detection in mpMRI
Script:         Preprocessing + NumPy I/O
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         06/04/2020

'''


# Image Whitening (Mean=0; Standard Deviation=1) [Ref:DLTK]
def whitening(image):
    image = image.astype(np.float32)
    mean  = np.mean(image)
    std   = np.std(image)
    if std > 0: ret = (image - mean) / std
    else:       ret = image * 0.
    return ret

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

# Center Crop NumPy Volumes
def center_crop(img,cropz,cropx,cropy,center_2d_coords=None,multi_channel=False):
    if center_2d_coords: x,y = center_2d_coords
    else:                x,y = img.shape[1]//2,img.shape[2]//2
    startz = img.shape[0]//2    - (cropz//2)
    startx = int(x) - (cropx//2)
    starty = int(y) - (cropy//2)
    if (multi_channel==True): return img[startz:startz+cropz,startx:startx+cropx,starty:starty+cropy,:]
    else:                     return img[startz:startz+cropz,startx:startx+cropx,starty:starty+cropy]

# Patch Extraction (Guided) [Ref:DLTK]
def extract_class_balanced_example_array(image, label, example_size=[1, 64, 64],
                                         n_examples=1, classes=2, class_weights=None):
    assert image.shape[:-1] == label.shape, 'Image and label shape must match'
    assert image.ndim - 1   == len(example_size), \
        'Example size doesnt fit image size'
    rank = len(example_size)
    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)

    if class_weights is None:
        n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(n_examples / n_classes))
    else:
        assert len(class_weights) == n_classes, \
            'Class_weights must match number of classes'
        class_weights  = np.array(class_weights)
        n_ex_per_class = np.round((class_weights / class_weights.sum()) * n_examples).astype(int)

    # Compute Example Radius to Define Region to Extract around Center Location
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0),
                               np.ceil(np.array(example_size) / 2.0))), dtype=np.int)
    class_ex_images = []
    class_ex_lbls   = []
    min_ratio       = 1.
    
    for c_idx, c in enumerate(classes):
        # Get Valid, Random Center Locations of Given Class
        idx       = np.argwhere(label == c)
        ex_images = []
        ex_lbls   = []

        if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
            class_ex_images.append([])
            class_ex_lbls.append([])
            continue

        # Extract Random locations
        r_idx_idx = np.random.choice(len(idx), size=min(n_ex_per_class[c_idx], len(idx)), replace=False).astype(int)
        r_idx     = idx[r_idx_idx]

        # Shift Random to Valid Locations (if necessary)
        r_idx = np.array([np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                                        ex_rad[dim][0]) for dim in range(rank)]) for r in r_idx])
        for i in range(len(r_idx)):
            # Extract Class-Balanced Examples from Original Image
            slicer   = tuple(slice(r_idx[i][dim] - ex_rad[dim][0], r_idx[i][dim] + ex_rad[dim][1]) for dim in range(rank))
            ex_image = image[slicer][np.newaxis, :]
            ex_lbl   = label[slicer][np.newaxis, :]

            # Concatenate and Return Examples
            ex_images = np.concatenate((ex_images, ex_image), axis=0) \
                if (len(ex_images) != 0) else ex_image
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) \
                if (len(ex_lbls) != 0) else ex_lbl

        class_ex_images.append(ex_images)
        class_ex_lbls.append(ex_lbls)
        ratio     = n_ex_per_class[c_idx] / len(ex_images)
        min_ratio = ratio if ratio < min_ratio else min_ratio

    indices   = np.floor(n_ex_per_class * min_ratio).astype(int)
    ex_images = np.concatenate([cimage[:idxs] for cimage, idxs in zip(class_ex_images, indices)
                                if len(cimage) > 0], axis=0)
    ex_lbls   = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices)
                                if len(clbl) > 0], axis=0)
    return ex_images, ex_lbls

# Circular Mask Creator for Pseudo-Segmentation of Prostate
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

# Decompose Image Volume into Quadrants
def decompose_3d_quads(img,multi_channel=False):
    if multi_channel: tiles = [img[:,x:x+img.shape[1]//2,y:y+img.shape[2]//2,:] for x in range(0,img.shape[1],img.shape[1]//2) for y in range(0,img.shape[2],img.shape[2]//2)]        
    else:             tiles = [img[:,x:x+img.shape[1]//2,y:y+img.shape[2]//2]   for x in range(0,img.shape[1],img.shape[1]//2) for y in range(0,img.shape[2],img.shape[2]//2)]
    return np.array(tiles)

# Decompose Image Volume into Octants Plus Coverage
def decompose_3d_octants_plus(img,multi_channel=False,plus=[0,0,0],center_oct=True):
    half_x = img.shape[1]//2
    half_y = img.shape[2]//2
    half_z = img.shape[0]//2
    if multi_channel:
        # Create Stack of 3D Octant Patches
        tiles = np.vstack((np.expand_dims(img[       0:half_z+plus[0],            0:half_x+plus[1],          0:half_y+plus[2],           :],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],      0:half_x+plus[1],          0:half_y+plus[2],           :],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],        half_x-plus[1]:img.shape[1],   0:half_y+plus[2],           :],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],  half_x-plus[1]:img.shape[1],   0:half_y+plus[2],           :],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],            0:half_x+plus[1],         half_y-plus[2]:img.shape[2], :],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],      0:half_x+plus[1],         half_y-plus[2]:img.shape[2], :],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],        half_x-plus[1]:img.shape[1],  half_y-plus[2]:img.shape[2], :],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],  half_x-plus[1]:img.shape[1],  half_y-plus[2]:img.shape[2], :],axis=0)))
        # Include Additional Center Patch
        if (center_oct==True): return np.concatenate((tiles,np.expand_dims(center_crop(img,half_z+plus[0],half_x+plus[1],half_y+plus[2],multi_channel=True),axis=0)),axis=0)
        else:                  return tiles
    else: 
        # Create Stack of 3D Octant Patches
        tiles = np.vstack((np.expand_dims(img[       0:half_z+plus[0],            0:half_x+plus[1],          0:half_y+plus[2]          ],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],      0:half_x+plus[1],          0:half_y+plus[2]          ],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],        half_x-plus[1]:img.shape[1],   0:half_y+plus[2]          ],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],  half_x-plus[1]:img.shape[1],   0:half_y+plus[2]          ],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],            0:half_x+plus[1],         half_y-plus[2]:img.shape[2]],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],      0:half_x+plus[1],         half_y-plus[2]:img.shape[2]],axis=0),
                           np.expand_dims(img[       0:half_z+plus[0],        half_x-plus[1]:img.shape[1],  half_y-plus[2]:img.shape[2]],axis=0),
                           np.expand_dims(img[  half_z-plus[0]:img.shape[0],  half_x-plus[1]:img.shape[1],  half_y-plus[2]:img.shape[2]],axis=0)))
        # Include Additional Center Patch
        if (center_oct==True): return np.concatenate((tiles,np.expand_dims(center_crop(img,half_z+plus[0],half_x+plus[1],half_y+plus[2],multi_channel=False),axis=0)),axis=0)
        else:                  return tiles

# Resize Image with Crop/Pad [Ref:DLTK]
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    rank = len(img_size)  # Image Dimensions

    # Placeholders for New Shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding   = [[0, 0] for dim in range(rank)]
    slicer       = [slice(None)] * rank

    # For Each Dimension Determine Process (Cropping/Padding)
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]
        # Create Slicer Object to Crop/Leave Each Dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad Cropped Image to Extend Missing Dimension
    return np.pad(image[slicer], to_padding, **kwargs)







# Master Directories
split          = 'train'   # 'train'/'validation'/'test'
annot_path     = './dataset/prime_annotations/'+split
zonal_seg_path = './dataset/zonal_segmentations'
MRI_path       = '../matin/Data/MHDs'
save_path      = './dataset/numpy/patch_128'

# Patch Parameters
patch_res      = [0.5, 0.5, 3.6]
patch_dims     = [12, 128, 128]
decompose      = None              # 'basic'/'advanced'/None
overlap_px     = [2,8,8]


# Create Directory List
files = []
for r, d, f in os.walk(annot_path):
    for file in f:
        if '.nii' in file:
            files.append(file)


error_list = []
for f in files:
    # I/O Directories -----------------------------------------------------------------------------------------------------------------------------
    subject_id = str(f).split('.nii')[0]        
    lbl_io     = annot_path+'/'+subject_id+'.nii.gz'
    mask_io    = zonal_seg_path+'/'+subject_id+'_zones.nii.gz'

    # Verify if Target Results Already Exist
    if not ((os.path.exists(save_path+'/images/'+split+'/'+subject_id+'.npy'))&(os.path.exists(save_path+'/labels/'+split+'/'+subject_id+'.npy'))):
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
        try:
            # Preprocessing mpMRI and Annotations ---------------------------------------------------------------------------------------------------------
            # Load Data
            img_T2W    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_T2W_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))
            img_ADC    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_ADC_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))
            img_DWI    = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(img_DWI_io, sitk.sitkFloat32),out_spacing=patch_res,is_label=False)))
            lbl        = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(lbl_io),out_spacing=patch_res,is_label=True)).astype(np.uint8))

            # Center Crop ADC,DWI,Annotation Scans to Same Scope
            zdim       = min(img_T2W.shape[0],img_ADC.shape[0],img_DWI.shape[0])
            xydim      = min(img_T2W.shape[1],img_ADC.shape[1],img_DWI.shape[1],img_T2W.shape[2],img_ADC.shape[2],img_DWI.shape[2])
            img_T2W    = center_crop(img_T2W, zdim,xydim,xydim)
            img_ADC    = center_crop(img_ADC, zdim,xydim,xydim)
            img_DWI    = center_crop(img_DWI, zdim,xydim,xydim)
            lbl        = center_crop(lbl,     zdim,xydim,xydim)
                                    
            # Preprocess and Clean Labels (from Possible Border Interpolation Errors)
            lbl[(lbl!=0)&(lbl!=1)]    = 0

            # Linear Normalization to Retain Numerical Significance of ADC 
            img_ADC    = img_ADC/3000          

            # Image Whitening (Mean=0; Standard Deviation=1)
            img_T2W    = whitening(img_T2W)   
            img_DWI    = whitening(img_DWI)  
         
            # Preprocessing Zonal Segmentations ----------------------------------------------------------------------------------------------------------- 
            segm_flag  = 0   # Ensure Zonal Segmentation is Not Empty
            while True:
                if (os.path.exists(mask_io))&(segm_flag==0):
                    mask         = np.array(sitk.GetArrayFromImage(resample_img(sitk.ReadImage(mask_io),out_spacing=patch_res,is_label=True)).astype(np.uint8))
                    mask[mask>2] = 0
                    mask         = center_crop(mask, zdim,xydim,xydim)
                    if (np.sum(mask)==0): segm_flag = 1
                    else:                 break
                else:
                    # Pseudo-Segmentation
                    mask            = np.zeros(shape=(zdim,xydim,xydim),dtype=np.uint8)
                    for z in range(zdim):     
                        mask[z,:,:] = np.subtract(create_circular_mask(xydim,xydim,radius=48).astype(np.uint8)*2,
                                                  create_circular_mask(xydim,xydim,radius=24).astype(np.uint8))
                    break

            # Patch Extraction (Guided) -------------------------------------------------------------------------------------------------------------------
            # Determine Maximum Prostate Area Slice
            max_area_slice        = 0
            bin_mask              = mask.copy()
            bin_mask[bin_mask!=0] = 1
            for z in range(bin_mask.shape[0]):
                if (np.sum(bin_mask[z,:,:])>np.sum(bin_mask[max_area_slice,:,:])): max_area_slice = z
                else:                                                              continue

            # Padding if Z-Dimension is Below Minimum Center-Crop Dimension
            if (lbl.shape[0]<patch_dims[0])|(img_T2W.shape[0]<patch_dims[0])|(img_ADC.shape[0]<patch_dims[0])|(img_DWI.shape[0]<patch_dims[0]):     
                img_T2W        = resize_image_with_crop_or_pad(img_T2W, img_size=(patch_dims[0],img_T2W.shape[1],img_T2W.shape[2]))
                img_ADC        = resize_image_with_crop_or_pad(img_ADC, img_size=(patch_dims[0],img_ADC.shape[1],img_ADC.shape[2]))
                img_DWI        = resize_image_with_crop_or_pad(img_DWI, img_size=(patch_dims[0],img_DWI.shape[1],img_DWI.shape[2]))
                lbl            = resize_image_with_crop_or_pad(lbl,     img_size=(patch_dims[0],lbl.shape[1],lbl.shape[2]))
                mask           = resize_image_with_crop_or_pad(mask,    img_size=(patch_dims[0],mask.shape[1],mask.shape[2]))

            # Center Crop Volumes to ROI with Same Dimensions
            crop_xy_dims   = patch_dims[1]
            crop_z_dims    = patch_dims[0]
            center_coords  = regionprops(bin_mask[max_area_slice,:,:])[0].centroid
            img_T2W        = center_crop(img_T2W, crop_z_dims,crop_xy_dims,crop_xy_dims, center_2d_coords=center_coords)
            img_ADC        = center_crop(img_ADC, crop_z_dims,crop_xy_dims,crop_xy_dims, center_2d_coords=center_coords)
            img_DWI        = center_crop(img_DWI, crop_z_dims,crop_xy_dims,crop_xy_dims, center_2d_coords=center_coords)
            lbl            = center_crop(lbl,     crop_z_dims,crop_xy_dims,crop_xy_dims, center_2d_coords=center_coords)
            mask           = center_crop(mask,    crop_z_dims,crop_xy_dims,crop_xy_dims, center_2d_coords=center_coords)


            # Concatenate to Multi-Modal MRI Input
            img            = np.concatenate((np.expand_dims(img_T2W, axis=3),
                                             np.expand_dims(img_ADC, axis=3),
                                             np.expand_dims(img_DWI, axis=3)), axis=3)

            # Multi-Label Mask with Annotation
            mask[lbl==1]   = 3

            # Patch Extraction (Guided)
            if (decompose=='basic'):
                patch_img  = decompose_3d_quads(img,  multi_channel=True)
                patch_lbl  = decompose_3d_quads(mask, multi_channel=False)
            elif (decompose=='advanced'):
                patch_img  = decompose_3d_octants_plus(img,  multi_channel=True,  plus=overlap_px, center_oct=True)
                patch_lbl  = decompose_3d_octants_plus(mask, multi_channel=False, plus=overlap_px, center_oct=True)
            else:
                patch_img  = img
                patch_lbl  = mask

            # Export Patches as NumPy Arrays
            if (np.any(patch_lbl==3).astype(np.int32)==0): np.save(save_path+'/images/benign/'+subject_id+'.npy',patch_img)
            else:                                          np.save(save_path+'/images/malignant/'+subject_id+'.npy',patch_img)
            np.save(save_path+'/labels/'+subject_id+'.npy',patch_lbl)
        except:
            print('ERROR: '+subject_id)
            error_list.append(subject_id)
    else:
        pass
        print('SKIPPING: '+subject_id)

# Error Log
np.save('./dataset/error.npy', np.array(error_list))
