from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import warnings
import csv
import argparse
import sys
import os
import numpy as np
import scipy.optimize
from PIL import Image
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
from models.model_utils.compute_overlap import compute_overlap
from models.model_utils.anchors import generate_anchors, AnchorParameters, anchors_for_shape
warnings.simplefilter("ignore")


'''
Prostate Detection in mpMRI
Script:         Redundancy Check (Anchor Optimization)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         10/04/2020

'''


SIZES               = [32,64,128,256,512]
STRIDES             = [8,16,32,64,128]
state               = {'best_result': sys.maxsize}
dataset_path        = './dataset/numpy/patch_128/'  
arg_scales          = 3 
arg_ratios          = 3
arg_objective       = 'focal'
arg_popsize         = 25


# Generates Class Bounding Box (Returns arr[ymin,xmin,ymax,xmax] from TL Corner)
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array((cmin,rmin,cmax,rmax))

# Estimate Shapes Based on Pyramid Levels
def __patch_shapes(image_shape, pyramid_levels):
    return [np.array([128,128]), np.array([64,64]), np.array([32,32]), np.array([16,16]), np.array([8,8])]


def calculate_config(values, ratio_count):
    split_point = int((ratio_count - 1) / 2)
    ratios      = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1/values[i])

    scales = values[split_point:]

    return AnchorParameters(SIZES, STRIDES, ratios, scales)


def base_anchors_for_shape(pyramid_levels=None, anchor_params=None):
    if pyramid_levels is None: pyramid_levels = [3,4,5,6,7]
    if anchor_params is None:  anchor_params = AnchorParameters.default

    # Compute Anchors For All Pyramid Levels
    all_anchors = np.zeros((0,4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size = anchor_params.sizes[idx],
            ratios    = anchor_params.ratios,
            scales    = anchor_params.scales)
        all_anchors = np.append(all_anchors, anchors, axis=0)
    return all_anchors


def average_overlap(values, entries, state, image_shape, mode='focal', ratio_count=3, include_stride=False):
    anchor_params              = calculate_config(values, ratio_count)
    if include_stride: anchors = anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=__patch_shapes)
    else:              anchors = base_anchors_for_shape(anchor_params=anchor_params)

    overlap     = compute_overlap(entries, anchors)
    max_overlap = np.amax(overlap, axis=1)
    not_matched = len(np.where(max_overlap<0.5)[0])

    if mode == 'avg':     result = 1 - np.average(max_overlap)
    elif mode == 'ce':    result = np.average(-np.log(max_overlap))
    elif mode == 'focal': result = np.average(-(1 - max_overlap) ** 2 * np.log(max_overlap))

    if result < state['best_result']:
        state['best_result'] = result

        print('Current best anchor configuration')
        print(f'Ratios: {sorted(np.round(anchor_params.ratios, 3))}')
        print(f'Scales: {sorted(np.round(anchor_params.scales, 3))}')
        if include_stride: print(f'Average overlap: {np.round(np.average(max_overlap), 3)}')
        print(f'Number of labels that don\'t have any matching anchor: {not_matched}')
        print()
    return result, not_matched






# For Loading Object Dimensions
if ((arg_ratios%2)!=1): raise Exception('Number of Ratios Must Be Odd')
seed    = np.random.RandomState(8)
entries = np.zeros((0, 4))
max_x   = 0
max_y   = 0

print('Loading Object Dimensions.')

# For Each Malignant Scan
for f in os.listdir(dataset_path+'labels/'):
    if '.npy' in f:
        lbl         = np.load(dataset_path+'labels/'+f)
        lbl[lbl!=0] = 1
        subject_id  = f.split('.npy')[0]

        # For Each Malignant Slice
        for e in range(lbl.shape[0]):
            if (np.any(lbl[e]==1).astype(np.int32)==0):  continue   # Skip Slices with Absence of Prostate
            else: 
                bbox_label            = bbox(lbl[e])
                x1                    = float(bbox_label[0])
                y1                    = float(bbox_label[1]) 
                x2                    = float(bbox_label[2])
                y2                    = float(bbox_label[3])
                max_x                 = max(x2, max_x)
                max_y                 = max(y2, max_y)

                # For Stride Computation
                entry   = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
                entries = np.append(entries, entry, axis=0)

print('Complete: ', str(entries.shape), ' Entries.')

image_shape = [max_y, max_x]
bounds      = []
best_result = sys.maxsize
for i in range(int((arg_ratios-1)/2)):  bounds.append((1,4))
for i in range(arg_scales):             bounds.append((0.4,2))

print('Optimizing Anchors.')

# Optimization
result = scipy.optimize.differential_evolution(lambda x: average_overlap(x, entries, state, image_shape, arg_objective, arg_ratios, 
	                                           include_stride=True)[0], bounds=bounds, popsize=arg_popsize, seed=8)
# User Feedback
if hasattr(result, 'success') and result.success: print('Optimization ended successfully!')
elif not hasattr(result, 'success'):              print('Optimization ended!')
else:
    print('Optimization ended unsuccessfully!')
    print(f'Reason: {result.message}')

values             = result.x
anchor_params      = calculate_config(values, arg_ratios)
(avg, not_matched) = average_overlap(values, entries, {'best_result': 0}, image_shape, 'avg', arg_ratios, include_stride=True)

# Display Results
print()
print('Final best anchor configuration')
print(f'Ratios: {sorted(np.round(anchor_params.ratios, 3))}')
print(f'Scales: {sorted(np.round(anchor_params.scales, 3))}')
print(f'Average overlap: {np.round(1 - avg, 3)}')
print(f'Number of labels that don\'t have any matching anchor: {not_matched}')