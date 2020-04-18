from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
import os
from models.model_utils.anchors import AnchorParameters
from models.model_utils.anchors import anchor_targets_bbox, anchors_for_shape, compute_gt_annotations

'''
Prostate Detection in mpMRI
Script:         Redundancy Check (Bounding Box Normalization)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         08/04/2020

'''

dataset_path        = './dataset/numpy/patch_128/'  
subject_id_list     = []
patch_list          = []      
bbox_X1_list        = []
bbox_Y1_list        = []
bbox_X2_list        = []
bbox_Y2_list        = []


# Anchor Definition
AnchorParam = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                               ratios  = np.array([0.705, 1.000, 1.419], tf.keras.backend.floatx()),
                               scales  = np.array([0.400, 0.644, 1.031], tf.keras.backend.floatx()))

anchor_deltas_mean = [-0.062645554, 0.016282263, 0.062880478, -0.016470429]
anchor_deltas_std  = [ 0.126736264, 0.113858276, 0.126964753,  0.113627249]

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

# For Each Weights Checkpoint
for f in os.listdir(dataset_path+'labels/'):
	if '.npy' in f:
		try:
			try:    
				img     = np.load(dataset_path+'images/malignant/'+f)
			except: 
				img     = np.load(dataset_path+'images/benign/'+f)
			lbl         = np.load(dataset_path+'labels/'+f)
			lbl[lbl!=0] = 1
			subject_id  = f.split('.npy')[0]
			for e in range(lbl.shape[0]):
				if (np.any(lbl[e]==1).astype(np.int32)==0): continue     # Skip Slices with Absence of Prostate 
				else:
					annotations  = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

					# Annotate Prostate
					bbox_label            = bbox(lbl[e])
					annotations['labels'] = np.concatenate((annotations['labels'], [0]))
					annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[float(bbox_label[0]), float(bbox_label[1]),
					                                                                 float(bbox_label[2]), float(bbox_label[3])]]))
					# Construct Anchors and Full Annotations
					anchors               = anchors_for_shape(img[e].shape, anchor_params=AnchorParam, shapes_callback=__patch_shapes)
					full_annotations      = anchor_targets_bbox(anchors, [img[e].astype(np.float32)], [annotations], 
					                                            num_classes=1, negative_overlap=0.4, positive_overlap=0.5,
					                                            anchor_deltas_mean=None, anchor_deltas_std=None)
					regression_deltas     = full_annotations[0][0]
					indices               = np.where(np.equal(regression_deltas[:,-1],1))
					foreground_deltas     = regression_deltas[indices]

					subject_id_list.append(subject_id)
					patch_list.append(e)
					bbox_X1_list.append(np.mean(foreground_deltas[:,0]))
					bbox_Y1_list.append(np.mean(foreground_deltas[:,1]))
					bbox_X2_list.append(np.mean(foreground_deltas[:,2]))
					bbox_Y2_list.append(np.mean(foreground_deltas[:,3]))
					print('Complete: ',subject_id,' ',str(e))
		except:
			print(str(f)+'DID NOT WORK')

# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(subject_id_list, patch_list, bbox_X1_list, bbox_Y1_list, bbox_X2_list, bbox_Y2_list)),
columns      =  ['subject_id','patch','bbox_dX1','bbox_dY1','bbox_dX2','bbox_dY2'])
CSVdata_name =  './models/mykonos/TumorsdBBOX-32_3S-3R.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)
