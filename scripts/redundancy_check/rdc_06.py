from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SimpleITK as sitk
import os
from models.model_utils.anchors import anchor_targets_bbox, anchors_for_shape, compute_gt_annotations

'''
Prostate Detection in mpMRI
Script:         Redundancy Check (Anchor Optimization)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         22/03/2020

'''

partition           = 'train/'
dataset_path        = './dataset/numpy/patch_112bo/'   
error_subject_list  = []
error_patch_list    = []     

# Generates Class Bounding Box (Returns arr[ymin,xmin,ymax,xmax] from TL Corner)
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array((cmin,rmin,cmax,rmax))

# For Each Weights Checkpoint
for f in os.listdir(dataset_path+'labels/'+partition):
	if '.npy' in f:
		img         = np.load(dataset_path+'images/'+partition+f)
		lbl         = np.load(dataset_path+'labels/'+partition+f)
		lbl[lbl<3]  = 0
		lbl[lbl==3] = 1
		subject_id  = f.split('.npy')[0]
		for e in range(lbl.shape[0]):
			if (np.any(lbl[e]==1).astype(np.int32)!=0): 
				# Label Each Tumor Blob
				blobs_labels = measure.label(lbl[e], background=0)   
				annotations  = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

				# Annotate Each Tumor Blob
				for blob_id in range(np.max(blobs_labels)):
					blobs = blobs_labels.copy()
					blobs[blobs!=(blob_id+1)] = 0
					bbox_label            = bbox(blobs)
					annotations['labels'] = np.concatenate((annotations['labels'], [0]))
					annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[float(bbox_label[0]), float(bbox_label[1]),
					                                                                 float(bbox_label[2]), float(bbox_label[3])]]))
				# Construct Anchors and Full Annotations
				anchors               = anchors_for_shape(img[e].shape, anchor_params=None, shapes_callback=None)
				full_annotations      = anchor_targets_bbox(anchors, [img[e].astype(np.float32)], [annotations], 
				                                            num_classes=1, negative_overlap=0.4, positive_overlap=0.5)

				# Feedback (Anchor Optimization)
				positive_indices, _, max_indices = compute_gt_annotations(anchors, annotations['bboxes'])

				# Display Image
				fig,ax = plt.subplots(figsize=(20, 10))
				ax.imshow(img[e,:,:,0], cmap='gray')

				# Display Anchors on Image
				for b in anchors[positive_indices]:
					b = np.array(b).astype(int)
					rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=4,edgecolor=(0,1,1),facecolor='none')
					ax.add_patch(rect)

				# Display Annotation in Red
				for i in range(annotations['bboxes'].shape[0]):
					b = np.array(annotations['bboxes'][i]).astype(int)
					rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=4,edgecolor=(1,0,0),facecolor='none')
					ax.add_patch(rect)

				# Display Regressed Anchors on Image
				for b in annotations['bboxes'][max_indices[positive_indices],:]:
					b = np.array(b).astype(int)
					rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=4,edgecolor=(0,1,0),facecolor='none')
					ax.add_patch(rect)

				plt.savefig('./models/ezra/anchors/'+subject_id+'_'+str(e)+'.png', bbox_inches='tight')
				plt.cla()
				plt.clf()

				if (len(annotations['bboxes'][max_indices[positive_indices],:])==0):
					error_subject_list.append(subject_id)
					error_patch_list.append(e)

# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(error_subject_list,
                                      error_patch_list)),
columns      =  ['subject_id', 'patch'])
CSVdata_name =  './models/ezra/anchors/TrainingAnchorError.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)
