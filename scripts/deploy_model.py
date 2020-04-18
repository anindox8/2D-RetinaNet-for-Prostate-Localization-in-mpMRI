from __future__ import division
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from data_reader import Reader,read_fn
from models.model_utils.anchors import AnchorParameters


'''
Prostate Detection in mpMRI
Script:         Deploy 2D RetinaNet
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         14/04/2020

'''


# Anchor Definition
AnchorParam = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                               ratios  = np.array([0.705, 1.000, 1.419], tf.keras.backend.floatx()),
                               scales  = np.array([0.400, 0.644, 1.031], tf.keras.backend.floatx()))

anchor_deltas_mean = [-0.079931999, 0.005310305, 0.075939696, 0.011442137]
anchor_deltas_std  = [ 0.081362432, 0.070324861, 0.080778639, 0.073184822]

counter = 0

def predict(args, display=False):
   # Read CSV with Validation Set
    file_names = pd.read_csv(args.csv, dtype=object, keep_default_na=False, na_values=[]).values

    # Load Trained Model
    export_dir = \
        [os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
         if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]
    
    print('Loading from {}'.format(export_dir))    
    my_predictor = tf.contrib.predictor.from_saved_model(export_dir=export_dir)
    print('Complete')

    counter = 0
    
    # Iterate through Files, Predict on the Full Volumes, Compute Dice
    for output in read_fn(file_references = file_names,
                          mode            = tf.estimator.ModeKeys.EVAL,
                          params          = {'anchor_params':   None,
                                             'anchor_mean':     anchor_deltas_mean,
                                             'anchor_std':      anchor_deltas_std,
                                             'display':         False,
                                             'deploy_mode':     True}):
        t0          = time.time() 
        counter    += 1

        # Parse Data Reader Output
        img         = np.expand_dims((output['features']['x']),axis=0)
        lbl         = output['labels']['mask']
        bbox_gt     = output['labels']['y_rg']
        subject_id  = output['img_id']
        
        # Generate Predictions
        y_boxes  = my_predictor.session.run(
            fetches    =  my_predictor._fetch_tensors['detection_boxes'],
            feed_dict  = {my_predictor._feed_tensors['x']: img})
        y_scores = my_predictor.session.run(
            fetches    =  my_predictor._fetch_tensors['detection_scores'],
            feed_dict  = {my_predictor._feed_tensors['x']: img})
        y_labels = my_predictor.session.run(
            fetches    =  my_predictor._fetch_tensors['detection_labels'],
            feed_dict  = {my_predictor._feed_tensors['x']: img})

        # Visualize Detections
        fig,ax              = plt.subplots(figsize=(20, 10))
        lbl8                = np.zeros(shape=(lbl.shape[0],lbl.shape[1],3))
        lbl8[:,:,0]         = ((lbl/lbl.max())*0).astype(np.uint8)
        lbl8[:,:,1]         = ((lbl/lbl.max())*0).astype(np.uint8)
        lbl8[:,:,2]         = ((lbl/lbl.max())*255).astype(np.uint8)
        ax.imshow(img[0,:,:,0], cmap='gray')                                 # Display Base Image
        ax.imshow(lbl8, alpha=0.20)                                          # Display Ground-Truth Segmentation Mask Annotation                                
 
        '''
        for i in range(bbox_gt.shape[0]):                                    # Display Ground-Truth Bounding Box Annotation
            b    = np.array(bbox_gt[i]).astype(int)
            print('GT',b)
            rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],
                                      linewidth=4,edgecolor=(0,0,1),
                                      facecolor='none')
            ax.add_patch(rect)
        '''

        max_score = 0
        for box, score, label in zip(y_boxes[0], y_scores[0], y_labels[0]):  # Display Predicted Detections
            if (score>max_score):
                max_score = score
                b         = box.astype(int)
        if ((b[0]==b[2])|(b[1]==b[3])): continue
        rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=5,edgecolor=(0,1,0),facecolor='none')
        ax.add_patch(rect)
        plt.text(b[0], b[1], 'prostate | {:.4f}'.format(score), bbox=dict(facecolor=(0,1,0), alpha=1.0), fontsize=12)    

        plt.savefig(args.save_path+subject_id+'_'+str(counter)+'.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        
        counter += 1
        #if (counter>=0): break

        # Print Outputs
        if display: print('ID={}; y_boxes={}; y_scores={}; y_labels={}; Run Time={:0.2f} s;'.format(
            subject_id, y_boxes, y_scores, y_labels, time.time()-t0))

    

if __name__ == '__main__':
    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Prostate Detection in mpMRI')
    parser.add_argument('--verbose',            default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path',   '-m', default='./models/mykonos/weights/002_14042020/Epoch-240_ValLoss-2.7697_FocalLoss-0.0906_L1Loss-2.6790/')
    parser.add_argument('--csv',          '-d', default='./models/mykonos/feed/prostate-mpMRI-128_validation-fold-1.csv')
    parser.add_argument('--save_path',    '-s', default='./models/mykonos/inference/case2/')
    args = parser.parse_args()
                
    # Set Verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # GPU Allocation Options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Allow GPU Usage Growth
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # Verify TensorFlow Integration
    print('TensorFlow Version:', tf.VERSION)
    print('TensorFlow-GPU:',     tf.test.is_built_with_cuda())

    # Inference
    predict(args)

    
    session.close()

