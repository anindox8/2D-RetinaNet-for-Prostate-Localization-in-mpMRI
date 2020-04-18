from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import json
import warnings
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from data_reader import Reader,read_fn
from models.retinanet import retinanet_2d
from models.model_utils.anchors import AnchorParameters
from models.model_utils.optimizer_utils import cyclic_learning_rate, leaky_relu, AdaBoundOptimizer
from models.model_utils.focal_l1_losses import focal, smooth_l1


'''
Prostate Detection in mpMRI
Script:         Train 2D RetinaNet
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         14/04/2020

'''


# Image Parameters
PATCH_DIMS          =  [128,128]
SLICE_NUM           =   12
NUM_CHANNELS        =   3
NUM_CLASSES         =   1
TRAIN_SIZE          =   1559*SLICE_NUM
VAL_SIZE            =   391*SLICE_NUM

# Data Augmentation Hyperparameters
AUG_PROB            =   0.00                              # 0-1
AUG_ROT_DEGREE      =   10                                # 0-360
AUG_TRANS_FACTOR    =   0.20                              # 0-1

# Training Hyperparameters
MAX_EPOCHS          =   350
VAL_POINTS          =   350
BATCH_SIZE          =   32
LR_MODE             =  'eLR'                              # 'CLR'/'eLR'
OPTIM               =  'momentum'                         # 'adabound'/'adam'/'momentum'/'rmsprop'
CACHE_TDS_PATH      =  '/home/user/prostate_tds_cache'    # None/'/home/user/prostate_tds_cache'
CACHE_VDS_PATH      =  '/home/user/prostate_vds_cache'    # None/'/home/user/prostate_vds_cache'

# Exponentially Decaying LR Hyperparameters
eLR_INITIAL         =   1e-4
eLR_DECAY_EPOCHS    =   5     
eLR_DECAY_RATE      =   0.80

# Cyclic LR Hyperparameters
CLR_STEPFACTOR      =   2.5     
CLR_MODE            =  'exp_range'
CLR_GAMMA           =   0.9975
CLR_MINLR           =   1e-6
CLR_MAXLR           =   1e-4

# Derived Operational Hyperparameters
PREFETCH_CACHE_SIZE =   8
SHUFFLE_CACHE_SIZE  =   BATCH_SIZE*8 
MAX_STEPS           =   int(np.ceil((TRAIN_SIZE/BATCH_SIZE)*MAX_EPOCHS))
EVAL_EVERY_N_STEPS  =   int(np.ceil(MAX_STEPS/VAL_POINTS))
EVAL_STEPS          =   int(np.ceil(VAL_SIZE/BATCH_SIZE))
eLR_DECAY_STEPS     =   int(np.floor((TRAIN_SIZE/BATCH_SIZE)*eLR_DECAY_EPOCHS))
CLR_STEPSIZE        =   int(np.ceil((TRAIN_SIZE/BATCH_SIZE)*CLR_STEPFACTOR))
COUNT_EPOCHS        =   []
COUNT_LOSS          =   []

# Anchor Definition
AnchorParam = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                               ratios  = np.array([0.705, 1.000, 1.419], tf.keras.backend.floatx()),
                               scales  = np.array([0.400, 0.644, 1.031], tf.keras.backend.floatx()))

anchor_deltas_mean = [-0.079931999, 0.005310305, 0.075939696, 0.011442137]
anchor_deltas_std  = [ 0.081362432, 0.070324861, 0.080778639, 0.073184822]


def model_fn(features, labels, mode, params):
    with tf.device('/gpu:0'):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            
            # Model Definition
            model_output_ops = retinanet_2d(
                inputs                  =   features['x'],        
                num_classes             =   NUM_CLASSES,
                anchor_params           =   AnchorParam,
                anchor_deltas_mean      =   anchor_deltas_mean,
                anchor_deltas_std       =   anchor_deltas_std,
                backbone                =  'seresnet',
                backbone_filters        =  (128,256,512,512),
                backbone_stride         = ((1,1),(2,2),(2,2),(2,2)),
                FPN_channels            =   256, 
                regression_channels     =   128,
                regression_layers       =   1, 
                classification_channels =   128,
                classification_layers   =   1,
                class_specific_filter   =   False,
                nms                     =   True,
                max_detections          =   300,
                score_threshold         =   0.05,
                mode                    =   mode)         


            # Prediction Mode
            if (mode==tf.estimator.ModeKeys.PREDICT):
                return tf.estimator.EstimatorSpec(
                    mode            = mode,
                    predictions     = model_output_ops,
                    export_outputs  = {'out': tf.estimator.export.PredictOutput(model_output_ops)})


            # Loss Function
            focal_loss, fo_ind  = focal(labels['y_cl'], model_output_ops['classification'], alpha=0.25, gamma=2.0)
            l1_loss,    l1_ind  = smooth_l1(labels['y_rg'], model_output_ops['regression'], sigma=3.00)
            loss                = tf.reduce_sum(focal_loss+l1_loss)
            global_step         = tf.train.get_global_step()


            # Learning Rate
            if   (LR_MODE=='eLR'):
                # Exponential Learning Rate Decay
                learning_rate = tf.train.exponential_decay(eLR_INITIAL, global_step, decay_steps=eLR_DECAY_STEPS, decay_rate=eLR_DECAY_RATE, staircase=True)           
            elif (LR_MODE=='CLR'):
                # Cyclic Learning Rate 
                learning_rate = cyclic_learning_rate(global_step=global_step, learning_rate=CLR_MINLR, max_lr=CLR_MAXLR, step_size=CLR_STEPSIZE, gamma=CLR_GAMMA, mode=CLR_MODE)


            # Optimizer
            if   (OPTIM == 'adabound'):
                optimiser = AdaBoundOptimizer(
                    learning_rate=learning_rate, final_lr=1e-2, gamma=1e-3, beta1=0.9, beta2=0.999, amsbound=True)
                optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
            elif  (OPTIM == 'adam'):
                optimiser = tf.train.AdamOptimizer(
                    learning_rate=learning_rate, epsilon=1e-5)
                optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
            elif (OPTIM == 'momentum'):
                optimiser = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=0.9, use_nesterov=False)
                optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)
            elif (OPTIM == 'rmsprop'):
                optimiser = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate, momentum=0.9)
                optimiser = tf.contrib.estimator.TowerOptimizer(optimiser)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimiser.minimize(loss, global_step=global_step)


            # Custom Image Summaries (TensorBoard)
            image_summaries              = {}
            image_summaries['T2W_Patch'] = features['x'][0,:,:,0]
            image_summaries['ADC_Patch'] = features['x'][0,:,:,1]
            image_summaries['DWI_Patch'] = features['x'][0,:,:,2]        
            expected_output_size         = [1,PATCH_DIMS[0],PATCH_DIMS[1],1]  # [B,W,H,C]
            [tf.summary.image(name, tf.reshape(image, expected_output_size))
             for name, image in image_summaries.items()]

            # Track Metrics
            eval_metric_ops = {"focal_loss":     tf.metrics.mean(focal_loss),
                               "smooth_l1_loss": tf.metrics.mean(l1_loss),
                               "positive_fo":    tf.metrics.mean(fo_ind),
                               "positive_l1":    tf.metrics.mean(l1_ind)}

            # Return EstimatorSpec Object
            return tf.estimator.EstimatorSpec(mode            = mode,
                                              predictions     = model_output_ops,
                                              loss            = loss,
                                              train_op        = train_op,
                                              eval_metric_ops = eval_metric_ops)


def train(args):
    np.random.seed(8)
    tf.set_random_seed(8)

    print('Setting Up...')

    # Read Training-Fold.csv
    train_filenames = pd.read_csv(
        args.train_csv, dtype=object, keep_default_na=False,
        na_values=[]).values

    # Read Validation-Fold.csv
    val_filenames = pd.read_csv(
        args.val_csv, dtype=object, keep_default_na=False,
        na_values=[]).values

    # Set Reader Parameters (#Patches,Patch Dimensions) 
    reader_params       = {'patch_size':         PATCH_DIMS, 
                           'anchor_params':      AnchorParam, 
                           'anchor_mean':        anchor_deltas_mean,
                           'anchor_std':         anchor_deltas_std,
                           'display':            True,
                           'deploy_mode':        False }
    reader_patch_shapes = {'features': {'x':     reader_params['patch_size'] + [NUM_CHANNELS]},
                           'labels':   {'y_rg':  [None,5],
                                        'y_cl':  [None,2]}}
    
    # Initiate Data Reader + Patch Extraction
    reader = Reader(read_fn, {'features': {'x':    tf.float32},
                              'labels':   {'y_rg': tf.float32,
                                           'y_cl': tf.float32}})

    # Create Input Functions + Queue Initialisation Hooks for Training/Validation Data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references       = train_filenames,
        mode                  = tf.estimator.ModeKeys.TRAIN,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        cache_file            = CACHE_TDS_PATH,
        aug_prob              = AUG_PROB,
        aug_rot_degree        = AUG_ROT_DEGREE,
        aug_trans_factor      = AUG_TRANS_FACTOR,
        params                = reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references       = val_filenames,
        mode                  = tf.estimator.ModeKeys.EVAL,
        example_shapes        = reader_patch_shapes,
        batch_size            = BATCH_SIZE,
        shuffle_cache_size    = SHUFFLE_CACHE_SIZE,
        prefetch_cache_size   = PREFETCH_CACHE_SIZE,
        cache_file            = CACHE_VDS_PATH,
        aug_prob              = False,
        params                = reader_params)

    # Instantiate Neural Network Estimator
    nn = tf.estimator.Estimator(
        model_fn             = tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir            = args.model_path,
        config               = tf.estimator.RunConfig())                                         

    # Hooks for Validation Summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(os.path.join(args.model_path, 'eval'))
    step_cnt_hook    = tf.train.StepCounterHook(every_n_steps = EVAL_EVERY_N_STEPS,
                                                output_dir    = args.model_path)

    # Training Script
    print('Begin Training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            session            = tf.Session(config=config)
            nn.train(input_fn  = train_input_fn,
                     hooks     = [train_qinit_hook, step_cnt_hook],
                     steps     = EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val   = nn.evaluate(
                    input_fn  = val_input_fn,
                    hooks     = [val_qinit_hook, val_summary_hook],
                    steps     = EVAL_STEPS)

                
                EPOCH_DISPLAY = int( int(results_val['global_step']) / (TRAIN_SIZE/BATCH_SIZE))
                print('Epoch = {}; Step = {} / ValLoss = {:.5f}'.format(
                    EPOCH_DISPLAY, 
                    results_val['global_step'], 
                    results_val['loss']))
                
                export_dir                 = nn.export_savedmodel(
                export_dir_base            = args.model_path + 'Epoch-{}_ValLoss-{:.4f}_FocalLoss-{:.4f}_L1Loss-{:.4f}'.format(
                                             EPOCH_DISPLAY,results_val['loss'],results_val['focal_loss'],results_val['smooth_l1_loss']),
                serving_input_receiver_fn  = reader.serving_input_receiver_fn(reader_patch_shapes))
                print('Model saved to {}.'.format(export_dir))
                COUNT_EPOCHS.append(EPOCH_DISPLAY)
                COUNT_LOSS.append(results_val['loss'])

            tf.keras.backend.clear_session()
            session.close()

    except KeyboardInterrupt:
        pass

    # Define Expected Input Shape during Export
    export_dir = nn.export_savedmodel(
        export_dir_base           = args.model_path,
        serving_input_receiver_fn = reader.serving_input_receiver_fn(
            {'features': {'x':    [None, None, NUM_CHANNELS]},
             'labels':   {'y_rg': [None, 5],
                          'y_cl': [None, 2]}}))
    print('Model saved to {}.'.format(export_dir))
    # Export Validation Loss Values
    Step_Loss      = pd.DataFrame(list(zip(COUNT_EPOCHS,COUNT_LOSS)),
    columns        = ['epoch','val_loss'])
    Step_Loss.to_csv(args.model_path+"ValidationMetrics.csv", encoding='utf-8', index=False)



if __name__ == '__main__':
    # Suppress Future Warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Argument Parser Setup
    parser = argparse.ArgumentParser(description='Prostate Detection in mpMRI')
    parser.add_argument('--run_validation',     default=True)
    parser.add_argument('--restart',            default=False, action='store_true')
    parser.add_argument('--verbose',            default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path',   '-p', default='./models/mykonos/weights/002_14042020/')
    parser.add_argument('--train_csv',    '-t', default='./models/mykonos/feed/prostate-mpMRI-128_training-fold-1.csv')
    parser.add_argument('--val_csv',      '-v', default='./models/mykonos/feed/prostate-mpMRI-128_validation-fold-1.csv')
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

    # Handle Restarting/Resuming Training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))
    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Verify TensorFlow Integration
    print('TensorFlow Version:', tf.VERSION)
    print('TensorFlow-GPU:',     tf.test.is_built_with_cuda())

    # Train
    train(args)
