from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import traceback
import os
import numpy as np
import scipy.ndimage
import time
import math
import multiprocessing
from skimage import measure
from models.model_utils.anchors import anchor_targets_bbox, anchors_for_shape, compute_gt_annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
Prostate Detection in mpMRI
Script:         Data Reader - NumPy I/O
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         04/04/2020

'''


# Data Augmentation via Tensors -----------------------------------------------------------------------------------------------------------------------------------------
def augment_ds(ds, aug_prob, aug_rot_degree=20, aug_trans_factor=0.20):
    image = ds['features']['x']   # Extract Image from Dataset
    # Axial-Plane Augmentations
    image = tf.cond(tf.random_uniform([],0,1)>(1-aug_prob), lambda: tf.image.flip_left_right(image),                               lambda: image)  # Horizontal Flip
    image = tf.cond(tf.random_uniform([],0,1)>(1-aug_prob), lambda: rotate_4D_tensor(image,    ROTATION_DEGREE=aug_rot_degree),    lambda: image)  # Rotation
    image = tf.cond(tf.random_uniform([],0,1)>(1-aug_prob), lambda: translate_4D_tensor(image, TRANSLATE_FACTOR=aug_trans_factor), lambda: image)  # XY Translation
    ds['features']['x'] = image
    return ds


# Translation Augmentation w/ 4D Tensors
def translate_4D_tensor(input_tensor, TRANSLATE_FACTOR=0.20, PAD_MODE='SYMMETRIC'):
    # Translation Offset Values
    pad_top      = tf.random_uniform([],0,tf.cast(input_tensor.get_shape()[1].value*TRANSLATE_FACTOR, dtype=tf.int32), dtype=tf.int32)
    pad_bottom   = tf.random_uniform([],0,tf.cast(input_tensor.get_shape()[1].value*TRANSLATE_FACTOR, dtype=tf.int32), dtype=tf.int32)
    pad_right    = tf.random_uniform([],0,tf.cast(input_tensor.get_shape()[2].value*TRANSLATE_FACTOR, dtype=tf.int32), dtype=tf.int32)
    pad_left     = tf.random_uniform([],0,tf.cast(input_tensor.get_shape()[2].value*TRANSLATE_FACTOR, dtype=tf.int32), dtype=tf.int32)
    
    # Translation + Padding
    x            = pad_to_bounding_box(input_tensor, pad_top, pad_left, 
                                       input_tensor.get_shape()[1] + pad_bottom + pad_top, 
                                       input_tensor.get_shape()[2] + pad_right  + pad_left,
                                       PAD_MODE = PAD_MODE)
    # Cropping to Original Shape
    output       = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, 
                                                 input_tensor.get_shape()[1].value, 
                                                 input_tensor.get_shape()[2].value)
    return output


# Translation Augmentation w/ 4D Tensors
def rotate_4D_tensor(input_tensor, ROTATION_DEGREE=20, PAD_MODE='SYMMETRIC'):    
    diagonal    = ((input_tensor.get_shape()[1].value)**2+(input_tensor.get_shape()[2].value)**2)**0.5
    pad         = np.ceil((diagonal-min((input_tensor.get_shape()[1].value),(input_tensor.get_shape()[2].value)))/2).astype(np.int32)
    
    # Preliminary Padding
    x           = pad_to_bounding_box(input_tensor, pad, pad, 
                                      input_tensor.get_shape()[1] + (2*pad), 
                                      input_tensor.get_shape()[2] + (2*pad),
                                      PAD_MODE = PAD_MODE)
    # Rotation
    angle       = tf.random_uniform([],-ROTATION_DEGREE,ROTATION_DEGREE, dtype=tf.float32)
    x           = tf.contrib.image.rotate(x,angle*math.pi/180, interpolation='BILINEAR')
    
    # Cropping to Original Shape
    ctr_frac    = input_tensor.get_shape()[1].value/x.get_shape()[1].value
    output      = tf.image.central_crop(x,ctr_frac)
    return output


# Modified Native 'tf.image.pad_to_bounding_box' Function
def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, PAD_MODE='CONSTANT'):

    image = ops.convert_to_tensor(image, name='image')

    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError('\'image\' must have either 3 or 4 dimensions.')

    batch, height, width, depth = _ImageDimensions(image, rank=4)

    after_padding_width  = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    paddings = array_ops.reshape(
        array_ops.stack([
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = array_ops.pad(image, paddings, mode=PAD_MODE)

    padded_shape = [
        None if isinstance(i,(ops.Tensor, variables.Variable)) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded

# Extract Image Dimensions [Native to TF]
def _ImageDimensions(image, rank):
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape  = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [ s if s is not None else d for s, d in zip(static_shape, dynamic_shape) ]

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Dataset Generator/Wrapper ---------------------------------------------------------------------------------------------------------------------------------------------
# Hook to Initialise Data Iterator [Ref:DLTK]
class IteratorInitializerHook(tf.estimator.SessionRunHook):
    def __init__(self):                              
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):  # Initialise Iterator
        self.iterator_initializer_func(session)


# Wrapper for Dataset Generation via Read Function [Ref:DLTK]
class Reader(object):
    def __init__(self, read_fn, dtypes):
        self.dtypes  = dtypes
        self.read_fn = read_fn

    def get_inputs(self,file_references,mode,
                   example_shapes      = None,
                   shuffle_cache_size  = 128,
                   prefetch_cache_size = 128,
                   batch_size          = 4,
                   cache_file          = None,
                   aug_prob            = 0,
                   aug_rot_degree      = 0,
                   aug_trans_factor    = 0,
                   params              = None):

        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            def f():
                def clean_ex(ex, compare):
                    # Clean Example Dictionary by Recursively Deleting Non-Relevant Entries
                    for k in list(ex.keys()):
                        if k not in list(compare.keys()):
                            del ex[k]
                        elif isinstance(ex[k], dict) \
                                and isinstance(compare[k], dict):
                            clean_ex(ex[k], compare[k])
                        elif (isinstance(ex[k], dict)
                              and not isinstance(compare[k], dict)) \
                                or (not isinstance(ex[k], dict)
                                    and isinstance(compare[k], dict)):
                            raise ValueError('Entries between example and '
                                             'dtypes incompatible for key {}'
                                             ''.format(k))
                        elif ((isinstance(ex[k], list)
                               and not isinstance(compare[k], list))
                              or (not isinstance(ex[k], list)
                                  and isinstance(compare[k], list))
                              or (isinstance(ex[k], list)
                                  and isinstance(compare[k], list)
                                  and not len(ex[k]) == len(compare[k]))):
                            raise ValueError('Entries between example and '
                                             'dtypes incompatible for key {}'
                                             ''.format(k))
                    for k in list(compare):
                        if k not in list(ex.keys()):
                            raise ValueError('Key {} not found in ex but is '
                                             'present in dtypes')
                    return ex

                fn = self.read_fn(file_references, mode, params)
                # Iterate over All Entries
                while True:
                    try:
                        ex = next(fn)
                        if ex.get('labels') is None:
                            ex['labels'] = None
                        if not isinstance(ex, dict):
                            raise ValueError('The read_fn has to return '
                                             'dictionaries')
                        ex = clean_ex(ex, self.dtypes)
                        yield ex
                    except (tf.errors.OutOfRangeError, StopIteration):  return    # Updated for Python3.7; uses 'raise' for Python3.6
                    except Exception as e:
                        print('got error `{} from `_read_sample`:'.format(e))
                        print(traceback.format_exc())
                        raise

            # TensorFlow Dataset Generation
            dataset       = tf.data.Dataset.from_generator(f,self.dtypes,example_shapes)
            if cache_file:             # Cache Dataset on Remote Server Local Memory/RAM for Efficiency
                dataset   = dataset.cache(filename=cache_file)
            if (aug_prob>0):  # Data Augmentation
                dataset   = dataset.map(lambda x: augment_ds(x,aug_prob,aug_rot_degree,aug_trans_factor), 
                                        num_parallel_calls=multiprocessing.cpu_count())
            dataset       = dataset.repeat(None)
            dataset       = dataset.shuffle(shuffle_cache_size)
            dataset       = dataset.batch(batch_size)
            dataset       = dataset.prefetch(prefetch_cache_size)
            iterator      = dataset.make_initializable_iterator()
            next_dict     = iterator.get_next()

            # Set Runhook to Initialize Iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(iterator.initializer)
            # Return Paired Input (Features,Labels)
            return next_dict['features'], next_dict.get('labels')
        # Return Function and Hook
        return train_inputs, iterator_initializer_hook

    def serving_input_receiver_fn(self, placeholder_shapes):
        # Function to be passed to tf.estimator.Estimator Instance 
        # when exporting Saved Model with tf.estimator.export_savedmodel
        def f():
            inputs = {k: tf.placeholder(
                shape=[None] + list(placeholder_shapes['features'][k]),
                dtype=self.dtypes['features'][k])
                      for k in list(self.dtypes['features'].keys())}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        return f
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Read Function ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Generates Class Bounding Box (Returns arr[ymin,xmin,ymax,xmax] from TL Corner)
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array((cmin,rmin,cmax,rmax))

# Return Pyramid Level Feature Map Shapes
def __patch_shapes(image_shape, pyramid_levels):
    return [np.array([128,128]), np.array([64,64]), np.array([32,32]), np.array([16,16]), np.array([8,8])]

def read_fn(file_references, mode, params=None):
  for f in file_references:
    t0         = time.time()  
    subject_id = str(f[0])
    img        = np.load(f[2])

    # Testing Mode with No Labels
    if (mode == tf.estimator.ModeKeys.PREDICT):
      yield {'features':  {'x': img.astype(np.float32)},
             'labels':      None,
             'img_id':      subject_id}

    # Load Labels for Training/Validation Mode
    elif (mode == tf.estimator.ModeKeys.TRAIN)|(mode == tf.estimator.ModeKeys.EVAL):
      lbl         = np.load(f[3])
      lbl[lbl!=0] = 1

      # Return Training Examples (Image, Class Label, Bounding Box Coordinates)
      for e in range(img.shape[0]):
        if (np.any(lbl[e]==1).astype(np.int32)==0):  continue    # Skip Slices with Absence of Prostate
        else:

          annotations  = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

          # Annotate Prostate
          bbox_label            = bbox(lbl[e])
          annotations['labels'] = np.concatenate((annotations['labels'], [0]))
          annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[float(bbox_label[0]), float(bbox_label[1]),
                                                                           float(bbox_label[2]), float(bbox_label[3])]]))
          # Construct Anchors and Full Annotations
          anchors               = anchors_for_shape(img[e].shape, anchor_params=params['anchor_params'], shapes_callback=__patch_shapes)
          full_annotations      = anchor_targets_bbox(anchors, [img[e].astype(np.float32)], [annotations], 
                                                      num_classes=1, negative_overlap=0.4, positive_overlap=0.5,
                                                      anchor_deltas_mean=params['anchor_mean'], anchor_deltas_std=params['anchor_std'])
          # Display Image,Label Pairs in Console
          regression_deltas     = full_annotations[0][0]
          foreground_deltas     = regression_deltas[np.where(np.equal(regression_deltas[:,-1],1))]
          classification_labels = full_annotations[1][0]
          background_labels     = classification_labels[np.where(np.equal(classification_labels[:,-1],0))]
          foreground_labels     = classification_labels[np.where(np.equal(classification_labels[:,-1],1))]

          if params['display']:
            print('------------------------------------------------------------------------------') 
            print('Loaded Scan {}; Slice {}; I/O Time = {:.4f}s'.format(subject_id,e,(np.any(lbl==1).astype(np.int32)),(time.time()-t0)))
            print('Prostate Bounding Box: [', str(bbox_label[0]), str(bbox_label[1]), str(bbox_label[2]), str(bbox_label[3]),']')
            print('Background Anchors: ', background_labels.shape, '; Foreground Anchors: ', foreground_labels.shape)

          if (params['deploy_mode']==True):                            # Inference/Evaluation Labels (Raw Annotations)
            yield {'features': {'x':    img[e].astype(np.float32)},
                   'labels':   {'y_rg': annotations['bboxes'],
                                'y_cl': annotations['labels'],
                                'mask': lbl[e]},
                   'img_id':     subject_id}
          else:                                                        # Training/Validation Labels (Anchors)
            yield {'features': {'x':    img[e].astype(np.float32)},
                   'labels':   {'y_rg': full_annotations[0][0],
                                'y_cl': full_annotations[1][0]},
                   'img_id':     subject_id}
  return
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------



