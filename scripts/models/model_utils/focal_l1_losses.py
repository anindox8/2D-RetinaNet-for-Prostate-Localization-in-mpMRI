from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np



# Focal Loss
def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    alpha:  Scale the focal weight with alpha.
    gamma:  Take the power of the focal weight with gamma.
    y_true: Tensor of target data from the generator with shape  (B, N, num_classes).
    y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
    """
    classification = y_pred
    labels         = y_true[:,:,:-1]
    anchor_state   = y_true[:,:,-1]    # [-1,0,1] : [ignore,background,object]

    # Filter Out "ignore" Anchors
    indices        = tf.where(tf.keras.backend.not_equal(anchor_state,-1))
    labels         = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # Compute Focal Loss
    alpha_factor   = tf.keras.backend.ones_like(labels) * alpha
    alpha_factor   = tf.where(tf.keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight   = tf.where(tf.keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight   = alpha_factor * focal_weight ** gamma
    cls_loss       = focal_weight * tf.keras.backend.binary_crossentropy(labels, classification)

    # Compute Normalizer: Number of Positive Anchors
    normalizer     = tf.where(tf.keras.backend.equal(anchor_state, 1))
    normalizer     = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
    normalizer     = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)
    
    return tf.keras.backend.sum(cls_loss)/normalizer, tf.reduce_sum(tf.where(tf.keras.backend.equal(anchor_state,1)))



# L1 Loss
def smooth_l1(y_true, y_pred, sigma=3.0):
    """
    sigma:  This argument defines the point where the loss changes from L2 to L1.
    y_true: Tensor from the generator of shape (B, N, 5). Last value per box is the state (ignore/negative/positive).
    y_pred: Tensor from the network of shape   (B, N, 4).
    """
    sigma_squared     = sigma ** 2

    # Separate Target and State
    regression        = y_pred
    regression_target = y_true[:,:,:-1]
    anchor_state      = y_true[:,:,-1]    # [-1,0,1] : [ignore,background,object]

    # Filter Out "ignore" Anchors
    indices           = tf.where(tf.keras.backend.equal(anchor_state,1))
    regression        = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # Compute Smooth L1 Loss
    regression_diff   = regression - regression_target
    regression_diff   = tf.keras.backend.abs(regression_diff)
    regression_loss   = tf.where(tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
                                 0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
                                 regression_diff - 0.5 / sigma_squared)

    # Compute Normalizer: Number of Positive Anchors
    normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
    normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
    
    return tf.keras.backend.sum(regression_loss)/normalizer, tf.reduce_sum(tf.where(tf.keras.backend.equal(anchor_state,1)))