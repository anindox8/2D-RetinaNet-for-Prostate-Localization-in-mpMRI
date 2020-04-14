from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from . import anchors as utils_anchors


# TensorFlow Keras Layer for Generating Anchors for Given Shape
class Anchors(tf.keras.layers.Layer):
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """
        size:   Base size of the anchors to generate.
        stride: Stride of the anchors to generate.
        ratios: Ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
        scales: Scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if isinstance(ratios, list):   self.ratios  = np.array(ratios)
        if isinstance(scales, list):   self.scales  = np.array(scales)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors     = tf.keras.backend.variable(utils_anchors.generate_anchors(base_size=self.size, ratios=self.ratios, scales=self.scales))
        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features       = inputs
        features_shape = tf.keras.backend.shape(features)
        anchors        = shift(features_shape[1:3], self.stride, self.anchors)         # Generate Proposals from Bounding Box Deltas and Shifted Anchors
        anchors        = tf.keras.backend.tile(tf.keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))
        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({'size'   : self.size,
                       'stride' : self.stride,
                       'ratios' : self.ratios.tolist(),
                       'scales' : self.scales.tolist()})
        return config



# TensorFlow Keras Layer for Upsampling Tensor to Given Shape
class UpsampleLike(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape   = tf.keras.backend.shape(target)
        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)



# TensorFlow Keras Layer for Applying Regression Values to Boxes
class RegressBoxes(tf.keras.layers.Layer):
    def __init__(self, mean=None, std=None, *args, **kwargs):
        """
        mean: Mean value of the regression values which was used for normalization.
        std:  Standard value of the regression values which was used for normalization.
        """
        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({'mean': self.mean.tolist(),
                       'std' : self.std.tolist()})
        return config



# TensorFlow Keras Layer to Clip Box Values to Fall Within Given Shape
class ClipBoxes(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        image, boxes        = inputs
        shape               = tf.keras.backend.cast(tf.keras.backend.shape(image), tf.keras.backend.floatx())
        _, height, width, _ = tf.unstack(shape, axis=0)
        x1, y1, x2, y2      = tf.unstack(boxes, axis=-1)
        x1                  = tf.clip_by_value(x1, 0, width  - 1)
        y1                  = tf.clip_by_value(y1, 0, height - 1)
        x2                  = tf.clip_by_value(x2, 0, width  - 1)
        y2                  = tf.clip_by_value(y2, 0, height - 1)
        return tf.keras.backend.stack([x1,y1,x2,y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]



# Generate Shifted Anchors based on Map Shape and Stride Size
def shift(shape, stride, anchors):
    """
    shape  : Shape to shift the anchors over.
    stride : Stride to shift the anchors with over the shape.
    anchors: Anchors to apply at each location.
    """
    shift_x           = (tf.keras.backend.arange(0, shape[1], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride
    shift_y           = (tf.keras.backend.arange(0, shape[0], dtype=tf.keras.backend.floatx()) + tf.keras.backend.constant(0.5, dtype=tf.keras.backend.floatx())) * stride
    shift_x, shift_y  = tf.meshgrid(shift_x, shift_y)
    shift_x           = tf.keras.backend.reshape(shift_x, [-1])
    shift_y           = tf.keras.backend.reshape(shift_y, [-1])
    shifts            = tf.keras.backend.stack([shift_x,shift_y,shift_x,shift_y], axis=0)
    shifts            = tf.keras.backend.transpose(shifts)
    number_of_anchors = tf.keras.backend.shape(anchors)[0]
    k                 = tf.keras.backend.shape(shifts)[0]  # Number of Base Points = feat_H * feat_W
    shifted_anchors   = tf.keras.backend.reshape(anchors, [1,number_of_anchors,4]) + tf.keras.backend.cast(tf.keras.backend.reshape(shifts,[k,1,4]), tf.keras.backend.floatx())
    shifted_anchors   = tf.keras.backend.reshape(shifted_anchors, [k*number_of_anchors,4])
    return shifted_anchors



# Applies Regressions Results to Bounding Boxes/Anchors
def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """
    boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
    deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
    mean  : Mean value used when computing deltas (defaults to [0, 0, 0, 0]).
    std   : Standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
    """
    if (mean==None)|(std==None):
        width      = boxes[:,:,2] - boxes[:,:,0]
        height     = boxes[:,:,3] - boxes[:,:,1]
        x1         = boxes[:,:,0] + deltas[:,:,0] * width
        y1         = boxes[:,:,1] + deltas[:,:,1] * height
        x2         = boxes[:,:,2] + deltas[:,:,2] * width
        y2         = boxes[:,:,3] + deltas[:,:,3] * height
    else:    
        width      = boxes[:,:,2] - boxes[:,:,0]
        height     = boxes[:,:,3] - boxes[:,:,1]
        x1         = boxes[:,:,0] + (deltas[:,:,0] * std[0] + mean[0]) * width
        y1         = boxes[:,:,1] + (deltas[:,:,1] * std[1] + mean[1]) * height
        x2         = boxes[:,:,2] + (deltas[:,:,2] * std[2] + mean[2]) * width
        y2         = boxes[:,:,3] + (deltas[:,:,3] * std[3] + mean[3]) * height

    return tf.keras.backend.stack([x1,y1,x2,y2], axis=2)