from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import math
from .model_utils.anchors import AnchorParameters
from .model_utils.ops import RegressBoxes, UpsampleLike, Anchors, ClipBoxes
from .model_utils.filter_detections import FilterDetections


'''
Prostate Detection in mpMRI
Script:         2D RetinaNet Model Definitions
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         10/04/2020

'''



# Apply Prior Probability to Weights
class PriorProbability(tf.keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {'probability': self.probability}

    def __call__(self, shape, dtype):
        # Bias: -log((1-p)/p) for Foreground
        return np.ones(shape) * -math.log((1-self.probability) / self.probability)



# Classification Sub-Model (Predicts Classes for Each Anchor)
def default_classification_model(inputs, num_classes, num_anchors, num_layers=8,
                                 prior_probability=0.01, classification_feature_size=64,
                                 name='classification_submodel'):
    x       =   inputs
    options = {'kernel_size' : 3,
               'strides'     : 1,
               'padding'     : 'same'}
    # Convolutional Layers
    for i in range(num_layers):
        x = tf.keras.layers.Conv2D(filters             = classification_feature_size, 
                                   kernel_initializer  = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                                   kernel_regularizer  = tf.contrib.layers.l2_regularizer(1e-3),
                                   bias_initializer    = tf.zeros_initializer(), **options)(x)
        x = tf.keras.layers.Activation("relu")(x)
    # Final Convolutional Layer
    x = tf.keras.layers.Conv2D(filters                 = num_classes * num_anchors,
                               kernel_initializer      = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None,),
                               kernel_regularizer      = tf.contrib.layers.l2_regularizer(1e-3),
                               bias_initializer        = PriorProbability(probability=prior_probability),**options)(x)
    # Reshape Output and Apply Sigmoid
    x       = tf.keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(x)
    outputs = tf.keras.layers.Activation("sigmoid", name='pyramid_classification_sigmoid')(x)
    return outputs



# Regression Sub-Model (Predicts Regression Values for Each Anchor)
def default_regression_model(inputs, num_values, num_anchors, num_layers=8, 
                             regression_feature_size=64, 
                             name='regression_submodel'):
    x       = inputs
    options = {'kernel_size'        : 3,
               'strides'            : 1,
               'padding'            : 'same',
               'kernel_initializer' : tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
               'kernel_regularizer' : tf.contrib.layers.l2_regularizer(1e-3),
               'bias_initializer'   : tf.zeros_initializer()}
    # Convolutional Layers
    for i in range(num_layers):
        x = tf.keras.layers.Conv2D(filters=regression_feature_size,**options)(x)
        x = tf.keras.layers.Activation("relu")(x)    
    # Final Convolutional Layer and Output
    x       = tf.keras.layers.Conv2D(filters=num_anchors*num_values, **options)(x)
    outputs = tf.keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(x)
    return outputs



# Feature Pyramid Network (Returns List of Feature Levels [P2,P3,P4,P5,P6] on Backbone Features)
def create_pyramid_features(C2, C3, C4, C5, feature_size=128):
    """
       Bottom-Down           Top-Up
    Backbone Features    Feature Pyramid
            XX                - P6 -             [Strided Convolution on C5]
         -- C5 --            -- P5 --            [Convolution on C5]    
       ---- C4 ----        ---- P4 ----          [Upsampled P5 + Reduced C4]
     ------ C3 ------    ------ P3 ------        [Upsampled P4 + Reduced C3]
    ------- C2 -------  ------- P2 -------       [Upsampled P3 + Reduced C2]
    """
    P6           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P5           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5,C4])
    P5           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)
   
    P4           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = tf.keras.layers.Add(name='P4_merged')([P5_upsampled,P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4,C3])
    P4           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
   
    P3           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = tf.keras.layers.Add(name='P3_merged')([P4_upsampled,P3])
    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3,C2])
    P3           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
   
    P2           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2           = tf.keras.layers.Add(name='P2_merged')([P3_upsampled,P2])
    P2           = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    return [P2,P3,P4,P5,P6]



# Generates List of Sub-Models for Object Detection
def default_submodels(num_classes, num_anchors):
    return [('regression',     default_regression_model(4, num_anchors)),
            ('classification', default_classification_model(num_classes, num_anchors))]

# Applies Single Sub-Model to Each FPN Level (Returns Tensor with Sub-Model Response for FPN Features)
def __build_model_pyramid(name, model, features):
    return tf.keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

# Applies All Sub-Models to Each FPN Level (Returns Tensor/Sub-Model Response)
def __build_pyramid(models, features):
    return [__build_model_pyramid(n,m,features) for n,m in models]

# Builds Anchors for FPN Feature Shapes (Returns Tensor Containing Anchors [Shape:(batch_size,num_anchors,4)])
def __build_anchors(anchor_parameters, features):
    anchors = [Anchors(size=anchor_parameters.sizes[i], stride=anchor_parameters.strides[i],
                       ratios=anchor_parameters.ratios, scales=anchor_parameters.scales,
                       name='anchors_{}'.format(i))(f) for i, f in enumerate(features)]
    return tf.keras.layers.Concatenate(axis=1, name='anchors')(anchors)



# 2D RetinaNet Model ---------------------------------------------------------------------------------------------------------------------------------------------------
def retinanet_2d(inputs, num_classes=1, 
                 anchor_params=None, anchor_deltas_mean=None, anchor_deltas_std=None, 
                 backbone                = 'resnet', 
                 backbone_filters        = (128,256,512,512), 
                 backbone_stride         = ((1,1),(2,2),(2,2),(2,2)), 
                 FPN_channels            = 128, 
                 regression_channels     = 64, 
                 regression_layers       = 8,
                 classification_channels = 64,
                 classification_layers   = 8, 
                 class_specific_filter   = True, 
                 nms=True, score_threshold=0.01, max_detections=300, 
                 mode=tf.estimator.ModeKeys.EVAL):

    if (anchor_params==None):  anchor_params = AnchorParameters.default
    num_anchors                              = anchor_params.num_anchors()

    # Preliminary Convolutional Layer
    x  = tf.keras.layers.Conv2D(filters=backbone_filters[0]//2, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same', kernel_initializer='he_uniform')(inputs)
    x  = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x  = tf.keras.layers.Activation('relu')(x)

    if (backbone=='resnet'):
      # Backbone Residual Features
      C2 = residual_block_2d(input_tensor=x,  input_channels=backbone_filters[0]//2, output_channels=backbone_filters[0], kernel_size=(3,3), stride=backbone_stride[0], mode=mode)
      C3 = residual_block_2d(input_tensor=C2, input_channels=backbone_filters[0],    output_channels=backbone_filters[1], kernel_size=(3,3), stride=backbone_stride[1], mode=mode)
      C4 = residual_block_2d(input_tensor=C3, input_channels=backbone_filters[1],    output_channels=backbone_filters[2], kernel_size=(3,3), stride=backbone_stride[2], mode=mode)
      C5 = residual_block_2d(input_tensor=C4, input_channels=backbone_filters[3],    output_channels=backbone_filters[3], kernel_size=(3,3), stride=backbone_stride[3], mode=mode)

    if (backbone=='seresnet'):
      C2 = se_residual_block_2d(input_tensor=x,  filters=backbone_filters[0], reduction=8, strides=backbone_stride[0], mode=mode)
      C3 = se_residual_block_2d(input_tensor=C2, filters=backbone_filters[1], reduction=8, strides=backbone_stride[1], mode=mode)
      C4 = se_residual_block_2d(input_tensor=C3, filters=backbone_filters[2], reduction=8, strides=backbone_stride[2], mode=mode)
      C5 = se_residual_block_2d(input_tensor=C4, filters=backbone_filters[3], reduction=8, strides=backbone_stride[3], mode=mode)
 
    # Feature Pyramid Network (Compute Pyramid Features, Sub-Model Responses on Features)
    features   = create_pyramid_features(C2,C3,C4,C5, feature_size=FPN_channels)                             
    response   = [tf.keras.layers.Concatenate(axis=1, name='regression')([
                        default_regression_model(inputs=f, num_values=4, num_anchors=num_anchors, 
                          regression_feature_size=regression_channels, num_layers=regression_layers) for f in features]),
                  tf.keras.layers.Concatenate(axis=1, name='classification')([
                        default_classification_model(inputs=f, num_classes=num_classes, num_anchors=num_anchors, 
                          classification_feature_size=classification_channels, num_layers=classification_layers) for f in features])]
    
    # --------------------------------------------------------------------------------------------------------------------------------
    print('Bottom-Down Backbone Features:', '\n',C2.get_shape(),'\n',C3.get_shape(),'\n',C4.get_shape(),'\n',C5.get_shape())
    print('Top-Up Feature Pyramid Network:','\n',features[4].get_shape(),'\n',features[3].get_shape(),'\n',features[2].get_shape(),
                                            '\n',features[1].get_shape(),'\n',features[0].get_shape())
    # --------------------------------------------------------------------------------------------------------------------------------

    # For Inference Stage Only
    with tf.device('/cpu:0'):
        # Compute Anchors and Extract Sub-Model Model Outputs
        anchors        = __build_anchors(anchor_params, features)    
        regression     = response[0]
        classification = response[1]
      
        # Apply Predicted Regression Values to Anchors
        boxes = RegressBoxes(name='boxes', mean=anchor_deltas_mean, std=anchor_deltas_std)([anchors,regression])
        boxes = ClipBoxes(name='clipped_boxes')([inputs,boxes])
      
        # Filter Detections (NMS/Score-Threshold/Top-K Selection)
        detection_boxes, detection_scores, detection_labels = FilterDetections(nms=nms, class_specific_filter=class_specific_filter, score_threshold=score_threshold, 
                                                                               max_detections=max_detections, name='filtered_detections')([boxes,classification])
    return {'regression':       regression,
            'classification':   classification,
            'detection_boxes':  detection_boxes,
            'detection_scores': detection_scores,
            'detection_labels': detection_labels}
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------








# Pre-Activation 2D Residual Block -------------------------------------------------------------------------------------------------------------------------------------
def residual_block_2d(input_tensor, input_channels=None, output_channels=None, kernel_size=(3,3), stride=(1,1), 
                      kernel_initializer = tf.initializers.variance_scaling(distribution='uniform'), 
                      kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), 
                      mode               = tf.estimator.ModeKeys.EVAL):
    # Define Target Channel Shapes
    if output_channels is None:
      output_channels = input_tensor.get_shape()[-1].value
    if input_channels is None:
      input_channels  = output_channels // 4

    conv_params  = {'padding':           'same',
                    'kernel_initializer': kernel_initializer,
                    'kernel_regularizer': kernel_regularizer,
                    'data_format':       'channels_last'}

    # First Convolutional Layer
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1,1), strides=(1,1), **conv_params)(x)

    # Second Convolutional Layer    
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=kernel_size, strides=stride, **conv_params)(x)

    # Third Convolutional Layer
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=(1,1), strides=(1,1), **conv_params)(x)

    if (input_channels!=output_channels)|(stride!=1):
      input_tensor = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=kernel_size, strides=stride, **conv_params)(input_tensor)

    # Residual Addition
    x = tf.keras.layers.Add()([x,input_tensor])
    return x
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 2D SE-Residual Block -------------------------------------------------------------------------------------------------------------------------------------------------
def se_residual_block_2d(input_tensor, filters=16, reduction=16, strides=(1,1), 
                         mode=tf.estimator.ModeKeys.EVAL):
    x        = input_tensor
    residual = input_tensor

    # Bottleneck
    x = tf.keras.layers.Conv2D(filters=filters//4, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
    x = tf.keras.layers.Conv2D(filters=filters//4, kernel_size=(3,3), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(x)

    # Replicate Operations with Residual Connection (change in #num_filters or spatial_dims)
    x_channels = x.get_shape()[-1].value
    r_channels = residual.get_shape()[-1].value
    if (strides!=1)|(x_channels!=r_channels):
        residual = tf.keras.layers.Conv2D(filters=x_channels, kernel_size=(1,1), strides=strides, use_bias=False, kernel_initializer='he_uniform')(residual)
        residual = tf.keras.layers.BatchNormalization(trainable=mode==tf.estimator.ModeKeys.TRAIN, epsilon=9.999999747378752e-06)(residual)

    # Attention Module
    x = ChannelSE(input_tensor=x, reduction=reduction)

    # Residual Addition
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Squeeze-and-Excitation Block
def ChannelSE(input_tensor, reduction=16):
    channels = input_tensor.get_shape()[-1].value
    x        = input_tensor

    # Squeeze-and-Excitation Block (originally derived from PyTorch)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Lambda(function=se_expand_dims)(x)
    x = tf.keras.layers.Conv2D(filters=channels//reduction, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1,1), strides=(1,1), use_bias=False, kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    # Attention
    x = tf.keras.layers.Multiply()([input_tensor, x])
    return x

def se_expand_dims(x):
    return x[:,None,None,:]
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


