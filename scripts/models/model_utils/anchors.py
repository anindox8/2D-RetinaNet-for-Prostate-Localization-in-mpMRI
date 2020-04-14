from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .compute_overlap import compute_overlap


# Anchor Definitions
class AnchorParameters:
    """
    sizes   : List of sizes to use. Each size corresponds to one feature level.
    strides : List of strides to use. Each stride correspond to one feature level.
    ratios  : List of ratios to use per location in a feature map.
    scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales
    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

'''
# Default Anchor Parameters
AnchorParameters.default = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                                            ratios  = np.array([0.35,0.70,1.20,2.25,3.50], tf.keras.backend.floatx()),
                                            scales  = np.array([0.25,0.50,0.75,1.00,1.50], tf.keras.backend.floatx()))

# Anchor Parameters (Average Overlap: 0.522)
AnchorParameters.default = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                                            ratios  = np.array([0.779,1.000,1.284], tf.keras.backend.floatx()),
                                            scales  = np.array([0.496,1.124,1.383], tf.keras.backend.floatx()))
'''
# Anchor Parameters (Average Overlap: 0.522)
AnchorParameters.default = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                                            ratios  = np.array([0.442,0.778,1.000,1.286,2.265], tf.keras.backend.floatx()),
                                            scales  = np.array([0.496,0.741,1.162], tf.keras.backend.floatx()))
'''
# Anchor Parameters (Average Overlap: 0.522)
AnchorParameters.default = AnchorParameters(sizes   = [32,64,128,256,512],  strides = [8,16,32,64,128],
                                            ratios  = np.array([0.270,0.778,1.000,1.286,3.701], tf.keras.backend.floatx()),
                                            scales  = np.array([0.496,0.699,0.853,1.062,1.497], tf.keras.backend.floatx()))

# Anchor Parameters (Average Overlap: 0.768)
AnchorParameters.default = AnchorParameters(sizes   = [16,32,64,128,256],  strides = [4,8,16,32,64],
                                            ratios  = np.array([0.778, 1.000, 1.286], tf.keras.backend.floatx()),
                                            scales  = np.array([0.992, 1.118, 1.424], tf.keras.backend.floatx()))

# Anchor Parameters (Average Overlap: 0.880)
AnchorParameters.default = AnchorParameters(sizes   = [8,16,32,64,128],  strides = [2,4,8,16,32],
                                            ratios  = np.array([0.667, 1.000, 1.500], tf.keras.backend.floatx()),
                                            scales  = np.array([0.903, 1.837, 1.843], tf.keras.backend.floatx()))

# Anchor Parameters (Average Overlap: 0.880)
AnchorParameters.default = AnchorParameters(sizes   = [8,16,32,64,128],  strides = [2,4,8,16,32],
                                            ratios  = np.array([0.665,0.940,1.000,1.064,1.503], tf.keras.backend.floatx()),
                                            scales  = np.array([0.903,1.837,1.843], tf.keras.backend.floatx()))
'''


# Generates Anchor Targets for Bounding Box Detection
def anchor_targets_bbox(anchors,     image_group,          annotations_group,
                        num_classes, negative_overlap=0.4, positive_overlap=0.5,
                        anchor_deltas_mean=None, anchor_deltas_std=None):
    """
    anchors:           np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
    image_group:       List of BGR images.
    annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
    num_classes:       Number of classes to predict.
    mask_shape:        If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
    negative_overlap:  IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
    positive_overlap:  IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    """
    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0),                 "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size        = len(image_group)
    regression_batch  = np.zeros((batch_size, anchors.shape[0], 4 + 1),           dtype=tf.keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=tf.keras.backend.floatx())

    # Compute Labels and Regression Targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            
            # Obtain Indices of Ground Truth Annotations with Greatest Overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     =  1
            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] =  1

            # Compute Target Class Labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1
            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :], mean=anchor_deltas_mean, std=anchor_deltas_std)

        # Ignore Annotations Outside of Image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices         = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])
            labels_batch[index, indices, -1]     = -1
            regression_batch[index, indices, -1] = -1
    return regression_batch, labels_batch


# Compute Ground Truth Annotation
def compute_gt_annotations(anchors, annotations, negative_overlap=0.4, positive_overlap=0.5):
    """
    anchors:          np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
    annotations:      np.array of shape (N, 5) for (x1, y1, x2, y2, label).
    negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
    positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    """
    overlaps             = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
    # Assign Redundant Labels
    positive_indices     = max_overlaps >= positive_overlap
    ignore_indices       = (max_overlaps > negative_overlap) & ~positive_indices
    return positive_indices, ignore_indices, argmax_overlaps_inds


# Computes Layer Shapes Given Input Image Shape and Target Model
def layer_shapes(image_shape, model):
    shape          = {model.layers[0].name: (None,) + image_shape}
    for layer in model.layers[1:]:
        nodes      = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)
    return shape


# Estimate Shapes Based on Pyramid Levels
def guess_shapes(image_shape, pyramid_levels):
    image_shape  = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


# Generate Anchors for Given Shape
def anchors_for_shape(image_shape, pyramid_levels=None, anchor_params=None, shapes_callback=None):
    """
    image_shape:     The shape of the image.
    pyramid_levels:  List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
    anchor_params:   Struct containing anchor parameters. If None, default values are used.
    shapes_callback: Function to call for getting the shape of the image at different pyramid levels.
    """
    if pyramid_levels is None:    pyramid_levels  = [3, 4, 5, 6, 7]
    if anchor_params is None:     anchor_params   = AnchorParameters.default
    if shapes_callback is None:   shapes_callback = guess_shapes
    
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # Compute Anchors Over All Pyramid Levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=anchor_params.sizes[idx], ratios=anchor_params.ratios, scales=anchor_params.scales)
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors


# Produce Shifted Anchors Based on Map Shape and Stride Size
def shift(shape, stride, anchors):
    """
    shape  : Shape to shift the anchors over.
    stride : Stride to shift the anchors with over the shape.
    anchors: Anchors to apply at each location.
    """
    # Create Grid starting from Half Stride from TL Corner
    shift_x          = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y          = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts      = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    A           = anchors.shape[0]
    K           = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors


# Generate Reference Anchor Windows by Enumerating Aspect Ratios
def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:   ratios = AnchorParameters.default.ratios
    if scales is None:   scales = AnchorParameters.default.scales

    num_anchors    = len(ratios) * len(scales)
    anchors        = np.zeros((num_anchors, 4))                          # Initialize Output Anchors
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T     # Scale 'base_size'
    areas          = anchors[:, 2] * anchors[:, 3]                       # Compute Areas of Anchors

    # Correct for Ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # Transform from (x_ctr,y_ctr,w,h) -> (x1,y1,x2,y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


# Compute Bounding Box Regression Targets for Given Image
def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    anchor_widths  = anchors[:, 2]   - anchors[:, 0]
    anchor_heights = anchors[:, 3]   - anchors[:, 1]
    targets_dx1    = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1    = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2    = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2    = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    targets        = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets        = targets.T
    if (mean!=None)&(std!=None):   targets = (targets - mean) / std
    return targets
