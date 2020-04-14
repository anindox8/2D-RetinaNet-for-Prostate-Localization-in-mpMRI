from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np


# Filter Detections using Bounding Boxes and Classification Values
def filter_detections(boxes, classification, other=[],
                      class_specific_filter = True,
                      nms                   = True,
                      score_threshold       = 0.05,
                      max_detections        = 300,
                      nms_threshold         = 0.5):
    """
    boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
    classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
    other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
    class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
    nms                   : Flag to enable/disable non maximum suppression.
    score_threshold       : Threshold used to prefilter the boxes with.
    max_detections        : Maximum number of detections to keep.
    nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
    """
    def _filter_detections(scores, labels):
        indices = tf.where(tf.keras.backend.greater(scores, score_threshold))  # Threshold Based on Score

        # Perform Non-Maximum Suppression (NMS)
        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = tf.keras.backend.gather(scores, indices)[:, 0]
            nms_indices     = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            indices         = tf.keras.backend.gather(indices, nms_indices)

        # Add Indices to List of All Indices
        labels  = tf.gather_nd(labels, indices)
        indices = tf.keras.backend.stack([indices[:, 0], labels], axis=1)
        return indices

    # Per Class Filtering
    if class_specific_filter:
        all_indices = []
        for c in range(int(classification.shape[1])):
            scores  = classification[:, c]
            labels  = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))
        indices = tf.keras.backend.concatenate(all_indices, axis=0)     # Concatenate to Single Tensor
    else:
        scores  = tf.keras.backend.max(classification,    axis = 1)
        labels  = tf.keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # Top-K Selection
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # Stop Gradients to Prevent Training Lower Layers
    scores              = tf.stop_gradient(scores)
    top_indices         = tf.stop_gradient(top_indices)

    # Filter Input using Final Set of Indices
    indices             = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes               = tf.keras.backend.gather(boxes, indices)
    labels              = tf.keras.backend.gather(labels, top_indices)
    other_              = [tf.keras.backend.gather(o, indices) for o in other]

    # Zero-Pad Outputs
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes    = tf.pad(boxes,  [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]],         constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]],         constant_values=-1)
    labels   = tf.keras.backend.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # Preset Shapes
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(tf.keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_




# TensorFlow Keras Layer for Filtering Detections using Score Threshold/NMS
class FilterDetections(tf.keras.layers.Layer):
    def __init__(self, nms=True, class_specific_filter=True, nms_threshold=0.5,
                 score_threshold=0.05, max_detections=300, parallel_iterations=32, **kwargs):
        """
        nms                   : Flag to enable/disable NMS.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    # Constructs NMS Graph
    def call(self, inputs, **kwargs):
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # Wrap NMS with Custom Parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(boxes, classification, other,
                                     nms                   = self.nms,
                                     class_specific_filter = self.class_specific_filter,
                                     score_threshold       = self.score_threshold,
                                     max_detections        = self.max_detections,
                                     nms_threshold         = self.nms_threshold)
        # Call 'filter_detections' on Each Batch
        outputs = tf.map_fn(_filter_detections, elems=[boxes, classification, other],
                            dtype=[tf.keras.backend.floatx(), tf.keras.backend.floatx(),'int32'] + [o.dtype for o in other],
                            parallel_iterations=self.parallel_iterations)
        return outputs

    # Compute Output Shapes for Given Input Shapes
    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections)] + [tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))]

    def compute_mask(self, inputs, mask=None):
        return (len(inputs)+1) * [None]

    def get_config(self):
        config = super(FilterDetections, self).get_config()
        config.update({'nms'                   : self.nms,
                       'class_specific_filter' : self.class_specific_filter,
                       'nms_threshold'         : self.nms_threshold,
                       'score_threshold'       : self.score_threshold,
                       'max_detections'        : self.max_detections,
                       'parallel_iterations'   : self.parallel_iterations})
        return config