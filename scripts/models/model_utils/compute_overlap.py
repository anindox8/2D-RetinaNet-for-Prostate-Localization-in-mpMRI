from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np


# Computes Overlap Between Two Patterns [Returns (N,K) array of overlap between 'boxes','query_boxes']
def compute_overlap(boxes,query_boxes):
    """
    boxes:       (N,4) ndarray of float64
    query_boxes: (K,4) ndarray of float64
    """
    N        = boxes.shape[0]
    K        = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        box_area = ((query_boxes[k,2] - query_boxes[k,0] + 1) * (query_boxes[k,3] - query_boxes[k,1] + 1))
        for n in range(N):
            iw = (min(boxes[n,2], query_boxes[k,2]) -
                  max(boxes[n,0], query_boxes[k,0]) + 1)
            if iw > 0:
                ih = (min(boxes[n,3], query_boxes[k,3]) -
                      max(boxes[n,1], query_boxes[k,1]) + 1)
                if ih > 0:
                    ua = np.float32((boxes[n, 2] - boxes[n, 0] + 1) *
                                    (boxes[n, 3] - boxes[n, 1] + 1) +
                                     box_area - iw * ih)
                    overlaps[n,k] = iw * ih / ua
    return overlaps