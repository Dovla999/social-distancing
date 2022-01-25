import json

import numpy as np
from mean_average_precision import MetricBuilder


def calculate_map():
    predictions = json.load(open("predictions.txt", 'r'))
    gt = json.load(open("ground_truth.txt", 'r'))

    def transform(v):
        for v_i in v:
            v_i[4] = 0
        return v

    predictions = {int(k): transform(v) for k, v in predictions.items()}
    gt = {int(k): transform(v) for k, v in gt.items()}

    m_ap = 0

    for i in range(len(gt)):
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
        for j in range(10):
            metric_fn.add(np.array(predictions[i]), np.array(gt[i]))
        m_ap += metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01),
                                mpolicy='soft')['mAP']
    m_ap /= len(gt)

    return m_ap


# calculate_map()
