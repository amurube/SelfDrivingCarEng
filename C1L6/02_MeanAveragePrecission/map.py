import copy
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_iou, check_results

def sort_based_on_prediction(preds):
    boxes = preds['boxes']
    classes = preds['classes']
    scores = preds['scores']
    preds_sorted = {}
    preds_sorted['boxes'] = []
    preds_sorted['classes'] = []
    preds_sorted['scores'] = []
    preds_sorted['filename'] = preds['filename']
    while len(scores)>0:
        m = np.argmax(scores)
        preds_sorted['boxes'].append(boxes.pop(m))
        preds_sorted['classes'].append(classes.pop(m))
        preds_sorted['scores'].append(scores.pop(m))
        
    return preds_sorted


if __name__ == '__main__':
    # load data 
    with open('data/predictions.json', 'r') as f:
        preds = json.load(f)

    with open('data/ground_truths.json', 'r') as f:
        gts = json.load(f)
    
    # TODO IMPLEMENT THIS SCRIPT
    preds_sorted = sort_based_on_prediction(preds[0])
    gts = gts[0]
    
    tps = 0
    fps = 0
    recall_denom = len(gts['classes'])
    
    precision = []
    recall = []
    
    # --------------- evaluate for each prediction the prediction and recall
    for pred_box, pred_class, pred_score in zip(preds_sorted['boxes'], preds_sorted['classes'], preds_sorted['scores']):
        for gt_box, gt_class in zip(gts['boxes'], gts['classes']):
            # calculate iou
            iou = calculate_iou(gt_box, pred_box)
            # is the iou > 0.5?
            if iou > 0.5:
                if pred_class == gt_class:
                    tps += 1
                else:
                    fps += 1
            
        # calculate Precision and Recall
        precision.append(tps / (tps + fps))
        recall.append(tps / recall_denom)
    
    
    # -------------- calculate the smoothed curve
    # keep first point
    new_max = precision[0]
    curve = []
    for i in range(len(precision)):
        if precision[i] < new_max:
            m = np.argmax(precision[i:])
            new_max = precision[m+i]
            curve.append([new_max, recall[i]])
        else:
            curve.append([new_max, recall[i]])
    
    

    # -------------- calculate the area under the smoothed curve
    mAP = 0
    
    smoothed = []
    smoothed.append([1, 0])
    smoothed.extend(curve)
    for i in range(len(smoothed)-1):
        if smoothed[i+1][0] == smoothed[i][0]:
            mAP += (smoothed[i+1][1] -  smoothed[i][1]) * smoothed[i][0]
        
    
    check_results(mAP)