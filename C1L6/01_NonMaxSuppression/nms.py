import json

from utils import calculate_iou, check_results
import numpy as np


def nms(predictions):
    """
    non max suppression
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    # IMPLEMENT THIS FUNCTION
    data = []
    for box, score in zip(predictions['boxes'], predictions['scores']):
        data.append([box,score])
        
    # pick the one box
    for bi in range(len(data)):
        #pick the next box     
        discard = False
        for bj in range(len(data)):
            if bi == bj:
                continue
            # calculate iou
            iou = calculate_iou(data[bi][0], data[bj][0])
            # if iou > 0.5
            if iou > 0.5:
                if data[bj][1] > data[bi][1]:
                    discard = True
                # keep the box with the highest probability
        if discard == False:
            filtered.append(data[bi])
    
    
    return filtered

def nms2(predictions):
    """
    non max suppression - to me more intuitive
    args:
    - predictions [dict]: predictions dict 
    returns:
    - filtered [list]: filtered bboxes and scores
    """
    filtered = []
    # IMPLEMENT THIS FUNCTION
    boxes = []
    scores = []
    for box, score in zip(predictions['boxes'], predictions['scores']):
        # remove all boxes that have a probability lower than 0.5
        if score >= 0.5:
            boxes.append(box)
            scores.append(score)
        
    # find the box with the largest score
    while len(scores) > 0:
        m = np.argmax(scores)
        b_m = boxes.pop(m)
        s_m = scores.pop(m)
        filtered.append([b_m, s_m])
        keep_boxes = []
        keep_scores = []
        # compare maximum prediction score with all other boxes
        for bi in range(len(scores)):
            iou = calculate_iou(b_m, boxes[bi])
            # if the iou is below the threshold it might be a different object and we need to keep it
            if iou < 0.5:
                keep_boxes.append(boxes[bi])
                keep_scores.append(scores[bi])
            
        boxes = keep_boxes
        scores = keep_scores

    return filtered


if __name__ == '__main__':
    with open('data/predictions_nms.json', 'r') as f:
        predictions = json.load(f)
    
    filtered = sms(predictions)
    check_results(filtered)