import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    ## IMPLEMENT THIS FUNCTION
    #print('pred_bbox: {}', pred_bbox)
    #print('gt_bbox: {}', gt_bbox)

    x1 = 0  # 
    y1 = 1  # 
    x2 = 2  # 
    y2 = 3  # 
    
    calculate_iou = True
    iou = 0
    
    if calculate_iou:
        xi1 = max(pred_bbox[x1], gt_bbox[x1])
        xi2 = min(pred_bbox[x2], gt_bbox[x2])
        yi1 = max(pred_bbox[y1], gt_bbox[y1])
        yi2 = min(pred_bbox[y2], gt_bbox[y2])    

        intersection = max((xi2 - xi1),0) * max((yi2-yi1),0)
        #print('intersection: ' , intersection)

        pred_area = (pred_bbox[x2]-pred_bbox[x1]) * (pred_bbox[y2] - pred_bbox[y1])
        gt_area = (gt_bbox[x2]-gt_bbox[x1]) * (gt_bbox[y2] - gt_bbox[y1])     
        #print(pred_area, gt_area)
        union = pred_area + gt_area - intersection
        #print('union: ', union)

        iou = intersection / union
        #print(iou)
        #print('#####################')    
    return iou


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)