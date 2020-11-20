import glob
import os
import sys

import json
import argparse
import time

import cv2
from skimage import io
import numpy as np
from torch.backends import cudnn
from pprint import pprint

import craft_utils
import imgproc
import file_utils
import json
import zipfile
import torch

from craft import CRAFT
from collections import OrderedDict
from eval.script import *

# Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


class Rectangle:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def area(self):
        if self.xmax - self.xmin <= 0:
            return 0
        if self.ymax - self.ymin <= 0:
            return 0
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)

    pointMat = resBoxes[0].reshape([2, 4]).T

    return plg.Polygon(pointMat)


def rectangle_to_points(rect):
    points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin),
              int(rect.xmin), int(rect.ymin)]
    return points


def get_intersection_over_union(pD, pG):
    intersection = get_intersection(pD, pG)
    if intersection == 0.0:
        return 0
    else:
        return intersection / (pD.area() + pG.area() - intersection)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def get_intersection_over_union_rec(pD, pG):
    intersection = get_intersection_rec(pD, pG)
    if intersection == 0:
        return 0
    else:
        return intersection / (pD.area() + pG.area() - intersection)


def get_intersection_rec(pD, pG):
    IntRec = Rectangle(max(pD.xmin, pG.xmin), max(pD.ymin, pG.ymin), min(pD.xmax, pG.xmax), min(pD.ymax, pG.ymax))
    return IntRec.area()


def compute_ap(confList, matchList, numGtCare):
    correct = 0
    AP = 0
    if len(confList) > 0:
        confList = np.array(confList)
        matchList = np.array(matchList)
        sorted_ind = np.argsort(-confList)
        confList = confList[sorted_ind]
        matchList = matchList[sorted_ind]
        for n in range(len(confList)):
            match = matchList[n]
            if match:
                correct += 1
                AP += float(correct) / (n + 1)

        if numGtCare > 0:
            AP /= numGtCare

    return AP


'''
def custom_evaluate_method(gt, subm, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    for module, alias in evaluation_imports().items():
        globals()[alias] = importlib.import_module(module)

    perSampleMetrics = {}

    matchedSum = 0

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in gt:

        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""

        pointsList, _, transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
            gtFile, evaluationParams['CRLF'], evaluationParams['LTRB'], True, False
        )

        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = (transcription == "###")
            if evaluationParams['LTRB']:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        if resFile in subm:
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])
            pointsList, confidencesList, _ = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(
                detFile, evaluationParams['CRLF'], evaluationParams['LTRB'], False, evaluationParams['CONFIDENCES']
            )
            for n in range(len(pointsList)):
                points = pointsList[n]

                if evaluationParams['LTRB']:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area()
                        precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision > evaluationParams['AREA_PRECISION_CONSTRAINT']):
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += "DET polygons: " + str(len(detPols)) + (
                " (" + str(len(detDontCarePolsNum)) + " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 \
                                and detRectMat[detNum] == 0 \
                                and gtNum not in gtDontCarePolsNum \
                                and detNum not in detDontCarePolsNum:
                            if iouMat[gtNum, detNum] > evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                detMatched += 1
                                pairs.append({'gt': gtNum, 'det': detNum})
                                detMatchedNums.append(detNum)
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        # we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            sampleAP = precision
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare
            if evaluationParams['CONFIDENCES'] and evaluationParams['PER_SAMPLE_RESULTS']:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        if evaluationParams['PER_SAMPLE_RESULTS']:
            perSampleMetrics[resFile] = {
                'precision': precision,
                'recall': recall,
                'hmean': hmean,
                'pairs': pairs,
                'AP': sampleAP,
                'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
                'gtPolPoints': gtPolPoints,
                'detPolPoints': detPolPoints,
                'gtDontCare': gtDontCarePolsNum,
                'detDontCare': detDontCarePolsNum,
                'evaluationParams': evaluationParams,
                'evaluationLog': evaluationLog
            }

    # Compute MAP and MAR
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean, 'AP': AP}

    resDict = {'calculated': True, 'Message': '', 'method': methodMetrics, 'per_sample': perSampleMetrics}

    return resDict
'''


def parse_args():
    parser = argparse.ArgumentParser()

    # Model Architecture
    parser.add_argument('--ocr_type', choices=['normal', 'single_char', 'force_column'], default='normal',
                        help='ocr_type')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    # GT
    parser.add_argument('--ext', type=str, choices=['json', 'xml'], required=True)
    # input
    parser.add_argument('--input_path', type=str, required=True)
    # Threshold
    parser.add_argument('--iou_constraint', default=0.5, type=float)
    parser.add_argument('--per_sample', action='store_true')
    args = parser.parse_args()
    return args


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, ocr_type):

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().detach().numpy()
    score_link = y[0, :, :, 1].cpu().detach().numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link,
        text_threshold, link_threshold,
        low_text, poly, ocr_type
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    if ocr_type == 'single_char':
        boxes = craft_utils.cluster_sort(boxes)

    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def read_gt(gt_path, ext):
    bboxes = []
    if ext == 'json':
        json_data = json.load(open(gt_path, 'r', encoding='utf-8'))
        regions = json_data['regions']
        for region in regions:
            bbox = region['boundingBox']
            bbox = (bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
            bboxes.append(bbox)
    if ext == 'xml':
        # TODO xml parser
        pass
    return bboxes


def test(args):
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint {}'.format(args.trained_model))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = False

    net.eval()

    t = time.time()
    image_list = []
    image_list.extend(glob.glob(os.path.join(args.input_path, '*.jpg')))
    image_list.extend(glob.glob(os.path.join(args.input_path, '*.png')))
    image_list.extend(glob.glob(os.path.join(args.input_path, '*.tif')))

    gt_list = [os.path.splitext(img)[0] + '.' + args.ext for img in image_list]

    perSampleMetrics = {}

    matchedSum = 0

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    # load data
    for k, (image_path, gt_path) in enumerate(zip(image_list, gt_list)):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)

        detMatched = 0

        pairs = []

        # gtPols = []
        gtRects = []
        gt_bboxes = read_gt(gt_path, args.ext)
        for bbox in gt_bboxes:
            gtRect = Rectangle(*bbox)
            # gtPol = rectangle_to_polygon(gtRect)
            # gtPols.append(gtPol)
            gtRects.append(gtRect)

        # detPols = []
        detRects = []
        det_bboxes, polys, score_text = test_net(
            net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args.ocr_type
        )
        for bbox in det_bboxes:
            '''
            bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[2][1])
            delRect = Rectangle(*bbox)
            detPol = rectangle_to_polygon(delRect)
            '''
            bbox = (min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1]))
            # detPol = plg.Polygon(bbox)
            # detPols.append(detPol)
            detRect = Rectangle(*bbox)
            detRects.append(detRect)

        # if len(gtPols) > 0 and len(detPols) > 0:
        if len(gtRects) > 0 and len(detRects) > 0:
            '''
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        if iouMat[gtNum, detNum] > args.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})

            numGtCare = len(gtPols)
            numDetCare = len(detPols)
            '''
            # Calculate IoU and precision matrixs
            outputShape = [len(gtRects), len(detRects)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtRects), np.int8)
            detRectMat = np.zeros(len(detRects), np.int8)
            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    pG = gtRects[gtNum]
                    pD = detRects[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union_rec(pD, pG)

            for gtNum in range(len(gtRects)):
                for detNum in range(len(detRects)):
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        if iouMat[gtNum, detNum] > args.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})

            numGtCare = len(gtRects)
            numDetCare = len(detRects)
            if numGtCare == 0:
                recall = float(1)
                precision = float(0) if numDetCare > 0 else float(1)
            else:
                recall = float(detMatched) / numGtCare
                precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare

            hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

            matchedSum += detMatched
            numGlobalCareGt += numGtCare
            numGlobalCareDet += numDetCare

            if args.per_sample:
                perSampleMetrics[image_path] = {
                    'precision': precision,
                    'recall': recall,
                    'hmean': hmean,
                    'pairs': pairs,
                    # 'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
                }

    # Compute MAP and MAR
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    resDict = {'calculated': True, 'method': methodMetrics, 'per_sample': perSampleMetrics}

    print("elapsed time : {}s".format(time.time() - t))

    return resDict


if __name__ == '__main__':
    args = parse_args()
    with torch.no_grad():
        resDict = test(args)
    pprint(resDict)
