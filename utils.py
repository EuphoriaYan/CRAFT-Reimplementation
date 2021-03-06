# -*- coding: utf-8 -*-
import itertools
from functools import reduce

import numpy as np
import cv2
import math
import collections
from PIL import Image
from typing import List
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, MeanShift, OPTICS, Birch

""" auxilary functions """


# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, ocr_type):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    if ocr_type == 'force_column' or ocr_type == 'single_char':
        text_score_comb = text_score.copy()
    else:
        text_score_comb = np.clip(text_score + link_score, 0, 1)

    text_score_comb = text_score_comb.astype(np.uint8)
    # Image.fromarray(text_score_comb * 255).show()

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb,
        connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        # segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel1, iterations=1)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def getPoly_core(boxes, labels, mapper, linkmap):
    # configs
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 30 or h < 30:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2: continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = [warpCoord(Minv, (spp[0], spp[1]))]
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, ocr_type='normal'):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, ocr_type)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        # convert single char box to colume box
        if ocr_type == 'force_column':
            boxes = adjustColumeBoxes(boxes)
        polys = [None] * len(boxes)

    return boxes, polys


def adjustColumeBoxes(boxes: List, row_threshold=0.67, col_threshold=1.35, union_threshold=0.8):
    def cal_IoU(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        in_l = max(array1_r, array2_r) - min(array1_l, array2_l)
        # in_l = max(0, in_l)
        un_l = min(array1_r, array2_r) - max(array1_l, array2_l)
        un_l = max(0, un_l)
        return un_l / in_l

    def check_vertical_dis(array1, array2):
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        dis = max(array1_u, array2_u) - min(array1_d, array2_d)
        array2_ver = array2_d - array2_u
        if dis < array2_ver * col_threshold:
            return True
        else:
            return False

    def mergeArray(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        array1_area = (array1_r - array1_l) * (array1_d - array1_u)
        array2_area = (array2_r - array2_l) * (array2_d - array2_u)
        new_array_l = (array1_l * array1_area + array2_l * array2_area) / (array1_area + array2_area)
        new_array_r = (array1_r * array1_area + array2_r * array2_area) / (array1_area + array2_area)
        new_array_u = min(array1_u, array2_u)
        new_array_d = max(array1_d, array2_d)
        new_array = [[new_array_l, new_array_u], [new_array_r, new_array_u],
                     [new_array_r, new_array_d], [new_array_l, new_array_d]]
        new_array = np.array(new_array, dtype=np.float)
        return new_array

    def getArea(array1):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array1_area = (array1_r - array1_l) * (array1_d - array1_u)
        return array1_area

    def checkArrayUnion(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        arrayU_l = max(array1_l, array2_l)
        arrayU_r = min(array1_r, array2_r)
        arrayU_u = max(array1_u, array2_u)
        arrayU_d = min(array1_d, array2_d)
        if arrayU_r > arrayU_l and arrayU_d > arrayU_u:
            arrayU_area = (arrayU_r - arrayU_l) * (arrayU_d - arrayU_u)
            array1_area = (array1_r - array1_l) * (array1_d - array1_u)
            array2_area = (array2_r - array2_l) * (array2_d - array2_u)
            if arrayU_area > array1_area * union_threshold or arrayU_area > array2_area * union_threshold:
                return True
            else:
                return False
        else:
            return False

    def getArrayUnion(array1, array2):
        array1_l = array1[0, 0]
        array1_r = array1[1, 0]
        array1_u = array1[0, 1]
        array1_d = array1[2, 1]
        array2_l = array2[0, 0]
        array2_r = array2[1, 0]
        array2_u = array2[0, 1]
        array2_d = array2[2, 1]
        arrayU_l = max(array1_l, array2_l)
        arrayU_r = min(array1_r, array2_r)
        arrayU_u = max(array1_u, array2_u)
        arrayU_d = min(array1_d, array2_d)
        if arrayU_r > arrayU_l and arrayU_d > arrayU_u:
            arrayU_area = (arrayU_r - arrayU_l) * (arrayU_d - arrayU_u)
            return arrayU_area
        else:
            return 0

    # 上下合并box
    new_boxes = []
    vis = [False for _ in range(len(boxes))]
    for i in range(len(boxes)):
        if vis[i]:
            continue
        cur_box = boxes[i]
        flag = True
        while flag:
            flag = False
            for j in range(i + 1, len(boxes)):
                if vis[j]:
                    continue
                if cal_IoU(cur_box, boxes[j]) > row_threshold and check_vertical_dis(cur_box, boxes[j]):
                    vis[j] = True
                    cur_box = mergeArray(cur_box, boxes[j])
                    flag = True
        vis[i] = True
        new_boxes.append(cur_box)
    # 合并交集比例过大的box
    merge_boxes = []
    vis = [False for _ in range(len(new_boxes))]
    for i in range(len(new_boxes)):
        if vis[i]:
            continue
        cur_box = new_boxes[i]
        flag = True
        while flag:
            flag = False
            for j in range(i + 1, len(new_boxes)):
                if vis[j]:
                    continue
                if checkArrayUnion(cur_box, new_boxes[j]):
                    vis[j] = True
                    cur_box = mergeArray(cur_box, new_boxes[j])
                    flag = True
        vis[i] = True
        merge_boxes.append(cur_box)
    # 和其他所有box有交集且比例过大的删除
    final_boxes = []
    cover_area = [0 for _ in range(len(merge_boxes))]
    for i in range(len(merge_boxes)):
        for j in range(i + 1, len(merge_boxes)):
            if i == j:
                continue
            union_area = getArrayUnion(merge_boxes[i], merge_boxes[j])
            cover_area[i] += union_area
            cover_area[j] += union_area
    for i in range(len(merge_boxes)):
        if cover_area[i] <= getArea(merge_boxes[i]) * union_threshold:
            final_boxes.append(merge_boxes[i])
    return final_boxes


def projection_split(shape, boxes, type='DBSCAN'):
    width, height, channel = shape
    switch = {
        'DBSCAN': DBSCAN(min_samples=1, eps=15),
        'MeanShift': MeanShift(bandwidth=0.3),
        'OPTICS': OPTICS(min_samples=1, eps=20),
        'Birch': Birch(n_clusters=None)
    }
    cluster = switch[type]
    boxes_data = [(b['r'] + b['l']) / 2 for b in boxes]
    boxes_data = np.array(boxes_data).reshape((-1, 1))
    labels = cluster.fit_predict(boxes_data)

    plt.scatter(boxes_data[:, 0], boxes_data[:, 0], s=1, c=labels)
    plt.show()

    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)
    return classified_box_ids


def cluster_boxes(boxes, type='DBSCAN'):
    switch = {
        'DBSCAN': DBSCAN(min_samples=1, eps=7),
        'MeanShift': MeanShift(bandwidth=0.3),
        'OPTICS': OPTICS(min_samples=1, eps=20),
        'Birch': Birch(n_clusters=None)
    }
    cluster = switch[type]
    boxes_data = [[b['l'], b['r']] for b in boxes]
    boxes_data = np.array(boxes_data)
    labels = cluster.fit_predict(boxes_data)
    '''
    plt.scatter(boxes_data[:, 0], boxes_data[:, 1], s=1, c=labels)
    plt.show()
    '''
    classified_box_ids = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        classified_box_ids[label].append(idx)
    return classified_box_ids


def list_sort(box_list):
    r = [b['r'] for b in box_list]
    length = [b['r'] - b['l'] for b in box_list]
    r = np.mean(r)
    length = np.mean(length)
    return r + length


def box_sort(box):
    u = box['u']
    d = box['d']
    return (u + d) / 2


def convert_bbox_to_lrud(bbox):
    l = min(bbox[:, 0])
    r = max(bbox[:, 0])
    u = min(bbox[:, 1])
    d = max(bbox[:, 1])
    return l, r, u, d


def cluster_sort(shape, boxes):
    """
    :param boxes:
    :return: cluster then sorted boxes
        l = array[0, 0]
        r = array[1, 0]
        u = array[0, 1]
        d = array[2, 1]
    """
    boxes_lrud = []
    for id, box in enumerate(boxes):
        l, r, u, d = convert_bbox_to_lrud(box)
        boxes_lrud.append({'id': id, 'l': l, 'r': r, 'u': u, 'd': d})
    # boxes_lrud = [{'l': b[0, 0], 'r': b[1, 0], 'u': b[0, 1], 'd': b[2, 1], 'id': id} for id, b in enumerate(boxes)]
    '''
    classified_box_ids = projection_split(shape, boxes_lrud)
    classified_boxes = []
    for k in classified_box_ids.keys():
        box_ids = classified_box_ids[k]
        classified_boxes.append([boxes_lrud[box_id] for box_id in box_ids])
    '''
    classified_box_ids = cluster_boxes(boxes_lrud)
    classified_boxes = []
    for k in classified_box_ids.keys():
        box_ids = classified_box_ids[k]
        classified_boxes.append([boxes_lrud[box_id] for box_id in box_ids])
    classified_boxes = sorted(classified_boxes, key=list_sort, reverse=True)
    new_classifier_boxes = []
    for box_list in classified_boxes:
        new_classifier_boxes.append(sorted(box_list, key=box_sort, reverse=False))
    new_classifier_boxes = list(itertools.chain.from_iterable(new_classifier_boxes))
    new_classifier_boxes = [boxes[b['id']] for b in new_classifier_boxes]
    return new_classifier_boxes


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
